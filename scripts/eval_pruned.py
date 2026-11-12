"""Does retraining heal a student with transformer blocks removed? (README, "Removing transformer blocks")

On the streaming path (StreamEditor, carry-over + temporal filter, student merged in PyTorch bf16, taesana engines),
the 4 evaluation clips x --keys, --frames frames:
  full       the original student on the full transformer (the reference)
  removed    the original student with --blocks removed (its LoRA entries for them dropped), no retraining
  healed     --healed, a student trained with those blocks removed (train_student.py --remove_blocks)
Each is scored against the reference (LPIPS, PSNR) and on its own (CLIP direction, flicker). Writes --out (JSON) and a
sheet next to it (the first key on the first clip, a late frame: full | removed | healed).

Usage: python scripts/eval_pruned.py --student runs/klein_sav_vnight_cont_768/lora_640.safetensors \
           --healed runs/klein640_prune14_18/lora.safetensors --blocks 14,18 --tae_engine out/trt/taesana_640_b8 \
           --keys oil anime night ukiyoe gouache crayon fog --out out/eval_pruned/klein640.json
"""
import argparse
import gc
import json
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch

import _paths  # noqa: F401
from vibe_stream import lora
from vibe_stream.carry import Carry
from vibe_stream.metrics import LPIPS, CLIPScore, flicker, read_video_frames
from vibe_stream.presets import ALL_PROMPTS, CLIP_CAPTIONS, EVAL_CLIPS, EVAL_CLIPS_DIR
from vibe_stream.stream import StreamEditor
from vibe_stream.temporal_filter import TemporalFilter
from vibe_stream.trt_tae import TRTTinyAutoencoder
from vibe_stream.vibe_model import ROOT, VibeModel


def psnr(a, b):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float(10 * np.log10(255 ** 2 / max(mse, 1e-6)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--student", required=True, help="the full student")
    ap.add_argument("--healed", required=True, help="the student trained with --blocks removed")
    ap.add_argument("--blocks", required=True, type=lambda s: [int(x) for x in s.split(",")])
    ap.add_argument("--tae_engine", required=True)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--filter", type=float, nargs=2, default=[0.7, 0.3])
    ap.add_argument("--keys", nargs="+", default=["oil", "anime", "night"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    size = int(lora.load(a.student)[1]["size"])
    tae = TRTTinyAutoencoder(a.tae_engine)
    lp, clip = LPIPS(), CLIPScore()
    clips = {c: read_video_frames(ROOT / EVAL_CLIPS_DIR / f"{c}.mp4", size, a.frames) for c in EVAL_CLIPS}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)

    def run(student, removed):
        """Outputs {(clip, key): frames} of one student, on a freshly loaded model."""
        vibe = VibeModel(size=size, remove_blocks=a.blocks if removed else ())
        if removed and "removed_blocks" not in lora.load(student)[1]:
            state, meta = lora.load(student)      # the full student: its entries for the removed blocks dropped
            kept = {k: v for k, v in state.items()
                    if not any(k.startswith(f"transformer_blocks.{i}.") for i in a.blocks)}
            student = str(Path(a.out).with_suffix(".tmp.safetensors"))
            lora.save(student, kept, meta)
        ed = StreamEditor(vibe, tae, student=student, temporal_filter=TemporalFilter(*a.filter), carry=Carry())
        outs = {}
        for c, fr in clips.items():
            for k in a.keys:
                ed.set_prompt(ALL_PROMPTS[k], fr[0])
                with torch.inference_mode():
                    outs[(c, k)] = ed.run(fr, 0)[0]
        del ed, vibe
        gc.collect()
        torch.cuda.empty_cache()
        return outs

    def score(outs, ref):
        def cd(c, k):
            return mean(clip(clips[c][b:b + 8:2], outs[(c, k)][b:b + 8:2], *CLIP_CAPTIONS[k])["clip_dir"]
                        for b in range(0, a.frames, 8))
        r = dict(clip_dir=mean(cd(c, k) for c, k in outs),
                 flicker=mean(flicker(clips[c], outs[(c, k)])["valid"] for c, k in outs),
                 clip_by_key={k: mean(cd(c, k) for c in clips) for k in a.keys})
        if ref is not None:
            r["lpips"] = mean(lp(outs[ck], ref[ck]) for ck in outs)
            r["psnr"] = mean(psnr(np.stack(outs[ck]), np.stack(ref[ck])) for ck in outs)
            r["lpips_by_key"] = {k: mean(lp(outs[(c, k)], ref[(c, k)]) for c in clips) for k in a.keys}
        return r

    ref = run(a.student, False)
    res = dict(blocks=a.blocks, student=a.student, healed=a.healed, full=score(ref, None))
    print(f"[ep] full: clip_dir {res['full']['clip_dir']:.4f} flicker {res['full']['flicker']:.2f}", flush=True)
    sheet = [ref[(EVAL_CLIPS[0], a.keys[0])][a.frames - 4]]
    for name, student in (("removed", a.student), ("healed", a.healed)):
        outs = run(student, True)
        r = score(outs, ref)
        res[name] = r
        print(f"[ep] {name}: LPIPS {r['lpips']:.3f} PSNR {r['psnr']:.1f} clip_dir {r['clip_dir'] - res['full']['clip_dir']:+.4f} "
              f"flicker {r['flicker'] - res['full']['flicker']:+.2f}  LPIPS by key "
              f"{ {k: round(v, 3) for k, v in r['lpips_by_key'].items()} }", flush=True)
        sheet.append(outs[(EVAL_CLIPS[0], a.keys[0])][a.frames - 4])
        Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    tiles = [cv2.resize(x, (384, 384), interpolation=cv2.INTER_AREA) for x in sheet]
    cv2.imwrite(str(Path(a.out).with_suffix(".jpg")), cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


if __name__ == "__main__":
    main()
