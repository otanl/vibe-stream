"""Score students against VIBE's K-step teacher on held-out clips (all through DC-AE, PyTorch bf16).

For each clip and prompt (the prompts stored in the first student file, else presets.PROMPTS): the teacher (K steps
of VIBE's scheduler), optionally the base model run as a 1- and 2-step student, and each student. Metrics per row:
warping error, LPIPS to the teacher, Sobel, edge agreement with the input and, for prompts with captions in
presets.CLIP_CAPTIONS, the CLIP directional similarity and output-caption similarity (see vibe_stream.metrics).
Writes <out>/eval.json and a strip of four frames per row.

Usage: python scripts/eval_student.py --out out/eval --runs runs/lam4_2step runs/lam4 --base
"""
import argparse
import json
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch

import _paths  # noqa: F401
from vibe_stream import lora
from vibe_stream.metrics import LPIPS, CLIPScore, edge_agreement, read_video_frames, sobel, warping_error
from vibe_stream.presets import CLIP_CAPTIONS, EVAL_CLIPS, EVAL_CLIPS_DIR, PROMPTS
from vibe_stream.vibe_model import ROOT, VibeModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--runs", nargs="*", default=[])
    p.add_argument("--clips", nargs="+", default=list(EVAL_CLIPS))
    p.add_argument("--clips_dir", default=str(ROOT / EVAL_CLIPS_DIR))
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--teacher_steps", type=int, default=8)
    p.add_argument("--base", action="store_true", help="also score the base model as a 1- and 2-step student")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = lora.load(Path(a.runs[0]) / "lora.safetensors")[1].get("prompts", PROMPTS) if a.runs else PROMPTS
    m = VibeModel(size=a.size)
    lp = LPIPS()
    clips = {c: read_video_frames(Path(a.clips_dir) / f"{c}.mp4", a.size, a.frames) for c in a.clips}
    cases = {}
    with torch.inference_mode():
        for c, fr in clips.items():
            il = m.encode(fr)
            for key, prompt in prompts.items():
                emb = m.embed(prompt, fr[0])
                cases[(c, key)] = dict(il=il, emb=emb, teacher=m.decode_u8(m.sample(il, emb, a.teacher_steps)))
    m.offload_text_encoder()
    clip = CLIPScore()
    res = {}

    def score(label, outputs_of):
        for (c, key), cs in cases.items():
            outs = list(outputs_of(cs))
            fr = clips[c]
            r = dict(warp=warping_error(fr, outs), lpips_to_teacher=lp(outs, cs["teacher"]), sobel=sobel(outs),
                     edge=mean(edge_agreement(x, y) for x, y in zip(fr[::4], outs[::4])))
            if key in CLIP_CAPTIONS:
                r.update(clip(fr, outs, *CLIP_CAPTIONS[key]))
            res[f"{label} | {c} | {key}"] = r
            cl = f"  clip_dir {r['clip_dir']:.3f}  clip_t {r['clip_t']:.3f}" if "clip_dir" in r else ""
            print(f"[eval] {label:<28} {c:<9} {key:<6} warp {r['warp']:6.2f}  lpips->teacher {r['lpips_to_teacher']:.3f}  "
                  f"sobel {r['sobel']:6.2f}  edge {r['edge']:.3f}{cl}", flush=True)
            strip = np.concatenate([cv2.resize(o, (256, 256)) for o in outs[:8:2]], axis=1)
            cv2.imwrite(str(out / f"strip_{label.replace(' ', '_')}_{c}_{key}.jpg"), cv2.cvtColor(strip, cv2.COLOR_RGB2BGR))

    def student_outputs(k):
        return lambda cs: m.decode_u8(torch.cat([m.student(cs["il"][i:i + 8], cs["emb"], k)
                                                 for i in range(0, len(cs["il"]), 8)]))

    score(f"teacher {a.teacher_steps} steps", lambda cs: cs["teacher"])
    if a.base:
        with torch.inference_mode():
            score("base 2 steps", student_outputs(2))
            score("base 1 step", student_outputs(1))
    for rp in a.runs:
        state, meta = lora.load(Path(rp) / "lora.safetensors")
        if lora.NOISE_KEY in state and not torch.equal(state[lora.NOISE_KEY].to(m.device), m.noise):
            raise SystemExit(f"{rp} was trained on a different init noise than the teacher here")
        wrappers = lora.attach(m.tr, int(meta["rank"]))
        lora.load_into(wrappers, state)
        k = int(meta.get("student_steps", 1))
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            score(f"{Path(rp).name} ({k} step)", student_outputs(k))
        lora.detach(m.tr, wrappers)
    (out / "eval.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    labels = list(dict.fromkeys(k.split(" | ")[0] for k in res))
    print("\n[eval] mean over clips and prompts (clip_dir per prompt)")
    for lab in labels:
        rows = {k: v for k, v in res.items() if k.startswith(lab + " | ")}
        vals = list(rows.values())
        per_prompt = "  ".join(f"{key} {mean(v['clip_dir'] for k, v in rows.items() if k.endswith(' | ' + key)):.3f}"
                               for key in prompts if key in CLIP_CAPTIONS)
        print(f"  {lab:<28} warp {mean(r['warp'] for r in vals):6.2f}  lpips->teacher {mean(r['lpips_to_teacher'] for r in vals):.3f}  "
              f"sobel {mean(r['sobel'] for r in vals):6.2f}  edge {mean(r['edge'] for r in vals):.3f}  clip_dir {per_prompt}")


if __name__ == "__main__":
    main()
