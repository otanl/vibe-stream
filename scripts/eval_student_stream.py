"""One student through the PyTorch streaming path at its own size: CLIP direction (and its worst batch), flicker,
warping error, Sobel and edge agreement, on the 2-step path and with carry-over + filter.

The 4 evaluation clips played forward then backward (--frames frames, no cut), --keys (by default 4 trained and 2 held
out). The student's size and removed transformer blocks come from its metadata (--remove_blocks for a file that does
not record them). A student with self-attention state carried across frames (vibe_stream.state_attn, Negative results)
gets it unless --no_state. Writes <out_dir>/<name>.json and, per clip x key, the carry-over + filter output on the frame
of largest motion.

Usage: python scripts/eval_student_stream.py --name klein_sav2 --student runs/klein_sav_vnight_cont_768/lora.safetensors
"""
import argparse
import copy
import json
from pathlib import Path
from statistics import mean

import cv2
import numpy as np
import torch

import _paths  # noqa: F401
from vibe_stream import lora, state_attn
from vibe_stream.carry import Carry
from vibe_stream.metrics import CLIPScore, edge_agreement, flicker, input_flows, read_video_frames, sobel, warping_error
from vibe_stream.presets import ALL_PROMPTS, CLIP_CAPTIONS, EVAL_CLIPS, EVAL_CLIPS_DIR
from vibe_stream.stream import StreamEditor
from vibe_stream.temporal_filter import TemporalFilter
from vibe_stream.trt_tae import TRTTinyAutoencoder
from vibe_stream.vibe_model import ROOT, VibeModel, student_model_args

KEYS = ("oil", "anime", "ukiyoe", "night", "gouache", "autumn")


def pingpong(frames, n):
    seq = list(frames) + list(frames[-2:0:-1])
    while len(seq) < n:
        seq += seq
    return seq[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--student", required=True)
    ap.add_argument("--keys", nargs="+", default=list(KEYS))
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--filter", type=float, nargs=2, default=[0.85, 0.5])
    ap.add_argument("--tae_engine", default=None, help="default: out/trt/taesana_<size>_b8")
    ap.add_argument("--remove_blocks", type=lambda s: [int(x) for x in s.split(",")], default=None)
    ap.add_argument("--no_state", action="store_true")
    ap.add_argument("--out_dir", default=str(ROOT / "out/eval_student_stream"))
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state, meta = lora.load(a.student)
    args = student_model_args(meta)
    if a.remove_blocks is not None:
        args["remove_blocks"] = a.remove_blocks
    size = args["size"]
    vibe = VibeModel(**args)
    tae = TRTTinyAutoencoder(a.tae_engine or str(ROOT / f"out/trt/taesana_{size}_b8"))
    plain = StreamEditor(vibe, tae, student=a.student)
    sc = None
    if state_attn.ALPHA_KEY in state and not a.no_state:
        sc = state_attn.StateCarry(vibe.tr, state[state_attn.ALPHA_KEY].float(), learn=False)
        velocity = vibe.velocity

        def keyed(x, il, emb, t):
            sc.slot = round(float(t.flatten()[0]), 1)
            return velocity(x, il, emb, t)
        vibe.velocity = keyed
        print(f"[eval] {a.name}: state carry, alpha {[round(v, 2) for v in sc.alphas().tolist()]}", flush=True)
    cf = copy.copy(plain)
    cf.tfilter, cf.carry, cf.stream = TemporalFilter(*a.filter), Carry(0), torch.cuda.Stream()
    clip = CLIPScore()
    clips = {c: pingpong(read_video_frames(ROOT / EVAL_CLIPS_DIR / f"{c}.mp4", size, a.frames), a.frames)
             for c in EVAL_CLIPS}
    flows = {c: input_flows(fr) for c, fr in clips.items()}
    fast = {c: 8 + int(np.argmax([np.linalg.norm(f, axis=2).mean() for f in flows[c][8:]])) for c in clips}
    res = {}
    for pname, ed in (("2step", plain), ("carry_filter", cf)):
        for c, fr in clips.items():
            for key in a.keys:
                if sc is not None:
                    sc.reset()
                ed.set_prompt(ALL_PROMPTS[key], fr[0])
                with torch.inference_mode():
                    o = ed.run(fr, 0)[0]
                per = [clip(fr[b:b + 8:2], o[b:b + 8:2], *CLIP_CAPTIONS[key])["clip_dir"] for b in range(0, len(fr), 8)]
                r = dict(clip_dir=mean(per), clip_batches=per, sobel=sobel(o), warp=warping_error(fr, o, flows[c]),
                         e_now=mean(edge_agreement(fr[t], o[t]) for t in range(2, len(fr), 3)),
                         e_old=mean(edge_agreement(fr[t - 2], o[t]) for t in range(2, len(fr), 3)),
                         **flicker(fr, o, flows=flows[c]))
                res.setdefault(pname, {}).setdefault(c, {})[key] = r
                if pname == "carry_filter":
                    cv2.imwrite(str(out_dir / f"{a.name}_{c}_{key}_f{fast[c]}.jpg"),
                                cv2.cvtColor(o[fast[c]], cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])
        rows = [res[pname][c][k] for c in res[pname] for k in a.keys]
        print(f"[eval] {a.name} {pname:<13} clip_dir {mean(x['clip_dir'] for x in rows):.3f} worst "
              f"{mean(min(x['clip_batches']) for x in rows):.3f}  flicker {mean(x['valid'] for x in rows):.2f} (low "
              f"{mean(x['low'] for x in rows):.2f} high {mean(x['high'] for x in rows):.2f})  warp "
              f"{mean(x['warp'] for x in rows):.2f}  sobel {mean(x['sobel'] for x in rows):.2f}  edge "
              f"{mean(x['e_now'] for x in rows):.3f} / t-2 {mean(x['e_old'] for x in rows):.3f}", flush=True)
    (out_dir / f"{a.name}.json").write_text(json.dumps(dict(student=a.student, fast=fast, res=res,
                                                            alpha=None if sc is None else sc.alphas().tolist()), indent=1))


if __name__ == "__main__":
    main()
