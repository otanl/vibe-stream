"""Flicker of one student on the streaming path (taesana, PyTorch bf16, StreamEditor), with and without the temporal
filter, and its response to sub-token shifts.

Per clip and prompt the clip is streamed twice, plain and through `temporal_filter.TemporalFilter`. Metrics per run:
warping error, `metrics.flicker` (output change where the input's flow explains the input, and its low / high band
split), Sobel and CLIP directional similarity. The shift probe crops a wider first frame at horizontal offsets of
0.5-32 px, edits the crops as one batch and compares each output with the unshifted output moved by the same offset:
a shift-equivariant editor scores 0, while texture anchored to the screen (the shared init noise, the 32-pixel latent
grid) shows up already at sub-pixel offsets. Offsets of whole 32-pixel tokens are also run with the noise rolled by
the same number of tokens, which removes the noise's share and leaves the rest.

One student per run (the LoRA is merged into the weights); --table prints several result files side by side.

Usage:
  python scripts/eval_flicker.py --student runs/lam4_2step_640/lora.safetensors --out out/flicker/lam4_2step_640.json
  python scripts/eval_flicker.py --table out/flicker/lam4_2step_640.json out/flicker/lam4_eq4_2step_640.json
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
from vibe_stream.carry import Carry
from vibe_stream.metrics import CLIPScore, flicker, read_video_frames, sobel, warping_error, write_video
from vibe_stream.presets import CLIP_CAPTIONS, EVAL_CLIPS_DIR, PROMPTS
from vibe_stream.stream import StreamEditor
from vibe_stream.taesana import TinyAutoencoder
from vibe_stream.temporal_filter import TemporalFilter
from vibe_stream.vibe_model import ROOT, VibeModel

CLIPS = ("blackswan", "parkour", "dance-twirl", "camel", "goat", "libby")
PROBE_CLIPS = ("blackswan", "camel", "parkour")
SHIFTS = (0.5, 1, 2, 4, 8, 16, 32)


def wide_first_frame(path, size):
    """First frame scaled to height `size`, full width (wider than the square crop the editor sees)."""
    cap = cv2.VideoCapture(str(path))
    ok, bgr = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    s = size / rgb.shape[0]
    return cv2.resize(rgb, (round(rgb.shape[1] * s), size), interpolation=cv2.INTER_AREA)


def shift_probe(ed, clips_dir, size, prompts):
    """Mean |edit(crop at +d) - edit(crop at 0) moved by d| per offset d (0-255, interior). Keys "<d>" use the fixed
    noise; "<d> rolled" are whole-token offsets with the noise rolled by the same number of tokens (what an editor
    whose noise moved with the content would see)."""
    res = {}
    m = size * 3 // 40                                                    # margin left out of the comparison
    whole = (32, 64, 96)

    def compare(o0, od, d, key):
        ref = cv2.warpAffine(o0, np.float32([[1, 0, d], [0, 1, 0]]), (size, size),
                             flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        diff = np.abs(od.astype(np.float32) - ref.astype(np.float32)).mean(2)
        res.setdefault(key, []).append(float(diff[m:size - m, m:size - m - int(np.ceil(d))].mean()))

    noise = ed.noise
    for clip in PROBE_CLIPS:
        wf = wide_first_frame(Path(clips_dir) / f"{clip}.mp4", size)
        cx, cy = wf.shape[1] / 2 - 0.5, size / 2 - 0.5
        crop = lambda d: cv2.getRectSubPix(wf, (size, size), (cx + d, cy))  # noqa: E731
        sub = np.stack([crop(d) for d in (0,) + SHIFTS])
        tok = np.stack([crop(d) for d in (0,) + whole] * 2)
        for prompt in prompts.values():
            ed.set_prompt(prompt, sub[0])
            o = ed.edit(torch.from_numpy(sub).to(ed.device)).cpu().numpy()
            for d, od in zip(SHIFTS, o[1:]):
                compare(o[0], od, d, str(d))
            # first half: noise rolled with the content (crop at +d sees the noise at +d / 32 tokens); second: fixed
            ed.noise = torch.cat([torch.roll(noise, -d // 32, dims=3) for d in (0,) + whole] + [noise] * (len(whole) + 1))
            try:
                o = ed.edit(torch.from_numpy(tok).to(ed.device)).cpu().numpy()
            finally:
                ed.noise = noise
            for j, d in enumerate(whole, 1):
                compare(o[0], o[j], d, f"{d} rolled")
                if d not in SHIFTS:
                    compare(o[0], o[len(whole) + 1 + j], d, str(d))
    order = [str(d) for d in SHIFTS + whole[1:]] + [f"{d} rolled" for d in whole]
    return {k: mean(res[k]) for k in order}


def evaluate(a):
    _, meta = lora.load(a.student)
    size = int(meta["size"])
    prompts = {k: v for k, v in meta.get("prompts", PROMPTS).items() if k in CLIP_CAPTIONS}
    vibe = VibeModel(size=size)
    carry = Carry(a.keyframe_every, tau=a.carry_tau, sigma=a.carry_sigma, fill=a.carry_fill) if a.carry else None
    ed = StreamEditor(vibe, TinyAutoencoder(), student=a.student, carry=carry)
    frames = {c: read_video_frames(Path(a.clips_dir) / f"{c}.mp4", size, a.frames) for c in a.clips}
    res, vids = {"student": str(a.student), "filter": a.filter, "frames": a.frames, "clips": list(a.clips),
                 "carry": a.keyframe_every if a.carry else None, "carry_fill": ed.carry_fill() if a.carry else None}, {}
    for c, fr in frames.items():
        for key, prompt in prompts.items():
            ed.set_prompt(prompt, fr[0])
            for name, tf in (("plain", None), ("filter", TemporalFilter(*a.filter))):
                ed.tfilter = tf
                outs, _ = ed.run(fr, warmup_batches=0)
                res[f"{name}|{c}|{key}"] = dict(warp=warping_error(fr, outs), **flicker(fr, outs), sobel=sobel(outs))
                vids[(c, key, name)] = outs
            ed.tfilter = None
        print(f"[flicker] {c}: " + "  ".join(f"{n}/{k} {res[f'{n}|{c}|{k}']['valid']:.1f}"
                                            for k in prompts for n in ("plain", "filter")), flush=True)
    res["shift_probe"] = shift_probe(ed, a.clips_dir, size, prompts)
    vibe.offload_text_encoder()
    torch.cuda.empty_cache()
    clip = CLIPScore()
    for (c, key, name), outs in vids.items():
        res[f"{name}|{c}|{key}"].update(clip(frames[c], outs, *CLIP_CAPTIONS[key]))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    if a.videos:
        for c in a.clips:
            for key in prompts:
                write_video(out.with_suffix("") / f"{c}_{key}.mp4",
                            [np.concatenate([frames[c][t], vids[(c, key, "plain")][t], vids[(c, key, "filter")][t]], 1)
                             for t in range(len(frames[c]))], 24.0)
    table([out])


def table(paths):
    cols = ("warp", "valid", "low", "high", "sobel", "clip_dir")
    print(f"\n{'student | variant | prompt':<46}" + "".join(f"{c:>9}" for c in cols))
    for p in paths:
        r = json.loads(Path(p).read_text())
        keys = sorted({k.split("|")[2] for k in r if k.count("|") == 2})
        for key in keys:
            for v in ("plain", "filter"):
                rows = [r[f"{v}|{c}|{key}"] for c in r["clips"]]
                print(f"{Path(p).stem[:28] + ' | ' + v + ' | ' + key:<46}"
                      + "".join(f"{mean(x[m] for x in rows):9.{3 if m == 'clip_dir' else 2}f}" for m in cols))
        print(f"{Path(p).stem[:28]} shift probe (px): " + "  ".join(f"{d}: {v:.1f}" for d, v in r["shift_probe"].items()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--student", help="student LoRA file")
    p.add_argument("--out", help="result JSON (videos, with --videos, in the directory of the same name)")
    p.add_argument("--clips", nargs="+", default=list(CLIPS))
    p.add_argument("--clips_dir", default=str(ROOT / EVAL_CLIPS_DIR))
    p.add_argument("--frames", type=int, default=40)
    p.add_argument("--filter", type=float, nargs=2, default=(0.7, 0.3), metavar=("LOW", "HIGH"))
    p.add_argument("--carry", action="store_true", help="stream with the carry-over (vibe_stream.carry)")
    p.add_argument("--keyframe_every", type=int, default=0, help="with --carry: a keyframe every N batches (0: first only)")
    p.add_argument("--carry_sigma", type=float, default=None,
                   help="with --carry: noise level of the carried start "
                        "(default: the student's, as recorded in it or in the engine's spec)")
    p.add_argument("--carry_fill", choices=["frame", "lut"], default=None,
                   help="with --carry: fill where the flow fails with the frame or the frame through the keyframe's "
                        "colour map (default: the student's, as recorded in it or in the engine's spec)")
    p.add_argument("--carry_tau", type=float, default=12.0,
                   help="with --carry: tolerance of the flow check, in 8-bit levels (Carry.tau)")
    p.add_argument("--videos", action="store_true", help="also write input | plain | filtered per clip and prompt")
    p.add_argument("--table", nargs="+", metavar="JSON", help="print a table of earlier results and exit")
    a = p.parse_args()
    if a.table:
        table(a.table)
    elif a.student and a.out:
        evaluate(a)
    else:
        p.error("give --student and --out, or --table")


if __name__ == "__main__":
    main()
