"""Streaming-path quality of a TensorRT transformer engine over many prompts: the 2-step path and the carry-over +
filter path at one or more filter blends, on clips played forward then backward (no cut, 96 frames by default).

Per path, clip and prompt: CLIP direction as a clip mean and per batch of 8 (the carry-over can fade where the motion
is fast, which the clip mean hides; the worst batch shows it), warping error, `metrics.flicker` (valid / low / high)
and Sobel. With --teacher the 2-step path is also compared with the 8-step teacher (PyTorch, DC-AE decoder, same init
noise) on frames 0, 4, ..., 28: LPIPS and CLIP direction. The default prompts are seven the multi-prompt student was
trained on and the four kept out (presets.HELD_OUT); --table reports the two groups apart.

Usage:
  python scripts/eval_stream.py --engine out/trt/vibe_multi_carry_s05_lut_2step_768_b8_int8.engine \
      --filters 0.7 0.3 0.85 0.5 --teacher --out out/eval_stream/multi_int8.json
  python scripts/eval_stream.py --table out/eval_stream/lut_int8.json out/eval_stream/multi_int8.json
"""
import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

import _paths  # noqa: F401
from vibe_stream.carry import Carry
from vibe_stream.metrics import LPIPS, CLIPScore, flicker, input_flows, read_video_frames, sobel, warping_error
from vibe_stream.presets import ALL_PROMPTS, CLIP_CAPTIONS, EVAL_CLIPS, EVAL_CLIPS_DIR, HELD_OUT
from vibe_stream.stream import StreamEditor
from vibe_stream.temporal_filter import TemporalFilter
from vibe_stream.trt_engine import TRTTransformer
from vibe_stream.trt_tae import TRTTinyAutoencoder
from vibe_stream.vibe_model import ROOT, VibeModel

PROMPTS = ("oil", "anime", "watercolor", "pencil", "ukiyoe", "winter", "night") + HELD_OUT
TEACHER_FRAMES = list(range(0, 32, 4))


def pingpong(frames, n):
    """The clip forward, then backward without repeating the end frames, until n frames."""
    seq = list(frames) + list(frames[-2:0:-1])
    while len(seq) < n:
        seq += seq
    return seq[:n]


def evaluate(a):
    engine = TRTTransformer(str(ROOT / a.engine))
    sp = engine.spec
    vibe = VibeModel(size=sp.size)
    tae = TRTTinyAutoencoder(str(ROOT / (a.tae_engine or f"out/trt/taesana_{sp.size}_b{sp.batch}")))
    clip = CLIPScore()
    clips = {c: pingpong(read_video_frames(ROOT / a.clips_dir / f"{c}.mp4", sp.size, a.frames), a.frames)
             for c in a.clips}
    flows = {c: input_flows(fr) for c, fr in clips.items()}
    res = dict(engine=a.engine, clips=a.clips, prompts=a.prompts, frames=a.frames, runs={})
    teacher = {}
    if a.teacher:
        lp = LPIPS()
        res["teacher"] = {}
        with torch.inference_mode():
            for c, fr in clips.items():
                lat = vibe.encode([fr[i] for i in TEACHER_FRAMES], chunk=2)
                for key in a.prompts:
                    teacher[c, key] = vibe.decode_u8(vibe.sample(lat, vibe.embed(ALL_PROMPTS[key], fr[0]), 8), chunk=2)
                    res["teacher"].setdefault(c, {})[key] = clip([fr[i] for i in TEACHER_FRAMES], teacher[c, key],
                                                                 *CLIP_CAPTIONS[key])["clip_dir"]
    # one editor per path kind (each has its own CUDA stream and buffers); the filter blends share the carry editor
    plain, carry = StreamEditor(vibe, tae, engine=engine), StreamEditor(vibe, tae, engine=engine, carry=Carry(0))
    paths = {"2step": (plain, None)}
    for low, high in zip(a.filters[::2], a.filters[1::2]):
        paths[f"carry + filter {low:g} / {high:g}"] = (carry, TemporalFilter(low, high))
    with torch.inference_mode():
        for name, (ed, tf) in paths.items():
            if tf is not None:
                ed.tfilter = tf
            run = res["runs"][name] = {}
            for c, fr in clips.items():
                for key in a.prompts:
                    ed.set_prompt(ALL_PROMPTS[key], fr[0])
                    outs = ed.run(fr, 0)[0]
                    per = [clip(fr[b:b + 8:2], outs[b:b + 8:2], *CLIP_CAPTIONS[key])["clip_dir"]
                           for b in range(0, len(fr), 8)]
                    r = dict(clip_dir=mean(per), clip_batches=per, warp=warping_error(fr, outs, flows[c]),
                             sobel=sobel(outs), **flicker(fr, outs, flows=flows[c]))
                    if teacher and name == "2step":
                        so = [outs[i] for i in TEACHER_FRAMES]
                        r["lpips_teacher"] = lp(so, teacher[c, key])
                        r["clip_dir_teacher_frames"] = clip([fr[i] for i in TEACHER_FRAMES], so,
                                                            *CLIP_CAPTIONS[key])["clip_dir"]
                    run.setdefault(c, {})[key] = r
            print(f"[eval] {name} done", flush=True)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    table([out])


def table(paths):
    cols = ("clip_dir", "worst", "valid", "low", "high", "warp", "sobel", "lpips_t")
    for p in paths:
        r = json.loads(Path(p).read_text())
        groups = {"trained": [k for k in r["prompts"] if k not in HELD_OUT],
                  "held out": [k for k in r["prompts"] if k in HELD_OUT]}
        print(f"\n{Path(r['engine']).stem}")
        print(f"{'path | prompts':<44}" + "".join(f"{c:>9}" for c in cols))
        for g, keys in groups.items():
            if not keys:
                continue
            if "teacher" in r:
                t = mean(r["teacher"][c][k] for c in r["clips"] for k in keys)
                print(f"{'8-step teacher (frames 0-28) | ' + g:<44}{t:9.3f}")
            for name, run in r["runs"].items():
                rows = [run[c][k] for c in r["clips"] for k in keys]
                vals = [mean(x["clip_dir"] for x in rows), mean(min(x["clip_batches"]) for x in rows)]
                vals += [mean(x[m] for x in rows) for m in ("valid", "low", "high", "warp", "sobel")]
                lpt = mean(x["lpips_teacher"] for x in rows) if "lpips_teacher" in rows[0] else float("nan")
                print(f"{name + ' | ' + g:<44}" + "".join(f"{v:9.3f}" if i < 2 else f"{v:9.2f}"
                                                         for i, v in enumerate(vals)) + f"{lpt:9.3f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", help="transformer engine (build_trt.py)")
    p.add_argument("--tae_engine", default=None, help="taesana engine prefix (default out/trt/taesana_<size>_b<batch>)")
    p.add_argument("--out", help="result JSON")
    p.add_argument("--clips", nargs="+", default=list(EVAL_CLIPS))
    p.add_argument("--clips_dir", default=EVAL_CLIPS_DIR)
    p.add_argument("--prompts", nargs="+", default=list(PROMPTS), help="keys of presets.ALL_PROMPTS")
    p.add_argument("--frames", type=int, default=96)
    p.add_argument("--filters", type=float, nargs="+", default=[0.7, 0.3], help="LOW HIGH pairs for the carry path")
    p.add_argument("--teacher", action="store_true", help="also compare the 2-step path with the 8-step teacher")
    p.add_argument("--table", nargs="+", metavar="JSON", help="print a table of earlier results and exit")
    a = p.parse_args()
    if a.table:
        table(a.table)
    elif a.engine and a.out:
        if len(a.filters) % 2:
            p.error("--filters takes LOW HIGH pairs")
        evaluate(a)
    else:
        p.error("give --engine and --out, or --table")


if __name__ == "__main__":
    main()
