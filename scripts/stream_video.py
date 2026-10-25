"""Stream a video file through a student: taesana encode -> k steps -> taesana decode, batched, on the GPU.

The instruction embedding is computed once from the first frame. Prints loop fps and step-only fps (defined in
vibe_stream.stream), the embedding time, peak VRAM, and the warping error, Sobel and edge agreement of the output.

Usage:
  python scripts/stream_video.py --input assets/davis_mp4/blackswan.mp4 --output out/stream/blackswan_oil.mp4 \
      --engine out/trt/vibe_lam4_2step_768_b8.engine --prompt oil
  python scripts/stream_video.py ... --student runs/lam4_2step/lora.safetensors      (PyTorch, no TensorRT)
  --loop_frames 480 repeats the clip for a sustained throughput figure; --compare writes input | output.
"""
import argparse
import json
from itertools import cycle, islice
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from PIL import Image

import _paths  # noqa: F401
from vibe_stream import lora
from vibe_stream.carry import Carry
from vibe_stream.metrics import edge_agreement, read_video_frames, sobel, video_fps, warping_error, write_video
from vibe_stream.presets import PROMPTS
from vibe_stream.stream import StreamEditor
from vibe_stream.taesana import TinyAutoencoder
from vibe_stream.temporal_filter import TemporalFilter
from vibe_stream.vibe_model import VibeModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--prompt", default="oil", help=f"an instruction, or a preset key: {', '.join(PROMPTS)}")
    p.add_argument("--style_image", default=None,
                   help="instead of --prompt: an image whose style VIBE's Qwen3-VL turns into the instruction")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--engine", help="TensorRT engine from scripts/build_trt.py")
    g.add_argument("--student", help="student LoRA file (PyTorch bf16 path)")
    p.add_argument("--batch", type=int, default=8, help="PyTorch path only; an engine fixes its batch")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--loop_frames", type=int, default=None, help="cycle the clip to this many frames")
    p.add_argument("--warmup_batches", type=int, default=2)
    p.add_argument("--compare", action="store_true", help="write input | output side by side")
    p.add_argument("--temporal_filter", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                   help="display-side temporal filter with these blend weights for the low and high band, e.g. 0.7 0.3")
    p.add_argument("--carry", action="store_true",
                   help="start batches after a keyframe from the previous output (vibe_stream.carry)")
    p.add_argument("--keyframe_every", type=int, default=0, help="with --carry: a keyframe every N batches (0: first only)")
    p.add_argument("--carry_sigma", type=float, default=None,
                   help="with --carry: noise level of the carried start "
                        "(default: the student's, as recorded in it or in the engine's spec)")
    p.add_argument("--carry_fill", choices=["frame", "lut"], default=None,
                   help="with --carry: fill where the flow fails with the frame or the frame through the keyframe's "
                        "colour map (default: the student's, as recorded in it or in the engine's spec)")
    p.add_argument("--out_json", default=None)
    a = p.parse_args()
    prompt = PROMPTS.get(a.prompt, a.prompt)

    if a.engine:
        from vibe_stream.trt_engine import TRTTransformer
        engine = TRTTransformer(a.engine)
        size, trained = engine.spec.size, engine.spec.prompts
    else:
        engine = None
        meta = lora.load(a.student)[1]
        size, trained = int(meta.get("size", 768)), meta.get("prompts", {})
    frames = read_video_frames(a.input, size, a.max_frames)
    print(f"[input] {a.input}: {len(frames)} frames at {size}", flush=True)
    vibe = VibeModel(size=size)
    if a.style_image:
        prompt = vibe.style_prompt(Image.open(a.style_image).convert("RGB"))
        print(f"[style] {a.style_image}: {prompt}", flush=True)
    if trained and prompt not in trained.values():
        print(f"[note] the student was trained on {list(trained.values())}; other instructions may come out weaker")
    tae = TinyAutoencoder()
    tf = TemporalFilter(*a.temporal_filter) if a.temporal_filter else None
    ed = StreamEditor(vibe, tae, engine=engine, student=a.student, batch_size=a.batch, temporal_filter=tf,
                      carry=Carry(a.keyframe_every, sigma=a.carry_sigma, fill=a.carry_fill) if a.carry else None)
    embed_ms = ed.set_prompt(prompt, frames[0])
    torch.cuda.reset_peak_memory_stats()
    run_frames = list(islice(cycle(frames), a.loop_frames)) if a.loop_frames else frames
    outs, tm = ed.run(run_frames, a.warmup_batches)
    outs = outs[:len(frames)]
    backend = f"TensorRT {engine.spec.tensorrt}" if engine else "PyTorch bf16"
    if tf is not None:
        backend += f", temporal filter {tf.low:g}/{tf.high:g}"
    if ed.carry is not None:
        backend += (f", carry-over ({ed.carry.keyframes} keyframe batches of {-(-len(run_frames) // ed.batch_size)}, "
                    f"fill {ed.carry_fill()})")
    print(f"[run] {backend}, {ed.steps}-step student, batch {ed.batch_size}, {len(run_frames)} frames", flush=True)
    print(f"[run] loop      {tm.loop_fps:6.1f} fps  ({tm.frames} frames in {tm.loop_s:.2f} s, steady state)")
    print(f"[run] step-only {tm.step_fps:6.1f} fps  ({mean(tm.step_ms) / ed.batch_size:.1f} ms per frame)")
    print(f"[run] embedding {embed_ms:.0f} ms once per clip; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB "
          f"(allocated by torch; the engine's own memory is not included)")
    res = dict(input=a.input, prompt=prompt, backend=backend, steps=ed.steps, batch=ed.batch_size, size=size,
               frames=len(run_frames), loop_fps=tm.loop_fps, step_fps=tm.step_fps, embed_ms=embed_ms,
               warp=warping_error(frames, outs), sobel=sobel(outs), sobel_input=sobel(frames),
               edge=mean(edge_agreement(x, y) for x, y in zip(frames[::4], outs[::4])))
    print(f"[quality] warp {res['warp']:.2f}  sobel {res['sobel']:.2f} (input {res['sobel_input']:.2f})  edge {res['edge']:.3f}")
    if a.output:
        vid = [np.concatenate([f, o], axis=1) for f, o in zip(frames, outs)] if a.compare else outs
        write_video(a.output, vid, video_fps(a.input))
        print(f"[out] {a.output}")
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
