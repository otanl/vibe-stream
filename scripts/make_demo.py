"""Demo video through the real-time path: input | one panel per prompt, clip after clip.

Each clip is streamed once per prompt exactly as scripts/stream_video.py does it (embedding from the clip's first
frame, batched taesana encode -> k steps -> taesana decode). The input panel names the clip, the student, the backend
and the GPU; each output panel its prompt key and the loop fps of its run (short clips: a few batches after one
warm-up batch, so a rougher figure than stream_video.py --loop_frames). Writes <out>/<clip>.mp4 per clip and
<out>/demo.mp4 with every clip in sequence; H.264 through ffmpeg when it is on PATH, else mp4v.

Usage: python scripts/make_demo.py --engine out/trt/vibe_lam4_2step_640_b8.engine --out out/demo_640
"""
import argparse
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import torch

import _paths  # noqa: F401
from vibe_stream import lora
from vibe_stream.carry import Carry
from vibe_stream.metrics import read_video_frames, video_fps, write_video
from vibe_stream.presets import EVAL_CLIPS_DIR, PROMPTS
from vibe_stream.stream import StreamEditor
from vibe_stream.taesana import TinyAutoencoder
from vibe_stream.temporal_filter import TemporalFilter
from vibe_stream.vibe_model import ROOT, VibeModel


def label(img: np.ndarray, text: str, bottom: bool = False) -> np.ndarray:
    """img with `text` (lines separated by newlines) in white on a black box, top-left (or bottom-left)."""
    img = img.copy()
    scale = img.shape[0] / 1200
    lines = text.split("\n")
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, scale, 1) for t in lines]
    lh = max(h + base for (_, h), base in sizes) + 6
    bw, bh = max(w for (w, _), _ in sizes) + 16, lh * len(lines) + 8
    y0 = img.shape[0] - bh if bottom else 0
    cv2.rectangle(img, (0, y0), (bw, y0 + bh), (0, 0, 0), -1)
    for i, (t, ((_, h), _)) in enumerate(zip(lines, sizes)):
        cv2.putText(img, t, (8, y0 + i * lh + h + 7), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def save(path: Path, frames, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        write_video(path, frames, fps)
        return
    h, w = frames[0].shape[:2]
    cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    if proc.wait():
        raise RuntimeError(f"ffmpeg failed for {path}")


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--engine", help="TensorRT engine from scripts/build_trt.py")
    g.add_argument("--student", help="student LoRA file (PyTorch bf16 path)")
    p.add_argument("--out", required=True)
    p.add_argument("--clips_dir", default=str(ROOT / EVAL_CLIPS_DIR))
    p.add_argument("--clips", nargs="*", default=None, help="clip names (default: every mp4 in --clips_dir)")
    p.add_argument("--prompts", nargs="+", default=list(PROMPTS), help="preset keys")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--temporal_filter", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                   help="display-side temporal filter (vibe_stream.temporal_filter), e.g. 0.7 0.3")
    p.add_argument("--carry", action="store_true",
                   help="start batches after a keyframe from the previous output (vibe_stream.carry)")
    p.add_argument("--keyframe_every", type=int, default=0, help="with --carry: a keyframe every N batches (0: first only)")
    p.add_argument("--carry_sigma", type=float, default=None,
                   help="with --carry: noise level of the carried start "
                        "(default: the student's, as recorded in it or in the engine's spec)")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if a.engine:
        from vibe_stream.trt_engine import TRTTransformer
        engine = TRTTransformer(a.engine)
        size, backend = engine.spec.size, "TensorRT"
    else:
        engine = None
        size, backend = int(lora.load(a.student)[1].get("size", 768)), "PyTorch"
    vibe = VibeModel(size=size)
    tf = TemporalFilter(*a.temporal_filter) if a.temporal_filter else None
    ed = StreamEditor(vibe, TinyAutoencoder(), engine=engine, student=a.student, temporal_filter=tf,
                      carry=Carry(a.keyframe_every, sigma=a.carry_sigma) if a.carry else None)
    clips = a.clips or sorted(f.stem for f in Path(a.clips_dir).glob("*.mp4"))
    gpu = torch.cuda.get_device_name().replace("NVIDIA ", "").replace("GeForce ", "")
    footer = f"{ed.steps}-step VIBE student, {size}px, {backend}, {gpu}"
    options = (["carry-over"] if ed.carry is not None else []) + (
        [f"temporal filter {tf.low:g}/{tf.high:g}"] if tf is not None else [])
    if options:
        footer += "\n" + " + ".join(options)
    demo, fps = [], 24.0
    for clip in clips:
        src = Path(a.clips_dir) / f"{clip}.mp4"
        frames = read_video_frames(src, size, a.max_frames)
        fps = video_fps(src)
        panels = [[label(label(f, f"input: {clip}"), footer, bottom=True) for f in frames]]
        for key in a.prompts:
            ed.set_prompt(PROMPTS[key], frames[0])
            outs, tm = ed.run(frames, warmup_batches=1)
            panels.append([label(o, f"{key}: {tm.loop_fps:.0f} fps loop") for o in outs])
            print(f"[demo] {clip:<14} {key:<6} {len(frames)} frames, loop {tm.loop_fps:.1f} fps", flush=True)
        rows = [np.concatenate(fs, axis=1) for fs in zip(*panels)]
        save(out / f"{clip}.mp4", rows, fps)
        demo += rows
    save(out / "demo.mp4", demo, fps)
    print(f"[demo] {out / 'demo.mp4'}: {len(clips)} clips, {len(demo)} frames", flush=True)


if __name__ == "__main__":
    main()
