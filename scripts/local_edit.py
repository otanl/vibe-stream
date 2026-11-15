"""Local edit of a video file from a text instruction (vibe_stream.local): VIBE's own Qwen3-VL plans the edit and finds
the object on the first frame, SAM 2.1 follows it, a student engine at the crop size edits a window around it, and the
edit is pasted back; the rest of the frame stays as it is. Prints the plan, the one-off start (plan, grounding,
embedding) and ms per frame per stage; writes the output (with --compare, input | output) and <output>.json.

The crop engines are the stream's student copied to the crop size (scripts/resize_student.py) and built at it
(scripts/build_trt.py): 384 for additions on a part (sunglasses, a cap), 512 for removals and changes.

Usage:
  python scripts/local_edit.py --input assets/davis_mp4/parkour.mp4 --instruction "remove the man" \
      --engine_384 out/trt/vibe_local_384_b8_int8.engine --engine_512 out/trt/vibe_local_512_b8_int8.engine \
      --output out/local/parkour_remove.mp4 --compare
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
from vibe_stream.local import YUNET, LocalEditor
from vibe_stream.local_plan import plan_edit
from vibe_stream.metrics import read_video_frames, video_fps, write_video
from vibe_stream.sam_tracker import load_sam2
from vibe_stream.trt_engine import TRTTransformer
from vibe_stream.trt_tae import TRTTinyAutoencoder
from vibe_stream.vibe_model import ROOT, VibeModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--instruction", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--compare", action="store_true", help="write input | output side by side")
    p.add_argument("--size", type=int, default=768, help="frame size (centre square)")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--engine_384", required=True, help="the student engine at 384 (additions on a part)")
    p.add_argument("--engine_512", required=True, help="the student engine at 512 (removals and changes)")
    p.add_argument("--tae_dir", default=str(ROOT / "out/trt"), help="taesana_<size>_b<batch> engines")
    p.add_argument("--sam_res", type=int, default=512, help="SAM 2's input resolution (a multiple of 32)")
    p.add_argument("--no_bgfill", action="store_true", help="removals: paste VIBE's fill instead of the canvas")
    p.add_argument("--no_face_gate", action="store_true")
    p.add_argument("--face_model", default=str(YUNET))
    p.add_argument("--rounds", type=int, default=1, help="run the clip this many times (timing from the last)")
    a = p.parse_args()

    frames = np.stack(read_video_frames(a.input, a.size, a.max_frames))
    vibe = VibeModel(size=a.size, keep_text_encoder=True, denoiser_on_gpu=False)
    plan = plan_edit(vibe, a.instruction, frames[0])
    print(f"[local] plan: scope {plan.scope}, target {plan.target!r}, part {plan.part!r}, kind {plan.kind}, "
          f"face {plan.face}; objects {[(o['role'], [round(v) for v in o['box']]) for o in plan.objects]}; "
          f"seconds {plan.seconds}", flush=True)
    if not plan.local:
        raise SystemExit("not a local edit (or nothing found): use scripts/stream_video.py for a global one")
    engine = TRTTransformer(a.engine_384 if plan.kind == "add" else a.engine_512)
    Z, B = engine.spec.size, engine.spec.batch
    tae = TRTTinyAutoencoder(str(Path(a.tae_dir) / f"taesana_{Z}_b{B}"))
    sam, proc = load_sam2(a.sam_res)
    le = LocalEditor(vibe, engine, tae, sam, proc, sam_res=a.sam_res, bgfill=not a.no_bgfill,
                     face_gate=not a.no_face_gate, face_model=a.face_model)
    for _ in range(a.rounds):
        outs, timing = le.run(frames, plan)
    ms = timing.ms_per_frame()
    print(f"[local] crop size {Z}: " + ", ".join(f"{k} {v}" for k, v in ms.items())
          + f" -> {timing.fps:.1f} fps over {timing.frames} frames; embedding {timing.embed_ms:.0f} ms; "
          f"peak VRAM {torch.cuda.max_memory_allocated() / 2 ** 30:.1f} GB", flush=True)
    if a.output:
        out = [np.hstack([f, o]) for f, o in zip(frames, outs)] if a.compare else outs
        write_video(a.output, out, video_fps(a.input))
        Path(a.output).with_suffix(".json").write_text(json.dumps(dict(
            plan=plan.to_json(), crop_size=Z, sam_res=a.sam_res, ms_per_frame=ms, fps=round(timing.fps, 1),
            frames_timed=timing.frames, embed_ms=round(timing.embed_ms)), indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
