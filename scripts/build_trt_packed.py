"""The packed engine of a student for the face pass riding along in the stream's call (vibe_stream.packed): the frame
at the student's size and one face crop (default 256) in one TensorRT call, batch 1.

  export  the packed graph as ONNX and the spec (needs the VIBE weights; with --int8, ModelOpt)
  build   the strongly typed engine from it (TensorRT only)

--int8 quantizes as build_trt.py --int8 does, calibrated on frames of --calib_clips training clips (two each) with
face-sized crops of their centres. Run the stream with --packed_engine next to --engine and --face_engine main
(stream_video.py, measure_latency.py).

Usage:
  python scripts/build_trt_packed.py --student runs/klein640_prune14_18/lora.safetensors \
      --engine out/trt/vibe_klein640_prune14_18_packed256_int8.engine --int8
"""
import argparse

import _paths  # noqa: F401
from vibe_stream import lora, packed
from vibe_stream.metrics import read_video_frames
from vibe_stream.presets import EVAL_CLIPS_DIR, TRAIN_CLIPS_DIR
from vibe_stream.vibe_model import ROOT, VibeModel, student_model_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", nargs="?", default="all", choices=["export", "build", "all"])
    p.add_argument("--student", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--int8", action="store_true")
    p.add_argument("--calib_clips", type=int, default=25)
    p.add_argument("--workspace_gb", type=float, default=6.0)
    a = p.parse_args()
    if a.stage in ("export", "all"):
        if a.int8:
            from vibe_stream import trt_int8  # (before the weights are loaded, see trt_int8)
        args = student_model_args(lora.load(a.student)[1])
        m = VibeModel(**args, keep_text_encoder=False)
        if a.int8:
            frames = [f for fr in trt_int8.calibration_frames(ROOT / TRAIN_CLIPS_DIR, m.size, 2, a.calib_clips) for f in fr]
        else:
            frames = read_video_frames(ROOT / EVAL_CLIPS_DIR / "blackswan.mp4", m.size, 1)
        packed.export(m, a.student, a.engine, a.crop, a.int8, frames)
        del m
    if a.stage in ("build", "all"):
        import torch
        torch.cuda.empty_cache()
        packed.build(a.engine, a.workspace_gb)


if __name__ == "__main__":
    main()
