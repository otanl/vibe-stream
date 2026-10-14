"""Build the TensorRT engine of a student (interpreter with tensorrt; see vibe_stream.trt_engine).

  export  merge the student into VIBE's transformer, write the ONNX graph, the check file, the spec and the noise
  build   TensorRT only (fresh process)
  bench   engine against the bf16 PyTorch velocities, and ms per frame per step
  all     the three in order, each in its own process (the default)

Usage: python scripts/build_trt.py --student runs/lam4_2step/lora.safetensors --engine out/trt/vibe_lam4_2step_768_b8.engine
"""
import argparse
import subprocess
import sys
from pathlib import Path

import torch

import _paths  # noqa: F401
from vibe_stream import lora, trt_engine
from vibe_stream.metrics import read_video_frames
from vibe_stream.presets import EVAL_CLIPS_DIR, PROMPTS
from vibe_stream.vibe_model import ROOT, VibeModel


def export(a):
    meta = lora.load(a.student)[1]
    size = int(meta.get("size", 768))
    m = VibeModel(size=size)
    # real inputs for the check file: the first frames of a clip and the student's first prompt
    frames = read_video_frames(ROOT / EVAL_CLIPS_DIR / "blackswan.mp4", size, a.batch)
    prompt = next(iter(meta.get("prompts", PROMPTS).values()))
    emb = m.embed(prompt, frames[0])
    m.offload_text_encoder()
    img_lat = m.encode(frames)
    trt_engine.export(m, a.student, a.engine, a.batch, img_lat, emb)


def bench(a):
    eng = trt_engine.TRTTransformer(a.engine)
    errs = eng.check(a.engine)
    print(f"[bench] engine vs bf16 PyTorch velocity, rel err per step: {', '.join(f'{e:.4f}' for e in errs)}", flush=True)
    sp = eng.spec
    lat = sp.size // 32
    x = torch.randn(sp.batch, 64, lat, lat, device="cuda", dtype=torch.float16)
    from safetensors.torch import load_file
    enc = load_file(str(trt_engine.check_path(a.engine)))["enc"].cuda()
    t = torch.full((sp.batch,), sp.timesteps[0], device="cuda")
    ms = trt_engine.median_ms(lambda: eng(x, enc, t))
    print(f"[bench] {ms:.1f} ms per call of batch {sp.batch} = {ms / sp.batch:.2f} ms per frame per step "
          f"({sp.student_steps} steps per frame)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", nargs="?", default="all", choices=["export", "build", "bench", "all"])
    p.add_argument("--student", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workspace_gb", type=float, default=6.0)
    a = p.parse_args()
    if a.stage == "all":
        rest = [x for x in sys.argv[1:] if x != "all"]
        for stage in ("export", "build", "bench"):
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), stage, *rest])
            if r.returncode:
                raise SystemExit(r.returncode)
        return
    {"export": export, "build": lambda a: trt_engine.build(a.engine, a.workspace_gb), "bench": bench}[a.stage](a)


if __name__ == "__main__":
    main()
