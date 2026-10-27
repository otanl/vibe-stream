"""Build the TensorRT engine of a student (interpreter with tensorrt; see vibe_stream.trt_engine).

  export  merge the student into VIBE's transformer, write the ONNX graph, the check file, the spec and the noise
  build   TensorRT only (fresh process)
  bench   engine against the bf16 PyTorch velocities, and ms per frame per step
  all     the three in order, each in its own process (the default)

--int8 quantizes the transformer (vibe_stream.trt_int8; needs nvidia-modelopt), calibrated on --calib_clips training
clips.

Usage: python scripts/build_trt.py --student runs/lam4_2step/lora.safetensors --engine out/trt/vibe_lam4_2step_768_b8.engine
       python scripts/build_trt.py --student ... --engine out/trt/vibe_lam4_2step_768_b8_int8.engine --int8
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
from vibe_stream import lora, trt_engine
from vibe_stream.metrics import read_video_frames
from vibe_stream.presets import EVAL_CLIPS_DIR, PROMPTS, TRAIN_CLIPS_DIR
from vibe_stream.vibe_model import ROOT, VibeModel


def export(a):
    if a.int8:
        from vibe_stream import trt_int8  # noqa: F401  (before the weights are loaded, see trt_int8)
    meta = lora.load(a.student)[1]
    size = int(meta.get("size", 768))
    m = VibeModel(size=size)
    # real inputs for the check file: the first frames of a clip and the student's first prompt
    frames = read_video_frames(ROOT / EVAL_CLIPS_DIR / "blackswan.mp4", size, a.batch)
    prompts = list(meta.get("prompts", PROMPTS).values())
    emb = m.embed(prompts[0], frames[0])
    quantize = None
    if a.int8:
        from vibe_stream.taesana import TinyAutoencoder
        tae = TinyAutoencoder()
        calib = []   # (taesana latents of the frames, raw embedding); projected inside `quantize`, after the merge
        with torch.inference_mode():
            for i, fr in enumerate(trt_int8.calibration_frames(ROOT / TRAIN_CLIPS_DIR, size, a.batch, a.calib_clips)):
                il = tae.encode_u8(torch.from_numpy(np.stack(fr)).to(m.device))
                extra = [trt_int8.EXTRA_PROMPTS[(i // 2) % 2]] if i % 2 == 0 else []
                calib += [(il, m.embed(p, fr[0])) for p in prompts + extra]

        def quantize(graph, timesteps, sigmas):
            with torch.inference_mode():
                batches = [(il, m.project_embedding(e).half().expand(a.batch, -1, -1).contiguous()) for il, e in calib]
                s = meta.get("carry_sigma") or sigmas[-2]
                calls = trt_int8.calibration_calls(graph, batches, m.noise, timesteps, sigmas, 1000.0 * s * m.tscale, s)
            return trt_int8.quantize(graph, calls)
    m.offload_text_encoder()
    img_lat = m.encode(frames)
    trt_engine.export(m, a.student, a.engine, a.batch, img_lat, emb, quantize=quantize)


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
    p.add_argument("--int8", action="store_true", help="INT8 transformer (vibe_stream.trt_int8)")
    p.add_argument("--calib_clips", type=int, default=10, help="with --int8: training clips to calibrate on")
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
