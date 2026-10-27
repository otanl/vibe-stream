"""Build taesana's TensorRT engines (interpreter with tensorrt; see vibe_stream.trt_tae).

  export  the two ONNX graphs and the spec
  build   TensorRT only (fresh process)
  bench   engines against PyTorch taesana: latent error, decode PSNR between the two, round-trip PSNR, ms per frame
  all     the three in order, each in its own process (the default)

Usage: python scripts/build_trt_tae.py --size 768 --batch 8 --prefix out/trt/taesana_768_b8
"""
import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
from vibe_stream import trt_tae
from vibe_stream.metrics import read_video_frames
from vibe_stream.presets import EVAL_CLIPS_DIR
from vibe_stream.taesana import TinyAutoencoder
from vibe_stream.trt_engine import median_ms, rel_err
from vibe_stream.vibe_model import ROOT


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = ((a.float() - b.float()) ** 2).mean().item()
    return 10 * np.log10(255 ** 2 / max(mse, 1e-10))


def bench(a):
    eng = trt_tae.TRTTinyAutoencoder(a.prefix)
    sp = eng.spec
    tae = TinyAutoencoder()
    frames = read_video_frames(ROOT / EVAL_CLIPS_DIR / "blackswan.mp4", sp.size, sp.batch)
    x = torch.from_numpy(np.stack(frames)).cuda()
    st = torch.cuda.Stream()
    with torch.cuda.stream(st):
        lat_pt, lat_trt = tae.encode_u8(x), eng.encode_u8(x)
        dec_pt, dec_trt = tae.decode_u8(lat_pt), eng.decode_u8(lat_pt)
        rt_trt = eng.decode_u8(lat_trt)
        torch.cuda.synchronize()
        print(f"[bench] latent rel err {rel_err(lat_trt, lat_pt):.4f}; decode of the same latent: PSNR engine vs "
              f"PyTorch {psnr(dec_trt, dec_pt):.1f} dB, mean |diff| {(dec_trt.float() - dec_pt.float()).abs().mean():.2f}"
              f" of 255", flush=True)
        print(f"[bench] round trip PSNR to the input: PyTorch {psnr(dec_pt, x):.2f} dB, engines {psnr(rt_trt, x):.2f} dB",
              flush=True)
        n = sp.batch
        for name, fn in (("encode", lambda: tae.encode_u8(x)), ("decode", lambda: tae.decode_u8(lat_pt))):
            ms_pt = median_ms(fn)
            ms_trt = median_ms(lambda: (eng.encode_u8 if name == "encode" else eng.decode_u8)(
                x if name == "encode" else lat_pt))
            print(f"[bench] {name}: PyTorch {ms_pt / n:.2f} ms per frame, TensorRT {ms_trt / n:.2f} ms per frame "
                  f"(batch {n})", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", nargs="?", default="all", choices=["export", "build", "bench", "all"])
    p.add_argument("--prefix", required=True, help="engine files are <prefix>.enc.engine, <prefix>.dec.engine")
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workspace_gb", type=float, default=4.0)
    a = p.parse_args()
    if a.stage == "all":
        rest = [x for x in sys.argv[1:] if x != "all"]
        for stage in ("export", "build", "bench"):
            r = subprocess.run([sys.executable, str(Path(__file__).resolve()), stage, *rest])
            if r.returncode:
                raise SystemExit(r.returncode)
        return
    {"export": lambda a: trt_tae.export(a.prefix, a.size, a.batch),
     "build": lambda a: trt_tae.build(a.prefix, a.workspace_gb), "bench": bench}[a.stage](a)


if __name__ == "__main__":
    main()
