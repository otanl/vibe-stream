"""taesana as two strongly typed fp16 TensorRT engines, encoder and decoder, with uint8 frames in and out.

In PyTorch, taesana's convolutions run in cuDNN with fp32 accumulation (half rate on GeForce Ampere), and every ReLU,
residual add, upsample and permute is a kernel of its own; in the 768 streaming loop that was about a quarter of the
GPU time (Nsight profile, README). TensorRT fuses conv + bias + ReLU (+ the residual add), picks fp16-accumulating
kernels, and takes the uint8 <-> fp16 conversion and the NHWC <-> NCHW permutes into the engines.

`export` writes the two ONNX graphs and a spec, `trt_engine.build_onnx` builds each engine (TensorRT only), and
`TRTTinyAutoencoder` stands in for `taesana.TinyAutoencoder` in `StreamEditor` (same `encode_u8` / `decode_u8`).
The engines are fixed to one size, one batch size and one TensorRT version; `<prefix>.json` records them and a
mismatch is refused. Files: `<prefix>.enc.engine`, `<prefix>.dec.engine`, `<prefix>.json`, `<prefix>_onnx/`.
"""

from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

from .taesana import TinyAutoencoder
from .trt_engine import capture, enqueue


@dataclasses.dataclass
class TaeSpec:
    tensorrt: str
    platform: str
    size: int
    batch: int

    @staticmethod
    def path_for(prefix) -> Path:
        return Path(f"{prefix}.json")

    def save(self, prefix) -> None:
        self.path_for(prefix).write_text(json.dumps(dataclasses.asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, prefix) -> "TaeSpec":
        return cls(**json.loads(cls.path_for(prefix).read_text(encoding="utf-8")))


def engine_path(prefix, part: str) -> Path:
    return Path(f"{prefix}.{part}.engine")


def onnx_path(prefix, part: str) -> Path:
    p = Path(prefix)
    return p.parent / f"{p.name}_onnx" / f"{part}.onnx"


class _Encode(nn.Module):
    """(B, H, W, 3) uint8 -> (B, 32, H / 32, W / 32) fp16 scaled latents, as `TinyAutoencoder.encode_u8`."""

    def __init__(self, enc: nn.Module):
        super().__init__()
        self.enc = enc

    def forward(self, frames):
        # TensorRT permutes no uint8 tensors: cast first, permute after
        return self.enc(frames.to(torch.float16).permute(0, 3, 1, 2) / 255)


class _Decode(nn.Module):
    """Scaled fp16 latents -> (B, H, W, 3) uint8, as `TinyAutoencoder.decode_u8`."""

    def __init__(self, dec: nn.Module):
        super().__init__()
        self.dec = dec

    def forward(self, lat):
        return (self.dec(lat).clamp(0, 1) * 255).round().permute(0, 2, 3, 1).to(torch.uint8)


def export(prefix, size: int, batch: int) -> None:
    """Stage 1: both ONNX graphs (fp16 weights) and the spec; the TensorRT version is filled in by the build."""
    tae = TinyAutoencoder()
    frames = torch.zeros((batch, size, size, 3), dtype=torch.uint8, device=tae.device)
    lat = torch.zeros((batch, 32, size // 32, size // 32), dtype=torch.float16, device=tae.device)
    for part, net, x, names in (("enc", _Encode(tae.enc), frames, (["frames"], ["lat"])),
                                ("dec", _Decode(tae.dec), lat, (["lat"], ["frames"]))):
        onnx = onnx_path(prefix, part)
        onnx.parent.mkdir(parents=True, exist_ok=True)
        with torch.inference_mode():
            torch.onnx.export(net.to(memory_format=torch.contiguous_format).eval(), (x,), str(onnx),
                              input_names=names[0], output_names=names[1], opset_version=17, do_constant_folding=True,
                              dynamo=False)
        print(f"[export] {onnx}", flush=True)
    TaeSpec(tensorrt="", platform="", size=size, batch=batch).save(prefix)


def build(prefix, workspace_gb: float = 4.0) -> None:
    """Stage 2: TensorRT only."""
    from .trt_engine import build_onnx
    spec = TaeSpec.load(prefix)
    for part in ("enc", "dec"):
        spec.tensorrt = build_onnx(onnx_path(prefix, part), engine_path(prefix, part), workspace_gb)
    spec.platform = sys.platform
    spec.save(prefix)


class _Runner:
    """One engine; with `cuda_graph` its call is replayed as a CUDA graph on fixed buffers (`trt_engine.capture`)."""

    def __init__(self, runtime, path: Path, cuda_graph: bool = True):
        self.engine = runtime.deserialize_cuda_engine(path.read_bytes())
        self.cuda_graph = cuda_graph
        self.configure(self.engine.create_execution_context())

    def configure(self, ctx) -> None:
        """Take `ctx` as the execution context (also after `trt_engine.share_scratch`)."""
        self.ctx = ctx
        self._graph = self._x = self._out = None

    def __call__(self, x: torch.Tensor, out: torch.Tensor, names, device) -> torch.Tensor:
        if not self.cuda_graph:
            self.ctx.set_tensor_address(names[0], x.data_ptr())
            self.ctx.set_tensor_address(names[1], out.data_ptr())
            enqueue(self.ctx, device)
            return out
        if self._graph is None or self._x.shape != x.shape or self._out.shape != out.shape:
            self._x, self._out = torch.empty_like(x), torch.empty_like(out)
            self.ctx.set_tensor_address(names[0], self._x.data_ptr())
            self.ctx.set_tensor_address(names[1], self._out.data_ptr())
            self._x.copy_(x)
            self._graph = capture(self.ctx, device)
        else:
            self._x.copy_(x)
            self._graph.replay()
        return out.copy_(self._out)


class TRTTinyAutoencoder:
    """`TinyAutoencoder`'s interface on the two engines, on the current CUDA stream. Each call returns a new tensor
    (the streaming loop keeps several encodings of one batch alive at once).

    `dec_prefix`: the decoder from engines of a smaller batch that divides this one (e.g. built with --batch 4),
    called once per part of the batch: its scratch memory is a fraction of the batch-8 decoder's (1.2 GB at 640), at
    42.7 dB from the batch-8 output and the same speed (README, "8 GB GPUs")."""

    def __init__(self, prefix, device: str = "cuda", dec_prefix=None):
        import tensorrt as trt
        self.spec = TaeSpec.load(prefix)
        specs = [self.spec] + ([TaeSpec.load(dec_prefix)] if dec_prefix else [])
        for sp, p in zip(specs, (prefix, dec_prefix)):
            if sp.tensorrt != trt.__version__ or sp.platform != sys.platform:
                raise RuntimeError(f"{p} was built with TensorRT {sp.tensorrt} on {sp.platform}; "
                                   f"this is TensorRT {trt.__version__} on {sys.platform}")
        self.device = torch.device(device)
        self.size, self.batch = self.spec.size, self.spec.batch
        self.dec_batch = specs[-1].batch
        if specs[-1].size != self.size or self.batch % self.dec_batch:
            raise ValueError(f"decoder engines for batch {self.dec_batch} at {specs[-1].size} do not divide batch "
                             f"{self.batch} at {self.size}")
        self._log = trt.Logger(trt.Logger.WARNING)
        rt = trt.Runtime(self._log)
        self.enc = _Runner(rt, engine_path(prefix, "enc"))
        self.dec = _Runner(rt, engine_path(dec_prefix or prefix, "dec"))

    def _check(self, shape, want):
        if tuple(shape) != want:
            raise ValueError(f"taesana engines take {want}, got {tuple(shape)}")

    @torch.inference_mode()
    def encode_u8(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, H, W, 3) uint8 on the GPU -> (B, 32, H / 32, W / 32) fp16 scaled latents."""
        n, lat = self.batch, self.size // 32
        self._check(frames.shape, (n, self.size, self.size, 3))
        out = torch.empty((n, 32, lat, lat), dtype=torch.float16, device=self.device)
        return self.enc(frames.to(torch.uint8).contiguous(), out, ("frames", "lat"), self.device)

    @torch.inference_mode()
    def decode_u8(self, lat: torch.Tensor) -> torch.Tensor:
        """Scaled latents -> (B, H, W, 3) uint8 on the GPU."""
        n, d = self.batch, self.dec_batch
        self._check(lat.shape, (n, 32, self.size // 32, self.size // 32))
        out = torch.empty((n, self.size, self.size, 3), dtype=torch.uint8, device=self.device)
        lat = lat.to(torch.float16).contiguous()
        for i in range(0, n, d):
            self.dec(lat[i:i + d], out[i:i + d], ("lat", "frames"), self.device)
        return out
