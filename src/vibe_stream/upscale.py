"""Display-side upscaling of the stream's output (2x: 640 -> 1280), on the GPU.

`RTXVSR` calls NVIDIA RTX Video Super Resolution (the RTX Video SDK's CUDA path) through out/rtx_vsr/rtxvsr.dll,
which scripts/build_rtx_vsr.py builds from the SDK; RTX GPUs only. On the commercial student's output (oil, anime,
watercolor; 640 -> 1280, RTX 3090 Ti) quality level 4 takes 1.3 ms per frame and gives 1.48x the Sobel of a
bicubic upscale with 1.09x its flicker, with crisp edges and brush texture and no ringing. Sharpening of any kind
raises the flicker roughly in proportion to the detail it adds: it amplifies the flicker the stream already has.
`Upscaler("bicubic")` is the fallback where the DLL or an RTX GPU is missing.
"""

from __future__ import annotations

import ctypes
from pathlib import Path

import torch
import torch.nn.functional as F

from .vibe_model import ROOT

DLL = ROOT / "out" / "rtx_vsr" / "rtxvsr.dll"


class _Rect(ctypes.Structure):
    _fields_ = [("left", ctypes.c_uint32), ("top", ctypes.c_uint32), ("right", ctypes.c_uint32),
                ("bottom", ctypes.c_uint32)]


class _VSR(ctypes.Structure):
    _fields_ = [("QualityLevel", ctypes.c_uint32)]


class _THDR(ctypes.Structure):
    _fields_ = [("Contrast", ctypes.c_uint32), ("Saturation", ctypes.c_uint32), ("MiddleGray", ctypes.c_uint32),
                ("MaxLuminance", ctypes.c_uint32)]


class RTXVSR:
    """RTX Video Super Resolution on PyTorch's CUDA context and current stream. quality 1-4 (4: best, 1.3 ms per
    640 -> 1280 frame on an RTX 3090 Ti; 1: 0.6 ms)."""

    def __init__(self, quality: int = 4, dll=DLL):
        if not Path(dll).exists():
            raise FileNotFoundError(f"{dll} not found; build it with scripts/build_rtx_vsr.py")
        torch.zeros(1, device="cuda")                       # PyTorch's context current on this thread
        self.lib = ctypes.CDLL(str(dll))
        self.lib.rtx_video_api_cuda_create.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_uint,
                                                       ctypes.c_uint]
        self.lib.rtx_video_api_cuda_create.restype = ctypes.c_uint
        self.lib.rtx_video_api_cuda_evaluate_deviceptr.argtypes = [ctypes.c_void_p, ctypes.c_void_p, _Rect, _Rect,
                                                                   ctypes.POINTER(_VSR), ctypes.POINTER(_THDR)]
        self.lib.rtx_video_api_cuda_evaluate_deviceptr.restype = ctypes.c_uint
        stream = torch.cuda.current_stream().cuda_stream
        if not self.lib.rtx_video_api_cuda_create(None, ctypes.c_void_p(stream), 0, 0, 1):
            raise RuntimeError("RTX Video Super Resolution is not available (an RTX GPU and driver R570+ are needed)")
        self.vsr, self.thdr = _VSR(quality), _THDR()
        self.buf = {}

    @torch.no_grad()
    def __call__(self, img: torch.Tensor, scale: int = 2) -> torch.Tensor:
        """img (H, W, 3) uint8 RGB on the GPU -> (H * scale, W * scale, 3) uint8 RGB on the GPU."""
        h, w = img.shape[:2]
        key = (h, w, scale)
        if key not in self.buf:
            self.buf[key] = (torch.full((h, w, 4), 255, dtype=torch.uint8, device="cuda"),
                             torch.empty((h * scale, w * scale, 4), dtype=torch.uint8, device="cuda"))
        src, dst = self.buf[key]
        src[..., :3] = img
        ok = self.lib.rtx_video_api_cuda_evaluate_deviceptr(
            ctypes.c_void_p(src.data_ptr()), ctypes.c_void_p(dst.data_ptr()), _Rect(0, 0, w, h),
            _Rect(0, 0, w * scale, h * scale), ctypes.byref(self.vsr), ctypes.byref(self.thdr))
        if not ok:
            raise RuntimeError("RTX Video Super Resolution failed")
        return dst[..., :3].contiguous()


class Upscaler:
    """method "vsr" (RTXVSR) or "bicubic"; a batch (N, H, W, 3) uint8 on the GPU in, (N, sH, sW, 3) out."""

    def __init__(self, method: str = "vsr", scale: int = 2, quality: int = 4):
        if method not in ("vsr", "bicubic"):
            raise ValueError(f"unknown upscale method {method!r}")
        self.method, self.scale = method, scale
        self.vsr = RTXVSR(quality) if method == "vsr" else None

    @torch.no_grad()
    def __call__(self, frames: torch.Tensor) -> torch.Tensor:
        if self.vsr is not None:
            return torch.stack([self.vsr(f, self.scale) for f in frames])
        x = frames.permute(0, 3, 1, 2).float()
        x = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return x.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
