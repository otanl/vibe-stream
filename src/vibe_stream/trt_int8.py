"""INT8 (W8A8) quantization of the student's transformer for the TensorRT engine, with NVIDIA ModelOpt.

`trt_engine.export(..., quantize=...)` hands its fp16 graph to `quantize` here before writing the ONNX graph. The
graph gets ModelOpt quantizers (weights per output channel, activations per tensor, static) calibrated on
stream-like inputs, and is exported with Q/DQ nodes that the strongly typed TensorRT build turns into INT8 GEMMs and
1x1 convolutions. On an RTX 3090 Ti (INT8 tensor cores at twice the fp16 rate) at 768, batch 8: 15.1 ms per frame per
step against 20.5 for fp16, velocity within 2.8-4.9 % of bf16 (fp16: 1.2-1.7 %), and through the streaming editor
LPIPS 0.008-0.015 to the fp16 engine's output with CLIP direction, flicker and warping error unchanged (README).

What it takes, measured layer group by layer group:
- SmoothQuant (scale activation outliers into the weights) for every quantized layer. ModelOpt's covers Linear
  layers only; the feed-forward's 1x1 convolutions (`GLUMBConv`) carry the worst outliers (`conv_point` alone: 12-19 %
  velocity error) and are smoothed here: `conv_point`'s 1 / s folds exactly into the x half of the depthwise
  convolution before the gate, `conv_inverted`'s input (the timestep-modulated norm, from a timestep projection
  shared by all blocks) gets an explicit per-channel multiply. alpha 0.8 there, 0.5 for the Linear layers.
- Left in fp16: the depthwise convolutions, patch embedding, timestep embedding, output projection, and the
  cross-attention key / value projections (they see only the 224 context tokens).
- Q/DQ in fp16 (opset 19). With ModelOpt's default (fp32 around every Q/DQ) the layers between them run in fp32 and
  the engine is slower.

Calibration: frames of the training clips (presets.TRAIN_CLIPS_DIR) with the student's prompts and two others, as
the streaming editor feeds the transformer: each student step from the shared noise, and the carried step.

ModelOpt is an optional dependency (`pip install nvidia-modelopt==0.39.0`; later releases require a newer torch than
the TensorRT environment's 2.6). Import this module before the VIBE weights are loaded: imported later, ModelOpt's
pyarrow import has crashed the process on Windows.
"""

from __future__ import annotations

import time
from typing import List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

import modelopt.torch.quantization as mtq
import modelopt.torch.quantization.tensor_quant as tq

from .vibe_model import flow_step

EXTRA_PROMPTS = ("transfer this to watercolor painting", "make it winter, covered in snow")
KEEP_FP16 = ("*patch_embed*", "*time_embed*", "*proj_out*", "*norm_out*", "*conv_depth*", "*attn2.to_k*",
             "*attn2.to_v*", "*edit_head*")


def _pure_torch(*args, **kw):
    raise AttributeError("use ModelOpt's pure-torch fake quantization")


# ModelOpt's CUDA extension needs a compiler at first use; without one its fallback op crashes torch.onnx's tracer
# (seen on Windows). The pure-torch path computes the same.
tq.quantize_op = _pure_torch


class ChannelScale(nn.Module):
    """x * scale per channel of an NCHW tensor (the smoothing of conv_inverted's input)."""

    def __init__(self, scale: torch.Tensor):
        super().__init__()
        self.register_buffer("scale", scale.view(1, -1, 1, 1))

    def forward(self, x):
        return x * self.scale


class Fp16Out(nn.Module):
    """The velocity in fp16 whatever the quantized graph computes last, as `trt_engine.TRTTransformer` expects."""

    def __init__(self, graph: nn.Module):
        super().__init__()
        self.graph = graph

    def forward(self, x, enc, t):
        return self.graph(x, enc, t).to(torch.float16)


Call = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]   # (noisy latent ++ reference latent, projected context, t)


@torch.inference_mode()
def calibration_calls(graph: nn.Module, batches: Sequence[Tuple[torch.Tensor, torch.Tensor]], noise: torch.Tensor,
                      timesteps: Sequence[float], sigmas: Sequence[float], carry_t: float, carry_sigma: float) -> List[Call]:
    """Transformer inputs as the streaming editor makes them: for each (reference latents (B, 32, h, w) fp16,
    projected context (B, L, D) fp16), every student step from the noise along the fp16 graph's own velocities, and
    the carried step from (1 - s) * output + s * noise."""
    calls = []
    for il, enc in batches:
        b = len(il)
        x = noise.expand(b, -1, -1, -1).clone()
        for t, s0, s1 in zip(timesteps, sigmas[:-1], sigmas[1:]):
            tt = torch.full((b,), float(t), device=il.device)
            calls.append((torch.cat([x.half(), il], 1), enc, tt))
            x = flow_step(x, graph(*calls[-1]).float(), s0, s1)
        xc = (1 - carry_sigma) * x + carry_sigma * noise.expand(b, -1, -1, -1)
        calls.append((torch.cat([xc.half(), il], 1), enc, torch.full((b,), float(carry_t), device=il.device)))
    return calls


@torch.no_grad()
def smooth_ffn(graph: nn.Module, calls: Sequence[Call], alpha: float) -> int:
    """SmoothQuant for each block's GLUMBConv, s = max|x|^alpha / max|w|^(1 - alpha) per input channel; returns the
    number of feed-forwards changed. The graph's function is unchanged up to rounding."""
    ffs = [mod for name, mod in graph.named_modules() if name.endswith(".ff") and hasattr(mod, "conv_point")]
    amax = {}

    def hook(key):
        def record(mod, inp, out):
            v = inp[0].abs().amax(dim=(0, 2, 3)).float()
            amax[key] = torch.maximum(amax[key], v) if key in amax else v
        return record

    handles = [ff.conv_inverted.register_forward_hook(hook((i, "inv"))) for i, ff in enumerate(ffs)]
    handles += [ff.conv_point.register_forward_hook(hook((i, "pt"))) for i, ff in enumerate(ffs)]
    with torch.inference_mode():
        for call in calls:
            graph(*call)
    for h in handles:
        h.remove()

    def factor(x_max, conv):
        w_max = conv.weight.abs().amax(dim=(0, 2, 3)).float()
        return (x_max.clamp_min(1e-5) ** alpha / w_max.clamp_min(1e-5) ** (1 - alpha)).clamp(1e-4, 1e4)

    for i, ff in enumerate(ffs):
        s = factor(amax[(i, "pt")], ff.conv_point)
        n, dw, pt = s.numel(), ff.conv_depth, ff.conv_point
        dw.weight[:n] = (dw.weight[:n].float() / s.view(-1, 1, 1, 1)).to(dw.weight.dtype)
        dw.bias[:n] = (dw.bias[:n].float() / s).to(dw.bias.dtype)
        pt.weight.copy_((pt.weight.float() * s.view(1, -1, 1, 1)).to(pt.weight.dtype))
        conv = ff.conv_inverted
        s = factor(amax[(i, "inv")], conv)
        conv.weight.copy_((conv.weight.float() * s.view(1, -1, 1, 1)).to(conv.weight.dtype))
        ff.conv_inverted = nn.Sequential(ChannelScale((1 / s).to(conv.weight.dtype)), conv)
    return len(ffs)


def quantize(graph: nn.Module, calls: Sequence[Call], alpha: float = 0.5, ffn_alpha: float = 0.8,
             keep_fp16: Sequence[str] = KEEP_FP16) -> nn.Module:
    """Smooth and quantize the fp16 graph in place; returns the module to export (fp16 output, fp16 Q/DQ)."""
    n = smooth_ffn(graph, calls, ffn_alpha)
    cfg = {"quant_cfg": dict(mtq.INT8_SMOOTHQUANT_CFG["quant_cfg"]), "algorithm": {"method": "smoothquant", "alpha": alpha}}
    for pat in keep_fp16:
        cfg["quant_cfg"][pat] = {"enable": False}

    def forward_loop(model):
        for call in calls:
            model(*call)

    t0 = time.perf_counter()
    mtq.quantize(graph, cfg, forward_loop)
    on = sum(1 for name, mod in graph.named_modules() if name.endswith("_quantizer") and mod.is_enabled)
    print(f"[int8] {n} feed-forwards smoothed (alpha {ffn_alpha}), {on} quantizers calibrated on {len(calls)} calls "
          f"(SmoothQuant alpha {alpha}) in {time.perf_counter() - t0:.0f} s", flush=True)
    for name, mod in graph.named_modules():
        if name.endswith("_quantizer"):
            mod.trt_high_precision_dtype = "Half"
    return Fp16Out(graph)


def calibration_frames(clips_dir, size: int, batch: int, n_clips: int) -> List[List[np.ndarray]]:
    """`batch` frames spread over the first 48 of each of the first `n_clips` clips in `clips_dir`."""
    from pathlib import Path

    from .metrics import read_video_frames
    out = []
    for clip in sorted(Path(clips_dir).glob("*.mp4"))[:n_clips]:
        fr = read_video_frames(clip, size, 48)
        out.append(fr[::max(1, len(fr) // batch)][:batch])
    return out
