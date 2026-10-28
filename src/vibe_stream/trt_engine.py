"""VIBE's transformer as a strongly typed fp16 TensorRT engine with the student LoRA merged, and its runtime wrapper.

Stage 1, `export` (needs the VIBE weights): merge the LoRA, take caption_projection + caption_norm out of the graph
(`VibeModel.project_embedding` computes them once per clip in fp32), cast to fp16, and write the ONNX graph with
external data plus a check file holding real inputs and the bf16 PyTorch velocity of every student step.
Stage 2, `build`: TensorRT only, from those files, in a fresh process (building while PyTorch still holds the models
has overflowed 24 GB).

An engine is fixed to one resolution, one batch size and one TensorRT version. `<engine>.json` (`EngineSpec`)
records them together with the student it was built from (and the student's `carry_sigma` and `carry_fill`, read by
`StreamEditor`), and `TRTTransformer` refuses a mismatch. The student's init noise is stored next to the engine
(`<engine>.noise.safetensors`), so the engine alone is enough to run.

The cross-attention projects its K / V from the context once per call, not once per frame
(`SharedContextAttnProcessor`): the streaming editor gives every frame of a batch the same instruction.

Measured on an RTX 3090 Ti with TensorRT 10.16 at 768, batch 8: 2.98 GB engine built in 100 s, velocity within
1.3-2.0 % of bf16 PyTorch, 18.8 ms per frame per step (21.3 with K / V per frame; bf16 PyTorch eager: 36 ms).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn

from . import lora as lora_mod
from .vibe_model import VibeModel, flow_step


@dataclasses.dataclass
class EngineSpec:
    tensorrt: str
    platform: str
    size: int
    batch: int
    student: str
    student_sha256: str
    student_steps: int
    timesteps: List[float]       # already multiplied by the transformer's timestep_scale
    sigmas: List[float]
    prompts: dict
    carry_sigma: Optional[float] = None   # the student's trained start of the carried step (vibe_stream.carry)
    carry_fill: Optional[str] = None      # and what it was trained to fill in with where the flow fails
    precision: str = "fp16"               # or "int8" (vibe_stream.trt_int8)
    shared_context: bool = False          # cross-attention K / V from the first row only (SharedContextAttnProcessor)

    @staticmethod
    def path_for(engine) -> Path:
        return Path(engine).with_suffix(".json")

    def save(self, engine) -> None:
        self.path_for(engine).write_text(json.dumps(dataclasses.asdict(self), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, engine) -> "EngineSpec":
        return cls(**json.loads(cls.path_for(engine).read_text(encoding="utf-8")))


def noise_path(engine) -> Path:
    return Path(engine).with_suffix(".noise.safetensors")


def check_path(engine) -> Path:
    return Path(engine).with_suffix(".check.safetensors")


def onnx_path(engine) -> Path:
    e = Path(engine)
    return e.parent / f"{e.stem}_onnx" / f"{e.stem}.onnx"


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


class SharedContextAttnProcessor:
    """Sana's cross-attention (diffusers' SanaAttnProcessor2_0, no mask) for a batch whose rows share one context, as
    in the streaming editor (one instruction per clip): K / V are projected from the first row only, and the batch's
    queries attend to them as one sequence, so the 20 layers' K / V projections run once per call instead of once per
    frame. Exact up to fp16 rounding; an engine exported with it ignores every row of `enc` but the first.
    """

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        if encoder_hidden_states is None or attention_mask is not None:
            raise ValueError("cross-attention without a mask expected")
        b, n, _ = hidden_states.shape
        ctx = encoder_hidden_states[:1]
        query, key, value = attn.to_q(hidden_states), attn.to_k(ctx), attn.to_v(ctx)
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        h = attn.heads
        d = key.shape[-1] // h
        query = query.view(b, n, h, d).permute(2, 0, 1, 3).reshape(1, h, b * n, d)
        key = key.view(1, -1, h, d).transpose(1, 2)
        value = value.view(1, -1, h, d).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(query, key, value)
        out = out.view(h, b, n, d).permute(1, 2, 0, 3).reshape(b, n, h * d).to(query.dtype)
        return attn.to_out[1](attn.to_out[0](out)) / attn.rescale_output_factor


class _Graph(nn.Module):
    """The exported function: velocity(x ++ reference latent, projected context, scaled timestep)."""

    def __init__(self, tr: nn.Module, batch: int):
        super().__init__()
        self.tr = tr
        self.batch = batch

    def forward(self, x, enc, t):
        return self.tr(hidden_states=x, encoder_hidden_states=enc, timestep=t, return_dict=False,
                       t2i_samples=[False] * self.batch)[0]


def export(model: VibeModel, student, engine, batch: int, img_lat: torch.Tensor, emb: torch.Tensor,
           quantize=None) -> None:
    """Stage 1. `img_lat` (batch, 32, h, w) and `emb` (1, L, D) are real inputs for the check file. `quantize`, if
    given, takes the fp16 graph, the scaled timesteps and the sigmas and returns the quantized module to export
    (`trt_int8`).

    Modifies `model.tr` in place (LoRA merged, projection removed, fp16); the model is not usable afterwards.
    """
    from safetensors.torch import save_file
    state, meta = lora_mod.load(student)
    k = int(meta.get("student_steps", 1))
    if int(meta.get("size", model.size)) != model.size:
        raise ValueError(f"student trained at {meta['size']}, exporting at {model.size}")
    noise = state.get(lora_mod.NOISE_KEY)
    if noise is not None:
        model.noise = noise.to(model.device, torch.float32)
    n = lora_mod.merge(model.tr, state)
    print(f"[export] merged {n} LoRA layers from {student} ({k}-step student)", flush=True)
    ts, sg = model.schedules[k]
    xs, vs = [], []
    with torch.inference_mode():
        x = model.noise.expand(batch, -1, -1, -1).clone()
        for t, s0, s1 in zip(ts, sg[:-1], sg[1:]):
            v = model.velocity(x, img_lat, emb, t)
            xs.append(x.clone())
            vs.append(v.clone())
            x = flow_step(x, v, s0, s1)
        enc = model.project_embedding(emb)
    print(f"[export] embedding |max| {emb.float().abs().max().item():.1f}, projected |max| {enc.abs().max().item():.1f}",
          flush=True)
    tr = model.tr
    tr.caption_projection, tr.caption_norm = nn.Identity(), nn.Identity()
    tr.half()
    for block in tr.transformer_blocks:
        block.attn2.set_processor(SharedContextAttnProcessor())
    graph = _Graph(tr, batch).eval()
    enc16 = enc.half().expand(batch, -1, -1).contiguous()
    tsc = [(t.float().expand(batch) * model.tscale).contiguous() for t in ts]
    with torch.inference_mode():
        for i, (x, v) in enumerate(zip(xs, vs)):
            v16 = graph(torch.cat([x, img_lat], 1).half(), enc16, tsc[i]).float()
            print(f"[export] step {i}: PyTorch fp16 vs bf16 velocity rel err {rel_err(v16, v):.4f}, "
                  f"finite {bool(torch.isfinite(v16).all())}", flush=True)
    if quantize is not None:
        graph = quantize(graph, [float(t[0]) for t in tsc], sg)
        with torch.inference_mode():
            for i, (x, v) in enumerate(zip(xs, vs)):
                vq = graph(torch.cat([x, img_lat], 1).half(), enc16, tsc[i]).float()
                print(f"[export] step {i}: quantized vs bf16 velocity rel err {rel_err(vq, v):.4f}", flush=True)
    check = {"enc": enc16, "il": img_lat.half()}
    for i, (x, v) in enumerate(zip(xs, vs)):
        check[f"x{i}"], check[f"v{i}"], check[f"t{i}"] = x, v, tsc[i]
    engine = Path(engine)
    engine.parent.mkdir(parents=True, exist_ok=True)
    save_file({key: val.detach().cpu().contiguous() for key, val in check.items()}, str(check_path(engine)))
    save_file({lora_mod.NOISE_KEY: model.noise.cpu().contiguous()}, str(noise_path(engine)))
    # the TensorRT version and platform are filled in by `build`
    EngineSpec(tensorrt="", platform="", size=model.size, batch=batch, student=str(student),
               student_sha256=sha256(student), student_steps=k, timesteps=[float(t[0]) for t in tsc], sigmas=sg,
               prompts=meta.get("prompts", {}), carry_sigma=meta.get("carry_sigma"),
               carry_fill=meta.get("carry_fill"), precision="fp16" if quantize is None else "int8",
               shared_context=True).save(engine)
    onnx = onnx_path(engine)
    onnx.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    # Q/DQ in fp16 need opset 19; ModelOpt's quantizers export under no_grad, not inference mode
    with torch.inference_mode() if quantize is None else torch.no_grad():
        torch.onnx.export(graph, (torch.cat([xs[0], img_lat], 1).half(), enc16, tsc[0]), str(onnx),
                          input_names=["x", "enc", "t"], output_names=["v"], opset_version=17 if quantize is None else 19,
                          do_constant_folding=True, dynamo=False)
    print(f"[export] {onnx} written in {time.perf_counter() - t0:.0f} s", flush=True)


def build_onnx(onnx, engine, workspace_gb: float = 6.0) -> str:
    """A strongly typed engine from an ONNX graph; returns the TensorRT version."""
    import tensorrt as trt
    engine = Path(engine)
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED))
    parser = trt.OnnxParser(net, log)
    if not parser.parse_from_file(str(onnx)):
        raise RuntimeError("; ".join(str(parser.get_error(i)) for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, int(workspace_gb * (1 << 30)))
    t0 = time.perf_counter()
    blob = builder.build_serialized_network(net, cfg)
    if blob is None:
        raise RuntimeError("TensorRT build failed")
    engine.write_bytes(bytes(blob))
    print(f"[build] {engine} {engine.stat().st_size / 2**30:.2f} GB in {time.perf_counter() - t0:.0f} s, "
          f"TensorRT {trt.__version__}", flush=True)
    return trt.__version__


def build(engine, workspace_gb: float = 6.0) -> None:
    """Stage 2: strongly typed engine from the ONNX graph of stage 1."""
    spec = EngineSpec.load(engine)
    spec.tensorrt, spec.platform = build_onnx(onnx_path(engine), engine, workspace_gb), sys.platform
    spec.save(engine)


class TRTTransformer:
    """Runs the engine on the current CUDA stream. The output buffer is reused: consume it before the next call."""

    def __init__(self, engine, device: str = "cuda"):
        import tensorrt as trt
        self.spec = EngineSpec.load(engine)
        if self.spec.tensorrt != trt.__version__ or self.spec.platform != sys.platform:
            raise RuntimeError(f"{engine} was built with TensorRT {self.spec.tensorrt} on {self.spec.platform}; "
                               f"this is TensorRT {trt.__version__} on {sys.platform}")
        from safetensors.torch import load_file
        self.device = torch.device(device)
        self.noise = load_file(str(noise_path(engine)))[lora_mod.NOISE_KEY].to(self.device, torch.float32)
        self._log = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(self._log).deserialize_cuda_engine(Path(engine).read_bytes())
        self.ctx = self.engine.create_execution_context()
        lat = self.spec.size // 32
        self.out = torch.empty((self.spec.batch, 32, lat, lat), dtype=torch.float16, device=self.device)

    def __call__(self, x: torch.Tensor, enc: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x (B, 64, h, w) fp16 noisy ++ reference latent, enc (B, L, 2240) fp16 projected context, t (B,) fp32
        scaled timestep -> velocity (B, 32, h, w) fp16."""
        bufs = {"x": x.contiguous(), "enc": enc.contiguous(), "t": t.contiguous(), "v": self.out}
        for name, ten in bufs.items():
            self.ctx.set_tensor_address(name, ten.data_ptr())
        if not self.ctx.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        return self.out

    def check(self, engine) -> List[float]:
        """Relative error of the engine against the bf16 PyTorch velocities stored at export."""
        from safetensors.torch import load_file
        ck = {k: v.to(self.device) for k, v in load_file(str(check_path(engine))).items()}
        errs = []
        for i in range(self.spec.student_steps):
            v = self(torch.cat([ck[f"x{i}"].half(), ck["il"]], 1), ck["enc"], ck[f"t{i}"]).float()
            errs.append(rel_err(v, ck[f"v{i}"]))
        torch.cuda.synchronize()
        return errs


def median_ms(fn, n: int = 20, warm: int = 5) -> float:
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t))
    return float(np.median(ts))
