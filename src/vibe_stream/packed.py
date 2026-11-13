"""The face pass riding along in the stream's transformer call: one engine call for the frame and the face crop.

The face pass (`vibe_stream.face`) edits its 256 crop with a transformer call of its own. 64 tokens against the
frame's 400 at 640, but the call is a whole call of the 1.6B transformer, about a thousand short kernels, so it costs
far more than its share of the work: on an RTX 4060 Ti 9.1 ms against 18.1 for the frame (INT8, batch 1).

`PackedGraph` runs both images as one sequence. VIBE's transformer (Sana 1.5: channel_cat, patch 1, no positional
embedding) has per-token layers (patch embedding, the projections of self- and cross-attention, norms and timestep
modulation, the 1x1 convolutions of the Mix-FFN, the output projection) and three that mix tokens: the linear
self-attention (one K^T V sum over the image), the cross-attention (to the image's own instruction embedding) and the
Mix-FFN's 3x3 depthwise convolution (on the image's grid). The per-token layers run once over the concatenated
sequence, their matrix products a sixth longer and no more kernels; the mixing ones run per segment, each segment with
its own embedding and timestep. Each segment's output is the one its own call would give, up to fp16 rounding (velocity
within 0.1-0.2 %). The crop's extra cost drops to 4.9 ms on the 4060 Ti (INT8; fp16: 13.3 to 5.1 ms), README, "Faces".

`export` writes the packed graph of a student (fp16 or INT8 as `trt_int8` quantizes the stream's engines, calibrated
on frame / crop pairs), `PackedEngine` runs it as a CUDA graph, and `PackedStreamEditor` uses it: the frame's edit and
the crop's are generators (`StreamEditor.edit_steps`) whose transformer calls are paired. A call one of them makes
alone (a crop starting afresh while the frame is carried, a clip's first frames) goes to the ordinary engines. Batch 1
only (the live setting).
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .stream import StreamEditor
from .trt_engine import capture, sha256


# ---------------------------------------------------------------- the packed graph

def _linear_attn(attn, x: torch.Tensor, segs) -> torch.Tensor:
    """VIBE's SanaLinearAttnProcessor2_0 (no mask) with the K^T V sum taken per segment."""
    dtype = x.dtype
    q, k, v = attn.to_q(x), attn.to_k(x), attn.to_v(x)
    if attn.norm_q is not None:
        q = attn.norm_q(q)
    if attn.norm_k is not None:
        k = attn.norm_k(k)
    q = q.transpose(1, 2).unflatten(1, (attn.heads, -1))                      # (B, h, d, N)
    k = k.transpose(1, 2).unflatten(1, (attn.heads, -1)).transpose(2, 3)      # (B, h, N, d)
    v = v.transpose(1, 2).unflatten(1, (attn.heads, -1))                      # (B, h, d, N)
    q, k = F.relu(q).float(), F.relu(k).float()
    v = F.pad(v.float(), (0, 0, 0, 1), mode="constant", value=1.0)            # (B, h, d + 1, N)
    h = torch.cat([torch.matmul(torch.matmul(v[..., s:e], k[:, :, s:e]), q[..., s:e]) for s, e in segs], -1)
    h = h[:, :, :-1] / (h[:, :, -1:] + 1e-15)
    h = attn.to_out[1](attn.to_out[0](h.flatten(1, 2).transpose(1, 2).to(dtype)))
    return h.clip(-65504, 65504) if dtype == torch.float16 else h


def _cross_attn(attn, x: torch.Tensor, ctx: torch.Tensor, segs, ctx_len: int) -> torch.Tensor:
    """Softmax cross-attention (as trt_engine.SharedContextAttnProcessor, batch 1), segment i attending to rows
    i * ctx_len .. (i + 1) * ctx_len of `ctx`."""
    q, k, v = attn.to_q(x), attn.to_k(ctx), attn.to_v(ctx)
    if attn.norm_q is not None:
        q = attn.norm_q(q)
    if attn.norm_k is not None:
        k = attn.norm_k(k)
    h = attn.heads
    d = k.shape[-1] // h
    outs = []
    for i, (s, e) in enumerate(segs):
        qi = q[:, s:e].unflatten(-1, (h, d)).transpose(1, 2)
        ki = k[:, i * ctx_len:(i + 1) * ctx_len].unflatten(-1, (h, d)).transpose(1, 2)
        vi = v[:, i * ctx_len:(i + 1) * ctx_len].unflatten(-1, (h, d)).transpose(1, 2)
        outs.append(F.scaled_dot_product_attention(qi, ki, vi).transpose(1, 2).flatten(2))
    return attn.to_out[1](attn.to_out[0](torch.cat(outs, 1).to(q.dtype))) / attn.rescale_output_factor


def _ffn(ff, nh: torch.Tensor, segs, grids) -> torch.Tensor:
    """GLUMBConv: its 1x1 convolutions once over all tokens (as a 1 x N image), the 3x3 depthwise one per segment."""
    x = nh.transpose(1, 2).unsqueeze(2)                                       # (1, C, 1, N)
    res = x
    x = ff.nonlinearity(ff.conv_inverted(x))
    x = torch.cat([ff.conv_depth(x[..., s:e].reshape(1, -1, g[0], g[1])).flatten(2) for (s, e), g in zip(segs, grids)],
                  -1).unsqueeze(2)
    x, gate = torch.chunk(x, 2, dim=1)
    x = ff.conv_point(x * ff.nonlinearity(gate))
    if ff.norm_type == "rms_norm":
        x = ff.norm(x.movedim(1, -1)).movedim(-1, 1)
    if ff.residual_connection:
        x = x + res
    return x.squeeze(2).transpose(1, 2)


def _spread(mod: torch.Tensor, segs) -> torch.Tensor:
    """(k, C), one row per segment -> (1, N, C) over the segments' tokens."""
    return torch.cat([mod[i:i + 1].expand(e - s, -1) for i, (s, e) in enumerate(segs)])[None]


def _block(block, h: torch.Tensor, ctx: torch.Tensor, temb: torch.Tensor, segs, grids, ctx_len: int):
    """VIBE's SanaTransformerBlock.forward over a packed sequence; temb (k, 6 C), one row per segment."""
    mods = block.scale_shift_table[None] + temb.reshape(len(segs), 6, -1)
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = [_spread(mods[:, j], segs) for j in range(6)]
    nh = (block.norm1(h) * (1 + scale_msa) + shift_msa).to(h.dtype)
    h = h + gate_msa * _linear_attn(block.attn1, nh, segs)
    if block.attn2 is not None:
        h = _cross_attn(block.attn2, h, ctx, segs, ctx_len) + h
    nh = block.norm2(h) * (1 + scale_mlp) + shift_mlp
    return h + gate_mlp * _ffn(block.ff, nh, segs, grids)


class PackedGraph(nn.Module):
    """velocity(x_i ++ reference latent_i, projected context_i, scaled timestep_i) for each image i in one call, `tr`
    prepared as `trt_engine.export` prepares it (LoRA merged, caption projection outside, fp16). Arguments: the
    images' inputs (1, 64, h_i, w_i), then their contexts (1, L, D), then t (k,)."""

    def __init__(self, tr: nn.Module, grids: Sequence[Tuple[int, int]]):
        super().__init__()
        self.tr, self.grids = tr, [tuple(g) for g in grids]
        n = [g[0] * g[1] for g in self.grids]
        self.segs = [(sum(n[:i]), sum(n[:i + 1])) for i in range(len(n))]

    def forward(self, *args) -> List[torch.Tensor]:
        k = len(self.grids)
        xs, encs, t = args[:k], args[k:2 * k], args[2 * k]
        tr = self.tr
        h = torch.cat([tr.patch_embed(x) for x in xs], 1)
        temb, emb_t = tr.time_embed(t, batch_size=k, hidden_dtype=h.dtype)
        ctx = torch.cat(list(encs), 1)
        for block in tr.transformer_blocks:
            if hasattr(block, "attn1"):             # not a removed block (vibe_model.SkipBlock)
                h = _block(block, h, ctx, temb, self.segs, self.grids, encs[0].shape[1])
        shift, scale = (tr.scale_shift_table[None] + emb_t[:, None]).chunk(2, dim=1)      # SanaModulatedNorm
        h = tr.norm_out.norm(h) * (1 + _spread(scale[:, 0], self.segs)) + _spread(shift[:, 0], self.segs)
        h = tr.proj_out(h)
        return [h[:, s:e].unflatten(1, g).permute(0, 3, 1, 2) for (s, e), g in zip(self.segs, self.grids)]


class _Fp16Outs(nn.Module):
    def __init__(self, graph: nn.Module):
        super().__init__()
        self.graph = graph

    def forward(self, *args):
        return [v.to(torch.float16) for v in self.graph(*args)]


# ---------------------------------------------------------------- export and build

def spec_path(engine) -> Path:
    return Path(engine).with_suffix(".json")


def export(model, student, engine, crop: int = 256, quantize: bool = False, calib_frames=()) -> None:
    """Stage 1: the packed graph of `student` (frame at `model.size`, one crop at `crop`) as ONNX next to `engine`,
    and its spec. `quantize`: INT8 as `trt_int8`, calibrated on `calib_frames` (RGB uint8 frames at model.size), each
    with a face-sized crop of its centre zoomed to `crop` and its own embedding of one of the student's instructions,
    along every student step from the shared noise (the crop's: its centre) and the carried step. Modifies model.tr
    in place."""
    import cv2

    from . import lora as lora_mod
    from .presets import ALL_PROMPTS
    from .vibe_model import flow_step
    state, meta = lora_mod.load(student)
    if int(meta.get("size", model.size)) != model.size:
        raise ValueError(f"student trained at {meta['size']}, exporting at {model.size}")
    if sorted(meta.get("removed_blocks") or []) != model.removed_blocks:
        raise ValueError(f"student trained without blocks {meta.get('removed_blocks')}, model without "
                         f"{model.removed_blocks}")
    print(f"[packed] merged {lora_mod.merge(model.tr, state)} LoRA layers from {student}", flush=True)
    gf, gc = model.size // 32, crop // 32
    off = (gf - gc) // 2
    noise_f = state[lora_mod.NOISE_KEY].to(model.device, torch.float32)
    noise_c = noise_f[:, :, off:off + gc, off:off + gc].contiguous()
    ts, sg = model.schedules[int(meta.get("student_steps", 2))]
    tsc = [float(t) * float(model.tscale) for t in ts]
    keys = list(meta.get("prompts", {})) or ["anime", "oil"]

    def inputs(frame, key):
        side = model.size // 4
        c = (model.size - side) // 2
        zoom = cv2.resize(frame[c:c + side, c:c + side], (crop, crop), interpolation=cv2.INTER_CUBIC)
        prompt = ALL_PROMPTS.get(key, key)
        enc_f = model.project_embedding(model.embed(prompt, frame)).half()
        enc_c = model.project_embedding(model.embed(prompt, zoom)).half()
        il_f = model.encode([frame]).half()
        size, model.size = model.size, crop
        il_c = model.encode([zoom]).half()
        model.size = size
        return il_f, il_c, enc_f, enc_c

    frames = list(calib_frames)
    pairs = [inputs(f, keys[i % len(keys)]) for i, f in enumerate(frames)]
    tr = model.tr
    tr.caption_projection, tr.caption_norm = nn.Identity(), nn.Identity()
    tr.half()
    graph = PackedGraph(tr, [(gf, gf), (gc, gc)]).eval()
    if quantize:
        from . import trt_int8
        calls = []
        s = float(meta.get("carry_sigma") or 0.5)
        with torch.inference_mode():
            for il_f, il_c, enc_f, enc_c in pairs:
                xf, xc = noise_f.clone(), noise_c.clone()
                for t, s0, s1 in zip(tsc, sg[:-1], sg[1:]):
                    call = (torch.cat([xf.half(), il_f], 1), torch.cat([xc.half(), il_c], 1), enc_f, enc_c,
                            torch.full((2,), t, device=model.device).half())
                    calls.append(call)
                    vf, vc = graph(*call)
                    xf, xc = flow_step(xf, vf.float(), s0, s1), flow_step(xc, vc.float(), s0, s1)
                calls.append((torch.cat([((1 - s) * xf + s * noise_f).half(), il_f], 1),
                              torch.cat([((1 - s) * xc + s * noise_c).half(), il_c], 1), enc_f, enc_c,
                              torch.full((2,), 1000.0 * s * float(model.tscale), device=model.device).half()))
        graph = _Fp16Outs(trt_int8.quantize(graph, calls).graph)
    il_f, il_c, enc_f, enc_c = pairs[0]
    args = (torch.cat([noise_f.half(), il_f], 1), torch.cat([noise_c.half(), il_c], 1), enc_f, enc_c,
            torch.full((2,), tsc[-1], device=model.device).half())
    engine = Path(engine)
    onnx = engine.parent / f"{engine.stem}_onnx" / f"{engine.stem}.onnx"
    onnx.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    with torch.no_grad() if quantize else torch.inference_mode():
        torch.onnx.export(graph, args, str(onnx), input_names=["xa", "xb", "enca", "encb", "t"],
                          output_names=["va", "vb"], opset_version=19 if quantize else 17, do_constant_folding=True,
                          dynamo=False)
    print(f"[packed] {onnx} written in {time.perf_counter() - t0:.0f} s", flush=True)
    spec_path(engine).write_text(json.dumps(dict(
        student=str(student), student_sha256=sha256(student), size=model.size, crop=crop,
        removed_blocks=model.removed_blocks or None, precision="int8" if quantize else "fp16", tensorrt="",
        platform=""), indent=1), encoding="utf-8")


def build(engine, workspace_gb: float = 6.0) -> None:
    """Stage 2: the strongly typed engine from stage 1's ONNX graph."""
    from .trt_engine import build_onnx
    engine = Path(engine)
    spec = json.loads(spec_path(engine).read_text(encoding="utf-8"))
    spec["tensorrt"] = build_onnx(engine.parent / f"{engine.stem}_onnx" / f"{engine.stem}.onnx", engine, workspace_gb)
    spec["platform"] = sys.platform
    spec_path(engine).write_text(json.dumps(spec, indent=1), encoding="utf-8")


# ---------------------------------------------------------------- runtime

class PackedEngine:
    """The packed engine (frame + one crop, batch 1) replayed as a CUDA graph on fixed buffers. Its outputs are
    reused: consume them before the next call."""

    def __init__(self, engine, device: str = "cuda"):
        import tensorrt as trt
        self.spec = json.loads(spec_path(engine).read_text(encoding="utf-8"))
        if self.spec.get("tensorrt") not in ("", trt.__version__) or self.spec.get("platform") not in ("", sys.platform):
            raise RuntimeError(f"{engine} was built with TensorRT {self.spec['tensorrt']} on {self.spec['platform']}")
        self.rt = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = self.rt.deserialize_cuda_engine(Path(engine).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.buf = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            self.buf[name] = torch.empty(tuple(self.engine.get_tensor_shape(name)), dtype=torch.float16, device=device)
            self.ctx.set_tensor_address(name, self.buf[name].data_ptr())
        self.device = device
        self._graph = None
        self.calls = 0

    def __call__(self, xa, xb, enca, encb, t):
        for k, v in (("xa", xa), ("xb", xb), ("enca", enca), ("encb", encb), ("t", t)):
            self.buf[k].copy_(v)
        if self._graph is None:
            self._graph = capture(self.ctx, self.device)
        else:
            self._graph.replay()
        self.calls += 1
        return self.buf["va"], self.buf["vb"]


def _crop_steps(ed: StreamEditor, fb):
    """The face pass's crop editor's `process` for a batch (no cut, no face pass of its own) as a generator."""
    fill = ed.carry_fill()
    carry = None if fb.plan is None else ed.carry.start(fb.crops, fb.plan, ed.noise.shape[-1], fill)
    out = yield from ed.edit_steps(fb.crops, carry)
    if ed.carry is not None:
        ed.carry.update(fb.crops, out, fill, 0)
    return ed.tfilter(fb.crops, out, fb.flows) if ed.tfilter is not None else out


def _crop_hold(ed: StreamEditor, fb) -> torch.Tensor:
    """A batch whose crop is not edited: the crop editor's last output warped to it (the carry-over's guess), which is
    also its next reference."""
    fill = ed.carry_fill()
    guess, _, _ = ed.carry.start(fb.crops, fb.plan, ed.noise.shape[-1], fill)
    ed.carry.update(fb.crops, guess, fill, 0)
    return ed.tfilter(fb.crops, guess, fb.flows) if ed.tfilter is not None else guess


def _send(gen, v):
    try:
        return gen.send(v), False
    except StopIteration as done:
        return done.value, True


class PackedStreamEditor(StreamEditor):
    """A StreamEditor (batch 1, with a face pass) whose batches with a face to edit run the frame's and the crop's
    transformer calls through `packed`. Cut batches and the keyframe that takes an embedding made on the side take the
    ordinary path. `face_every` n > 1 edits the crop on every n-th batch only and pastes its last edit, warped, in
    between (the frame's call runs alone there)."""

    def __init__(self, *args, packed: Optional[PackedEngine] = None, face_every: int = 1, **kw):
        super().__init__(*args, **kw)
        if packed is not None and self.batch_size != 1:
            raise ValueError("the packed engine runs batch 1")
        self.packed = packed
        self.face_every = face_every
        self._face_n = 0

    def _process(self, frames, flows, plan, face, cut, fill):
        side_due = plan is None and self._side is not None and self._side.ready()
        if self.packed is None or face is None or cut is not None or side_due:
            return super()._process(frames, flows, plan, face, cut, fill)
        fp = self.face
        self._face_n += 1
        hold = (self.face_every > 1 and self._face_n % self.face_every != 0 and face.plan is not None
                and fp._emb is not None and fp._emb_for is self.emb)
        crop = None if hold or not fp.begin(face, self) else _crop_steps(fp.ed, face)
        out, crop_out = self._drive(
            self.edit_steps(frames, None if plan is None else self.carry.start(frames, plan, self.noise.shape[-1],
                                                                                 fill)), crop)
        if hold:
            crop_out = _crop_hold(fp.ed, face)
        if self.carry is not None:
            self.carry.update(frames, out, fill, 0)
        if self.tfilter is not None:
            out = self.tfilter(frames, out, flows)
        return fp.paste(out, face, crop_out) if crop_out is not None else out

    def _drive(self, main, crop):
        """Run the frame's and the crop's edits to their ends, pairing their transformer calls."""
        ced = self.face.ed
        ra, da = _send(main, None)
        rb, db = _send(crop, None) if crop is not None else (None, True)
        while not (da and db):
            if not da and not db:
                (xa, ila, ta), (xb, ilb, tb) = ra, rb
                va, vb = self.packed(torch.cat([xa.half(), ila], 1), torch.cat([xb.half(), ilb], 1), self.enc16,
                                     ced.enc16, torch.cat([ta.reshape(-1)[:1], tb.reshape(-1)[:1]]).half())
                ra, da = _send(main, va.float())
                rb, db = _send(crop, vb.float())
            elif not da:
                ra, da = _send(main, self.velocity(*ra))
            else:
                rb, db = _send(crop, ced.velocity(*rb))
        return ra, rb
