"""The streaming loop: batches of frames -> taesana encode -> k student steps -> taesana decode, all on the GPU.

Per clip, the instruction embedding is computed once, from the first frame (Qwen3-VL over instruction + frame), and
for the engine projected in fp32. Per batch only the tiny encoder, k transformer calls and the tiny decoder run.
The transformer is either a TensorRT engine (`trt_engine.TRTTransformer`, the fast path) or VIBE's own PyTorch
transformer in bf16 with the student merged.

An optional `temporal_filter.TemporalFilter` blends each decoded batch with the previously shown frames along the
input's optical flow; its flows are computed on the host while the GPU works on the previous batch. An optional
`carry.Carry` starts each batch after a keyframe from the previous batch's output (one transformer call instead of
k); its flows are computed the same way.

Throughput figures, defined as in dreamlite-stream:
  step-only  GPU time of `edit` (CUDA events) on uint8 frames already on the GPU.
  loop       wall time per batch in the steady state of `run`: host frames -> pinned buffer -> GPU -> edit -> host
             frames, with the next batch's preparation overlapping the current batch's compute. Video decoding is
             not included (frames are decoded beforehand), as in dreamlite-stream's sustained_throughput_test.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch

from . import lora as lora_mod
from .carry import Carry
from .taesana import TinyAutoencoder
from .temporal_filter import TemporalFilter
from .trt_engine import TRTTransformer
from .vibe_model import VibeModel, flow_step


@dataclass
class RunTiming:
    frames: int = 0                  # frames in the measured (steady-state, full) batches
    loop_s: float = 0.0
    step_ms: List[float] = field(default_factory=list)
    embed_ms: float = 0.0            # once per clip, not in either rate

    @property
    def loop_fps(self) -> float:
        return self.frames / self.loop_s if self.loop_s else 0.0

    @property
    def step_fps(self) -> float:
        return self.frames / (sum(self.step_ms) / 1000) if self.step_ms else 0.0


class StreamEditor:
    """Batched frame editor for one student. Give either `engine` (TensorRT) or `student` (a LoRA file, PyTorch)."""

    def __init__(self, vibe: VibeModel, tae: TinyAutoencoder, *, engine: Optional[TRTTransformer] = None,
                 student=None, batch_size: int = 8, temporal_filter: Optional[TemporalFilter] = None,
                 carry: Optional[Carry] = None):
        if (engine is None) == (student is None):
            raise ValueError("give exactly one of engine / student")
        self.vibe, self.tae, self.engine = vibe, tae, engine
        self.device = vibe.device
        if engine is not None:
            sp = engine.spec
            if sp.size != vibe.size:
                raise ValueError(f"engine built for {sp.size}, model at {vibe.size}")
            self.batch_size = sp.batch
            self.steps = sp.student_steps
            self.noise = engine.noise
            self.sigmas = sp.sigmas
            self.t_engine = [torch.full((sp.batch,), t, dtype=torch.float32, device=self.device) for t in sp.timesteps]
            self.prompts = sp.prompts
        else:
            state, meta = lora_mod.load(student)
            if int(meta.get("size", vibe.size)) != vibe.size:
                raise ValueError(f"student trained at {meta['size']}, model at {vibe.size}")
            lora_mod.merge(vibe.tr, state)
            self.batch_size = batch_size
            self.steps = int(meta.get("student_steps", 1))
            self.noise = state[lora_mod.NOISE_KEY].to(self.device, torch.float32)
            self.timesteps, self.sigmas = vibe.schedules[self.steps]
            self.prompts = meta.get("prompts", {})
        self.emb = None
        self.enc16 = None
        self.tfilter = temporal_filter
        if carry is not None and self.steps < 2:
            raise ValueError("the carry-over needs a student with at least 2 steps")
        self.carry = carry
        # the noise level a student was trained to start its carried step from (an engine's spec does not record it)
        self.trained_carry_sigma = None if engine is not None else meta.get("carry_sigma")
        self._t_carry = {}
        # TensorRT adds stream synchronisations when enqueued on the legacy default stream
        self.stream = torch.cuda.Stream(self.device)

    # ---------------------------------------------------------------- per clip
    def set_prompt(self, prompt: str, frame: np.ndarray) -> float:
        """Embed the instruction with this frame; returns the time it took in ms."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self.emb = self.vibe.embed(prompt, frame)
        for state in (self.tfilter, self.carry):
            if state is not None:
                state.reset()
        if self.engine is not None:
            self.enc16 = self.vibe.project_embedding(self.emb).half().expand(self.batch_size, -1, -1).contiguous()
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - t0)

    # ---------------------------------------------------------------- per batch
    def carry_start(self):
        """(timestep, sigma) of the carried step: Carry(sigma=...) if given, else the student's trained value, else
        the last step's (whose timestep is then reused exactly)."""
        s = self.carry.sigma if self.carry is not None and self.carry.sigma is not None else self.trained_carry_sigma
        if s is None or abs(s - self.sigmas[-2]) < 1e-6:
            i = self.steps - 1
            return (self.t_engine[i] if self.engine is not None else self.timesteps[i]), self.sigmas[-2]
        s = float(s)
        if s not in self._t_carry:
            self._t_carry[s] = (torch.full((self.batch_size,), 1000.0 * s, dtype=torch.float32, device=self.device)
                                if self.engine is not None else torch.tensor(1000.0 * s, device=self.device))
        return self._t_carry[s], s

    @torch.inference_mode()
    def edit(self, frames: torch.Tensor, carry=None) -> torch.Tensor:
        """(B, H, W, 3) uint8 on the GPU -> edited (B, H, W, 3) uint8 on the GPU. With `carry` = (warped previous
        output (B, H, W, 3) uint8, weight (B, 1, h, w) on the latent grid) one step runs, from
        (1 - s) * (w * enc(warped) + (1 - w) * enc(frames)) + s * noise, s from `carry_start` (see `carry`)."""
        if self.emb is None:
            raise RuntimeError("call set_prompt first")
        il = self.tae.encode_u8(frames)
        x = self.noise.expand(len(frames), -1, -1, -1)
        if carry is None:
            plan = [(self.t_engine[i] if self.engine is not None else self.timesteps[i], self.sigmas[i], self.sigmas[i + 1])
                    for i in range(self.steps)]
        else:
            guess, w = carry
            t, s = self.carry_start()
            x = (1 - s) * (w * self.tae.encode_u8(guess).float() + (1 - w) * il.float()) + s * x
            plan = [(t, s, self.sigmas[-1])]
        for t, s0, s1 in plan:
            if self.engine is not None:
                v = self.engine(torch.cat([x.half(), il], 1), self.enc16, t).float()
            else:
                v = self.vibe.velocity(x, il, self.emb, t)
            x = flow_step(x, v, s0, s1)
        return self.tae.decode_u8(x)

    @torch.inference_mode()
    def run(self, frames: Sequence[np.ndarray], warmup_batches: int = 2) -> tuple[List[np.ndarray], RunTiming]:
        """Edit a list of RGB uint8 frames in batches; a short last batch is padded with its last frame.

        Double-buffered: batch b is copied into a pinned buffer while the GPU still works on batch b - 1, and the
        host waits only for the result of b - 1 after b has been queued. Everything runs on `self.stream`. Each call
        is one clip: the temporal filter and the carry-over, if any, start afresh.
        """
        for state in (self.tfilter, self.carry):
            if state is not None:
                state.reset()
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.stream):
            return self._run(frames, warmup_batches)

    def _run(self, frames, warmup_batches):
        B = self.batch_size
        h, w = frames[0].shape[:2]
        pin_in = [torch.empty((B, h, w, 3), dtype=torch.uint8).pin_memory() for _ in range(2)]
        pin_out = [torch.empty((B, h, w, 3), dtype=torch.uint8).pin_memory() for _ in range(2)]
        done = [torch.cuda.Event() for _ in range(2)]
        ev0 = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        ev1 = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
        outs: List[np.ndarray] = []
        timing = RunTiming()
        n_batches = (len(frames) + B - 1) // B
        pending = None
        t_last = time.perf_counter()

        def collect(slot, count, b):
            nonlocal t_last
            done[slot].synchronize()
            outs.extend(pin_out[slot].numpy()[:count].copy())
            now = time.perf_counter()
            if b >= warmup_batches and count == B:
                timing.frames += B
                timing.loop_s += now - t_last
                timing.step_ms.append(ev0[slot].elapsed_time(ev1[slot]))
            t_last = now

        for b in range(n_batches):
            s = b % 2
            chunk = frames[b * B:(b + 1) * B]
            arr = pin_in[s].numpy()
            for j in range(B):
                arr[j] = chunk[min(j, len(chunk) - 1)]
            flows = self.tfilter.flows(arr) if self.tfilter is not None else None
            plan = self.carry.plan(arr) if self.carry is not None else None
            gpu = pin_in[s].to(self.device, non_blocking=True)
            ev0[s].record()
            out = self.edit(gpu, None if plan is None else self.carry.start(gpu, plan, self.noise.shape[-1]))
            if self.carry is not None:
                self.carry.update(gpu, out)
            if self.tfilter is not None:
                out = self.tfilter(gpu, out, flows)
            ev1[s].record()
            pin_out[s].copy_(out, non_blocking=True)
            done[s].record()
            if pending is not None:
                collect(*pending)
            pending = (s, len(chunk), b)
        if pending is not None:
            collect(*pending)
        return outs, timing
