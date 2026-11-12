"""The streaming loop: batches of frames -> taesana encode -> k student steps -> taesana decode, all on the GPU.

Per clip, the instruction embedding is computed once, from the first frame (Qwen3-VL over instruction + frame), and
for the engine projected in fp32. Per batch only the tiny encoder, k transformer calls and the tiny decoder run.
The transformer is either a TensorRT engine (`trt_engine.TRTTransformer`, the fast path) or VIBE's own PyTorch
transformer in bf16 with the student merged.

An optional `temporal_filter.TemporalFilter` blends each decoded batch with the previously shown frames along the
input's optical flow; its flows are computed on the host while the GPU works on the previous batch. An optional
`carry.Carry` starts each batch after a keyframe from the previous batch's output (one transformer call instead of
k); its flows are computed the same way. For a live source, where a batch exists only once its last frame has
arrived, `live.LiveEditor` computes them as the frames arrive instead.

The embedding depends on the frame it is made with, and a keyframe (k steps from noise) follows it closely: after a
scene cut, an embedding made with the old scene paints that scene into the new one (swans over a breakdance crowd),
while carried frames, which start mostly from the frame itself, barely depend on it. With the carry-over, which
detects cuts (`Carry.cut_at`), the instruction is embedded again with the first frame after the cut before the
keyframe batch runs (`reembed_at_cut`, on by default; `reset_clip` goes back to the first frame's). The engine attends
to one embedding per batch, so the batch's frames before the cut, which the new one would paint with the next scene,
are carried over with the old one in a pass of their own (one more transformer call on that batch); the colour map of
the LUT fill is fitted on the frames from the cut on, and the temporal filter does not blend the frame at the cut
with the one before it. On two sequences spliced from three DAVIS clips each (blackswan | breakdance | libby,
parkour | motorbike | bmx-trees), `lam4_carry_s05_lut_2step_640` (fp16 engine) with carry-over + filter 0.7 / 0.3,
oil / anime / ukiyoe / night, LPIPS to each scene streamed as a clip of its own, without / with: 0.30 / 0.10 over the
16 frames after a cut, 0.34 / 0.01 over the cut batch's frames before it, 0.17 / 0.05 over the whole sequence; CLIP
direction 0.016 / 0.002 below the separate clips' on average. A cut costs one embedding (170 ms on an RTX 3090 Ti) and
the extra pass, on the stream's thread, so a live stream stalls for them: at a 30 fps source (640 INT8, batch 8) the
cut batch comes out about 290 ms later and the next about 230 ms, and two batches on the delay is back to normal. On
35 DAVIS clips without a cut, 7 of 267 carried batches (in four clips with fast motion) count as cuts; there the
embedding is redone with a later frame of the same scene. Without the carry-over no cut is detected and the embedding
stays the first frame's.

An optional `face.FacePass` edits the main face again from a crop at a size where it covers enough latent tokens and
pastes it into the shown output (see `face`).

Throughput figures, defined as in dreamlite-stream:
  step-only  GPU time of `edit` (CUDA events) on uint8 frames already on the GPU.
  loop       wall time per batch in the steady state of `run`: host frames -> pinned buffer -> GPU -> edit -> host
             frames, with the next batch's preparation overlapping the current batch's compute. Video decoding is
             not included (frames are decoded beforehand), as in dreamlite-stream's sustained_throughput_test.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np
import torch

from . import lora as lora_mod
from .carry import Carry
from .face import FaceBatch, FacePass
from .taesana import TinyAutoencoder
from .temporal_filter import TemporalFilter
from .trt_engine import TRTTransformer
from .vibe_model import VibeModel, flow_step


class SideEmbedding:
    """An instruction embedding made on a thread and a CUDA stream of their own, so that the stream does not wait for
    it: `make(frame)` -> (emb, enc16). `ready` once the thread is done; `take` hands the result over with the taking
    stream made to wait for it on the GPU (no wait on the host)."""

    def __init__(self, make, frame: np.ndarray, stream: torch.cuda.Stream):
        self.done = torch.cuda.Event()
        self.result = self.error = None
        self.ms = None
        t0 = time.perf_counter()

        def work():
            try:
                with torch.cuda.stream(stream):
                    self.result = make(frame)
                    self.done.record(stream)
                stream.synchronize()
                self.ms = 1000 * (time.perf_counter() - t0)
            except BaseException as e:          # surfaced by `take`
                self.error = e

        self.thread = threading.Thread(target=work, daemon=True)
        self.thread.start()

    def ready(self) -> bool:
        return not self.thread.is_alive()

    def take(self, stream: torch.cuda.Stream):
        self.thread.join()
        if self.error is not None:
            raise self.error
        stream.wait_event(self.done)
        return self.result


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
                 carry: Optional[Carry] = None, reembed_at_cut: bool = True, reembed_async: bool = False,
                 face: Optional[FacePass] = None):
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
            if "removed_blocks" in meta and sorted(meta["removed_blocks"]) != vibe.removed_blocks:
                raise ValueError(f"student trained without transformer blocks {meta['removed_blocks']}, model without "
                                 f"{vibe.removed_blocks} (VibeModel(remove_blocks=...))")
            lora_mod.merge(vibe.tr, state)
            self.batch_size = batch_size
            self.steps = int(meta.get("student_steps", 1))
            self.noise = state[lora_mod.NOISE_KEY].to(self.device, torch.float32)
            self.timesteps, self.sigmas = vibe.schedules[self.steps]
            self.prompts = meta.get("prompts", {})
        tae_shape = (getattr(tae, "batch", self.batch_size), getattr(tae, "size", vibe.size))
        if tae_shape != (self.batch_size, vibe.size):
            raise ValueError(f"taesana engines built for batch {tae_shape[0]} at {tae_shape[1]}, "
                             f"streaming batch {self.batch_size} at {vibe.size}")
        self.emb = None
        self.enc16 = None
        self._prompt = self._reference = None
        self._clip_emb = (None, None)
        self.reembed_at_cut = reembed_at_cut
        self.reembed_ms: List[float] = []       # one entry per embedding redone at a cut
        self.tfilter = temporal_filter
        if carry is not None and self.steps < 2:
            raise ValueError("the carry-over needs a student with at least 2 steps")
        self.carry = carry
        # the noise level a student was trained to start its carried step from (engines built before it was recorded
        # in the spec have None)
        self.trained_carry_sigma = engine.spec.carry_sigma if engine is not None else meta.get("carry_sigma")
        self.trained_carry_fill = engine.spec.carry_fill if engine is not None else meta.get("carry_fill")
        self._t_carry = {}
        if face is not None and face.batch_size != self.batch_size:
            raise ValueError(f"face pass at batch {face.batch_size}, streaming batch {self.batch_size}")
        self.face = face
        if reembed_async and (carry is None or not reembed_at_cut):
            raise ValueError("reembed_async goes with reembed_at_cut and the carry-over (which detects the cuts)")
        self.reembed_async = reembed_async
        self.live = False
        self._side: Optional[SideEmbedding] = None       # an embedding being made on the side after a cut
        self._side_next: Optional[np.ndarray] = None     # the frame of a later cut, to embed with once that is done
        self.early_side = None      # (frame index, SideEmbedding) LiveEditor's host thread started at a cut's frame
        self.side_stream = torch.cuda.Stream(self.device, priority=-1)   # ahead of the stream's own kernels
        # TensorRT adds stream synchronisations when enqueued on the legacy default stream
        self.stream = torch.cuda.Stream(self.device)

    # ---------------------------------------------------------------- per clip
    def set_prompt(self, prompt: str, frame: np.ndarray, reference=None) -> float:
        """Embed the instruction with this frame, and with a reference image after it if one is given
        (`VibeModel.embed_with_reference`); returns the time it took in ms."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        self._prompt, self._reference = prompt, reference
        self._embed(frame)
        self._clip_emb = (self.emb, self.enc16)
        self.reset_clip()
        torch.cuda.synchronize()
        return 1000 * (time.perf_counter() - t0)

    def reset_clip(self) -> None:
        """Start a clip afresh: the temporal filter and the carry-over forget the previous frames, and an embedding
        redone at a cut goes back to `set_prompt`'s."""
        for state in (self.tfilter, self.carry, self.face):
            if state is not None:
                state.reset()
        self.emb, self.enc16 = self._clip_emb
        self._side = self._side_next = self.early_side = None   # an embedding still being made is dropped

    def _make_embedding(self, frame: np.ndarray):
        """(raw embedding, the engine's projected context or None) of the instruction given to `set_prompt` with this
        frame (and its reference image, if any)."""
        emb = (self.vibe.embed(self._prompt, frame) if self._reference is None
               else self.vibe.embed_with_reference(self._prompt, frame, self._reference))
        enc16 = (self.vibe.project_embedding(emb).half().expand(self.batch_size, -1, -1).contiguous()
                 if self.engine is not None else None)
        return emb, enc16

    def _embed(self, frame: np.ndarray) -> None:
        self.emb, self.enc16 = self._make_embedding(frame)

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
            self._t_carry[s] = (torch.full((self.batch_size,), 1000.0 * s * self.vibe.tscale, dtype=torch.float32,
                                           device=self.device)
                                if self.engine is not None else torch.tensor(1000.0 * s, device=self.device))
        return self._t_carry[s], s

    def carry_fill(self) -> str:
        """What fills in where the flow fails: Carry(fill=...) if given, else the student's, else the frame."""
        if self.carry is not None and self.carry.fill:
            return self.carry.fill
        return self.trained_carry_fill or "frame"

    @torch.inference_mode()
    def edit(self, frames: torch.Tensor, carry=None) -> torch.Tensor:
        """(B, H, W, 3) uint8 on the GPU -> edited (B, H, W, 3) uint8 on the GPU. With `carry` = (warped previous
        output (B, H, W, 3) uint8, weight (B, 1, h, w) on the latent grid, fill (B, H, W, 3) uint8 or None for the
        frames) one step runs, from (1 - s) * (w * enc(warped) + (1 - w) * enc(fill)) + s * noise, s from
        `carry_start` (see `carry`)."""
        if self.emb is None:
            raise RuntimeError("call set_prompt first")
        il = self.tae.encode_u8(frames)
        x = self.noise.expand(len(frames), -1, -1, -1)
        if carry is None:
            plan = [(self.t_engine[i] if self.engine is not None else self.timesteps[i], self.sigmas[i], self.sigmas[i + 1])
                    for i in range(self.steps)]
        else:
            guess, w, fill = carry
            t, s = self.carry_start()
            zf = il if fill is None else self.tae.encode_u8(fill)
            x = (1 - s) * (w * self.tae.encode_u8(guess).float() + (1 - w) * zf.float()) + s * x
            plan = [(t, s, self.sigmas[-1])]
        for t, s0, s1 in plan:
            if self.engine is not None:
                v = self.engine(torch.cat([x.half(), il], 1), self.enc16, t).float()
            else:
                v = self.vibe.velocity(x, il, self.emb, t)
            x = flow_step(x, v, s0, s1)
        return self.tae.decode_u8(x)

    @torch.inference_mode()
    def process(self, frames: torch.Tensor, flows=None, plan=None, face: Optional[FaceBatch] = None,
                host: Optional[np.ndarray] = None, live: bool = False) -> torch.Tensor:
        """GPU side of one staged batch: `edit` (carried when `plan` is given), the carry-over's bookkeeping, the
        temporal filter and the face pass. `flows` / `plan` / `face` come from the batch's `TemporalFilter.flows` /
        `Carry.plan` / `FacePass.stage`; `host` is the batch's frames on the host, if the caller has them. At a cut
        (`Carry.cut_at`) with `reembed_at_cut`, the frames before it are carried over with the old embedding and the
        batch is keyframed with one made from the frame at the cut (module docstring). With `reembed_async` and `live`
        (LiveEditor) the cut batch is carried over whole with the old embedding while the new one is made on the side,
        and the first batch after it is ready is a keyframe with it: the stream does not wait for the text encoder.
        Offline (`run`) the stall costs only throughput, and waiting gives the better frames after the cut."""
        cut = self.carry.cut_at if self.carry is not None and plan is None and self.reembed_at_cut else None
        fill = self.carry_fill()
        self.live = live                # the face pass embeds on the side too when this is set (and reembed_async)
        if cut is not None and self.reembed_async and live:
            out = self._process_cut_async(frames, flows, face, host, cut, fill)
        else:
            out = self._process(frames, flows, plan, face, cut, fill)
        self.early_side = None
        if self._side is not None and self._side.ready():
            if self._side_next is not None:          # a later cut came while it was made: embed with that one
                self._side = SideEmbedding(self._make_embedding, self._side_next, self.side_stream)
                self._side_next = None
            else:
                self.carry.force_key = True          # the next batch is a keyframe with it (`_process`)
        return out

    def side_check(self) -> None:
        """Just before a batch is planned (LiveEditor): an embedding made on the side that is ready by now makes this
        batch the keyframe that takes it, one batch earlier than the check at the end of `process`."""
        if self._side is not None and self._side_next is None and self._side.ready():
            self.carry.force_key = True

    def _process_cut_async(self, frames, flows, face, host, cut, fill):
        frame = host[cut].copy() if host is not None else frames[cut].cpu().numpy()
        early, self.early_side = self.early_side, None
        if self._side is None or self._side.ready():      # one made for an earlier scene is dropped
            self._side = (early[1] if early is not None and early[0] == cut
                          else SideEmbedding(self._make_embedding, frame, self.side_stream))
            self._side_next = None
        else:
            self._side_next = frame
        # the frames past the cut, whose flows do not hold, start from their fill; one transformer call for all
        out = self.edit(frames, self.carry.start(frames, self.carry.cut_flows, self.noise.shape[-1], fill))
        if flows is not None:
            flows[1][cut] = False
        self.carry.update(frames, out, fill, cut)
        if self.tfilter is not None:
            out = self.tfilter(frames, out, flows)
        if face is not None:
            out = self.face.apply(out, face, self)
        return out

    def _process(self, frames, flows, plan, face, cut, fill):
        if plan is None and self._side is not None and self._side.ready():   # the keyframe it waited for
            self.emb, self.enc16 = self._side.take(torch.cuda.current_stream(self.device))
            self.reembed_ms.append(self._side.ms)
            self._side = None
        before = None
        if cut is not None:
            if cut > 0:     # the frames before the cut, carried with the old scene's embedding
                before = self.edit(frames, self.carry.start(frames, self.carry.cut_flows, self.noise.shape[-1], fill))
            if flows is not None:
                flows[1][cut] = False                  # the filter starts afresh at the cut, as at a clip's start
            frame = frames[cut].cpu().numpy()          # waits for the batches queued before this one
            t0 = time.perf_counter()
            self._embed(frame)
            torch.cuda.current_stream(self.device).synchronize()
            self.reembed_ms.append(1000 * (time.perf_counter() - t0))
        out = self.edit(frames, None if plan is None else self.carry.start(frames, plan, self.noise.shape[-1], fill))
        if before is not None:
            out = torch.cat([before[:cut], out[cut:]])
        if self.carry is not None:
            self.carry.update(frames, out, fill, cut or 0)
        if self.tfilter is not None:
            out = self.tfilter(frames, out, flows)
        if face is not None:
            out = self.face.apply(out, face, self)
        return out

    @torch.inference_mode()
    def run(self, frames: Sequence[np.ndarray], warmup_batches: int = 2) -> tuple[List[np.ndarray], RunTiming]:
        """Edit a list of RGB uint8 frames in batches; a short last batch is padded with its last frame.

        Double-buffered: batch b is copied into a pinned buffer while the GPU still works on batch b - 1, and the
        host waits only for the result of b - 1 after b has been queued. The host work of the batches ahead (flows,
        the face pass's detection and crops) runs on worker threads. Everything on the GPU runs on `self.stream`.
        Each call is one clip: the temporal filter and the carry-over, if any, start afresh (`reset_clip`).
        """
        self.reset_clip()
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
        # The host work of the batches ahead (the filter's and the carry-over's flows, as LiveEditor's host thread does
        # it, and the face pass's detection, crops and flows) runs on worker threads, one per kind so that each keeps
        # the batches in order; this thread only stages and queues. Queueing a batch can wait for the GPU (the launch
        # queue holds less than a batch with the face pass), and host work done here would then not overlap the GPU's.
        pools, ahead = {}, {}
        if self.tfilter is not None or self.carry is not None:
            pools["main"] = ThreadPoolExecutor(1)
        if self.face is not None:
            pools["face"] = ThreadPoolExecutor(1)
        ref = [None]            # the carry-over's reference for the host work: the previous batch's last frame (gray)

        def padded(b):
            chunk = frames[b * B:(b + 1) * B]
            return [chunk[min(j, len(chunk) - 1)] for j in range(B)]

        def main_host(batch):
            tfl = [self.tfilter.host_flow(f) for f in batch] if self.tfilter is not None else None
            carry_host = None
            if self.carry is not None:
                grays = [self.carry.gray(f) for f in batch]
                carry_host = (grays, [self.carry.host_flow(g, ref[0]) for g in grays])
                ref[0] = grays[-1]
            return tfl, carry_host

        work = {"main": main_host, "face": self.face.host_batch if self.face is not None else None}

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
            for kind, pool in pools.items():
                for nb in (b, b + 1):
                    if nb < n_batches and (kind, nb) not in ahead:
                        ahead[kind, nb] = pool.submit(work[kind], padded(nb))
            s = b % 2
            chunk = frames[b * B:(b + 1) * B]
            arr = pin_in[s].numpy()
            for j in range(B):
                arr[j] = chunk[min(j, len(chunk) - 1)]
            tfl, carry_host = ahead.pop(("main", b)).result() if "main" in pools else (None, None)
            flows = self.tfilter.flows(arr, tfl) if self.tfilter is not None else None
            plan = self.carry.plan(arr, carry_host) if self.carry is not None else None
            face = self.face.stage(arr, ahead.pop(("face", b)).result()) if "face" in pools else None
            gpu = pin_in[s].to(self.device, non_blocking=True)
            ev0[s].record()
            out = self.process(gpu, flows, plan, face, host=arr)
            ev1[s].record()
            pin_out[s].copy_(out, non_blocking=True)
            done[s].record()
            if pending is not None:
                collect(*pending)
            pending = (s, len(chunk), b)
        if pending is not None:
            collect(*pending)
        for pool in pools.values():
            pool.shutdown()
        return outs, timing
