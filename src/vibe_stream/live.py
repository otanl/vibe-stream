"""Live input: frames arrive one at a time (a camera, a capture) and leave edited with as little delay as the batch
allows.

`StreamEditor.run` stages whole batches, and its host work (the temporal filter's and the carry-over's DIS flows,
about 15 ms per frame at 768) overlaps the previous batch's GPU work, which is what offline throughput needs. A live
batch cannot be staged before its last frame has arrived, so that host work would sit between the last frame and the
GPU: 117 ms per batch of 8 at 768, which with the GPU's 169 ms exceeds the 267 ms a 30 fps source leaves, and the
delay grows without bound. `LiveEditor` computes each frame's flows on a thread as the frame arrives
(`TemporalFilter.host_flow`, `Carry.host_flow`; the carry-over's reference, the previous batch's last frame, is known
by then), so only the last frame's flows lie between it and the GPU, the two computed side by side. It runs the same
operations as `StreamEditor.run` on the same frames; with the fp16 engines and the INT8 ones for 768 batch 8 the
output is identical bit for bit, also across a cut where the instruction is embedded again. The other INT8 engines (768
batch 4 and 2, 640 batch 8) are not bit-reproducible even from one offline run to the next: TensorRT's kernels for
their smaller matrix products sum in a varying order, so a transformer call can differ by up to about 0.1 in velocity,
and the difference fades over the following batches. At 640 batch 8 that happens on every call (LPIPS between two
offline runs of a clip about 0.01).

With a face pass (`vibe_stream.face`), its host work (face detection, the face crops and their flows, about 15 ms per
frame) runs on a second thread beside the first.

At a cut the instruction is embedded again on the thread that calls `step` (`StreamEditor.reembed_at_cut`), and the
frames before the cut get a pass of their own: at 640 on an RTX 3090 Ti that holds the cut batch back by about 290 ms
and the next one by about 230 ms. With `StreamEditor(reembed_async=True)` nothing waits: the host thread starts the
embedding on the side as soon as the frame at the cut arrives (`SideEmbedding`), the cut batch is carried over with
the old embedding, and the first batch planned after the new one is ready (`StreamEditor.side_check`) is a keyframe
with it (README, "Re-embedding at cuts without a stall"). Live output is then no longer identical to `run`'s, which
waits.

    live = LiveEditor(editor)          # after editor.set_prompt(...)
    live.start()
    # capture thread:  live.submit(frame) per frame (RGB uint8 at the editor's size)
    # pipeline thread: while (outs := live.step()) is not None: show(outs)   # the batch's edited frames
    live.stop()                        # from the capture side; a partial last batch is dropped

Delay at a 30 fps source, the INT8 engines with carry-over + filter (scripts/measure_latency.py), mean / 95th
percentile: at 768 batch 8 298 / 415 ms, batch 4 150 / 201 ms, batch 2 80 / 98 ms (batch 1, 26 fps, falls behind);
at 640 250 / 368, 128 / 178 and 68 / 87 ms. With the commercial student at 640 (pruned, INT8, `--fast_post`) and the
engines replayed as CUDA graphs: 229 / 346, 111 / 162 and 56 / 74 ms, batch 1 26 / 28 ms; with the face pass 262 /
379, 134 / 186, 73 / 91 and 44 / 47 ms (README, "Shorter paths inside a batch": the batch size barely changes the
picture).
"""

from __future__ import annotations

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import numpy as np
import torch

from .stream import SideEmbedding, StreamEditor


class LiveEditor:
    def __init__(self, editor: StreamEditor):
        self.ed = editor
        self.batch_size = editor.batch_size
        self._in: queue.Queue = queue.Queue()
        self._ready: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._face_thread: Optional[threading.Thread] = None
        self._pin_in = self._pin_out = None
        self._done = torch.cuda.Event()

    def start(self) -> None:
        """Begin a clip: the temporal filter and the carry-over start afresh (`StreamEditor.reset_clip`), and the host
        thread starts."""
        self.ed.reset_clip()
        self._in, self._ready = queue.Queue(), queue.Queue()
        self._face_in, self._face_ready = queue.Queue(), queue.Queue()
        self._thread = threading.Thread(target=self._host, daemon=True)
        self._thread.start()
        if self.ed.face is not None:
            self._face_thread = threading.Thread(target=self._face_host, daemon=True)
            self._face_thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """One arriving frame, RGB uint8 (H, W, 3) at the editor's size. Returns at once."""
        self._in.put(frame)

    def stop(self) -> None:
        """End the clip; `step` returns None once the complete batches before it are out."""
        self._in.put(None)
        for t in (self._thread, self._face_thread):
            if t is not None:
                t.join()
        self._thread = self._face_thread = None

    def _host(self) -> None:
        tf, cr, fp, B = self.ed.tfilter, self.ed.carry, self.ed.face, self.batch_size
        # the filter's flow on a worker beside the carry-over's (OpenCV releases the GIL): the last frame of a batch
        # then holds the GPU back by the longer of the two (about 6 ms at 640), not by both (about 11 ms)
        pool = ThreadPoolExecutor(1) if tf is not None and cr is not None else None
        ref = None
        while True:
            frames, tfl, grays, res, early = [], [], [], [], None
            for _ in range(B):
                f = self._in.get()
                if fp is not None:
                    self._face_in.put(f)
                if f is None:
                    self._ready.put(None)
                    if pool is not None:
                        pool.shutdown()
                    return
                frames.append(f)
                job = pool.submit(tf.host_flow, f) if pool is not None else None
                if tf is not None and job is None:
                    tfl.append(tf.host_flow(f))
                if cr is not None:
                    grays.append(cr.gray(f))
                    res.append(cr.host_flow(grays[-1], ref))
                    if early is None and self.ed.reembed_async and res[-1] is not None and not res[-1][1]:
                        # the frame at a cut (`Carry.plan` will find the same one): its embedding starts on the side
                        # now, not once the rest of the batch has arrived
                        early = (len(frames) - 1, SideEmbedding(self.ed._make_embedding, f.copy(), self.ed.side_stream))
                if job is not None:
                    tfl.append(job.result())
            if cr is not None:
                ref = grays[-1]
            self._ready.put((frames, tfl if tf is not None else None, (grays, res) if cr is not None else None, early))

    def _face_host(self) -> None:
        """The face pass's host work (detection, crops, their flows) on a thread of its own, beside `_host`."""
        fp, B = self.ed.face, self.batch_size
        while True:
            items = []
            for _ in range(B):
                f = self._face_in.get()
                if f is None:
                    self._face_ready.put(None)
                    return
                items.append(fp.host_frame(f))
            self._face_ready.put(fp.host_end(items))

    @torch.inference_mode()
    def step(self) -> Optional[List[np.ndarray]]:
        """Wait for the next complete batch, edit it and return its frames (RGB uint8 arrays); None after `stop`.
        Call it from one thread."""
        item = self._ready.get()
        if item is None:
            return None
        frames, tfl, carry_host, early = item
        ed = self.ed
        ed.early_side = early
        face_host = self._face_ready.get() if ed.face is not None else None
        h, w = frames[0].shape[:2]
        if self._pin_in is None or self._pin_in.shape[1:3] != (h, w):
            self._pin_in = torch.empty((self.batch_size, h, w, 3), dtype=torch.uint8).pin_memory()
            self._pin_out = torch.empty((self.batch_size, h, w, 3), dtype=torch.uint8).pin_memory()
        arr = self._pin_in.numpy()
        for j, f in enumerate(frames):
            arr[j] = f
        ed.stream.wait_stream(torch.cuda.current_stream(ed.device))
        with torch.cuda.stream(ed.stream):
            flows = ed.tfilter.flows(arr, tfl) if ed.tfilter is not None else None
            if ed.reembed_async:
                ed.side_check()
            plan = ed.carry.plan(arr, carry_host) if ed.carry is not None else None
            face = ed.face.stage(arr, face_host) if ed.face is not None else None
            out = ed.process(self._pin_in.to(ed.device, non_blocking=True), flows, plan, face, host=arr, live=True)
            self._pin_out.copy_(out, non_blocking=True)
            self._done.record()
        self._done.synchronize()
        return list(self._pin_out.numpy().copy())
