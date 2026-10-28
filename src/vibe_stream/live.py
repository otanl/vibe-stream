"""Live input: frames arrive one at a time (a camera, a capture) and leave edited with as little delay as the batch
allows.

`StreamEditor.run` stages whole batches, and its host work (the temporal filter's and the carry-over's DIS flows,
about 15 ms per frame at 768) overlaps the previous batch's GPU work, which is what offline throughput needs. A live
batch cannot be staged before its last frame has arrived, so that host work would sit between the last frame and the
GPU: 117 ms per batch of 8 at 768, which with the GPU's 169 ms exceeds the 267 ms a 30 fps source leaves, and the
delay grows without bound. `LiveEditor` computes each frame's flows on a thread as the frame arrives
(`TemporalFilter.host_flow`, `Carry.host_flow`; the carry-over's reference, the previous batch's last frame, is known
by then), so only the last frame's flows lie between it and the GPU. It runs the same operations as
`StreamEditor.run` on the same frames; with the batch-8 engines the output is identical bit for bit. The INT8 engines
for batch 4 and 2 are not bit-reproducible even from one offline run to the next: TensorRT's kernels for their
smaller matrix products sum in a varying order, so now and then one transformer call differs by up to about 0.1 in
velocity, and the difference fades over the following batches.

    live = LiveEditor(editor)          # after editor.set_prompt(...)
    live.start()
    # capture thread:  live.submit(frame) per frame (RGB uint8 at the editor's size)
    # pipeline thread: while (outs := live.step()) is not None: show(outs)   # the batch's edited frames
    live.stop()                        # from the capture side; a partial last batch is dropped

Delay at a 30 fps source, the INT8 engines with carry-over + filter (scripts/measure_latency.py), mean / 95th
percentile: at 768 batch 8 298 / 415 ms, batch 4 150 / 201 ms, batch 2 80 / 98 ms (batch 1, 26 fps, falls behind);
at 640 250 / 368, 128 / 178 and 68 / 87 ms.
"""

from __future__ import annotations

import queue
import threading
from typing import List, Optional

import numpy as np
import torch

from .stream import StreamEditor


class LiveEditor:
    def __init__(self, editor: StreamEditor):
        self.ed = editor
        self.batch_size = editor.batch_size
        self._in: queue.Queue = queue.Queue()
        self._ready: queue.Queue = queue.Queue()
        self._thread: Optional[threading.Thread] = None
        self._pin_in = self._pin_out = None
        self._done = torch.cuda.Event()

    def start(self) -> None:
        """Begin a clip: the temporal filter and the carry-over start afresh, and the host thread starts."""
        for state in (self.ed.tfilter, self.ed.carry):
            if state is not None:
                state.reset()
        self._in, self._ready = queue.Queue(), queue.Queue()
        self._thread = threading.Thread(target=self._host, daemon=True)
        self._thread.start()

    def submit(self, frame: np.ndarray) -> None:
        """One arriving frame, RGB uint8 (H, W, 3) at the editor's size. Returns at once."""
        self._in.put(frame)

    def stop(self) -> None:
        """End the clip; `step` returns None once the complete batches before it are out."""
        self._in.put(None)
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _host(self) -> None:
        tf, cr, B = self.ed.tfilter, self.ed.carry, self.batch_size
        ref = None
        while True:
            frames, tfl, grays, res = [], [], [], []
            for _ in range(B):
                f = self._in.get()
                if f is None:
                    self._ready.put(None)
                    return
                frames.append(f)
                if tf is not None:
                    tfl.append(tf.host_flow(f))
                if cr is not None:
                    grays.append(cr.gray(f))
                    res.append(cr.host_flow(grays[-1], ref))
            if cr is not None:
                ref = grays[-1]
            self._ready.put((frames, tfl if tf is not None else None, (grays, res) if cr is not None else None))

    @torch.inference_mode()
    def step(self) -> Optional[List[np.ndarray]]:
        """Wait for the next complete batch, edit it and return its frames (RGB uint8 arrays); None after `stop`.
        Call it from one thread."""
        item = self._ready.get()
        if item is None:
            return None
        frames, tfl, carry_host = item
        ed = self.ed
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
            plan = ed.carry.plan(arr, carry_host) if ed.carry is not None else None
            out = ed.process(self._pin_in.to(ed.device, non_blocking=True), flows, plan)
            self._pin_out.copy_(out, non_blocking=True)
            self._done.record()
        self._done.synchronize()
        return list(self._pin_out.numpy().copy())
