"""Delay of the live path (vibe_stream.live): frames of a video file are handed over at a camera's rate, in wall-clock
time, and each frame's delay is the time from its hand-over to its edited copy being back on the host.

Per run: mean, 95th percentile and worst delay over the frames after the first two batches, and the median time
from a batch's last frame to its result. A batch that takes longer than batch / fps makes the delay grow; the run then
says it falls behind. --check compares the outputs with those of StreamEditor.run on the same frames (bit for bit
with the batch-8 engines; the smaller batches' INT8 engines vary slightly from run to run, see vibe_stream.live).

Usage:
  python scripts/measure_latency.py --engine out/trt/vibe_lam4_carry_s05_lut_2step_768_b4_int8.engine \
      --tae_engine out/trt/taesana_768_b4 --carry --temporal_filter 0.7 0.3 --fps 30
"""
import argparse
import json
import threading
import time
from itertools import cycle, islice
from pathlib import Path

import numpy as np

import _paths  # noqa: F401
from vibe_stream.carry import Carry
from vibe_stream.live import LiveEditor
from vibe_stream.metrics import read_video_frames
from vibe_stream.presets import PROMPTS
from vibe_stream.stream import StreamEditor
from vibe_stream.temporal_filter import TemporalFilter
from vibe_stream.trt_engine import TRTTransformer
from vibe_stream.trt_tae import TRTTinyAutoencoder
from vibe_stream.vibe_model import ROOT, VibeModel


def wait_until(t: float) -> None:
    while (d := t - time.perf_counter()) > 0:
        time.sleep(d - 0.002 if d > 0.003 else 0)


def feed(live: LiveEditor, frames, fps: float, t0: float, arrived: list) -> None:
    for i, f in enumerate(frames):
        wait_until(t0 + i / fps)
        arrived.append(time.perf_counter())
        live.submit(f)
    live.stop()


def measure(ed: StreamEditor, frames, fps: float):
    live = LiveEditor(ed)
    live.start()
    arrived, ready, outs = [], [], []
    t0 = time.perf_counter() + 0.2
    threading.Thread(target=feed, args=(live, frames, fps, t0, arrived), daemon=True).start()
    while (batch := live.step()) is not None:
        ready.append(time.perf_counter())
        outs += batch
    B = ed.batch_size
    delay = np.array([ready[i // B] - arrived[i] for i in range(2 * B, len(ready) * B)])
    after = np.array([ready[b] - arrived[(b + 1) * B - 1] for b in range(2, len(ready))])
    return delay, after, outs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, help="TensorRT engine from scripts/build_trt.py (its batch is the one measured)")
    p.add_argument("--tae_engine", required=True, help="taesana engines of the same size and batch")
    p.add_argument("--input", default=str(ROOT / "assets/davis_mp4/blackswan.mp4"))
    p.add_argument("--frames", type=int, default=240, help="the clip cycled to this many frames")
    p.add_argument("--fps", type=float, nargs="+", default=[30.0], help="source rate(s)")
    p.add_argument("--prompt", default="oil")
    p.add_argument("--carry", action="store_true")
    p.add_argument("--temporal_filter", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"))
    p.add_argument("--check", action="store_true", help="compare the outputs with StreamEditor.run's")
    p.add_argument("--out_json", default=None)
    a = p.parse_args()
    eng = TRTTransformer(a.engine)
    size, B = eng.spec.size, eng.spec.batch
    frames = list(islice(cycle(read_video_frames(a.input, size)), a.frames // B * B))
    vibe = VibeModel(size=size)
    ed = StreamEditor(vibe, TRTTinyAutoencoder(a.tae_engine), engine=eng,
                      temporal_filter=TemporalFilter(*a.temporal_filter) if a.temporal_filter else None,
                      carry=Carry(0) if a.carry else None)
    ed.set_prompt(PROMPTS.get(a.prompt, a.prompt), frames[0])
    offline, tm = ed.run(frames, 2)   # warm-up; the same batch size's offline loop rate for reference
    path = "carry-over + filter" if a.carry and a.temporal_filter else "carry-over" if a.carry else \
        "filter" if a.temporal_filter else "2 steps"
    res = dict(engine=a.engine, size=size, batch=B, path=path, offline_loop_fps=tm.loop_fps, runs={})
    for fps in a.fps:
        delay, after, outs = measure(ed, frames, fps)
        behind = bool(after[-5:].mean() - after[:5].mean() > 1 / fps)
        r = dict(mean_ms=1000 * delay.mean(), p95_ms=1000 * np.percentile(delay, 95), max_ms=1000 * delay.max(),
                 after_last_frame_ms=1000 * np.median(after), falls_behind=behind)
        res["runs"][str(fps)] = r
        print(f"[latency] {size} batch {B} {path}, {fps:g} fps source: delay mean {r['mean_ms']:.0f} ms, "
              f"p95 {r['p95_ms']:.0f}, worst {r['max_ms']:.0f}; last frame to result {r['after_last_frame_ms']:.0f} ms"
              f"{'; FALLS BEHIND' if behind else ''} (offline loop {tm.loop_fps:.1f} fps)", flush=True)
        if a.check:
            same = all(np.array_equal(x, y) for x, y in zip(outs, offline))
            print(f"[latency] outputs {'identical to' if same else 'DIFFERENT from'} StreamEditor.run's", flush=True)
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
