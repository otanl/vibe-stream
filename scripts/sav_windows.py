"""Training windows from SA-V clips: --n clips drawn at random from those with usable motion, one window each.

Reads sav_probe.py's csv. A clip qualifies with a median motion (Farneback between t and t+2, 768-px units) in
[--min_motion, --max_motion] and at least --min_frames frames. Clips slower than --slow get stride 2 (every other
frame), the rest stride 1. The window starts at a random frame away from both ends. Frames are centre-cropped to a
square and resized to --size with INTER_AREA, as vibe_stream.metrics.read_video_frames does; windows darker than
mean 20/255 are skipped (the next drawn clip takes the slot).
Writes <out>/w###.npz (frames uint8 (window, size, size, 3), clip, start, stride, motion) and <out>/windows.json.

Usage: python scripts/sav_windows.py --probe data/sav/probe.csv --videos data/sav/videos --out data/sav_windows \
           --n 200
"""
import argparse
import csv
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def read_window(path, start, n, stride, size):
    cap = cv2.VideoCapture(str(path))
    try:
        for _ in range(start):
            if not cap.grab():
                return None
        out = []
        for i in range((n - 1) * stride + 1):
            ok, bgr = cap.read()
            if not ok:
                return None
            if i % stride == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                h, w, _ = rgb.shape
                s = min(h, w)
                rgb = rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
                out.append(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA))
        return np.stack(out)
    finally:
        cap.release()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--min_motion", type=float, default=1.5)
    ap.add_argument("--max_motion", type=float, default=30.0)
    ap.add_argument("--slow", type=float, default=5.0)
    ap.add_argument("--min_frames", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    paths = {p.name: p for p in Path(a.videos).rglob("*.mp4")}
    rows = [r for r in csv.DictReader(open(a.probe, newline=""))
            if a.min_motion <= float(r["motion_med"]) <= a.max_motion and int(r["n"]) >= a.min_frames
            and r["file"] in paths]
    rng = random.Random(a.seed)
    rng.shuffle(rows)
    print(f"{len(rows)} clips qualify; taking {a.n}", flush=True)
    plans = []
    for r in rows:
        stride = 2 if float(r["motion_med"]) < a.slow else 1
        span = (a.window - 1) * stride + 1
        start = rng.randrange(12, int(r["n"]) - span - 12)
        plans.append(dict(clip=Path(r["file"]).stem, file=r["file"], start=start, stride=stride,
                          motion=float(r["motion_med"])))

    def load(pl):
        fr = read_window(paths[pl["file"]], pl["start"], a.window, pl["stride"], a.size)
        return None if fr is None or fr.mean() < 20 else fr

    kept = []
    with ThreadPoolExecutor(8) as ex:
        for c in range(0, len(plans), 32):  # in chunks, so that only about --n clips are decoded
            for pl, fr in zip(plans[c:c + 32], ex.map(load, plans[c:c + 32])):
                if fr is None or len(kept) == a.n:
                    continue
                np.savez(out / f"w{len(kept):03d}.npz", frames=fr, clip=pl["clip"], start=pl["start"],
                         stride=pl["stride"], motion=pl["motion"])
                kept.append(pl)
            if len(kept) == a.n:
                break
    (out / "windows.json").write_text(json.dumps(dict(args=vars(a), windows=kept), indent=1))
    s1 = sum(p["stride"] == 1 for p in kept)
    print(f"{len(kept)} windows ({s1} at stride 1, {len(kept) - s1} at stride 2), "
          f"motion median {np.median([p['motion'] for p in kept]):.1f}", flush=True)


if __name__ == "__main__":
    main()
