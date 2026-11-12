"""Probe SA-V clips for training-window selection (sav_windows.py): size, fps, length, motion and cuts per clip.

For every mp4 under --root: width, height, fps, frame count; for 8 positions spread over the clip, the mean
Farneback flow magnitude between frame t and t+2 at 256 px width (in 768-px units, i.e. what a 768 crop would see)
and a grey-histogram distance between t and t+1 (a cut shows up as a spike). Writes --out (csv) and prints percentiles.

Usage: python scripts/sav_probe.py --root data/sav/videos --out data/sav/probe.csv
"""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np


def probe(path):
    cap = cv2.VideoCapture(str(path))
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mags, cuts = [], []
    scale = 256 / min(w, h)
    for t in np.linspace(0, max(n - 4, 0), 8).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        fr = []
        for _ in range(3):
            ok, f = cap.read()
            if not ok:
                break
            g = cv2.cvtColor(cv2.resize(f, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
            fr.append(g)
        if len(fr) < 3:
            continue
        flow = cv2.calcOpticalFlowFarneback(fr[0], fr[2], None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(float(np.linalg.norm(flow, axis=-1).mean()) * 768 / 256)
        h0, h1 = (cv2.calcHist([x], [0], None, [32], [0, 256]).ravel() for x in fr[:2])
        cuts.append(float(0.5 * np.abs(h0 / h0.sum() - h1 / h1.sum()).sum()))
    cap.release()
    return dict(file=path.name, w=w, h=h, fps=round(fps, 2), n=n,
                motion_med=round(float(np.median(mags)), 2) if mags else -1,
                motion_max=round(max(mags), 2) if mags else -1, hist_max=round(max(cuts), 3) if cuts else -1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    files = sorted(Path(a.root).rglob("*.mp4"))
    with ThreadPoolExecutor(8) as ex:
        rows = list(ex.map(probe, files))
    with open(a.out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    print(f"{len(rows)} clips")
    for k in ("w", "h", "fps", "n", "motion_med", "motion_max", "hist_max"):
        v = np.array([r[k] for r in rows], dtype=float)
        print(f"{k:11s} p5 {np.percentile(v, 5):8.2f}  p25 {np.percentile(v, 25):8.2f}  p50 {np.percentile(v, 50):8.2f}"
              f"  p75 {np.percentile(v, 75):8.2f}  p95 {np.percentile(v, 95):8.2f}")
    sizes = {}
    for r in rows:
        key = f"{max(r['w'], r['h'])}x{min(r['w'], r['h'])} {'landscape' if r['w'] >= r['h'] else 'portrait'}"
        sizes[key] = sizes.get(key, 0) + 1
    print("sizes:", sorted(sizes.items(), key=lambda x: -x[1])[:8])


if __name__ == "__main__":
    main()
