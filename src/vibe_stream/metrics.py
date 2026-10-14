"""Video I/O and the evaluation metrics, on RGB uint8 frames.

The definitions follow dreamlite-stream's `metrics.py`, so numbers are comparable across the two repositories:

  warping error   mean |out_t - warp(out_t+1, flow of the INPUT frames t -> t+1)| over the clip, 0-255 scale,
                  Farneback flow, no occlusion mask. Lower = less flicker.
  sobel           mean |Sobel_xy(gray)|, a texture / sharpness probe: an output can lower its warping error by
                  blurring, which shows up here.
  edge agreement  correlation of blurred Sobel magnitudes of input and output: how much of the source structure the
                  edit keeps.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence

import cv2
import numpy as np


def read_video_frames(path, size: int, max_frames: Optional[int] = None) -> List[np.ndarray]:
    """RGB uint8 frames, centre-cropped to a square and resized to (size, size) with INTER_AREA."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    frames: List[np.ndarray] = []
    try:
        while max_frames is None or len(frames) < max_frames:
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            h, w, _ = rgb.shape
            s = min(h, w)
            rgb = rgb[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
            frames.append(cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA))
    finally:
        cap.release()
    return frames


def video_fps(path) -> float:
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    cap.release()
    return fps


def write_video(path, frames: Sequence[np.ndarray], fps: float) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not wr.isOpened():
        raise RuntimeError(f"cannot open video writer: {path}")
    for f in frames:
        wr.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    wr.release()


def farneback(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Dense flow (H, W, 2) from RGB frame a to b: for each pixel x of a, b is sampled at x + flow(x)."""
    return cv2.calcOpticalFlowFarneback(cv2.cvtColor(a, cv2.COLOR_RGB2GRAY), cv2.cvtColor(b, cv2.COLOR_RGB2GRAY), None,
                                        pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2,
                                        flags=0)


def _warp(img: np.ndarray, flow: np.ndarray) -> np.ndarray:
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w), np.arange(h))
    return cv2.remap(img, (xs + flow[..., 0]).astype(np.float32), (ys + flow[..., 1]).astype(np.float32),
                     cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def warping_error(inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray]) -> float:
    if len(inputs) != len(outputs):
        raise ValueError(f"frame count mismatch: in={len(inputs)} out={len(outputs)}")
    errs = []
    for t in range(len(inputs) - 1):
        flow = farneback(inputs[t], inputs[t + 1])
        errs.append(float(np.mean(np.abs(outputs[t].astype(np.float32) - _warp(outputs[t + 1].astype(np.float32), flow)))))
    return float(np.mean(errs)) if errs else 0.0


def sobel(frames: Sequence[np.ndarray]) -> float:
    return float(np.mean([np.abs(cv2.Sobel(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_32F, 1, 1, ksize=3)).mean()
                          for f in frames]))


def edge_agreement(a: np.ndarray, b: np.ndarray) -> float:
    def e(x):
        g = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).astype(np.float32)
        m = np.hypot(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        return cv2.GaussianBlur(m, (0, 0), 2.0).ravel()
    return float(np.corrcoef(e(a), e(b))[0, 1])


class LPIPS:
    """LPIPS (AlexNet) between RGB uint8 frames, on the GPU. Needs the `lpips` package."""

    def __init__(self, device: str = "cuda"):
        import lpips
        import torch
        self.torch = torch
        self.device = device
        self.net = lpips.LPIPS(net="alex", verbose=False).to(device).eval().requires_grad_(False)

    def __call__(self, xs: Sequence[np.ndarray], ys: Sequence[np.ndarray]) -> float:
        torch = self.torch
        vals = []
        with torch.no_grad():
            for x, y in zip(xs, ys):
                tx = torch.from_numpy(x).to(self.device).permute(2, 0, 1)[None].float() / 127.5 - 1
                ty = torch.from_numpy(y).to(self.device).permute(2, 0, 1)[None].float() / 127.5 - 1
                vals.append(self.net(tx, ty).item())
        return float(np.mean(vals))
