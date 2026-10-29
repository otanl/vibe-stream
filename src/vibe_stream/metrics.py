"""Video I/O and the evaluation metrics, on RGB uint8 frames.

The definitions follow dreamlite-stream's `metrics.py`, so numbers are comparable across the two repositories:

  warping error   mean |out_t - warp(out_t+1, flow of the INPUT frames t -> t+1)| over the clip, 0-255 scale,
                  Farneback flow, no occlusion mask. Lower = less flicker.
  flicker         the same change, only where the flow explains the input, and its low / high band split (`flicker`).
                  Not in dreamlite-stream.
  sobel          mean |Sobel_xy(gray)|, a texture / sharpness probe: an output can lower its warping error by
                  blurring, which shows up here.
  edge agreement  correlation of blurred Sobel magnitudes of input and output: how much of the source structure the
                  edit keeps.
  clip_dir        CLIP directional similarity (as in InstructPix2Pix): cosine between the image-embedding change
                  input -> output and the text-embedding change source caption -> target caption. How far an edit
                  moved towards the instruction, whether or not it matches the teacher's picture. Not in
                  dreamlite-stream.
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


def input_flows(inputs: Sequence[np.ndarray]) -> List[np.ndarray]:
    """Farneback flows t -> t+1 of a clip, for `warping_error` / `flicker` on several outputs of the same input."""
    return [farneback(inputs[t], inputs[t + 1]) for t in range(len(inputs) - 1)]


def warping_error(inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray],
                  flows: Optional[Sequence[np.ndarray]] = None) -> float:
    if len(inputs) != len(outputs):
        raise ValueError(f"frame count mismatch: in={len(inputs)} out={len(outputs)}")
    errs = []
    for t in range(len(inputs) - 1):
        flow = flows[t] if flows is not None else farneback(inputs[t], inputs[t + 1])
        errs.append(float(np.mean(np.abs(outputs[t].astype(np.float32) - _warp(outputs[t + 1].astype(np.float32), flow)))))
    return float(np.mean(errs)) if errs else 0.0


def flicker(inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray], sigma: float = 4.0,
            valid_below: float = 6.0, flows: Optional[Sequence[np.ndarray]] = None) -> dict:
    """Output change where the input's motion is explained, split into bands.

    Like the warping error (Farneback flow on the input, output t+1 warped back to t), but averaged only over pixels
    where the same warp reproduces the input to within `valid_below` (mean over RGB, 0-255): occlusions, motion blur
    and flow failures are left out, so what remains is change the edit added. `low` / `high` split that change into
    a Gaussian low band (sigma px: colour blotches, the scale of the 32-pixel latent tokens) and the rest (strokes,
    grain). Returns dict(valid, low, high), each a clip mean on the 0-255 scale. `flows`: precomputed `input_flows`."""
    v, lo, hi = [], [], []
    for t in range(len(inputs) - 1):
        flow = flows[t] if flows is not None else farneback(inputs[t], inputs[t + 1])
        ok = np.abs(inputs[t].astype(np.float32) - _warp(inputs[t + 1].astype(np.float32), flow)).mean(2) < valid_below
        a, b = outputs[t].astype(np.float32), _warp(outputs[t + 1].astype(np.float32), flow)
        la, lb = cv2.GaussianBlur(a, (0, 0), sigma), cv2.GaussianBlur(b, (0, 0), sigma)
        v.append(np.abs(a - b).mean(2)[ok].mean())
        lo.append(np.abs(la - lb).mean(2)[ok].mean())
        hi.append(np.abs((a - la) - (b - lb)).mean(2)[ok].mean())
    return dict(valid=float(np.mean(v)), low=float(np.mean(lo)), high=float(np.mean(hi)))


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


class CLIPScore:
    """CLIP ViT-L/14 (transformers) scores of an edit: `clip_dir` (see the module docstring) and `clip_t`, the cosine
    between output and target caption. Means over frames."""

    def __init__(self, model: str = "openai/clip-vit-large-patch14", device: str = "cuda"):
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self.torch = torch
        self.device = device
        self.model = CLIPModel.from_pretrained(model, torch_dtype=torch.float16).to(device).eval()
        self.proc = CLIPProcessor.from_pretrained(model)

    def _images(self, frames: Sequence[np.ndarray]):
        x = self.proc(images=list(frames), return_tensors="pt")["pixel_values"].to(self.device, self.torch.float16)
        f = self.model.get_image_features(pixel_values=x).float()
        return f / f.norm(dim=-1, keepdim=True)

    def _texts(self, texts: Sequence[str]):
        x = self.proc(text=list(texts), return_tensors="pt", padding=True).to(self.device)
        f = self.model.get_text_features(**x).float()
        return f / f.norm(dim=-1, keepdim=True)

    def __call__(self, inputs: Sequence[np.ndarray], outputs: Sequence[np.ndarray], source: str, target: str) -> dict:
        torch = self.torch
        with torch.no_grad():
            fi, fo = self._images(inputs), self._images(outputs)
            ts, tt = self._texts([source, target])
            d_img, d_txt = fo - fi, tt - ts
            cos_dir = (d_img @ d_txt) / (d_img.norm(dim=-1) * d_txt.norm() + 1e-8)
            return dict(clip_dir=float(cos_dir.mean()), clip_t=float((fo @ tt).mean()))
