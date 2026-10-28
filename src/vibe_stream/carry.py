"""Carry-over between batches: start a batch from the previous batch's output instead of from noise.

The 2-step student's first step lays out composition and colour areas, the second paints the detail; most of the
flicker is the first step deciding afresh on every frame (README, "Flicker"). With the carry-over, a batch after the
first skips step 1: frame j starts step 2 at sigma s1 from

    x = (1 - s1) * z + s1 * noise,   z = w * enc(warp(previous output)) + (1 - w) * enc(frame j)

(s1 = the last step's sigma, 0.7496 for 2 steps, or the `carry_sigma` a student was trained with), where the
previous output is the last frame of the previous batch, warped to frame j along the input's optical flow
(OpenCV DIS, fast preset, at 320 px, frame j -> reference frame), and w = exp(-|frame j - warp(reference frame)| /
tau), averaged over each latent token, keeps the warp only where it explains the input: occlusions and new content
start from the frame itself. One transformer call per frame instead of two. A student trained for this
(`train_student.py --w_carry`) keeps its style; the plain one loses some (README).

Keyframes run the usual k steps from noise: the first batch of a clip, every `keyframe_every` batches if set, and
any batch with a frame that the flow explains on less than `cut_below` of its area on average (a cut). On the DAVIS
clips that mean is 0.03-0.29 (median 0.11) across cuts and 0.24 or more (median 0.42) within a clip 8 frames apart.

The frame that fills in where the flow fails can carry the style's colours instead of the source's (`fill="lut"`):
the keyframe batch's inputs and outputs give a colour map (`fit_colour_map`: mean output colour per input colour,
both blurred, on a 16-level RGB grid, smoothed), and the fill is the frame with its local colour moved through that map
(`apply_colour_map`). With the plain frame, about half of every carried start is the source (w is near 0.5 even where
the flow is right), and the single step does not repaint large colour changes: anime keeps a green pond that the
2-step path paints pale blue. A student is trained for one fill (`train_student.py --carry_fill`); it records it.
Swapping the fill at run time fixes saturation but not colour; a student trained with the map brings the carried
output's low-band colour a third closer to the 2-step path's (README, Flicker).

`plan` runs on the host when a batch is staged (flows into pinned memory, copied without a sync); `start` and
`update` run on the GPU stream in order with the batch's edit.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .temporal_loss import flow_grid, warp_grid


def holds(cur: np.ndarray, ref: np.ndarray, flow: np.ndarray, tau: float) -> np.ndarray:
    """Per-pixel weight in [0, 1] (host, uint8 images of one size): how well `flow` (cur -> ref) explains cur."""
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    wr = cv2.remap(ref, xs + flow[..., 0], ys + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    d = np.abs(cur.astype(np.float32) - wr.astype(np.float32))
    return np.exp(-(d.mean(2) if d.ndim == 3 else d) / tau)


def _gblur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """(B, C, H, W) float, separable Gaussian, replicated border."""
    r = max(1, int(3 * sigma))
    k = torch.exp(-(torch.arange(-r, r + 1, device=x.device, dtype=torch.float32) ** 2) / (2 * sigma ** 2))
    k = (k / k.sum()).view(1, 1, 1, -1).expand(x.shape[1], 1, 1, -1)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="replicate"), k, groups=x.shape[1])
    return F.conv2d(F.pad(x, (0, 0, r, r), mode="replicate"), k.transpose(2, 3), groups=x.shape[1])


def _blur_grid(v: torch.Tensor, sigma: float) -> torch.Tensor:
    """(C, D, D, D) float, Gaussian of `sigma` cells along each axis, zero border."""
    r = max(1, int(2 * sigma))
    k = torch.exp(-(torch.arange(-r, r + 1, device=v.device, dtype=torch.float32) ** 2) / (2 * sigma ** 2))
    k = k / k.sum()
    x = v.unsqueeze(1)
    for axis in range(3):
        shape = [1, 1, 1]
        shape[axis] = -1
        pad = [0, 0, 0]
        pad[axis] = r
        x = F.conv3d(x, k.view(1, 1, *shape), padding=pad)
    return x.squeeze(1)


def fit_colour_map(inputs: torch.Tensor, outputs: torch.Tensor, bins: int = 16, sigma_px: float = 4.0,
                   sigma_bin: float = 1.0, prior: float = 50.0) -> torch.Tensor:
    """Colour offsets (3, bins, bins, bins), indexed [R][G][B] by input colour: the mean of blur(output) -
    blur(input) over the pixels of that colour (every 4th pixel), smoothed over neighbouring colours and pulled towards
    the overall mean offset where a colour is rare (`prior` pixels' worth). inputs / outputs (B, H, W, 3) uint8."""
    a = _gblur(inputs.permute(0, 3, 1, 2).float(), sigma_px)[:, :, ::4, ::4]
    b = _gblur(outputs.permute(0, 3, 1, 2).float(), sigma_px)[:, :, ::4, ::4]
    idx = (a * (bins / 256)).long().clamp(0, bins - 1)
    flat = ((idx[:, 0] * bins + idx[:, 1]) * bins + idx[:, 2]).reshape(-1)
    d = (b - a).permute(1, 0, 2, 3).reshape(3, -1)
    # index_put_ with accumulate sorts by index, so the sums come out the same every run; index_add_ adds with atomics
    # in a varying order, and the last-digit differences grew to visible ones through the fill
    rows = torch.arange(3, device=a.device)[:, None].expand(3, flat.numel())
    sums = torch.zeros(3, bins ** 3, device=a.device).index_put_((rows, flat.expand(3, -1)), d, accumulate=True)
    cnt = torch.bincount(flat, minlength=bins ** 3).float().view(1, -1)
    sums = _blur_grid(sums.view(3, bins, bins, bins), sigma_bin)
    cnt = _blur_grid(cnt.view(1, bins, bins, bins), sigma_bin)
    return (sums + prior * d.mean(1).view(3, 1, 1, 1)) / (cnt + prior)


def apply_colour_map(cmap: torch.Tensor, frames: torch.Tensor, sigma_px: float = 4.0) -> torch.Tensor:
    """frames (B, H, W, 3) uint8 with the offset of their local (blurred) colour added, trilinear in the map."""
    x = frames.permute(0, 3, 1, 2).float()
    grid = (_gblur(x, sigma_px) * (2 / 256) - 1).permute(0, 2, 3, 1)[..., [2, 1, 0]].unsqueeze(1)
    off = F.grid_sample(cmap.unsqueeze(0).expand(len(frames), -1, -1, -1, -1), grid, mode="bilinear",
                        padding_mode="border", align_corners=False)[:, :, 0]
    return (x + off).clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()


class Carry:
    def __init__(self, keyframe_every: int = 0, tau: float = 12.0, cut_below: float = 0.2, flow_size: int = 320,
                 sigma: Optional[float] = None, fill: Optional[str] = None, device: str = "cuda"):
        self.keyframe_every, self.tau, self.cut_below, self.flow_size = keyframe_every, tau, cut_below, flow_size
        self.sigma = sigma        # noise level of the carried start; None: the student's (or its last step's)
        if fill not in (None, "frame", "lut"):
            raise ValueError(f"fill must be 'frame' or 'lut', not {fill!r}")
        self.fill = fill          # what fills in where the flow fails; None: the student's (or the frame)
        self.device = device
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self._pinned: List[torch.Tensor] = []
        self._copied = [torch.cuda.Event(), torch.cuda.Event()]
        self._slot = 0
        self.keyframes = 0
        self.reset()

    def reset(self) -> None:
        """Forget the reference (call between unrelated clips)."""
        self._ref_gray: Optional[np.ndarray] = None
        self._ref_in = self._ref_out = None
        self._since_key = 0
        self._was_key = False
        self._cmap: Optional[torch.Tensor] = None

    @staticmethod
    def flow_to(cur: np.ndarray, ref: np.ndarray, dis, size: int) -> np.ndarray:
        """DIS flow cur -> ref (RGB uint8) at size x size, upsampled to cur's resolution and scaled to its pixels."""
        g = [cv2.cvtColor(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY) for f in (cur, ref)]
        fl = dis.calc(g[0], g[1], None)
        h, w = cur.shape[:2]
        return cv2.resize(fl, (w, h), interpolation=cv2.INTER_LINEAR) * (w / size)

    def gray(self, frame: np.ndarray) -> np.ndarray:
        fs = self.flow_size
        return cv2.cvtColor(cv2.resize(frame, (fs, fs), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)

    def host_flow(self, gray: np.ndarray, ref: Optional[np.ndarray]):
        """Host part of one frame (its `gray`) against the reference, the gray of the previous batch's last frame
        (None for a clip's first batch): (flow at flow_size, whether it holds), or None without a reference. `plan`
        calls it per frame; a live source, which knows the reference once the previous batch is complete, can call it
        as each frame arrives (on another thread) and hand the batch's grays and results to `plan` instead."""
        if ref is None:
            return None
        fl = self.dis.calc(gray, ref, None)
        return fl, holds(gray, ref, fl, self.tau).mean() >= self.cut_below

    def plan(self, frames: np.ndarray, host=None):
        """Host side of a staged batch (B, H, W, 3) uint8. Returns None for a keyframe batch, else the flows of its
        frames to the reference frame on the GPU (B, flow_size, flow_size, 2), copied from pinned memory. `host`:
        the frames' (grays, `host_flow` results), computed ahead."""
        fs = self.flow_size
        grays = host[0] if host is not None else [self.gray(f) for f in frames]
        ref, self._ref_gray = self._ref_gray, grays[-1]
        key = ref is None or (self.keyframe_every > 0 and self._since_key >= self.keyframe_every)
        if not key:
            if not self._pinned or self._pinned[0].shape[0] != len(frames):
                self._pinned = [torch.zeros((len(frames), fs, fs, 2)).pin_memory() for _ in range(2)]
            s, self._slot = self._slot, self._slot ^ 1
            self._copied[s].synchronize()
            buf = self._pinned[s].numpy()
            for j, g in enumerate(grays):
                buf[j], ok = host[1][j] if host is not None else self.host_flow(g, ref)
                if not ok:
                    key = True
                    break
        self._was_key = key
        if key:
            self._since_key = 1
            self.keyframes += 1
            return None
        self._since_key += 1
        out = self._pinned[s].to(self.device, non_blocking=True)
        self._copied[s].record()
        return out

    @torch.inference_mode()
    def start(self, inputs: torch.Tensor, flows: torch.Tensor, lat_size: int, fill: str = "frame"):
        """GPU side: (warped previous output (B, H, W, 3) uint8, w (B, 1, lat_size, lat_size), fill) for the batch's
        inputs (B, H, W, 3) uint8 and the flows from `plan`; fill is None for the frame itself, else the frames through
        the keyframe's colour map (B, H, W, 3) uint8."""
        h, w = inputs.shape[1:3]
        fl = F.interpolate(flows.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False) * (h / self.flow_size)
        fl = fl.permute(0, 2, 3, 1)
        n = len(inputs)
        both = warp_grid(torch.cat([self._ref_out, self._ref_in], 1).expand(n, -1, -1, -1), flow_grid(fl))
        guess, wi = both[:, :3], both[:, 3:]
        m = torch.exp(-(inputs.permute(0, 3, 1, 2).float() - wi).abs().mean(1, keepdim=True) / self.tau)
        guess = guess.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        lut = apply_colour_map(self._cmap, inputs) if fill == "lut" and self._cmap is not None else None
        return guess, F.adaptive_avg_pool2d(m, lat_size), lut

    @torch.inference_mode()
    def update(self, inputs: torch.Tensor, outputs: torch.Tensor, fill: str = "frame") -> None:
        """Keep the batch's last input and output (before any display filter) as the next reference; after a keyframe
        batch with fill "lut", also its colour map."""
        self._ref_in = inputs[-1:].permute(0, 3, 1, 2).float()
        self._ref_out = outputs[-1:].permute(0, 3, 1, 2).float()
        if fill == "lut" and self._was_key:
            self._cmap = fit_colour_map(inputs, outputs)
