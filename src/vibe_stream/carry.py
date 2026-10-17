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

`plan` runs on the host when a batch is staged (flows into pinned memory, copied without a sync); `start` and
`update` run on the GPU stream in order with the batch's edit.
"""

from __future__ import annotations

from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .temporal_loss import warp


def holds(cur: np.ndarray, ref: np.ndarray, flow: np.ndarray, tau: float) -> np.ndarray:
    """Per-pixel weight in [0, 1] (host, uint8 images of one size): how well `flow` (cur -> ref) explains cur."""
    h, w = flow.shape[:2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    wr = cv2.remap(ref, xs + flow[..., 0], ys + flow[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    d = np.abs(cur.astype(np.float32) - wr.astype(np.float32))
    return np.exp(-(d.mean(2) if d.ndim == 3 else d) / tau)


class Carry:
    def __init__(self, keyframe_every: int = 0, tau: float = 12.0, cut_below: float = 0.2, flow_size: int = 320,
                 sigma: Optional[float] = None, device: str = "cuda"):
        self.keyframe_every, self.tau, self.cut_below, self.flow_size = keyframe_every, tau, cut_below, flow_size
        self.sigma = sigma        # noise level of the carried start; None: the student's (or its last step's)
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

    @staticmethod
    def flow_to(cur: np.ndarray, ref: np.ndarray, dis, size: int) -> np.ndarray:
        """DIS flow cur -> ref (RGB uint8) at size x size, upsampled to cur's resolution and scaled to its pixels."""
        g = [cv2.cvtColor(cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY) for f in (cur, ref)]
        fl = dis.calc(g[0], g[1], None)
        h, w = cur.shape[:2]
        return cv2.resize(fl, (w, h), interpolation=cv2.INTER_LINEAR) * (w / size)

    def plan(self, frames: np.ndarray):
        """Host side of a staged batch (B, H, W, 3) uint8. Returns None for a keyframe batch, else the flows of its
        frames to the reference frame on the GPU (B, flow_size, flow_size, 2), copied from pinned memory."""
        fs = self.flow_size
        grays = [cv2.cvtColor(cv2.resize(f, (fs, fs), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY) for f in frames]
        ref, self._ref_gray = self._ref_gray, grays[-1]
        key = ref is None or (self.keyframe_every > 0 and self._since_key >= self.keyframe_every)
        if not key:
            if not self._pinned or self._pinned[0].shape[0] != len(frames):
                self._pinned = [torch.zeros((len(frames), fs, fs, 2)).pin_memory() for _ in range(2)]
            s, self._slot = self._slot, self._slot ^ 1
            self._copied[s].synchronize()
            buf = self._pinned[s].numpy()
            for j, g in enumerate(grays):
                buf[j] = self.dis.calc(g, ref, None)
                if holds(g, ref, buf[j], self.tau).mean() < self.cut_below:
                    key = True
                    break
        if key:
            self._since_key = 1
            self.keyframes += 1
            return None
        self._since_key += 1
        out = self._pinned[s].to(self.device, non_blocking=True)
        self._copied[s].record()
        return out

    @torch.inference_mode()
    def start(self, inputs: torch.Tensor, flows: torch.Tensor, lat_size: int):
        """GPU side: (warped previous output (B, H, W, 3) uint8, w (B, 1, lat_size, lat_size)) for the batch's inputs
        (B, H, W, 3) uint8 and the flows from `plan`."""
        h, w = inputs.shape[1:3]
        fl = F.interpolate(flows.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False) * (h / self.flow_size)
        fl = fl.permute(0, 2, 3, 1)
        n = len(inputs)
        guess, _ = warp(self._ref_out.expand(n, -1, -1, -1), fl)
        wi, _ = warp(self._ref_in.expand(n, -1, -1, -1), fl)
        m = torch.exp(-(inputs.permute(0, 3, 1, 2).float() - wi).abs().mean(1, keepdim=True) / self.tau)
        guess = guess.clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
        return guess, F.adaptive_avg_pool2d(m, lat_size)

    @torch.inference_mode()
    def update(self, inputs: torch.Tensor, outputs: torch.Tensor) -> None:
        """Keep the batch's last input and output (before any display filter) as the next reference."""
        self._ref_in = inputs[-1:].permute(0, 3, 1, 2).float()
        self._ref_out = outputs[-1:].permute(0, 3, 1, 2).float()
