"""Display-side temporal filter: recursive blending of the output along the input's optical flow, in two bands.

Most of the student's flicker is content moving under texture that is anchored to the screen (the shared init noise
and the 32-pixel latent grid): token-sized colour blotches shift and repaint, and brush strokes re-sample, even for a
shift of a few pixels. After decoding, each frame is split into a low band (Gaussian, sigma 6 px: the blotches) and the
rest, and each band is blended with the previous shown frame warped to the current one:

    low_t  = m * a_low  * warp(low_t-1)  + (1 - m * a_low)  * low(out_t)
    high_t = m * a_high * warp(high_t-1) + (1 - m * a_high) * high(out_t)        shown_t = low_t + high_t

warp follows the flow from frame t to frame t-1 of the INPUT (OpenCV DIS, fast preset, on the CPU at 320 px), and
m = exp(-|in_t - warp(in_t-1)| / tau) turns the blend off where that flow does not explain the input (occlusions,
cuts). A hard blend of the low band and a light one of the high band removes most of the blotch flicker without the
softening and double images of a full-band blend (see README, "Temporal filter").

The flows of a batch are computed when the batch is staged, on the host thread, while the GPU still works on the
previous batch; the blending runs on the GPU after the decoder.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .temporal_loss import flow_grid, warp_grid


class TemporalFilter:
    def __init__(self, low: float = 0.7, high: float = 0.3, sigma: float = 6.0, tau: float = 12.0,
                 flow_size: int = 320, device: str = "cuda"):
        self.low, self.high, self.tau, self.flow_size, self.device = low, high, tau, flow_size, device
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        r = int(3 * sigma + 0.5)
        k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32) ** 2 / (2 * sigma ** 2))
        self.kernel = (k / k.sum()).to(device)
        self._pinned: List[torch.Tensor] = []                # two host buffers for the flows, used alternately
        self._copied = [torch.cuda.Event(), torch.cuda.Event()]
        self._slot = 0
        self.reset()

    def reset(self) -> None:
        """Forget the previous frame (call between unrelated clips)."""
        self._prev_gray: Optional[np.ndarray] = None
        self._prev_in = self._prev_low = self._prev_high = None

    def host_flow(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Host part of one frame, in order: its flow to the frame before it at flow_size (None for a clip's first
        frame). `flows` calls it per frame; a live source can call it as each frame arrives (on another thread) and
        hand the batch's results to `flows` instead."""
        fs = self.flow_size
        g = cv2.cvtColor(cv2.resize(frame, (fs, fs), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        fl = None if self._prev_gray is None else self.dis.calc(g, self._prev_gray, None)
        self._prev_gray = g
        return fl

    def flows(self, frames: np.ndarray,
              host: Optional[List[Optional[np.ndarray]]] = None) -> Tuple[torch.Tensor, List[bool]]:
        """Flows of a staged batch (B, H, W, 3) uint8: per frame the flow to the frame before it, computed on the host
        at flow_size (or taken from `host`, the frames' `host_flow` results) into pinned memory and copied to the GPU
        on the current stream without waiting for it. Returns the (B, flow_size, flow_size, 2) GPU tensor and per
        frame whether it has a predecessor (not the first frame of a clip)."""
        fs = self.flow_size
        if not self._pinned or self._pinned[0].shape[0] != len(frames):
            self._pinned = [torch.zeros((len(frames), fs, fs, 2)).pin_memory() for _ in range(2)]
        s, self._slot = self._slot, self._slot ^ 1
        self._copied[s].synchronize()                         # the copy from this buffer two batches ago is done
        buf, has = self._pinned[s].numpy(), []
        for j, f in enumerate(frames):
            fl = host[j] if host is not None else self.host_flow(f)
            has.append(fl is not None)
            if fl is not None:
                buf[j] = fl
        out = self._pinned[s].to(self.device, non_blocking=True)
        self._copied[s].record()
        return out, has

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        c, r = x.shape[1], len(self.kernel) // 2
        kx = self.kernel.view(1, 1, 1, -1).expand(c, 1, 1, -1)
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), kx, groups=c)
        return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), kx.transpose(2, 3), groups=c)

    @torch.inference_mode()
    def __call__(self, inputs: torch.Tensor, outputs: torch.Tensor, flows: Tuple[torch.Tensor, List[bool]]) -> torch.Tensor:
        """inputs / outputs (B, H, W, 3) uint8 on the GPU and the batch's `flows` -> filtered outputs, same layout."""
        h, w = outputs.shape[1:3]
        x_in = inputs.permute(0, 3, 1, 2).float()
        out = outputs.permute(0, 3, 1, 2).float()
        low = self._blur(out)
        high = out - low
        fl_all, has = flows
        fl_all = F.interpolate(fl_all.permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False)
        fl_all = (fl_all * (h / self.flow_size)).permute(0, 2, 3, 1)
        grid = flow_grid(fl_all)
        # the occlusion weights of all frames at once: frame j's predecessor is frame j - 1 (frame 0's the previous
        # batch's last); only the band blends are recursive
        prev_in = torch.cat([x_in[:1] if self._prev_in is None else self._prev_in, x_in[:-1]])
        m = torch.exp(-(x_in - warp_grid(prev_in, grid)).abs().mean(1, keepdim=True) / self.tau)
        prev = None if self._prev_low is None else torch.cat([self._prev_low, self._prev_high], 1)
        shown = []
        for j in range(len(out)):
            lo, hi = low[j:j + 1], high[j:j + 1]
            if has[j] and prev is not None:
                wp, mj = warp_grid(prev, grid[j:j + 1]), m[j:j + 1]
                lo = mj * self.low * wp[:, :3] + (1 - mj * self.low) * lo
                hi = mj * self.high * wp[:, 3:] + (1 - mj * self.high) * hi
            prev = torch.cat([lo, hi], 1)
            shown.append(lo + hi)
        self._prev_in, self._prev_low, self._prev_high = x_in[-1:], prev[:, :3], prev[:, 3:]
        return torch.cat(shown).clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
