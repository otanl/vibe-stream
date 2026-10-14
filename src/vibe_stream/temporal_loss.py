"""The temporal term of the student loss: masked L1 between output t and output t+1 warped back by the input flow.

It uses the same convention as `metrics.warping_error` (Farneback flow on the INPUT frames t -> t+1, sample t+1 at
x + flow(x)), so the training term and the reported warping error measure the same thing. Pixels the flow cannot
explain on the input itself (occlusions, flow failures) are down-weighted with a photometric mask,
exp(-|in_t - warp(in_t+1)| / 0.05), and pixels warped from outside the frame are dropped.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def warp(img: torch.Tensor, flow: torch.Tensor):
    """img (B, C, H, W) sampled at x + flow(x); flow (B, H, W, 2) in pixels. Returns (warped, in-bounds mask)."""
    b, _, h, w = img.shape
    ys, xs = torch.meshgrid(torch.arange(h, device=img.device, dtype=torch.float32),
                            torch.arange(w, device=img.device, dtype=torch.float32), indexing="ij")
    gx = xs[None] + flow[..., 0]
    gy = ys[None] + flow[..., 1]
    inb = ((gx >= 0) & (gx <= w - 1) & (gy >= 0) & (gy <= h - 1)).float()[:, None]
    grid = torch.stack([gx / (w - 1) * 2 - 1, gy / (h - 1) * 2 - 1], -1)
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True), inb


def occlusion_mask(in_a: torch.Tensor, in_b: torch.Tensor, flow: torch.Tensor, tau: float = 0.05) -> torch.Tensor:
    """(B, 1, H, W) weight for pairs of input frames in [0, 1]: in-bounds x photometric consistency of the flow."""
    wi, inb = warp(in_b, flow)
    return inb * torch.exp(-(in_a - wi).abs().mean(1, keepdim=True) / tau)


def masked_warp_l1(out_a: torch.Tensor, out_b: torch.Tensor, flow: torch.Tensor, mask: torch.Tensor,
                   margin: int = 0) -> torch.Tensor:
    """Masked mean of |out_a - warp(out_b)| (images in [0, 1]), ignoring `margin` pixels at each crop border."""
    wb, _ = warp(out_b, flow)
    err = (out_a - wb).abs().mean(1, keepdim=True)
    if margin:
        err, mask = err[..., margin:-margin, margin:-margin], mask[..., margin:-margin, margin:-margin]
    return (err * mask).sum() / mask.sum().clamp_min(1.0)
