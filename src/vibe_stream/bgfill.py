"""Background canvas for removals (local edits, `vibe_stream.local`), on the GPU, one frame at a time.

VIBE repaints a removed object's region on every frame with its own texture and tone, so a patch moves with the
object. The canvas instead keeps real background seen on earlier frames and shows it where the object was:
  host  `BgMotion`: ground motion frame t-1 -> t as a homography, fitted (RANSAC) on DIS flow at 384 px outside
        both frames' dilated object masks. It needs only the frames and the tracker's masks, so a live version runs
        it on a thread while the GPU edits the batch (about 9 ms).
  GPU   `BgFillGPU.step`: the canvas, its confidence and age warped by the homography (grid_sample); the region (the
        mask dilated, plus what VIBE changed strongly next to it, grown from the mask by iterated dilation inside the
        change map at `small` px, which takes the object's shadow along); real pixels written beyond a guard band
        (closer ones are skipped, or blurred fur drifts in as white ghosts); where the canvas has nothing, VIBE's
        pixels with a local tone offset (the frame minus VIBE on a ring just outside the region, spread inward by
        normalised blurs; a global mean / std match made a ghost of a man against the sky), carried for `fill_age`
        frames, then repainted (carried longer they smear); the feathered blend. One CUDA graph per step (3.5 ms
        per frame at 768 on an RTX 3090 Ti).
On DAVIS parkour 83 % of the removed man's region comes from real background, sharper than VIBE's fill; on the dog
29 %, the rest VIBE's fill toned to match.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F


def gauss1d(sigma: float, device) -> torch.Tensor:
    r = max(1, int(3 * sigma))
    k = torch.exp(-torch.arange(-r, r + 1, dtype=torch.float32, device=device) ** 2 / (2 * sigma ** 2))
    return k / k.sum()


def blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur of (N, C, H, W), replicate borders."""
    k = gauss1d(sigma, x.device)
    p, c = len(k) // 2, x.shape[1]
    x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="replicate"), k.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(F.pad(x, (0, 0, p, p), mode="replicate"), k.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


def dilate(m: torch.Tensor, r: int) -> torch.Tensor:
    """Square dilation of a (N, 1, H, W) float mask by r pixels, as a row then a column max."""
    if r <= 0:
        return m
    m = F.max_pool2d(m, (1, 2 * r + 1), stride=1, padding=(0, r))
    return F.max_pool2d(m, (2 * r + 1, 1), stride=1, padding=(r, 0))


class BgMotion:
    """Host side: homography frame t-1 -> t from DIS flow outside the objects."""

    def __init__(self, flow_size: int = 384, step: int = 8):
        self.flow_size, self.step = flow_size, step
        self.dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self.prev = None

    def reset(self) -> None:
        self.prev = None

    def __call__(self, frame: np.ndarray, keep_small: np.ndarray) -> np.ndarray:
        """frame (S, S, 3) uint8 RGB; keep_small (flow_size, flow_size) bool, background in this frame (anded with the
        previous frame's). Returns H (3 x 3, float64) for frame t-1 -> t; identity on the first frame."""
        fs, s = self.flow_size, frame.shape[0]
        g = cv2.cvtColor(cv2.resize(frame, (fs, fs), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)
        prev, self.prev = self.prev, (g, keep_small)
        if prev is None:
            return np.eye(3)
        flow = self.dis.calc(prev[0], g, None)
        k = keep_small & prev[1]
        ys, xs = np.mgrid[0:fs:self.step, 0:fs:self.step]
        ys, xs = ys.ravel(), xs.ravel()
        ok = k[ys, xs]
        if ok.sum() < 30:
            return np.eye(3)
        f = fs / s
        src = np.stack([xs[ok], ys[ok]], 1).astype(np.float32)
        dst = src + flow[ys[ok], xs[ok]]
        H, _ = cv2.findHomography(src / f, dst / f, cv2.RANSAC, 2.0 / f)
        return H if H is not None else np.eye(3)


class BgFillGPU:
    def __init__(self, size: int, device: str = "cuda", dilate_px: int = 15, guard: int = 40, ring: int = 24,
                 change_thr: float = 24.0, max_age: int = 90, fill_age: int = 8, small: int = 192,
                 grow_iters: int = 24, change_sigma: float = 12.0, graph: bool = True):
        self.s, self.dev = size, device
        self.dilate_px, self.guard, self.ring, self.change_thr = dilate_px, guard, ring, change_thr
        self.max_age, self.fill_age, self.small, self.grow_iters = max_age, fill_age, small, grow_iters
        # VIBE redraws its whole window: at a small blur the texture it redraws counts as changed and the region
        # spreads over the window
        self.change_sigma = change_sigma
        ys, xs = torch.meshgrid(torch.arange(size, device=device, dtype=torch.float32),
                                torch.arange(size, device=device, dtype=torch.float32), indexing="ij")
        self.pix = torch.stack([xs, ys, torch.ones_like(xs)], -1).view(-1, 3)      # (HW, 3) homogeneous
        # state and static inputs / output: fixed buffers, so the step can be replayed as one CUDA graph
        self.canvas = torch.zeros((1, 3, size, size), device=device)
        self.conf = torch.zeros((1, 1, size, size), device=device)
        self.age = torch.zeros((1, 1, size, size), device=device)
        self.f8 = torch.zeros((size, size, 3), dtype=torch.uint8, device=device)
        self.v8 = torch.zeros_like(self.f8)
        self.mk = torch.zeros((size, size), dtype=torch.bool, device=device)
        self.Hi = torch.eye(3, device=device)
        # pinned staging slots for H^-1, each reused only after its copy has run (one shared slot let the next
        # frame's matrix overwrite this frame's before the asynchronous copy read it)
        self.Hi_host = [torch.eye(3).pin_memory() for _ in range(8)]
        self.Hi_ev = [None] * 8
        self.Hi_slot = 0
        self.out8 = torch.zeros_like(self.f8)
        self.use_graph, self.graph = graph, None

    def reset(self) -> None:
        """Empty canvas (in place, so a captured graph stays valid)."""
        for x in (self.canvas, self.conf, self.age):
            x.zero_()

    def _warp(self, canvas, conf, age):
        """canvas / conf / age warped from frame t-1 to frame t: sampled at H^-1 x."""
        q = self.pix @ self.Hi.T
        q = q[:, :2] / q[:, 2:3].clamp(min=1e-6)
        grid = (q / (self.s - 1) * 2 - 1).view(1, self.s, self.s, 2)
        return (F.grid_sample(canvas, grid, mode="bilinear", padding_mode="zeros", align_corners=True),
                F.grid_sample(conf, grid, mode="nearest", padding_mode="zeros", align_corners=True),
                F.grid_sample(age, grid, mode="nearest", padding_mode="zeros", align_corners=True) + 1)

    def region(self, frame, vibe, mask):
        """(1, 1, S, S) float region: mask dilated, plus VIBE's strong changes connected to it."""
        r = dilate(mask, self.dilate_px)
        if self.change_thr <= 0:
            return r
        d = blur((vibe - frame).abs().mean(1, keepdim=True), self.change_sigma) > self.change_thr
        sm = (self.small, self.small)
        ch = F.interpolate(d.float(), size=sm, mode="area") > 0.5
        seed = (F.interpolate(r, size=sm, mode="area") > 0).float() * ch.float()
        for _ in range(self.grow_iters):                      # reconstruction by dilation inside the change map
            seed = dilate(seed, 1) * ch.float()
        grown = F.interpolate(seed, size=(self.s, self.s), mode="bilinear", align_corners=False) > 0.5
        return torch.maximum(r, grown.float())

    def offset(self, frame, vibe, R):
        """frame - VIBE on a ring around R, spread inward."""
        band = dilate(R, self.ring) * (1 - R)
        sm = (self.small, self.small)
        d = F.interpolate((frame - vibe) * band, size=sm, mode="area")
        w = F.interpolate(band, size=sm, mode="area")
        num, den = torch.zeros_like(d), torch.zeros_like(w)
        for sigma in (2, 4, 8, 16, 32):
            num += blur(d, sigma) / sigma
            den += blur(w, sigma) / sigma
        return F.interpolate(num / den.clamp(min=1e-6), size=(self.s, self.s), mode="bilinear", align_corners=False)

    def _core(self) -> None:
        """One frame from the static buffers (f8, v8, mk, Hi) into the state buffers and out8."""
        f = self.f8.permute(2, 0, 1)[None].float()
        v = self.v8.permute(2, 0, 1)[None].float()
        m = self.mk[None, None].float()
        canvas, conf, age = self._warp(self.canvas, self.conf, self.age)
        conf = torch.where((age > self.max_age) & (conf >= 1), 0.0, conf)
        conf = torch.where((age > self.fill_age) & (conf > 0) & (conf < 1), 0.0, conf)
        R = self.region(f, v, m)
        bg = dilate(R, self.guard) < 0.5
        canvas = torch.where(bg, f, canvas)
        conf = torch.where(bg, 1.0, conf)
        age = torch.where(bg, 0.0, age)
        hole = (R > 0.5) & (conf <= 0)
        canvas = torch.where(hole, v + self.offset(f, v, R), canvas)
        conf = torch.where(hole, 0.5, conf)
        age = torch.where(hole, 0.0, age)
        alpha = blur(R, self.dilate_px / 2)
        out = f * (1 - alpha) + torch.where(conf > 0, canvas, f) * alpha
        self.canvas.copy_(canvas)
        self.conf.copy_(conf)
        self.age.copy_(age)
        self.out8.copy_(out[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8))

    @torch.inference_mode()
    def precapture(self) -> None:
        """Capture the graph on blank inputs up front (before other threads use the GPU), then empty the canvas."""
        z = torch.zeros((self.s, self.s, 3), dtype=torch.uint8, device=self.dev)
        self.step(z, z, torch.zeros((self.s, self.s), dtype=torch.bool, device=self.dev), np.eye(3))
        torch.cuda.synchronize()
        self.reset()

    @torch.inference_mode()
    def step(self, frame_u8: torch.Tensor, vibe_u8: torch.Tensor, mask: torch.Tensor, H: np.ndarray) -> torch.Tensor:
        """frame_u8 / vibe_u8 (S, S, 3) uint8 on the GPU (VIBE's edit pasted over its window), mask (S, S) bool on the
        GPU, H from BgMotion. Returns the output frame (S, S, 3) uint8 on the GPU, a buffer the next step reuses."""
        k = self.Hi_slot = (self.Hi_slot + 1) % len(self.Hi_host)
        if self.Hi_ev[k] is not None:
            self.Hi_ev[k].synchronize()
        self.Hi_host[k].copy_(torch.from_numpy(np.linalg.inv(H)).float())
        self.Hi.copy_(self.Hi_host[k], non_blocking=True)
        self.Hi_ev[k] = torch.cuda.Event()
        self.Hi_ev[k].record()
        self.f8.copy_(frame_u8)
        self.v8.copy_(vibe_u8)
        self.mk.copy_(mask)
        if not self.use_graph:
            self._core()
        elif self.graph is None:
            # warm up on a side stream without touching the state, then capture
            keep = [x.clone() for x in (self.canvas, self.conf, self.age)]
            st = torch.cuda.Stream()
            st.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(st):
                for _ in range(2):
                    for x, k in zip((self.canvas, self.conf, self.age), keep):
                        x.copy_(k)
                    self._core()
            torch.cuda.current_stream().wait_stream(st)
            for x, k in zip((self.canvas, self.conf, self.age), keep):
                x.copy_(k)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph, capture_error_mode="thread_local"):
                self._core()
            for x, k in zip((self.canvas, self.conf, self.age), keep):   # capture does not run the work
                x.copy_(k)
            self.graph.replay()
        else:
            self.graph.replay()
        return self.out8
