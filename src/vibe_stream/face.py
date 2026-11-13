"""Face pass: the main face edited again at a size where it covers enough latent tokens, and pasted back.

At 640 a face 20-60 px wide covers one or two of the DiT's 32-pixel latent tokens. The 2-step student cannot draw a
face on so few tokens: eyes and mouth come out as blotches, and since the flow that the carry-over and the temporal
filter follow fails on a turning, talking face, they are drawn afresh on every frame, which reads as noise crawling
over the face. The 8-step teacher does no better at that size. Cut out as a square of `side` face widths around the
face and brought to the face pass's size (256), the face covers about three tokens, and the same student draws a
proper face in the instruction's style.

Per frame, YuNet (`face_detection_yunet_2023mar.onnx` from the OpenCV model zoo, MIT licence) finds the faces, through
OpenCV on the host (`CVYuNet`) or as a TensorRT engine on the GPU (`TRTYuNet`), and `FaceTracker` follows one main face
causally.
The crops of a batch are a video of that face held in place, which optical flow follows well; they go through a second
`StreamEditor` at the crop size with its own carry-over and temporal filter, and are pasted back on the GPU inside a
feathered ellipse, their low band below half a face width replaced by the main output's so that the face keeps the
colours around it. The instruction is embedded again with a face crop: the embedding depends on the image it is made
with, and the whole frame's gives the crop editor worse faces (darker, half photographic). That costs one more run of
the text encoder per clip and per cut, on the stream's thread, when the first face is pasted.

The main path stays as it is; only the paste changes its output, and only around one face.

On the 640 path (pruned Klein student, INT8 engine, carry-over + filter 0.7 / 0.3) with a 256 INT8 engine of the Klein
768 student (its LoRA with the size changed and the init noise centre-cropped), over 11 clip x instruction pairs from 6
DAVIS clips with people (india, lady-running, swing, dance-twirl, breakdance, parkour): the change of the face region's
high band from frame to frame (against the previous frame's, warped along the input's flow, where that flow holds) is
9.8 without the face pass and 5.6 with it (the input's own: 3.2), and the faces are drawn in the instruction's style
instead of as blotches. On an RTX 3090 Ti (`breakdance`, YuNet as a TensorRT engine): loop 75 -> 58 fps, the same as
step-only (4.0-4.7 ms more per frame); live at a 30 fps source, mean / 95th percentile delay 243 / 360 -> 284 / 400 ms;
one more run of the text encoder per clip, about 160 ms. YuNet through OpenCV on the host takes 11 ms per frame at
384, the TensorRT engine 4.4 ms at 640 with the same faces as OpenCV's at 640. A face engine of its own
holds a second copy of the transformer's weights (1.9 GB in INT8); the stream's engine built with a 256
profile (`build_trt.py --extra_sizes 256`, `TRTTransformer(size=256, share=...)`) serves both on one copy. The crop editor's student matters: the
DAVIS-trained `multi_carry_s05_lut_2step_768` at 256 (on `lam4_carry_s05_lut_2step_640`'s path) draws hollow dark
eyes and blurred faces (face change 7.8 -> 5.4, but visibly worse than the Klein student's). Detecting on every
second frame only (`detect_every=2`) makes the crop lag behind a moving face (face change 6.1); the crops' flows at
256 instead of 128 change nothing (5.5).
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .carry import Carry, _gblur, holds
from .temporal_filter import TemporalFilter

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
Box = Tuple[float, float, float]        # centre x, centre y, face width, in frame pixels


class CVYuNet:
    """YuNet through OpenCV's FaceDetectorYN on the host, on the frame resized to `size` (about 11 ms per frame at
    384 on 32 threads; recall at 384 is that at 640 for faces 16 px and wider)."""

    def __init__(self, model: str, size: int = 384, score: float = 0.6, nms: float = 0.3):
        self.det = cv2.FaceDetectorYN.create(str(model), "", (size, size), score, nms, 50)
        self.size = size

    def __call__(self, frame: np.ndarray) -> List[Box]:
        h, w = frame.shape[:2]
        s = self.size
        small = cv2.resize(frame, (s, s), interpolation=cv2.INTER_AREA)
        _, fs = self.det.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        if fs is None:
            return []
        return [((x + fw / 2) * w / s, (y + fh / 2) * h / s, fw * w / s) for x, y, fw, fh in fs[:, :4]]


class TRTYuNet:
    """YuNet as a TensorRT engine (scripts/build_trt_yunet.py) at 640, run on a CUDA stream of its own so that the host
    thread calling it waits only for its own small piece of GPU work. The outputs are decoded as OpenCV's
    FaceDetectorYN decodes them (score sqrt(cls * obj), boxes from the stride grids) and NMS'd with OpenCV."""

    STRIDES = (8, 16, 32)

    def __init__(self, engine: str, score: float = 0.6, nms: float = 0.3, device: str = "cuda"):
        import json
        import sys
        from pathlib import Path

        import tensorrt as trt
        spec = json.loads(Path(engine).with_suffix(".json").read_text(encoding="utf-8"))
        if spec["tensorrt"] != trt.__version__ or spec["platform"] != sys.platform:
            raise RuntimeError(f"{engine} was built with TensorRT {spec['tensorrt']} on {spec['platform']}; "
                               f"this is TensorRT {trt.__version__} on {sys.platform}")
        self.size, self.score, self.nms = spec["size"], score, nms
        self.device = torch.device(device)
        self._log = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(self._log).deserialize_cuda_engine(Path(engine).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.out = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
                self.out[name] = torch.empty(tuple(self.engine.get_tensor_shape(name)), dtype=torch.float32,
                                             device=self.device)
                self.ctx.set_tensor_address(name, self.out[name].data_ptr())
        self.grid = {}
        for s in self.STRIDES:
            n = self.size // s
            idx = torch.arange(n * n, device=self.device)
            self.grid[s] = ((idx % n).float(), (idx // n).float())
        # high priority: on a GPU busy with the stream's large kernels, the detector's small ones would otherwise
        # wait behind them, and the host thread waiting for the faces with them
        self.stream = torch.cuda.Stream(self.device, priority=-1)

    def _candidates(self, frame: np.ndarray) -> torch.Tensor:
        """Queue one frame on the detector's stream; its candidates (score, cx, cy, w, h at 640) stay on the GPU."""
        h, w = frame.shape[:2]
        S = self.size
        x = torch.from_numpy(np.ascontiguousarray(frame)).to(self.device, non_blocking=True).permute(2, 0, 1)[None]
        x = x.float()
        if (h, w) != (S, S):
            x = F.interpolate(x, size=(S, S), mode="area")
        x = x[:, [2, 1, 0]].contiguous()                                        # BGR, 0-255, as OpenCV feeds it
        self.ctx.set_tensor_address("input", x.data_ptr())
        if not self.ctx.execute_async_v3(self.stream.cuda_stream):
            raise RuntimeError("TensorRT execution failed")
        cand = []
        for s in self.STRIDES:
            c, r = self.grid[s]
            cls = self.out[f"cls_{s}"][0, :, 0].clamp(0, 1)
            obj = self.out[f"obj_{s}"][0, :, 0].clamp(0, 1)
            bb = self.out[f"bbox_{s}"][0]
            cand.append(torch.stack([(cls * obj).sqrt(), (c + bb[:, 0]) * s, (r + bb[:, 1]) * s,
                                     bb[:, 2].exp() * s, bb[:, 3].exp() * s], 1))
        return torch.cat(cand)

    def _faces(self, cand: np.ndarray, w: int, h: int) -> List[Box]:
        if not len(cand):
            return []
        xywh = [[float(cx - bw / 2), float(cy - bh / 2), float(bw), float(bh)] for _, cx, cy, bw, bh in cand]
        keep = cv2.dnn.NMSBoxes(xywh, cand[:, 0].tolist(), self.score, self.nms, top_k=50)
        sx, sy = w / self.size, h / self.size
        return [(float(cand[i, 1]) * sx, float(cand[i, 2]) * sy, float(cand[i, 3]) * sx) for i in np.ravel(keep)]

    @torch.inference_mode()
    def __call__(self, frame: np.ndarray) -> List[Box]:
        with torch.cuda.stream(self.stream):
            cand = self._candidates(frame)
            cand = cand[cand[:, 0] >= self.score].cpu().numpy()
        return self._faces(cand, frame.shape[1], frame.shape[0])

    @torch.inference_mode()
    def many(self, frames: Sequence[np.ndarray]) -> List[List[Box]]:
        """The faces of several frames, queued together and waited for once."""
        with torch.cuda.stream(self.stream):
            # each frame's candidates are copied out of the engine's buffers before the next call reuses them; all of
            # them come to the host in one copy (a mask on the GPU would wait for each frame)
            cand = torch.stack([self._candidates(f) for f in frames]).cpu().numpy()
        return [self._faces(c[c[:, 0] >= self.score], f.shape[1], f.shape[0]) for c, f in zip(cand, frames)]


def make_detector(model: str):
    """`TRTYuNet` for a .engine file, else `CVYuNet` for the ONNX file. The engine is faster where the GPU has room to
    spare (RTX 3090 Ti at 640: loop 57.0 against 50.6 fps); where the GPU is the bottleneck its small kernels wait
    behind the stream's and OpenCV on the host is faster (RTX 4060 Ti: 29.9 against 25.9 fps)."""
    return TRTYuNet(model) if str(model).endswith(".engine") else CVYuNet(model)


class FaceTracker:
    """One main face followed causally, frame by frame, from a detector's faces (`make_detector`).

    A detection joins a track when its centre lies within `link` face widths of the track's last one (per detected
    frame since, when the detector runs on every `detect_every`-th frame only); a track unseen for more than `hold`
    frames ends. The main face is the widest track seen on
    at least two frames, kept until it ends or another one is `switch` times as wide on `switch_frames` frames in a
    row. Its box is smoothed (each new detection moves the centre 0.6 and the width 0.3 of the way) and held while it
    is not detected; its paste weight is 0.5 on the track's first frame, then 1, and falls by `fade` per frame without
    a detection. Faces narrower than `min_width` or wider than `max_width` (frame pixels) get weight 0: below, the crop
    holds too little to go on; above, the face already covers enough tokens.
    """

    def __init__(self, detector, link: float = 0.8, hold: int = 5, fade: float = 0.6, switch: float = 1.5,
                 switch_frames: int = 3, min_width: float = 16.0, max_width: float = 200.0, detect_every: int = 1):
        self.detect = detector              # RGB uint8 frame -> faces (centre x, centre y, width) in its pixels
        self.link, self.hold, self.fade = link, hold, fade
        self.detect_every = detect_every
        self.switch, self.switch_frames = switch, switch_frames
        self.min_width, self.max_width = min_width, max_width
        self.reset()

    def reset(self) -> None:
        self.tracks: List[dict] = []
        self.t = 0
        self.main_id: Optional[int] = None
        self.box: Optional[Box] = None
        self.weight = 0.0
        self._next_id = 0
        self._rival = (None, 0)

    def step(self, dets: Optional[Sequence[Box]]) -> Tuple[Optional[Box], float, bool]:
        """One frame's detections -> (the main face's box, None before the first one; its paste weight; whether it is
        a new track since the previous frame). `dets` None: no detection ran on this frame, everything is held."""
        t = self.t
        self.t += 1
        if dets is None:
            if self.box is None:
                return None, 0.0, False
            return self.box, self.weight if self.min_width <= self.box[2] <= self.max_width else 0.0, False
        used = set()
        for d in sorted(dets, key=lambda d: -d[2]):
            # distance in face widths (scaled by the frames since the track was last seen, only when the detector skips
            # frames: with a detection every frame that scaling let other faces join the main track)
            gap = self.detect_every if self.detect_every > 1 else 1
            cand = [(np.hypot(tr["c"][0] - d[0], tr["c"][1] - d[1]) / d[2] / (min(t - tr["last"], gap)), i)
                    for i, tr in enumerate(self.tracks) if i not in used]
            dist, i = min(cand, default=(np.inf, None))
            if dist <= self.link:
                self.tracks[i].update(c=d, last=t, seen=self.tracks[i]["seen"] + 1)
            else:
                self.tracks.append(dict(c=d, last=t, seen=1, id=self._next_id))
                self._next_id += 1
                i = len(self.tracks) - 1
            used.add(i)
        self.tracks = [tr for tr in self.tracks if t - tr["last"] <= self.hold]
        main = next((tr for tr in self.tracks if tr["id"] == self.main_id), None)
        best = max((tr for tr in self.tracks if tr["seen"] >= 2 and tr["last"] == t), key=lambda tr: tr["c"][2],
                   default=None)
        if main is None:
            main = best
        elif best is not None and best is not main and best["c"][2] > self.switch * main["c"][2]:
            rid, n = self._rival
            n = n + 1 if rid == best["id"] else 1
            self._rival = (best["id"], n)
            if n >= self.switch_frames:
                main = best
        else:
            self._rival = (None, 0)
        new = main is not None and main["id"] != self.main_id
        self.main_id = None if main is None else main["id"]
        if main is not None and (new or self.box is None):
            self.box, self.weight = main["c"], 0.5
        elif main is not None and main["last"] == t:
            b, d = self.box, main["c"]
            self.box = (0.4 * b[0] + 0.6 * d[0], 0.4 * b[1] + 0.6 * d[1], 0.7 * b[2] + 0.3 * d[2])
            self.weight = min(1.0, self.weight + 0.5)
        else:
            self.weight *= self.fade
        if self.box is None:
            return None, 0.0, False
        ok = self.min_width <= self.box[2] <= self.max_width
        return self.box, self.weight if ok else 0.0, new


@dataclass
class FaceBatch:
    """One staged batch of the face pass (`FacePass.stage`)."""
    boxes: np.ndarray                   # (B, 3) centre x, centre y, face width, frame pixels
    weights: np.ndarray                 # (B,) paste weights
    crops: torch.Tensor                 # (B, Z, Z, 3) uint8 on the GPU
    probe: np.ndarray                   # the first crop with a face (host), to embed the instruction with
    boxes_gpu: torch.Tensor             # `boxes` on the GPU, copied from pinned memory when the batch is staged
    flows: object = None                # the crop editor's TemporalFilter.flows
    plan: object = None                 # and Carry.plan


class FacePass:
    """The face pass of a `StreamEditor` (its `face` argument). `engine` / `tae`: the transformer engine and taesana
    (engines or the PyTorch one) at the crop size, same batch as the main editor's; `model`: the YuNet ONNX file (run
    by OpenCV on the host) or its TensorRT engine (scripts/build_trt_yunet.py, run on the GPU).

    Host side, in frame order: `host_frame` per frame (detection, tracking, the crop and its flows) and `host_end` at
    the end of each batch; `StreamEditor.run` calls them inline, `LiveEditor` on its host thread as frames arrive.
    `stage` (the stream's thread) turns a batch of them into a `FaceBatch`, None when no face is to be pasted; `apply`
    (GPU) edits the crops and pastes them into the main output.
    """

    def __init__(self, vibe, engine, tae, model: str, *, side: float = 2.6, carry: bool = True,
                 temporal_filter: Optional[Tuple[float, float]] = (0.7, 0.3), flow_size: int = 128,
                 detect_every: int = 1, tracker: Optional[dict] = None):
        from .stream import StreamEditor
        Z = engine.spec.size
        v = copy.copy(vibe)             # the same model at the crop size (the engine and taesana do the size's work)
        v.size = Z
        self.size, self.side, self.flow_size, self.detect_every = Z, side, flow_size, detect_every
        self.ed = StreamEditor(v, tae, engine=engine, carry=Carry(flow_size=flow_size) if carry else None,
                               temporal_filter=(TemporalFilter(*temporal_filter, flow_size=flow_size)
                                                if temporal_filter else None),
                               reembed_at_cut=False)
        self.batch_size = self.ed.batch_size
        self.device = self.ed.device
        self.tracker = FaceTracker(make_detector(model), detect_every=detect_every, **(tracker or {}))
        # the paste mask in crop pixels, where the face is centred and `Z / side` wide
        u = Z / side
        m = np.zeros((Z, Z), np.float32)
        cv2.ellipse(m, (Z // 2, int(Z / 2 - 0.05 * u)), (int(0.62 * u), int(0.8 * u)), 0, 0, 360, 1.0, -1)
        self.mask = torch.from_numpy(cv2.GaussianBlur(m, (0, 0), 0.12 * u))[None, None].to(self.device)
        self.sigma_low = 0.5 * u
        self._dis_f = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self._dis_c = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
        self._pinned: List[torch.Tensor] = []
        self._copied = [torch.cuda.Event(), torch.cuda.Event()]
        self._slot = 0
        self.embed_ms: List[float] = []     # one entry per instruction embedded with a face crop
        self._side_stream = torch.cuda.Stream(self.device, priority=-1)
        self.reset()

    def reset(self) -> None:
        """Start afresh (a new clip): no face tracked yet."""
        self.tracker.reset()
        self._h_prev = self._h_ref = None
        self._h_n = 0
        self._restart = True
        self._emb = self._emb_for = None
        self._side = self._side_for = None      # a face embedding being made on the side (async re-embedding)

    def crop(self, frame: np.ndarray, box: Box) -> np.ndarray:
        cx, cy, w = box
        s = self.size / (self.side * w)
        M = np.array([[s, 0, self.size / 2 - cx * s], [0, s, self.size / 2 - cy * s]], np.float32)
        return cv2.warpAffine(frame, M, (self.size, self.size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)

    # ---------------------------------------------------------------- host
    def gray(self, crop: np.ndarray) -> np.ndarray:
        fs = self.flow_size
        return cv2.cvtColor(cv2.resize(crop, (fs, fs), interpolation=cv2.INTER_AREA), cv2.COLOR_RGB2GRAY)

    def host_frame(self, frame: np.ndarray, faces: Optional[List[Box]] = None) -> dict:
        """Detection (unless `faces` are given), tracking, the crop and its flows (to the previous crop for the filter,
        to the previous batch's last crop for the carry-over, at `flow_size`) for one frame, in order. While a main face
        is followed, the detector runs on every `detect_every`-th frame only and the box is held in between."""
        detect = self.tracker.main_id is None or self._h_n % self.detect_every == 0
        self._h_n += 1
        if detect and faces is None:
            faces = self.tracker.detect(frame)
        box, weight, new = self.tracker.step(faces if detect else None)
        it = dict(box=box, weight=weight, new=new, crop=None, gray=None, tfl=None, cres=None)
        if box is None:
            return it
        if new:
            self._h_prev = self._h_ref = None
        crop = self.crop(frame, box)
        g = self.gray(crop)
        it.update(crop=crop, gray=g)
        if self._h_prev is not None:
            it["tfl"] = self._dis_f.calc(g, self._h_prev, None)
        if self._h_ref is not None:
            fl = self._dis_c.calc(g, self._h_ref, None)
            c = self.ed.carry
            it["cres"] = (fl, holds(g, self._h_ref, fl, c.tau).mean() >= c.cut_below) if c is not None else None
        self._h_prev = g
        return it

    def host_end(self, items: List[dict]) -> List[dict]:
        """End of a batch: its last crop is the carry-over's reference for the next one."""
        self._h_ref = items[-1]["gray"]
        return items

    def host_batch(self, frames: Sequence[np.ndarray]) -> List[dict]:
        """`host_frame` for a batch's frames and `host_end`; a detector that can (`TRTYuNet.many`) runs on all of them
        at once."""
        many = getattr(self.tracker.detect, "many", None)
        faces = many(frames) if many is not None and self.detect_every == 1 else [None] * len(frames)
        return self.host_end([self.host_frame(f, d) for f, d in zip(frames, faces)])

    # ---------------------------------------------------------------- stream thread
    def stage(self, frames: np.ndarray, items: List[dict]) -> Optional[FaceBatch]:
        """A batch's host results -> a FaceBatch with the crops on the GPU (current stream), or None when no frame of
        it has a face to paste. A batch after such a gap, or in which a new main face appears, starts the crop editor
        afresh (a keyframe batch)."""
        if self._side is not None and self._side.ready():  # a face embedding made on the side: the crop editor
            side, self._side = self._side, None                                  # starts afresh with it
            self._emb, self._emb_for = side.take(torch.cuda.current_stream(self.device)), self._side_for
            self.embed_ms.append(side.ms)
            self._restart = True
        weights = np.array([it["weight"] for it in items], np.float32)
        if not (weights > 0.01).any():
            self._restart = True
            return None
        first = next(it["box"] for it in items if it["box"] is not None)
        restart = (self._restart or any(it["new"] or it["box"] is None for it in items)
                   or (self.ed.carry is not None and any(it["cres"] is None for it in items)))
        if restart:
            self.ed.reset_clip()
            self._restart = False
        B, Z = len(items), self.size
        if not self._pinned or self._pinned[0].shape[0] != B:
            self._pinned = [torch.empty((B, Z, Z, 3), dtype=torch.uint8).pin_memory() for _ in range(2)]
        s, self._slot = self._slot, self._slot ^ 1
        self._copied[s].synchronize()
        buf, grays, boxes = self._pinned[s].numpy(), [], []
        for j, it in enumerate(items):
            if it["box"] is None:           # before the first face of the clip: crops at its first box, weight 0
                buf[j] = self.crop(frames[j], first)
                grays.append(self.gray(buf[j]))
                boxes.append(first)
            else:
                buf[j] = it["crop"]
                grays.append(it["gray"])
                boxes.append(it["box"])
        ed = self.ed
        flows = ed.tfilter.flows(buf, [it["tfl"] for it in items]) if ed.tfilter is not None else None
        plan = ed.carry.plan(buf, (grays, [it["cres"] for it in items])) if ed.carry is not None else None
        crops = self._pinned[s].to(self.device, non_blocking=True)
        boxes = np.array(boxes, np.float32)
        # a copy from pageable memory would wait for all the GPU work queued before it: the paste would hold the host
        # until the batch's edits are done
        boxes_gpu = torch.from_numpy(boxes).pin_memory().to(self.device, non_blocking=True)
        self._copied[s].record()
        probe = buf[int(np.argmax(weights > 0.01))].copy()
        return FaceBatch(boxes, weights, crops, probe, boxes_gpu, flows, plan)

    # ---------------------------------------------------------------- GPU
    @torch.inference_mode()
    def apply(self, out: torch.Tensor, fb: FaceBatch, main) -> torch.Tensor:
        """Edit the batch's crops and paste them into `out` (B, H, W, 3) uint8 on the GPU, the shown output of `main`
        (the StreamEditor); returns the result, same layout. The instruction is embedded with a face crop (the
        embedding depends on the image, and one made with the whole frame gives worse faces), the first time a face
        is pasted after the main editor's own embedding was made (a clip's start, a cut)."""
        if not self.begin(fb, main):
            return out
        return self.paste(out, fb, self.ed.process(fb.crops, fb.flows, fb.plan))

    def begin(self, fb: FaceBatch, main) -> bool:
        """The crop editor's embedding for this batch (see `apply`); False when there is no face pass this batch
        (with `main.reembed_async` live, until the first face embedding made on the side is ready)."""
        ed = self.ed
        if self._emb is None or self._emb_for is not main.emb:
            ed._prompt, ed._reference = main._prompt, main._reference
            if main.reembed_async and main.live:    # made on the side (`stage` takes it); no face pass until
                                                    # the first one is ready
                if self._side is None:
                    from .stream import SideEmbedding
                    self._side = SideEmbedding(ed._make_embedding, fb.probe, self._side_stream)
                    self._side_for = main.emb
                if self._emb is None:
                    self._restart = True
                    return False
            else:
                t0 = time.perf_counter()
                ed._embed(fb.probe)
                torch.cuda.current_stream(self.device).synchronize()
                self.embed_ms.append(1000 * (time.perf_counter() - t0))
                self._emb, self._emb_for = (ed.emb, ed.enc16), main.emb
        ed.emb, ed.enc16 = self._emb
        return True

    @torch.inference_mode()
    def paste(self, out: torch.Tensor, fb: FaceBatch, crop_out: torch.Tensor) -> torch.Tensor:
        """The batch's edited crops (B, Z, Z, 3) uint8 pasted into `out` (see `apply`)."""
        B, H, W = out.shape[:3]
        Z = self.size
        cx, cy, fw = fb.boxes_gpu[:, 0], fb.boxes_gpu[:, 1], fb.boxes_gpu[:, 2]
        s = Z / (self.side * fw)                                    # crop pixels per frame pixel
        # frame -> crop in grid_sample's coordinates (align_corners=False): u = s (x - cx) + Z / 2
        ax, ay = s * W / Z, s * H / Z
        bx_, by_ = (s * (W - 1 - 2 * cx) + 1) / Z, (s * (H - 1 - 2 * cy) + 1) / Z
        zero = torch.zeros_like(ax)
        to_crop = torch.stack([torch.stack([1 / ax, zero, -bx_ / ax], 1), torch.stack([zero, 1 / ay, -by_ / ay], 1)], 1)
        to_frame = torch.stack([torch.stack([ax, zero, bx_], 1), torch.stack([zero, ay, by_], 1)], 1)
        res = out.permute(0, 3, 1, 2).float()
        around = F.grid_sample(res, F.affine_grid(to_crop, (B, 3, Z, Z), align_corners=False), mode="bilinear",
                               padding_mode="border", align_corners=False)
        co = crop_out.permute(0, 3, 1, 2).float()
        co = co + _gblur(around - co, self.sigma_low)               # the main output's colours below half a face
        src = torch.cat([co, self.mask.expand(B, -1, -1, -1)], 1)
        grid = F.affine_grid(to_frame, (B, 4, H, W), align_corners=False)
        rows = []
        for j in range(B):
            if fb.weights[j] <= 0.01:
                rows.append(res[j:j + 1])
                continue
            n = int(round(self.side * float(fb.boxes[j, 2])))      # the crop's size in the frame: area-average down
                                                                    # (from the host copy: no sync with the GPU)
            sj = F.interpolate(src[j:j + 1], size=(n, n), mode="area") if n < Z else src[j:j + 1]
            back = F.grid_sample(sj, grid[j:j + 1], mode="bilinear", padding_mode="zeros", align_corners=False)
            m = back[:, 3:] * float(fb.weights[j])
            rows.append(res[j:j + 1] * (1 - m) + back[:, :3] * m)
        return torch.cat(rows).clamp(0, 255).round().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
