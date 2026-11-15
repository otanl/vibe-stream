"""Local edits from a text instruction: remove, add to or change one object and leave the rest of the frame as it is.

A plain streaming edit with "put sunglasses on the girl on the left" redraws the whole frame, puts glasses on the
wrong people and changes the background. Here the edit runs on a crop that follows the object, with the rest of the
frame kept:
  plan     `vibe_stream.local_plan.plan_edit` (VIBE's own Qwen3-VL): the target, the part an added thing goes on, the
           kind of edit, and their boxes on the first frame. Once per instruction.
  track    SAM 2.1 hiera-tiny follows each box frame by frame, causally (`vibe_stream.sam_tracker.LeanTracker`, one
           CUDA graph per frame; 9 ms at 512 on an RTX 3090 Ti).
  window   a square window around the tracked mask (centre and size follow it with an EMA; the size never below 1.1x
           the mask's extent; held while the mask is lost), resized to the student's size: 384 for parts (sunglasses
           on a head; small faces come out as well as at 768 or clearer), 512 for whole objects (removals, colour
           changes; at 384 a dress change missed the top in the first frames).
  edit     the student on the crops, 2 steps from the shared noise, with the close-up instruction. Always 2 steps: the
           carry-over brings a removed object back as a ghost and fades a changed colour back, because it starts each
           frame from the previous output.
  paste    through the dilated, feathered mask; for a removal also through what the edit changed strongly next to the
           mask (it removes the shadow with the object; a mask-only paste put the shadow back), or, with `bgfill`,
           from a background canvas of real pixels seen on earlier frames (`vibe_stream.bgfill`). An edit on the face
           (plan.face: glasses, a mask, make-up) is pasted only while a face turned to the camera is seen in the
           part's window (`FaceGate`), so the back of a head does not get a black bar.
Outside the paste region the output is the input frame. The stages run one after the other per batch; on an RTX 3090
Ti that gives about 50 fps for additions and 35 for removals (`scripts/local_edit.py`).
"""

from __future__ import annotations

import copy
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .bgfill import BgFillGPU, BgMotion, blur, dilate
from .local_plan import LocalPlan
from .sam_tracker import LeanTracker
from .stream import StreamEditor
from .vibe_model import ROOT

YUNET = ROOT / "models" / "face_detection_yunet_2023mar.onnx"


class Window:
    """A causal square crop window following one object's box, frame by frame."""

    def __init__(self, size: int, margin: float, min_side: int, alpha: float = 0.5, beta: float = 0.2):
        self.size, self.margin, self.min_side, self.alpha, self.beta = size, margin, min_side, alpha, beta
        self.c = self.s = None

    def __call__(self, bbox):
        """bbox (x0, y0, x1, y1) or None (lost: the window is held). Returns (x, y, side) or None before any box."""
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            ext = max(x1 - x0, y1 - y0) + 1
            cn, sn = np.array([(x0 + x1) / 2, (y0 + y1) / 2]), ext * self.margin
            self.c = cn if self.c is None else self.c + self.alpha * (cn - self.c)
            self.s = sn if self.s is None else max(self.s + self.beta * (sn - self.s), 1.1 * ext)
        if self.c is None:
            return None
        side = int(np.clip(round(self.s), self.min_side, self.size))
        x = int(np.clip(round(self.c[0] - side / 2), 0, self.size - side))
        y = int(np.clip(round(self.c[1] - side / 2), 0, self.size - side))
        return x, y, side


def paste_batch(frames: torch.Tensor, edited: torch.Tensor, masks: Sequence[torch.Tensor], wins, remove_thr: float = 0.0,
                small: int = 160, weights: Optional[Sequence[float]] = None) -> torch.Tensor:
    """frames (B, S, S, 3) float, edited (B, Z, Z, 3) uint8, masks (S, S) bool, wins (x, y, side) or None: each window's
    edit pasted through the dilated, feathered mask. With remove_thr > 0 (removals) also through the regions where
    the edit changed the window by more than remove_thr (blurred mean absolute difference) that touch the dilated
    mask; their connected components are found by OpenCV on copies of at most small x small pixels, all frames of the
    batch in one transfer each way. weights: per frame, a factor on the alpha (the face gate)."""
    out, todo = frames.clone(), []
    for j, w in enumerate(wins):
        if w is None:
            continue
        x, y, side = w
        r = max(3, int(0.05 * side))
        back = F.interpolate(edited[j].permute(2, 0, 1)[None].float(), size=(side, side), mode="area")[0].permute(1, 2, 0)
        src = frames[j, y:y + side, x:x + side]
        m = (dilate(masks[j][y:y + side, x:x + side].float()[None, None], r) > 0).float()
        al = blur(m, r / 2) * (1.0 if weights is None else weights[j])
        item = dict(j=j, x=x, y=y, side=side, r=r, back=back, src=src, al=al)
        if remove_thr > 0:
            sz = (max(1, round(side * min(1.0, small / side))),) * 2
            reg = (blur((back - src).abs().mean(2)[None, None], r) > remove_thr).float()
            pad = (0, small - sz[1], 0, small - sz[0])
            item["reg"] = F.pad(F.interpolate(reg, size=sz, mode="area")[0, 0], pad) > 0.5
            item["near"] = F.pad(F.interpolate(m, size=sz, mode="area")[0, 0], pad) > 0
            item["sz"] = sz
        todo.append(item)
    if remove_thr > 0 and todo:
        reg = torch.stack([t["reg"] for t in todo]).to(torch.uint8).cpu().numpy()
        near = torch.stack([t["near"] for t in todo]).cpu().numpy()
        keep = np.zeros(reg.shape, np.float32)
        for i, t in enumerate(todo):
            h, w = t["sz"]
            _, lab = cv2.connectedComponents(reg[i, :h, :w])
            keep[i, :h, :w] = np.isin(lab, np.unique(lab[near[i, :h, :w] & (lab > 0)])) & (lab > 0)
        keep = torch.from_numpy(keep).to(frames.device)
        for i, t in enumerate(todo):
            h, w = t["sz"]
            k = F.interpolate(keep[i, :h, :w][None, None], size=(t["side"],) * 2, mode="bilinear", align_corners=False)
            t["al"] = torch.maximum(t["al"], blur(k, t["r"] / 2))
    for t in todo:
        al = t["al"][0, 0, ..., None]
        out[t["j"], t["y"]:t["y"] + t["side"], t["x"]:t["x"] + t["side"]] = t["src"] * (1 - al) + t["back"] * al
    return out


class FaceGate:
    """Whether to paste an edit on the face in this frame: a weight in [0, 1], called once per frame in order.

    YuNet (OpenCV's FaceDetectorYN, MIT) looks for a face in the part's crop window that
    - touches the part's mask: at least `min_overlap` of the face box inside the mask dilated by 15 % of the face's size
      (SAM 2's "head" is sometimes the hair alone with the face below it);
    - is centred within the mask's horizontal extent (+- `x_margin`) and at least `min_rel` of its width (a neighbour's
      face beside the head, or one further back, is not the target's);
    - is turned to the camera: eye spacing at least `min_eyes` of the face's width and the nose roughly between the
      eyes (YuNet scores the side or back of a head up to about 0.8).
    The gate opens once a face was found in `on_hits` of the last `on_window` frames, shuts after `keep` frames
    without one, and moves by 1 / `fade` per frame, so the edit fades in and out instead of popping.
    """

    def __init__(self, model=YUNET, keep: int = 6, fade: int = 3, size: int = 320, min_frac: float = 0.1,
                 min_overlap: float = 0.1, score: float = 0.7, min_eyes: float = 0.18, nose_range=(-0.3, 1.6),
                 x_margin: float = 0.15, min_rel: float = 0.35, on_hits: int = 2, on_window: int = 3):
        self.yunet = cv2.FaceDetectorYN.create(str(model), "", (size, size), score, 0.3, 50)
        self.keep, self.fade, self.size, self.min_frac, self.min_overlap = keep, fade, size, min_frac, min_overlap
        self.min_eyes, self.nose_range, self.x_margin, self.min_rel = min_eyes, nose_range, x_margin, min_rel
        self.on_hits, self.on_window = on_hits, on_window
        self.reset()

    def reset(self) -> None:
        self.seen, self.w, self.open = [], None, False

    def detect(self, crop: np.ndarray, mask: np.ndarray) -> List[tuple]:
        """crop (Z, Z, 3) RGB uint8, mask (Z, Z) bool of the part in the same crop -> the faces (x, y, w, h) at crop
        scale that qualify."""
        s = self.size
        f = crop.shape[0] / s
        small = cv2.resize(crop, (s, s), interpolation=cv2.INTER_AREA)
        m = cv2.resize(mask.astype(np.uint8), (s, s), interpolation=cv2.INTER_NEAREST)
        ms = int(self.min_frac * s)
        boxes = []
        _, faces = self.yunet.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        for face in (faces if faces is not None else []):
            x, y, w, h = (int(round(float(v))) for v in face[:4])
            eyes = abs(float(face[6] - face[4]))
            nose = (float(face[8]) - min(float(face[4]), float(face[6]))) / max(eyes, 1e-3)
            frontal = eyes >= self.min_eyes * w and self.nose_range[0] <= nose <= self.nose_range[1]
            if min(w, h) >= ms and frontal:
                boxes.append((max(0, x), max(0, y), w, h))
        xs = np.nonzero(m.any(0))[0]
        if not len(xs):
            return []
        mx0, mx1 = xs.min(), xs.max()
        mx0, mx1 = mx0 - self.x_margin * (mx1 - mx0), mx1 + self.x_margin * (mx1 - mx0)
        mw = (mx1 - mx0) / (1 + 2 * self.x_margin)
        keep = []
        for x, y, w, h in boxes:
            if not mx0 <= x + w / 2 <= mx1 or w < self.min_rel * mw:
                continue
            r = max(1, int(0.15 * w))
            md = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
            if md[y:y + h, x:x + w].mean() >= self.min_overlap:
                keep.append((x, y, w, h))
        return [tuple(int(round(v * f)) for v in b) for b in keep]

    def __call__(self, crop: np.ndarray, mask: np.ndarray) -> float:
        self.seen = (self.seen + [bool(self.detect(crop, mask))])[-max(self.keep, self.on_window):]
        if not self.open and sum(self.seen[-self.on_window:]) >= self.on_hits:
            self.open = True
        elif self.open and not any(self.seen[-self.keep:]):
            self.open = False
        target = 1.0 if self.open else 0.0
        self.w = target if self.w is None else self.w + float(np.clip(target - self.w, -1 / self.fade, 1 / self.fade))
        return self.w


@dataclass
class LocalTiming:
    """Milliseconds per stage, summed over the measured frames (all but the first `warmup_batches`)."""
    frames: int = 0
    loop_s: float = 0.0
    stages: Dict[str, float] = field(default_factory=dict)
    embed_ms: float = 0.0

    @property
    def fps(self) -> float:
        return self.frames / self.loop_s if self.loop_s else 0.0

    def ms_per_frame(self) -> Dict[str, float]:
        return {k: round(1000 * v / max(self.frames, 1), 2) for k, v in self.stages.items()}


class LocalEditor:
    """A local edit through a student at the crop size. `engine` / `tae`: the transformer engine and taesana at that
    size (384 for parts, 512 for whole objects; an engine of the stream's student built at that size, e.g. from
    scripts/resize_student.py, or a profile of the stream's engine, as the face pass uses); `sam` / `proc`: from
    `load_sam2(sam_res)`. One editor serves any number of clips and instructions of its size."""

    def __init__(self, vibe, engine, tae, sam, proc, *, sam_res: int = 512, bgfill: bool = True,
                 face_gate: bool = True, face_model=YUNET, remove_thr: float = 24.0, margin: float = 1.4,
                 part_margin: float = 2.5, part_min_side: int = 128):
        v = copy.copy(vibe)             # the same model at the crop size (the engine and taesana do the size's work)
        v.size = engine.spec.size
        self.ed = StreamEditor(v, tae, engine=engine, reembed_at_cut=False)
        self.Z, self.B = engine.spec.size, self.ed.batch_size
        self.sam, self.proc, self.sam_res = sam, proc, sam_res
        self.use_bgfill, self.use_gate, self.face_model = bgfill, face_gate, face_model
        self.remove_thr, self.margins = remove_thr, (margin, part_margin, part_min_side)
        self.trackers: List[LeanTracker] = []
        self.bgf: Optional[BgFillGPU] = None
        self.pool, self.mpool = ThreadPoolExecutor(1), ThreadPoolExecutor(1)

    def _trackers(self, n: int, size: int) -> List[LeanTracker]:
        while len(self.trackers) < n:
            self.trackers.append(LeanTracker(self.sam, self.proc, self.sam_res))
            self.trackers[-1].precapture(size)          # all graphs up front: frames 1-15 run as graphs too
        return self.trackers[:n]

    @torch.inference_mode()
    def run(self, frames: np.ndarray, plan: LocalPlan, warmup_batches: int = 2):
        """frames (N, S, S, 3) uint8 RGB; plan from `plan_edit` on frames[0]. Returns the output frames (list of
        uint8 arrays) and a LocalTiming. A last partial batch is padded with its last frame."""
        n, size = len(frames), frames.shape[1]
        if not plan.local:
            raise ValueError(f"not a local edit (scope {plan.scope!r}, {len(plan.objects)} objects)")
        margin, part_margin, part_min_side = self.margins
        sel, mg, min_side, text = plan.group(margin, part_margin, part_min_side)
        boxes = [plan.objects[j]["box"] for j in sel]
        trackers = self._trackers(len(boxes), size)
        has_part = any(plan.objects[j]["role"] == "part" for j in sel)
        gate = FaceGate(self.face_model) if self.use_gate and plan.face and has_part else None
        use_bg = self.use_bgfill and plan.kind == "remove"
        if use_bg and (self.bgf is None or self.bgf.s != size):
            self.bgf = BgFillGPU(size)
            self.bgf.precapture()
        if use_bg:
            self.bgf.reset()
        motion = BgMotion() if use_bg else None
        rthr = self.remove_thr if plan.kind == "remove" else 0.0
        win = Window(size, mg, min_side)
        ed, B, Z = self.ed, self.B, self.Z
        timing = LocalTiming(stages=dict.fromkeys(("track", "crop", "student", "gate_wait", "motion_wait", "paste"), 0.0))
        outs: List[np.ndarray] = []
        cs = torch.cuda.current_stream()
        for bi, b0 in enumerate(range(0, n, B)):
            idx = [min(b0 + j, n - 1) for j in range(B)]
            t_batch = time.perf_counter()
            st = dict.fromkeys(timing.stages, 0.0)
            gfr = torch.from_numpy(frames[idx]).cuda()
            # track: the union of the objects' masks, its box, the window
            t = time.perf_counter()
            wins, gmasks = [], []
            for j in range(B):
                logits = [tr.start(gfr[j], [bx]) if b0 == 0 and j == 0 else tr(gfr[j])
                          for tr, bx in zip(trackers, boxes)]
                m = (F.interpolate(torch.cat(logits).float(), size=(size, size), mode="bilinear",
                                   align_corners=False)[:, 0] > 0).any(0)
                ys, xs = torch.nonzero(m, as_tuple=True)
                bbox = tuple(torch.stack([xs.min(), ys.min(), xs.max(), ys.max()]).tolist()) if len(xs) >= 16 else None
                wins.append(win(bbox))
                gmasks.append(m)
            st["track"] = time.perf_counter() - t
            held = [w for w in wins if w is not None]
            if not held:
                outs.extend(frames[idx][:min(B, n - b0)])
                continue
            # crops at the student's size
            t = time.perf_counter()
            crops = []
            for j, w in enumerate(wins):
                x, y, side = w or held[0]
                c = gfr[j, y:y + side, x:x + side].permute(2, 0, 1)[None].float()
                crops.append(F.interpolate(c, size=(Z, Z), mode="bicubic", align_corners=False)
                             .clamp(0, 255).round().to(torch.uint8)[0].permute(1, 2, 0))
            crops = torch.stack(crops)
            arr = crops.cpu().numpy() if (bi == 0 or gate is not None) else None
            fut = mfut = None
            if gate is not None:
                mc = torch.stack([F.interpolate(gmasks[j][y:y + sd, x:x + sd][None, None].float(), size=(Z, Z),
                                                mode="nearest")[0, 0] > 0
                                  for j, (x, y, sd) in enumerate(w or held[0] for w in wins)]).cpu().numpy()
                fut = self.pool.submit(lambda A=arr, M=mc: [gate(c, mm) for c, mm in zip(A, M)])
            if motion is not None:
                keep = (F.interpolate(torch.stack(gmasks)[:, None].float(), size=(384, 384), mode="area")[:, 0] == 0)
                keep = [~(cv2.dilate((~k).astype(np.uint8), np.ones((9, 9), np.uint8)) > 0) for k in keep.cpu().numpy()]
                mfut = self.mpool.submit(lambda Fs=frames[idx], Ks=keep: [motion(f, k) for f, k in zip(Fs, Ks)])
            cs.synchronize()
            st["crop"] = time.perf_counter() - t
            # the student on the crops (the instruction is embedded from the first crop)
            if bi == 0:
                te = time.perf_counter()
                with torch.cuda.stream(ed.stream):
                    ed.set_prompt(text, arr[0])
                timing.embed_ms = 1000 * (time.perf_counter() - te)
            t = time.perf_counter()
            with torch.cuda.stream(ed.stream):
                ed.stream.wait_stream(cs)
                eo = ed.process(crops)
            cs.wait_stream(ed.stream)
            cs.synchronize()
            st["student"] = time.perf_counter() - t
            t = time.perf_counter()
            weights = fut.result() if fut is not None else None
            st["gate_wait"] = time.perf_counter() - t
            t = time.perf_counter()
            Hs = mfut.result() if mfut is not None else None
            st["motion_wait"] = time.perf_counter() - t
            # paste
            t = time.perf_counter()
            if motion is not None:
                ob = []
                for j, w in enumerate(wins):
                    full = gfr[j].clone()
                    if w is not None:
                        x, y, sd = w
                        full[y:y + sd, x:x + sd] = F.interpolate(eo[j].permute(2, 0, 1)[None].float(), size=(sd, sd),
                                                                 mode="area")[0].permute(1, 2, 0).round().clamp(0, 255).to(torch.uint8)
                    ob.append(self.bgf.step(gfr[j], full, gmasks[j], Hs[j]).clone())
                ob = torch.stack(ob).cpu().numpy()
            else:
                ob = paste_batch(gfr.float(), eo, gmasks, wins, rthr, weights=weights).round().clamp(0, 255) \
                    .to(torch.uint8).cpu().numpy()
            st["paste"] = time.perf_counter() - t
            outs.extend(ob[:min(B, n - b0)])
            if bi >= warmup_batches and b0 + B <= n:
                timing.frames += B
                timing.loop_s += time.perf_counter() - t_batch
                for k, v in st.items():
                    timing.stages[k] += v
        return outs, timing
