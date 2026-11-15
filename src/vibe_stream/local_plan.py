"""From a text instruction to what a local edit needs: which object, which part of it, which kind of edit, and where
they are on the first frame. The control stays text only; the boxes are internal.

VIBE's own Qwen3-VL-2B (`VibeModel.chat`, no extra model) does three things:
  plan    the instruction alone (no image), answered as JSON: scope (global / local), target (the whole object or
          person, with the words that say which one), part (where an added or changed thing goes, e.g. the head for
          sunglasses; "" for the whole target), kind (remove / add / change) and edit.
  face    for an addition or change, a separate question (FACE): is it on the face itself (glasses, a mask,
          make-up)? Such an edit is visible only while the face is turned to the camera, so the local editor pastes
          it only while a face is seen (`vibe_stream.local.FaceGate`). Asked inside PLAN as a "face" key, the model
          answered false even for sunglasses; asked apart it was right on 15 of 16 test instructions.
  ground  the target on the first frame (bbox_2d, Qwen3-VL's 0-1000 coordinates); with a part, the part inside a crop
          around the target box (not for removals); for a removal, the target's shadow on the whole frame, kept only
          if it is not the target itself, touches it and is not much larger (Qwen3-VL-2B mostly answers the target or
          the whole frame for shadows).
The crop around the target is edited with the instruction cut to a close-up (`close_up`: "the girl on the left" ->
"the girl"), since the crop no longer shows which one.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

PLAN = """The instruction below will be applied to every frame of a video.
Instruction: "{instr}"
Reply with JSON only, with these keys:
"scope": "global" if it changes the whole picture (art style, colours of everything, lighting, weather, season, time of day), "local" if it changes, adds to or removes particular objects or people.
"target": for a local edit, a short phrase naming the whole object or person the edit is about, keeping any words that say which one (for "put a hat on the man on the left": "the man on the left"); "" for a global edit.
"part": for a local edit that adds or changes something on one part of the target, that part as a short phrase (for the example: "the head"; for "make the girl's shoes red": "the shoes"); "" if the edit is about the whole target (removing it, recolouring all of it, turning it into something else) or for a global edit.
"kind": "remove" if the target is to be removed, "add" if something is added to it, "change" otherwise.
"edit": the instruction rewritten for a close-up picture that shows only that target, without the words that say which one (for the example: "put a hat on the man"); the original instruction for a global edit."""
GROUND = "Locate {target} in the image and output the bbox coordinates in JSON format."
FACE = ('Instruction: "{instr}"\nDoes this instruction put something over the eyes, nose or mouth of a person, or change '
        'the eyes, nose, mouth or skin of the face? Answer yes or no.')
QUALIFIERS = {"on", "in", "with", "at", "near", "wearing", "who", "that", "from", "behind", "next", "holding", "to"}

Box = List[float]            # x0, y0, x1, y1 in frame pixels


def close_up(instr: str, target: str) -> str:
    """The instruction for a crop of the target: the target phrase cut before its first qualifying word ("the girl on
    the left" -> "the girl"). Qwen3-VL-2B, asked for this rewrite in the plan, keeps the qualifiers."""
    words = target.split()
    cut = next((i for i, w in enumerate(words) if i > 0 and w.lower() in QUALIFIERS), len(words))
    return instr.replace(target, " ".join(words[:cut])) if target and target in instr else instr


def parse_json(s: str):
    s = re.sub(r"^```(json)?|```$", "", s.strip(), flags=re.M).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        m = re.search(r"(\[.*\]|\{.*\})", s, re.S)
        try:
            return json.loads(m.group(1)) if m else None
        except json.JSONDecodeError:
            return None


def boxes_from(ans, w: float, h: Optional[float] = None, x0: float = 0, y0: float = 0) -> List[Box]:
    """bbox_2d entries of a grounding answer, in pixels of a w x h image placed at (x0, y0) (Qwen3-VL answers 0-1000)."""
    h = w if h is None else h
    items = ans if isinstance(ans, list) else [ans] if isinstance(ans, dict) else []
    out = []
    for it in items:
        b = it.get("bbox_2d") if isinstance(it, dict) else None
        if b and len(b) == 4:
            out.append([x0 + float(np.clip(b[0] / 1000 * w, 0, w - 1)), y0 + float(np.clip(b[1] / 1000 * h, 0, h - 1)),
                        x0 + float(np.clip(b[2] / 1000 * w, 0, w - 1)), y0 + float(np.clip(b[3] / 1000 * h, 0, h - 1))])
    return out


def inter(a: Box, b: Box) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def area(a: Box) -> float:
    return (a[2] - a[0]) * (a[3] - a[1])


def grow(b: Box, f: float, size: int) -> Box:
    dx, dy = f * (b[2] - b[0]), f * (b[3] - b[1])
    return [max(0.0, b[0] - dx), max(0.0, b[1] - dy), min(size - 1.0, b[2] + dx), min(size - 1.0, b[3] + dy)]


@dataclass
class LocalPlan:
    """What `plan_edit` found. `objects`: dicts with role (target / part / shadow), of (the target's index) and box."""
    instr: str
    scope: str = "global"
    target: str = ""
    part: str = ""
    kind: str = "change"
    edit: str = ""
    face: bool = False
    objects: List[Dict] = field(default_factory=list)
    raw: Dict[str, object] = field(default_factory=dict)
    seconds: Dict[str, float] = field(default_factory=dict)

    @property
    def local(self) -> bool:
        return self.scope == "local" and bool(self.objects)

    def group(self, margin: float = 1.4, part_margin: float = 2.5, part_min_side: int = 128,
              min_side: int = 96) -> Tuple[List[int], float, int, str]:
        """The first target's tracked objects (its parts if it has any, else the target and its shadows), the crop
        window's margin and minimum side, and the crop's instruction."""
        k = next(i for i, o in enumerate(self.objects) if o["role"] == "target")
        text = close_up(self.instr, self.target)
        parts = [j for j, o in enumerate(self.objects) if o["role"] == "part" and o["of"] == k]
        if parts:
            return parts, part_margin, part_min_side, text
        shadows = [j for j, o in enumerate(self.objects) if o["role"] == "shadow" and o["of"] == k]
        return [k] + shadows, margin, min_side, text + (" and the shadow" if shadows else "")

    def to_json(self) -> Dict:
        return dict(instr=self.instr, scope=self.scope, target=self.target, part=self.part, kind=self.kind,
                    edit=self.edit, face=self.face, objects=self.objects, raw=self.raw, seconds=self.seconds)


def plan_edit(vibe, instr: str, frame: np.ndarray) -> LocalPlan:
    """Plan and ground `instr` on `frame` (S x S x 3 uint8 RGB, the first frame) with `vibe`'s Qwen3-VL."""
    size = frame.shape[0]
    t = time.perf_counter()
    raw = vibe.chat(PLAN.format(instr=instr))
    p = parse_json(raw) or {}
    plan = LocalPlan(instr=instr, scope=str(p.get("scope", "global")), target=str(p.get("target") or ""),
                     part=str(p.get("part") or ""), kind=str(p.get("kind") or "change"),
                     edit=str(p.get("edit") or instr), raw=dict(plan=raw))
    if plan.scope == "local" and plan.kind != "remove":
        ans = vibe.chat(FACE.format(instr=instr), max_new_tokens=4)
        plan.face, plan.raw["face"] = ans.strip().lower().startswith("yes"), ans
    plan.seconds["plan"] = round(time.perf_counter() - t, 2)
    if plan.scope != "local" or not plan.target:
        return plan
    t = time.perf_counter()
    plan.raw["ground"] = vibe.chat(GROUND.format(target=plan.target), frame)
    boxes = boxes_from(parse_json(plan.raw["ground"]), size)
    objects = [dict(role="target", of=k, box=b) for k, b in enumerate(boxes)]
    part = plan.part
    if plan.kind == "remove" or part.strip().lower() == plan.target.strip().lower():
        part = ""                    # Qwen3-VL-2B names the target itself as the part of a removal
    for k, b in enumerate(boxes if part else []):
        x0, y0, x1, y1 = (int(round(v)) for v in grow(b, 0.1, size))
        ans = vibe.chat(GROUND.format(target=part), frame[y0:y1 + 1, x0:x1 + 1])
        plan.raw.setdefault("part", []).append(ans)
        cand = boxes_from(parse_json(ans), x1 + 1 - x0, y1 + 1 - y0, x0, y0)
        if cand:
            objects.append(dict(role="part", of=k, box=max(cand, key=lambda c: (inter(c, b), area(c)))))
    if plan.kind == "remove" and boxes:
        plan.raw["shadow"] = vibe.chat(GROUND.format(target=f"the shadow of {plan.target}"), frame)
        for c in boxes_from(parse_json(plan.raw["shadow"]), size):
            k = max(range(len(boxes)), key=lambda j: inter(c, grow(boxes[j], 0.3, size)))
            own = inter(c, boxes[k]) / max(area(c) + area(boxes[k]) - inter(c, boxes[k]), 1.0)
            small = area(c) <= 1.5 * area(boxes[k]) and area(c) < 0.4 * size ** 2
            if own < 0.5 and small and inter(c, grow(boxes[k], 0.3, size)) > 0:
                objects.append(dict(role="shadow", of=k, box=c))
    plan.objects = objects
    plan.seconds["ground"] = round(time.perf_counter() - t, 2)
    return plan
