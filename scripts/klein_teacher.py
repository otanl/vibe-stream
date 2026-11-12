"""FLUX.2-klein-4B teacher outputs for training windows (sav_windows.py): each frame edited on its own, 4 steps,
seed 42.

Window i gets --per_window instruction keys, consecutive in a fixed seeded shuffle of the keys taken cyclically, so
every key is used about equally and a window never repeats one. Klein is given presets.KLEIN_PROMPTS[key] where there
is one, else presets.ALL_PROMPTS[key] (--klein_prompts: a JSON file of further overrides); the student is later trained
on ALL_PROMPTS[key]. Prompt embeddings are computed once per key (Klein's text encoder does not see the image). Keys in
--skip are assigned but not run here; pack_klein.py --vibe_keys fills them with VIBE's own teacher. Writes
<out>/w###_<key>.npz (frames uint8) and <out>/assign.json ({window: [keys]}, the prompts, the arguments); existing
outputs are kept, so a run resumes, and a key can be redone by deleting its files.

Needs a diffusers with Flux2KleinPipeline (run here with 0.39.0.dev0 under torch 2.11 / Python 3.14) and the weights
of black-forest-labs/FLUX.2-klein-4B (Apache-2.0). About 31 s per window x key (10 frames at 768) on an RTX 3090 Ti.

Usage: python scripts/klein_teacher.py --windows data/sav_windows --out data/sav_klein
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import _paths  # noqa: F401
from vibe_stream.presets import ALL_PROMPTS, HELD_OUT, KLEIN_PROMPTS

SIZE, SEED = 768, 42
TRAIN_KEYS = [k for k in ALL_PROMPTS if k not in HELD_OUT]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--klein_prompts", default=None, help="JSON {key: text} overriding presets.KLEIN_PROMPTS")
    ap.add_argument("--keys", nargs="+", default=TRAIN_KEYS)
    ap.add_argument("--per_window", type=int, default=5)
    ap.add_argument("--skip", nargs="*", default=[], help="keys left to VIBE's teacher (pack_klein.py --vibe_keys)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    extra = json.loads(Path(a.klein_prompts).read_text(encoding="utf-8")) if a.klein_prompts else {}
    over = {**KLEIN_PROMPTS, **extra}
    texts = {k: over.get(k, ALL_PROMPTS[k]) for k in a.keys if k not in a.skip}
    order = list(a.keys)
    random.Random(a.seed).shuffle(order)
    wins = sorted(Path(a.windows).glob("w*.npz"))
    assign = {w.stem: [order[(i * a.per_window + j) % len(order)] for j in range(a.per_window)]
              for i, w in enumerate(wins)}
    (out / "assign.json").write_text(json.dumps(dict(assign=assign, klein_prompts=texts, args=vars(a)), indent=1))
    todo = [(w, k) for w in wins for k in assign[w.stem] if k in texts and not (out / f"{w.stem}_{k}.npz").exists()]
    print(f"[klein] {len(wins)} windows x {a.per_window} keys, {len(todo)} to do", flush=True)
    if not todo:
        return
    import diffusers
    pipe = diffusers.Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B",
                                                        torch_dtype=torch.bfloat16).to("cuda")
    with torch.inference_mode():
        pe = {k: pipe.encode_prompt(t, device="cuda")[0] for k, t in texts.items()}
    t0 = time.time()
    for n, (w, k) in enumerate(todo):
        res = []
        for f in np.load(w)["frames"]:
            o = pipe(image=[Image.fromarray(f)], prompt_embeds=pe[k], height=SIZE, width=SIZE, num_inference_steps=4,
                     guidance_scale=1.0, generator=torch.Generator("cpu").manual_seed(SEED)).images[0]
            res.append(np.asarray(o.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)))
        np.savez_compressed(out / f"{w.stem}_{k}.npz", frames=np.stack(res))
        if n % 20 == 0 or n == len(todo) - 1:
            el = time.time() - t0
            print(f"[klein] {n + 1}/{len(todo)} {w.stem} {k}  {el / (n + 1):.1f} s per window-key, "
                  f"eta {el / (n + 1) * (len(todo) - n - 1) / 3600:.1f} h", flush=True)


if __name__ == "__main__":
    main()
