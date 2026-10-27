"""Evaluate a style fine-tune (scripts/train_style_ft.py) on the held-out references against the base model.

Every manifest pair of a held-out reference (<data>/split.json; selected or not, since at run time there is no
selection) is sampled again with the fine-tune, 8 steps of VIBE's scheduler from the noise of the base candidate 0
(`<way>0.jpg`), so the base output and its scores are already in the data. Each output is scored as in
make_style_data.py / select_style_targets.py (CLIP ref_gain and input_sim, edge agreement, LPIPS to the photo) and
counted as passing when it would have been selected. Reported per way and per group of references (the
--holdout_dirs ones, the random others). A way the data has no candidates for (C, the written instruction with the
reference next to the photo) gets its base output sampled here, from the same noise with the fine-tune switched off.

Plain instructions are checked too: a few held-out photos x --text_prompts, base and fine-tune from the same noise,
LPIPS between the two and, for the preset styles, CLIP directional similarity.

Writes <out>/results.json and contact sheets (photo | reference | base | fine-tune).
Usage: python scripts/eval_style_ft.py --data data/style_ft_768 --lora runs/style_ft_b/lora.safetensors \
           --out out/style_ft/style_ft_b
"""

import argparse
import copy
import json
import random
import time
import zlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

import _paths  # noqa: F401
from vibe_stream import lora
from vibe_stream.metrics import LPIPS, CLIPScore, edge_agreement
from vibe_stream.presets import CLIP_CAPTIONS, PROMPTS
from vibe_stream.vibe_model import LATENT_CHANNELS, REFERENCE_INSTRUCTION, VibeModel

S_CHANGE = 256   # LPIPS resolution of select_style_targets.py


def centre_square(path, size: int) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def small(img):
    return cv2.resize(img, (S_CHANGE, S_CHANGE), interpolation=cv2.INTER_AREA)


def tile(img, text, s=224):
    t = cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA)[:, :, ::-1].copy()
    cv2.rectangle(t, (0, 0), (6 + 7 * len(text), 16), (0, 0, 0), -1)
    cv2.putText(t, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1, cv2.LINE_AA)
    return t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--ways", nargs="+", choices=["A", "B", "C"], default=["B", "A"])
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--max_pairs", type=int, default=None, help="at most this many held-out pairs (random, fixed seed)")
    p.add_argument("--text_prompts", nargs="*",
                   default=list(PROMPTS.values()) + ["make it winter, covered in snow", "transfer this to watercolor painting"])
    p.add_argument("--text_photos", type=int, default=8)
    # selection thresholds, as select_style_targets.py
    p.add_argument("--min_edge", type=float, default=0.25)
    p.add_argument("--min_input", type=float, default=0.62)
    p.add_argument("--max_input", type=float, default=0.95)
    p.add_argument("--min_gain", type=float, default=0.05)
    p.add_argument("--min_change", type=float, default=0.25)
    p.add_argument("--sheet", type=int, default=24)
    a = p.parse_args()
    d, out = Path(a.data), Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    split = json.loads((d / "split.json").read_text(encoding="utf-8"))
    vr = set(split["val_refs"])
    rows = [json.loads(line) for line in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    rows = [r for r in rows if f"{Path(r['ref']).parent.name}/{Path(r['ref']).stem}" in vr]
    if a.max_pairs and len(rows) > a.max_pairs:
        rows = random.Random(0).sample(rows, a.max_pairs)
    change = json.loads((d / "change.json").read_text(encoding="utf-8"))
    size = rows[0]["size"]
    print(f"[eval] {len(rows)} held-out pairs, ways {a.ways}, {a.steps} steps", flush=True)

    m = VibeModel(size=size)
    wrappers = lora.attach(m.tr, int(lora.load(a.lora)[1].get("rank", 32)))
    lora.load_into(wrappers, lora.load(a.lora)[0])
    clip, lp = CLIPScore(), LPIPS()
    lat = size // 32

    def sample(il, emb, noise):
        sch = copy.deepcopy(m.pipe.scheduler)
        sch.set_timesteps(a.steps, device=m.device)
        x = noise.clone()
        for t in sch.timesteps:
            x = sch.step(m.velocity(x, il, emb, t), t, x, return_dict=False)[0]
        return m.decode_u8(x)[0]

    def passes(s):
        return (s["edge"] >= a.min_edge and a.min_input <= s["input_sim"] <= a.max_input and s["ref_gain"] >= a.min_gain
                and s["change"] >= a.min_change)

    res, t0 = [], time.time()
    for i, r in enumerate(rows):
        frame = centre_square(r["content"], size)
        ref = Image.open(r["ref"]).convert("RGB")
        noise = torch.randn((1, LATENT_CHANNELS, lat, lat), dtype=torch.float32,
                            generator=torch.Generator("cpu").manual_seed(r["seed0"])).to(m.device)
        with torch.inference_mode():
            il = m.encode([frame])
            fin, fref = clip._images([frame])[0], clip._images([np.asarray(ref)])[0]
            base_sim = float(fin @ fref)
            def score(img):
                fo = clip._images([img])[0]
                return dict(ref_gain=float(fo @ fref) - base_sim, input_sim=float(fo @ fin),
                            edge=edge_agreement(frame, img), change=lp([small(frame)], [small(img)]))

            for way in a.ways:
                instruction = REFERENCE_INSTRUCTION if way == "B" else r["caption"]
                emb = m.embed(instruction, frame) if way == "A" else m.embed_with_reference(instruction, frame, ref)
                img = sample(il, emb, noise)
                ft = score(img)
                Image.fromarray(img).save(out / "images" / f"{r['pair']}_{way}.jpg", quality=92)
                if f"{way}0" in r["scores"]:
                    base = dict(r["scores"][f"{way}0"], change=change[r["pair"]][f"{way}0"])
                    base_img = d / "pairs" / r["pair"] / f"{way}0.jpg"
                else:
                    with lora.use(wrappers, None):
                        bimg = sample(il, emb, noise)
                    base = {k: round(v, 4) for k, v in score(bimg).items()}
                    base_img = out / "images" / f"{r['pair']}_{way}_base.jpg"
                    Image.fromarray(bimg).save(base_img, quality=92)
                res.append(dict(pair=r["pair"], ref=r["ref"], group=Path(r["ref"]).parent.name, way=way,
                                base=base, ft={k: round(v, 4) for k, v in ft.items()}, base_img=str(base_img),
                                base_pass=passes(base), ft_pass=passes(ft)))
        if (i + 1) % 25 == 0:
            print(f"[eval] {i + 1}/{len(rows)} ({(time.time() - t0) / (i + 1):.1f} s per pair)", flush=True)

    summary = {}
    for way in a.ways:
        for group in ["all"] + sorted({x["group"] for x in res}):
            xs = [x for x in res if x["way"] == way and (group == "all" or x["group"] == group)]
            if not xs:
                continue
            row = {"n": len(xs), "base_pass": np.mean([x["base_pass"] for x in xs]),
                   "ft_pass": np.mean([x["ft_pass"] for x in xs])}
            for k in ("ref_gain", "input_sim", "edge", "change"):
                row[f"base_{k}"] = np.median([x["base"][k] for x in xs])
                row[f"ft_{k}"] = np.median([x["ft"][k] for x in xs])
            summary[f"{way}/{group}"] = {k: round(float(v), 4) for k, v in row.items()}
            print(f"[eval] {way} {group:<14} n {len(xs):3d}  pass {row['base_pass']:.2f} -> {row['ft_pass']:.2f}  "
                  + "  ".join(f"{k} {row[f'base_{k}']:.3f} -> {row[f'ft_{k}']:.3f}" for k in ("ref_gain", "input_sim", "edge", "change"))
                  + "  (medians)", flush=True)

    # plain instructions: base vs fine-tune from the same noise
    photos = sorted({r["content"] for r in rows})
    photos = random.Random(1).sample(photos, min(a.text_photos, len(photos)))
    frames = [centre_square(c, size) for c in photos]
    text = {}
    noise = m.noise.expand(1, -1, -1, -1)
    for prompt in a.text_prompts:
        outs = {None: [], "student": []}
        with torch.inference_mode():
            for f in frames:
                il, emb = m.encode([f]), m.embed(prompt, f)
                for which in outs:
                    with lora.use(wrappers, which):
                        outs[which].append(sample(il, emb, noise))
        t = dict(lpips_base_ft=lp(outs[None], outs["student"]))
        key = next((k for k, v in PROMPTS.items() if v == prompt), None)
        if key in CLIP_CAPTIONS:
            t["clip_dir_base"] = clip(frames, outs[None], *CLIP_CAPTIONS[key])["clip_dir"]
            t["clip_dir_ft"] = clip(frames, outs["student"], *CLIP_CAPTIONS[key])["clip_dir"]
        text[prompt] = {k: round(v, 4) for k, v in t.items()}
        print(f"[eval] text '{prompt[:40]}': {text[prompt]}", flush=True)
        strip = [np.concatenate([tile(f, "photo"), tile(b, "base"), tile(o, "fine-tune")], 0)
                 for f, b, o in zip(frames, outs[None], outs["student"])]
        cv2.imwrite(str(out / f"text_{len(text)}.jpg"), np.concatenate(strip, 1), [cv2.IMWRITE_JPEG_QUALITY, 85])

    (out / "results.json").write_text(json.dumps(dict(lora=a.lora, summary=summary, text=text, pairs=res), indent=1),
                                      encoding="utf-8")
    rng = random.Random(0)
    for way in a.ways:
        for group in sorted({x["group"] for x in res}):
            xs = [x for x in res if x["way"] == way and x["group"] == group]
            pick = rng.sample(xs, min(a.sheet, len(xs)))
            lines = []
            for x in pick:
                photo = centre_square(next(r["content"] for r in rows if r["pair"] == x["pair"]), size)
                lines.append(np.concatenate([
                    tile(photo, "photo"), tile(np.asarray(Image.open(x["ref"]).convert("RGB")), "ref " + Path(x["ref"]).stem),
                    tile(np.asarray(Image.open(x["base_img"])), f"base g{x['base']['ref_gain']:.2f} "
                         f"c{x['base']['change']:.2f}{' ok' if x['base_pass'] else ''}"),
                    tile(np.asarray(Image.open(out / "images" / f"{x['pair']}_{way}.jpg")), f"ft g{x['ft']['ref_gain']:.2f} "
                         f"c{x['ft']['change']:.2f}{' ok' if x['ft_pass'] else ''}")], 1))
            half = (len(lines) + 1) // 2
            cols = [np.concatenate(lines[:half], 0),
                    np.concatenate(lines[half:] + [np.zeros_like(lines[0])] * (2 * half - len(lines)), 0)]
            cv2.imwrite(str(out / f"sheet_{way}_{group}.jpg"), np.concatenate(cols, 1), [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"[eval] wrote {out / 'results.json'} and sheets", flush=True)


if __name__ == "__main__":
    main()
