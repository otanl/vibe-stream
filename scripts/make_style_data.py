"""Candidate data for fine-tuning VIBE on style from a reference image: content photo x style reference -> edits.

For every content photo (centre square at --size) and each of --refs_per_content references drawn for it (fixed
seed), VIBE runs --steps steps of its own scheduler from --seeds noise draws per way of conditioning:
  A  the reference's style as an instruction written by VIBE's Qwen3-VL (`VibeModel.style_prompt`, cached per
     reference in <out>/captions.json), with the photo as usual;
  B  "transfer this to the artistic style of the second image" with the photo and the reference both given to the
     encoder (`VibeModel.embed_with_reference`; carries the reference's content as well as its style);
  C  the instruction of A with the photo and the reference both given to the encoder.
--ways picks among them (A and B by default). --base_lora merges a fine-tune of the transformer
(scripts/train_style_ft.py) first, for a further round of data from the fine-tuned model; --skip_split leaves out the
pairs a split.json holds out, which that round does not need.
Each candidate is saved as <out>/pairs/<content>__<ref>/<way><seed>.jpg and scored: CLIP ViT-L/14 image similarity
to the reference minus the photo's (`ref_gain`; style, and content if copied), CLIP similarity to the photo
(`input_sim`; content kept) and edge agreement with the photo (`edge`). One JSON line per pair in
<out>/manifest.jsonl. Pairs already in the manifest are skipped, so the run can be stopped (touch --stop_file) and
resumed. Selecting targets from the candidates is left to the training side.
Usage: python scripts/make_style_data.py --content data/content_openimages --refs data/style_refs_aic \
           data/style_refs --out data/style_ft_768
       python scripts/make_style_data.py ... --out data/style_ft2_768 --ways B C \
           --base_lora runs/style_ft_b/lora.safetensors --skip_split data/style_ft_768/split.json
           (copy captions.json over first to reuse the instructions)
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
from vibe_stream.metrics import CLIPScore, edge_agreement
from vibe_stream.vibe_model import LATENT_CHANNELS, REFERENCE_INSTRUCTION, VibeModel


def centre_square(path: Path, size: int) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--content", required=True, help="directory of content photos (*.jpg)")
    p.add_argument("--refs", nargs="+", required=True, help="directories of style references (*.jpg)")
    p.add_argument("--out", required=True)
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--refs_per_content", type=int, default=3)
    p.add_argument("--limit", type=int, default=None, help="stop after this many pairs in the manifest")
    p.add_argument("--stop_file", default=None)
    p.add_argument("--ways", nargs="+", choices=["A", "B", "C"], default=["A", "B"])
    p.add_argument("--base_lora", default=None, help="fine-tune of the transformer merged before sampling")
    p.add_argument("--skip_split", default=None, help="split.json whose held-out references and photos are skipped")
    a = p.parse_args()
    out = Path(a.out)
    (out / "pairs").mkdir(parents=True, exist_ok=True)
    contents = sorted(Path(a.content).glob("*.jpg"))
    refs = sorted(f for d in a.refs for f in Path(d).glob("*.jpg"))
    print(f"[data] {len(contents)} content photos, {len(refs)} references", flush=True)

    m = VibeModel(size=a.size)
    if a.base_lora:
        lora.merge(m.tr, lora.load(a.base_lora)[0])
    cap_path = out / "captions.json"
    captions = json.loads(cap_path.read_text(encoding="utf-8")) if cap_path.exists() else {}
    for r in refs:
        key = f"{r.parent.name}/{r.stem}"
        if key not in captions:
            captions[key] = m.style_prompt(Image.open(r).convert("RGB"))
            cap_path.write_text(json.dumps(captions, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[data] {len(captions)} captions", flush=True)

    man_path = out / "manifest.jsonl"
    done = {json.loads(line)["pair"] for line in man_path.read_text(encoding="utf-8").splitlines()} if man_path.exists() else set()
    rng = random.Random(0)
    plan = [(c, r) for c in contents for r in rng.sample(refs, a.refs_per_content)]
    if a.skip_split:
        split = json.loads(Path(a.skip_split).read_text(encoding="utf-8"))
        vr, vc = set(split["val_refs"]), set(split["val_contents"])
        plan = [(c, r) for c, r in plan if f"{r.parent.name}/{r.stem}" not in vr and c.stem not in vc]
        print(f"[data] {len(plan)} pairs outside the held-out split", flush=True)
    clip = CLIPScore()
    lat = a.size // 32
    t0, n_new = time.time(), 0
    with man_path.open("a", encoding="utf-8") as man:
        for c, r in plan:
            if a.limit is not None and len(done) >= a.limit:
                break
            if a.stop_file and Path(a.stop_file).exists():
                print("[data] stop file found", flush=True)
                break
            pair = f"{c.stem}__{r.parent.name}-{r.stem}"
            if pair in done:
                continue
            frame = centre_square(c, a.size)
            ref = Image.open(r).convert("RGB")
            caption = captions[f"{r.parent.name}/{r.stem}"]
            seed0 = zlib.crc32(pair.encode())
            noise = torch.stack([torch.randn((LATENT_CHANNELS, lat, lat), dtype=torch.float32,
                                             generator=torch.Generator("cpu").manual_seed(seed0 + s))
                                 for s in range(a.seeds)]).to(m.device)
            d = out / "pairs" / pair
            d.mkdir(exist_ok=True)
            scores = {}
            with torch.inference_mode():
                il = m.encode([frame]).expand(a.seeds, -1, -1, -1)
                make = {"A": lambda: m.embed(caption, frame),
                        "B": lambda: m.embed_with_reference(REFERENCE_INSTRUCTION, frame, ref),
                        "C": lambda: m.embed_with_reference(caption, frame, ref)}
                embs = {w: make[w]() for w in a.ways}
                fin, fref = clip._images([frame])[0], clip._images([np.asarray(ref)])[0]
                base = float(fin @ fref)
                for way, emb in embs.items():
                    sch = copy.deepcopy(m.pipe.scheduler)
                    sch.set_timesteps(a.steps, device=m.device)
                    x = noise.clone()
                    for t in sch.timesteps:
                        x = sch.step(m.velocity(x, il, emb, t), t, x, return_dict=False)[0]
                    imgs = m.decode_u8(x)
                    fo = clip._images(list(imgs))
                    for s, img in enumerate(imgs):
                        Image.fromarray(img).save(d / f"{way}{s}.jpg", quality=95)
                        scores[f"{way}{s}"] = dict(ref_gain=round(float(fo[s] @ fref) - base, 4),
                                                   input_sim=round(float(fo[s] @ fin), 4),
                                                   edge=round(edge_agreement(frame, img), 4))
            man.write(json.dumps(dict(pair=pair, content=str(c), ref=str(r), caption=caption, size=a.size,
                                      steps=a.steps, seed0=seed0, base_lora=a.base_lora, scores=scores),
                                 ensure_ascii=False) + "\n")
            man.flush()
            done.add(pair)
            n_new += 1
            if n_new % 50 == 0:
                print(f"[data] {len(done)} pairs ({(time.time() - t0) / n_new:.2f} s per pair)", flush=True)
    print(f"[data] done: {len(done)} pairs in the manifest", flush=True)


if __name__ == "__main__":
    main()
