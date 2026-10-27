"""Fine-tune VIBE (rank-32 LoRA on the transformer) for style from a reference image, on selected targets.

Data: scripts/make_style_data.py candidates, one target per pair chosen by scripts/select_style_targets.py
(<data>/selected.jsonl). The model is conditioned as at run time with the reference handed to the encoder next to the
photo (`VibeModel.embed_with_reference`, way B), with the instruction written from the reference (`embed`, way A), or
with both (that instruction and the reference next to the photo, way C), and trained with VIBE's own flow-matching
objective: x_t = (1 - s) x0 + s eps, velocity target eps - x0, sigma from a logit-normal shifted by --flow_shift (VIBE
samples with flow shift 3). The targets come from the model itself (the base, and in a further round the fine-tuned
model), so this is fine-tuning on selected samples: it makes the outcomes the selection keeps (style taken, photo
kept, no copied content) the usual ones rather than teaching anything the base cannot produce.

Held out for evaluation (<data>/split.json, written on the first run and reused): every reference in --holdout_dirs
(the nine famous works) and --val_refs random others, with every photo paired with any of them. Their selected pairs
give a validation flow loss at fixed sigmas and noise; scripts/eval_style_ft.py samples them.

Embeddings and DC-AE latents (photo, target) are computed once per selected pair and cached in <data>/ft_cache/; the
text encoder is offloaded before training. The LoRA file has no init noise: it changes the base model, and a
streaming student learns from it through teacher data made with the deltas merged (make_teacher_data.py --base_lora).

Usage: python scripts/train_style_ft.py --data data/style_ft_768 --run runs/style_ft_b
       python scripts/train_style_ft.py --data data/style_ft2_768 --run runs/style_ft2_bc --cond BC
           (second round: data/style_ft2_768 holds split.json copied from the first and selected.jsonl from
           select_style_targets.py --data data/style_ft2_768 --extra data/style_ft_768)
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file, save_file

import _paths  # noqa: F401
from vibe_stream import lora
from vibe_stream.vibe_model import REFERENCE_INSTRUCTION, VibeModel

VAL_SIGMAS = (0.25, 0.5, 0.75, 0.95)


def ref_key(path) -> str:
    return f"{Path(path).parent.name}/{Path(path).stem}"


def centre_square(path, size: int) -> np.ndarray:
    """The photo as make_style_data.py gave it to VIBE."""
    img = np.asarray(Image.open(path).convert("RGB"))
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def make_split(d: Path, holdout_dirs, n_val: int, seed: int = 0) -> dict:
    """Held-out references and photos; written once to <data>/split.json so training and evaluation agree."""
    path = d / "split.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in (d / "manifest.jsonl").read_text(encoding="utf-8").splitlines()]
    refs = sorted({ref_key(r["ref"]) for r in rows})
    fixed = [k for k in refs if k.split("/")[0] in holdout_dirs]
    val_refs = sorted(fixed + random.Random(seed).sample([k for k in refs if k not in fixed], n_val))
    val_contents = sorted({Path(r["content"]).stem for r in rows if ref_key(r["ref"]) in val_refs})
    split = dict(val_refs=val_refs, val_contents=val_contents, holdout_dirs=list(holdout_dirs), n_val=n_val, seed=seed)
    path.write_text(json.dumps(split, indent=1), encoding="utf-8")
    return split


def build_cache(m: VibeModel, d: Path, sel: list, ways: str) -> None:
    """<data>/ft_cache/<pair>.safetensors: photo and target latents, embeddings of each way (bf16). A file missing
    the embedding of a way in `ways` is made again."""
    cache = d / "ft_cache"
    cache.mkdir(exist_ok=True)

    def complete(path):
        if not path.exists():
            return False
        with safe_open(str(path), "pt") as f:
            return all(f"emb_{w}" in f.keys() for w in ways)

    todo = [s for s in sel if not complete(cache / f"{s['pair']}.safetensors")]
    t0 = time.time()
    for i, s in enumerate(todo):
        photo = centre_square(s["content"], m.size)
        target = np.asarray(Image.open(s["target"]).convert("RGB"))
        ref = Image.open(s["ref"]).convert("RGB")
        lat = m.encode([photo, target])
        out = dict(photo=lat[:1].cpu().contiguous(), target=lat[1:].cpu().contiguous(),
                   emb_A=m.embed(s["caption"], photo).cpu().contiguous(),
                   emb_B=m.embed_with_reference(REFERENCE_INSTRUCTION, photo, ref).cpu().contiguous(),
                   emb_C=m.embed_with_reference(s["caption"], photo, ref).cpu().contiguous())
        save_file(out, str(cache / f"{s['pair']}.safetensors"))
        if (i + 1) % 200 == 0:
            print(f"[cache] {i + 1}/{len(todo)} ({(time.time() - t0) / (i + 1):.2f} s per pair)", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--cond", choices=["B", "A", "C", "AB", "BC", "ABC"], default="B",
                   help="conditioning: reference image (B), written instruction (A), both (C), or one of several at "
                        "random per sample")
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--flow_shift", type=float, default=3.0)
    p.add_argument("--holdout_dirs", nargs="*", default=["style_refs"])
    p.add_argument("--val_refs", type=int, default=16, help="random references held out besides --holdout_dirs")
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--stop_file", default=None, help="stop (and save) when this file exists")
    a = p.parse_args()
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    d, run = Path(a.data), Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(a), indent=1), encoding="utf-8")
    sel = [json.loads(line) for line in (d / "selected.jsonl").read_text(encoding="utf-8").splitlines()]
    split = make_split(d, a.holdout_dirs, a.val_refs)
    vr, vc = set(split["val_refs"]), set(split["val_contents"])
    val = [s for s in sel if ref_key(s["ref"]) in vr]
    train = [s for s in sel if ref_key(s["ref"]) not in vr and Path(s["content"]).stem not in vc]
    print(f"[ft] {len(train)} training pairs, {len(val)} held-out pairs ({len(vr)} references held out); "
          f"conditioning {a.cond}", flush=True)

    size = json.loads((d / "manifest.jsonl").open(encoding="utf-8").readline())["size"]
    m = VibeModel(size=size)
    build_cache(m, d, train + val, a.cond)
    m.offload_text_encoder()
    m.vae.to("cpu")
    torch.cuda.empty_cache()
    dev = m.device

    def batch(items, ways, gen=None):
        rows = [load_file(str(d / "ft_cache" / f"{s['pair']}.safetensors")) for s in items]
        embs = [r[f"emb_{w}"] for r, w in zip(rows, ways)]
        if len({e.shape[1] for e in embs}) > 1:
            raise SystemExit(f"embedding lengths differ: {[e.shape for e in embs]}")
        return (torch.cat([r["target"] for r in rows]).to(dev), torch.cat([r["photo"] for r in rows]).to(dev),
                torch.cat(embs).to(dev))

    def ways(n):
        return [random.choice(a.cond) for _ in range(n)]

    def fm_loss(x0, il, emb, s, eps):
        xt = (1 - s) * x0 + s * eps
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = m.velocity(xt, il, emb, s.flatten() * 1000)
        return F.mse_loss(v, eps - x0)

    # fixed noise and sigmas for the validation loss, one draw per held-out pair
    gval = torch.Generator("cpu").manual_seed(1)
    val_eps = [torch.randn((1, 32, size // 32, size // 32), generator=gval) for _ in val]

    @torch.no_grad()
    def validate(active):
        out = {}
        for w in sorted(set(a.cond)):
            tot = {s: 0.0 for s in VAL_SIGMAS}
            with lora.use(wrappers, active):
                for item, eps in zip(val, val_eps):
                    x0, il, emb = batch([item], [w])
                    for s in VAL_SIGMAS:
                        tot[s] += fm_loss(x0, il, emb, torch.full((1, 1, 1, 1), s, device=dev), eps.to(dev)).item()
            out[w] = {f"{s}": round(v / len(val), 5) for s, v in tot.items()}
        return out

    wrappers = lora.attach(m.tr, a.rank)
    params = [q for w in wrappers.values() for q in (w.A, w.B)]
    print(f"[ft] {len(wrappers)} LoRA layers, {sum(q.numel() for q in params) / 1e6:.1f} M params", flush=True)
    if a.grad_ckpt:
        m.tr.enable_gradient_checkpointing()
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * (0.5 * (1 + math.cos(math.pi * s / a.steps))))
    log = open(run / "log.jsonl", "w", encoding="utf-8")
    base_val = validate(None)
    print(f"[ft] validation flow loss, base: {base_val}", flush=True)
    log.write(json.dumps({"step": 0, "val": base_val}) + "\n")
    t0, acc, step = time.perf_counter(), 0.0, 0
    for step in range(a.steps):
        if a.stop_file and Path(a.stop_file).exists():
            print("[ft] stop file found", flush=True)
            break
        items = random.sample(train, a.batch)
        x0, il, emb = batch(items, ways(a.batch))
        u = torch.sigmoid(torch.randn(a.batch, 1, 1, 1, device=dev))
        s = a.flow_shift * u / (1 + (a.flow_shift - 1) * u)
        loss = fm_loss(x0, il, emb, s, torch.randn_like(x0))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        acc += loss.item()
        if (step + 1) % a.log_every == 0:
            el = time.perf_counter() - t0
            print(f"[ft] {step + 1:5d}/{a.steps} loss {acc / a.log_every:.4f}  {el / (step + 1):.2f} s/step  "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            log.write(json.dumps({"step": step + 1, "loss": acc / a.log_every}) + "\n")
            log.flush()
            acc = 0.0
        if (step + 1) % a.val_every == 0 or step + 1 == a.steps:
            v = validate("student")
            print(f"[ft] validation flow loss at {step + 1}: {v}", flush=True)
            log.write(json.dumps({"step": step + 1, "val": v}) + "\n")
            log.flush()
    state = lora.state_from_wrappers(wrappers)
    lora.save(run / "lora.safetensors", state, dict(base="VIBE-Image-Edit-DistilledCFG", kind="style_ft", size=size,
                                                     rank=a.rank, steps=step + 1, cond=a.cond, lr=a.lr,
                                                     batch=a.batch, flow_shift=a.flow_shift, data=str(d),
                                                     reference_instruction=REFERENCE_INSTRUCTION,
                                                     n_train=len(train)))
    print(f"[ft] saved {run / 'lora.safetensors'} after {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
