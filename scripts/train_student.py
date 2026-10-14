"""Train a k-step student of VIBE (rank-32 LoRA) on teacher data, with a temporal-consistency term.

loss = Huber(student latent, teacher latent)
       + w_lp * LPIPS(decoded crop, teacher crop)
       + lam  * masked L1(decoded crop t, decoded crop t+1 warped back by the input flow t -> t+1)

Each step takes `--pairs` pairs of consecutive frames (random window, prompt and t), runs the student on both frames
of each pair, decodes the same random crop with DC-AE (frame by frame, recomputed in backward), and applies the
three terms; the temporal term and its mask are in `vibe_stream.temporal_loss`. The student starts from the shared
init noise and runs k first-order steps on VIBE's own k-step schedule.

Recipe of the released 2-step student (runs/lam4_2step): --student_steps 2 --lam 4 --grad_ckpt, 1500 steps,
about 8 GB and 2 s/step (56 min) on an RTX 3090 Ti. One step: drop --grad_ckpt (16.5 GB, 1.0 s/step).

Usage: python scripts/train_student.py --data data/teacher_768 --run runs/lam4_2step --student_steps 2 --lam 4 --grad_ckpt
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import _paths  # noqa: F401
from vibe_stream import lora
from vibe_stream.temporal_loss import masked_warp_l1, occlusion_mask
from vibe_stream.vibe_model import VibeModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--student_steps", type=int, default=2, choices=[1, 2])
    p.add_argument("--lam", type=float, default=4.0, help="weight of the temporal term (0: plain distillation)")
    p.add_argument("--w_lp", type=float, default=2.0)
    p.add_argument("--rank", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--pairs", type=int, default=2)
    p.add_argument("--crop", type=int, default=384, help="decoded crop in pixels (multiple of 32)")
    p.add_argument("--margin", type=int, default=32, help="crop border left out of LPIPS and the temporal term")
    p.add_argument("--grad_ckpt", action="store_true", help="transformer gradient checkpointing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--no_save", action="store_true")
    a = p.parse_args()
    import lpips
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    run = Path(a.run)
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(a), indent=1), encoding="utf-8")
    meta = json.loads((Path(a.data) / "meta.json").read_text(encoding="utf-8"))
    size, prompts, nfr = meta["size"], meta["prompts"], meta["nfr"]
    data = [dict(np.load(f)) for f in sorted(Path(a.data).glob("w*.npz"))]
    keys = list(prompts)
    print(f"[train] {len(data)} windows x {len(keys)} prompts at {size}, {a.student_steps}-step student, lam {a.lam}", flush=True)

    m = VibeModel(size=size, keep_text_encoder=False)
    m.vae.requires_grad_(False)
    wrappers = lora.attach(m.tr, a.rank)
    params = [q for w in wrappers.values() for q in (w.A, w.B)]
    print(f"[train] {len(wrappers)} LoRA layers, {sum(q.numel() for q in params) / 1e6:.1f} M params", flush=True)
    if a.grad_ckpt:
        m.tr.enable_gradient_checkpointing()
    lp = lpips.LPIPS(net="alex", verbose=False).to(m.device).eval().requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * (0.5 * (1 + math.cos(math.pi * s / a.steps))))
    lat_n, cl, S, mg = size // 32, a.crop // 32, a.crop, a.margin
    dev = m.device
    log = open(run / "log.jsonl", "w", encoding="utf-8")
    acc = dict.fromkeys(("huber", "lpips", "temp", "temp_teacher"), 0.0)
    t0 = time.perf_counter()
    for step in range(a.steps):
        items = [(random.choice(data), random.choice(keys), random.randrange(nfr - 1)) for _ in range(a.pairs)]
        pick = [(d, k, t + j) for d, k, t in items for j in (0, 1)]
        img_lat = torch.from_numpy(np.stack([d["img_lat"][t] for d, _, t in pick])).to(dev)
        emb = torch.from_numpy(np.concatenate([d[f"emb_{k}"] for d, k, _ in pick])).to(dev, torch.bfloat16)
        tlat = torch.from_numpy(np.stack([d[f"tlat_{k}"][t] for d, k, t in pick])).to(dev)
        cy, cx = random.randrange(lat_n - cl + 1), random.randrange(lat_n - cl + 1)
        Y, X = cy * 32, cx * 32
        timg = torch.from_numpy(np.stack([d[f"timg_{k}"][t, Y:Y + S, X:X + S] for d, k, t in pick]))
        timg = timg.to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
        fin = torch.from_numpy(np.stack([d["frames"][t, Y:Y + S, X:X + S] for d, _, t in pick]))
        fin = fin.to(dev).permute(0, 3, 1, 2).float() / 255.0
        flow = torch.from_numpy(np.stack([d["flows"][t, Y:Y + S, X:X + S] for d, _, t in items])).to(dev).float()

        with torch.autocast("cuda", dtype=torch.bfloat16):
            x0 = m.student(img_lat, emb, a.student_steps)
            l_lat = F.smooth_l1_loss(x0, tlat, beta=0.1)
            z = (x0[:, :, cy:cy + cl, cx:cx + cl] / m.sf).to(torch.bfloat16)
            # one frame at a time, recomputed in backward: the decoder's full-resolution activations dominate memory
            dec = torch.cat([checkpoint(lambda zz: m.vae.decode(zz).sample, z[i:i + 1], use_reentrant=False)
                             for i in range(len(z))]).float()
        inner = (slice(None), slice(None), slice(mg, S - mg), slice(mg, S - mg))
        l_lp = lp(dec[inner].clamp(-1, 1), timg[inner]).mean()
        out01 = (dec.clamp(-1, 1) + 1) / 2
        mask = occlusion_mask(fin[0::2], fin[1::2], flow)
        l_t = masked_warp_l1(out01[0::2], out01[1::2], flow, mask, mg)
        with torch.no_grad():
            t01 = (timg + 1) / 2
            l_tt = masked_warp_l1(t01[0::2], t01[1::2], flow, mask, mg)
        loss = l_lat + a.w_lp * l_lp + a.lam * l_t
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        for key, val in (("huber", l_lat), ("lpips", l_lp), ("temp", l_t), ("temp_teacher", l_tt)):
            acc[key] += val.item()
        if (step + 1) % a.log_every == 0:
            r = {k: v / a.log_every for k, v in acc.items()}
            el = time.perf_counter() - t0
            print(f"[train] {step + 1:5d}/{a.steps} huber {r['huber']:.4f} lpips {r['lpips']:.4f} "
                  f"temp {r['temp'] * 255:.2f} (teacher {r['temp_teacher'] * 255:.2f}, /255)  {el / (step + 1):.2f} s/step  "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GB", flush=True)
            log.write(json.dumps({"step": step + 1, **r}) + "\n")
            log.flush()
            acc = dict.fromkeys(acc, 0.0)
    if a.no_save:
        return
    state = lora.state_from_wrappers(wrappers)
    state[lora.NOISE_KEY] = m.noise.cpu().contiguous()
    lora.save(run / "lora.safetensors", state, dict(base="VIBE-Image-Edit-DistilledCFG", size=size, rank=a.rank, lam=a.lam,
                                                     student_steps=a.student_steps, w_lp=a.w_lp, steps=a.steps,
                                                     prompts=prompts, teacher_steps=meta["teacher_steps"]))
    print(f"[train] saved {run / 'lora.safetensors'} after {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
