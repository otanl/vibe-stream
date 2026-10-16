"""Train a k-step student of VIBE (rank-32 LoRA) on teacher data, with a temporal-consistency term.

loss = Huber(student latent, teacher latent)
       + w_lp * LPIPS(decoded crop, teacher crop)
       + lam  * masked L1(decoded crop t, decoded crop t+1 warped back by the input flow t -> t+1)

Each step takes `--pairs` pairs of consecutive frames (random window, prompt and t), runs the student on both frames
of each pair, decodes the same random crop with DC-AE (frame by frame, recomputed in backward), and applies the
three terms; the temporal term and its mask are in `vibe_stream.temporal_loss`. The student starts from the shared
init noise and runs k first-order steps on VIBE's own k-step schedule.

--w_dmd > 0 adds distribution matching distillation (DMD, Yin et al. 2024) on the student latents: they are re-noised
at a random sigma, denoised by the teacher (the base weights; VIBE's DistilledCFG needs no guidance pass) and by a
"fake score" model that is trained alongside to denoise the student's own outputs, and the student moves along
(fake x0 - teacher x0), normalised per sample by |student - teacher x0|. Paired regression makes a student that
cannot redraw the frame in k steps settle near the input; this term instead penalises outputs the teacher would not
produce. The fake model is a second LoRA on the same frozen base (`lora.add_aux`), updated `--fake_updates` times per
step on the step's student latents with fresh noise.

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
    p.add_argument("--w_dmd", type=float, default=0.0, help="weight of the DMD term (0: off)")
    p.add_argument("--fake_rank", type=int, default=32)
    p.add_argument("--fake_lr", type=float, default=1e-4)
    p.add_argument("--fake_updates", type=int, default=2, help="fake score updates per student step")
    p.add_argument("--dmd_sigma", type=float, nargs=2, default=(0.02, 0.98), help="range of the re-noising sigma")
    p.add_argument("--init", default=None, help="start the student from this LoRA file (same rank and noise)")
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
    dev = m.device
    wrappers = lora.attach(m.tr, a.rank)
    if a.init:
        state, _ = lora.load(a.init)
        if lora.NOISE_KEY in state and not torch.equal(state[lora.NOISE_KEY].to(dev), m.noise):
            raise SystemExit(f"{a.init} was trained on a different init noise")
        lora.load_into(wrappers, state)
    params = [q for w in wrappers.values() for q in (w.A, w.B)]
    print(f"[train] {len(wrappers)} LoRA layers, {sum(q.numel() for q in params) / 1e6:.1f} M params", flush=True)
    if a.w_dmd > 0:
        fparams = lora.add_aux(wrappers, a.fake_rank)
        fopt = torch.optim.AdamW(fparams, lr=a.fake_lr, weight_decay=0.0)
        print(f"[train] DMD: weight {a.w_dmd}, fake score rank {a.fake_rank}, {a.fake_updates} updates/step, "
              f"sigma {a.dmd_sigma[0]}-{a.dmd_sigma[1]}", flush=True)
    if a.grad_ckpt:
        m.tr.enable_gradient_checkpointing()
    lp = lpips.LPIPS(net="alex", verbose=False).to(m.device).eval().requires_grad_(False)
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 50) * (0.5 * (1 + math.cos(math.pi * s / a.steps))))
    lat_n, cl, S, mg = size // 32, a.crop // 32, a.crop, a.margin
    log = open(run / "log.jsonl", "w", encoding="utf-8")
    acc = dict.fromkeys(("huber", "lpips", "temp", "temp_teacher") + (("dmd", "dmd_g", "fake") if a.w_dmd > 0 else ()), 0.0)

    def renoise(x0):
        """x0 re-noised at a random sigma per sample; returns (x_t, sigma, eps, VIBE timestep)."""
        s = torch.empty(len(x0), 1, 1, 1, device=dev).uniform_(*a.dmd_sigma)
        eps = torch.randn_like(x0)
        # VIBE's timestep is 1000 sigma (its schedules pair t 999 with sigma 0.9997 and 749 with 0.7496)
        return (1 - s) * x0 + s * eps, s, eps, s.flatten() * 1000

    def dmd_term(x0, img_lat, emb):
        """DMD loss on the student latents x0 (its gradient is the normalised fake - teacher x0) and mean |grad|."""
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            x0d = x0.detach().float()
            xt, s, _, t = renoise(x0d)
            with lora.use(wrappers, None):
                real = xt - s * m.velocity(xt, img_lat, emb, t)
            with lora.use(wrappers, "aux"):
                fake = xt - s * m.velocity(xt, img_lat, emb, t)
            g = (fake - real) / (x0d - real).abs().mean(dim=(1, 2, 3), keepdim=True).clamp_min(1e-6)
        return 0.5 * F.mse_loss(x0.float(), x0d - g), g.abs().mean()

    def update_fake(x0d, img_lat, emb):
        """Flow-matching steps for the fake score model on the student's latents. Returns the last loss."""
        with lora.use(wrappers, "aux"):
            for _ in range(a.fake_updates):
                xt, _, eps, t = renoise(x0d)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    l_f = F.mse_loss(m.velocity(xt, img_lat, emb, t), eps - x0d)
                fopt.zero_grad(set_to_none=True)
                l_f.backward()
                torch.nn.utils.clip_grad_norm_(fparams, 1.0)
                fopt.step()
        return l_f
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
        terms = [("huber", l_lat), ("lpips", l_lp), ("temp", l_t), ("temp_teacher", l_tt)]
        if a.w_dmd > 0:
            l_dmd, g_abs = dmd_term(x0, img_lat, emb)
            loss = loss + a.w_dmd * l_dmd
            terms += [("dmd", l_dmd), ("dmd_g", g_abs)]
        opt.zero_grad(set_to_none=True)
        loss.backward()   # with the student's delta active: checkpointed blocks recompute with it
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if a.w_dmd > 0:
            terms.append(("fake", update_fake(x0.detach().float(), img_lat, emb)))
        for key, val in terms:
            acc[key] += val.item()
        if (step + 1) % a.log_every == 0:
            r = {k: v / a.log_every for k, v in acc.items()}
            el = time.perf_counter() - t0
            dmd = f"dmd {r['dmd']:.4f} |g| {r['dmd_g']:.3f} fake {r['fake']:.4f}  " if a.w_dmd > 0 else ""
            print(f"[train] {step + 1:5d}/{a.steps} huber {r['huber']:.4f} lpips {r['lpips']:.4f} "
                  f"temp {r['temp'] * 255:.2f} (teacher {r['temp_teacher'] * 255:.2f}, /255)  {dmd}{el / (step + 1):.2f} s/step  "
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
                                                     prompts=prompts, teacher_steps=meta["teacher_steps"],
                                                     w_dmd=a.w_dmd, init=a.init))
    print(f"[train] saved {run / 'lora.safetensors'} after {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
