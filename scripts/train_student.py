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

--w_eq > 0 adds a shift-equivariance term: the first frame of each pair is also shifted by a random sub-token
translation d (up to --eq_shift pixels, bilinear, reflected border), DC-AE-encoded and run through the student, and
its decoded crop is compared with the original output sampled at x + d (masked L1, same margin as the temporal term).
The correspondence is exact, unlike the input flow of the temporal term; it targets the student's main flicker
mechanism, outputs that do not move with content shifted by less than one 32-pixel latent token.

--w_tex > 0 adds a texture-amount term: L1 between the local amplitude of detail (above a sigma-2 blur, energy
averaged over sigma 8 px) of the decoded crop and of the teacher crop. The teacher paints its texture at different
places on consecutive frames, so a student pulled towards both the teacher and temporal consistency averages it
away; this term asks for the teacher's amount of texture without asking for its placement.

--w_carry > 0 trains the student for the carry-over of `vibe_stream.carry` as well: the student's own 2-step output
for frame t of each pair (no gradient) is warped to a frame u up to --carry_gap frames later exactly as at run time
(DIS flow at 320 px, photometric weight pooled to the latent grid, the frame itself where the flow fails), and the
last step runs from it. That output gets the Huber and LPIPS terms against the teacher's frame u and lam times an L1
against the warped output, weighted by the flow's weight. With probability --carry_rollout the carried output is
itself carried once (through a frame between t and u), as in the steady state of the stream.

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

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

import _paths  # noqa: F401
from vibe_stream import lora
from vibe_stream.carry import Carry, holds
from vibe_stream.temporal_loss import masked_warp_l1, occlusion_mask, warp
from vibe_stream.vibe_model import VibeModel, flow_step


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
    p.add_argument("--w_eq", type=float, default=0.0, help="weight of the shift-equivariance term (0: off)")
    p.add_argument("--eq_shift", type=float, default=16.0, help="largest synthetic shift in pixels")
    p.add_argument("--w_tex", type=float, default=0.0, help="weight of the texture-amount term (0: off)")
    p.add_argument("--w_carry", type=float, default=0.0, help="weight of the carry-over terms (0: off; 2-step students)")
    p.add_argument("--carry_gap", type=int, default=8, help="largest distance in frames from the carried output")
    p.add_argument("--carry_rollout", type=float, default=0.5,
                   help="probability that the carried output is itself a carried one (one intermediate frame)")
    p.add_argument("--carry_sigma", type=float, default=None,
                   help="noise level the carried step starts from (default: the last step's, 0.7496 for 2 steps)")
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
    acc = dict.fromkeys(("huber", "lpips", "temp", "temp_teacher") + (("dmd", "dmd_g", "fake") if a.w_dmd > 0 else ())
                        + (("eq",) if a.w_eq > 0 else ()) + (("tex",) if a.w_tex > 0 else ())
                        + (("c_huber", "c_lpips", "c_temp") if a.w_carry > 0 else ()), 0.0)
    if a.w_carry > 0 and a.student_steps != 2:
        raise SystemExit("--w_carry needs --student_steps 2")
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    xs_g, ys_g = np.meshgrid(np.arange(size, dtype=np.float32), np.arange(size, dtype=np.float32))

    def carry_start(prev_out, prev_in, cur_in, tau=12.0):
        """Host side as in vibe_stream.carry: the previous output warped to the current frame, and the weight."""
        fl = Carry.flow_to(cur_in, prev_in, dis, 320)
        guess = cv2.remap(prev_out, xs_g + fl[..., 0], ys_g + fl[..., 1], cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
        return guess, holds(cur_in, prev_in, fl, tau).astype(np.float32)

    def carry_step(guesses, weights, img_lat_c, emb_c):
        """The last student step from (1 - s1) (w enc(guess) + (1 - w) enc(frame)) + s1 noise (DC-AE latents)."""
        ts2, sg2 = m.schedules[2]
        s, t = (sg2[1], ts2[1]) if a.carry_sigma is None else (a.carry_sigma, torch.tensor(1000.0 * a.carry_sigma, device=dev))
        zg = m.encode(guesses).clone()
        wl = F.adaptive_avg_pool2d(torch.from_numpy(np.stack(weights))[:, None].to(dev), lat_n)
        x = (1 - s) * (wl * zg + (1 - wl) * img_lat_c) + s * m.noise
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return flow_step(x, m.velocity(x, img_lat_c, emb_c, t), s, 0.0)

    def gblur(x, sigma):
        r = int(3 * sigma + 0.5)
        k = torch.exp(-torch.arange(-r, r + 1, device=x.device, dtype=x.dtype) ** 2 / (2 * sigma ** 2))
        k = (k / k.sum()).view(1, 1, 1, -1).expand(x.shape[1], 1, 1, -1)
        x = F.conv2d(F.pad(x, (r, r, 0, 0), mode="reflect"), k, groups=x.shape[1])
        return F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.transpose(2, 3), groups=x.shape[1])

    def texture_amount(x):
        """Local amplitude of the detail above a sigma-2 blur, averaged over sigma 8 px: how much texture, not where."""
        hp = x - gblur(x, 2.0)
        return (gblur(hp.pow(2).mean(1, keepdim=True), 8.0) + 1e-6).sqrt()

    def decode_crop(x0, cy, cx):
        z = (x0[:, :, cy:cy + cl, cx:cx + cl] / m.sf).to(torch.bfloat16)
        # one frame at a time, recomputed in backward: the decoder's full-resolution activations dominate memory
        return torch.cat([checkpoint(lambda zz: m.vae.decode(zz).sample, z[i:i + 1], use_reentrant=False)
                          for i in range(len(z))]).float()

    def shifted(frame, dx, dy):
        """frame(x + d): the frame with its content moved by -d, sub-pixel, border reflected."""
        mat = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, mat, (size, size), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                              borderMode=cv2.BORDER_REFLECT_101)

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
            dec = decode_crop(x0, cy, cx)
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
        if a.w_eq > 0:
            d = [(random.uniform(-a.eq_shift, a.eq_shift), random.uniform(-a.eq_shift, a.eq_shift)) for _ in items]
            il_s = m.encode([shifted(dd["frames"][t], dx, dy) for (dd, _, t), (dx, dy) in zip(items, d)]).clone()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out_s = (decode_crop(m.student(il_s, emb[0::2], a.student_steps), cy, cx).clamp(-1, 1) + 1) / 2
            flow_d = torch.tensor(d, device=dev, dtype=torch.float32)[:, None, None, :].expand(-1, S, S, -1)
            inb = warp(torch.ones_like(out_s[:, :1]), flow_d)[1]
            l_eq = masked_warp_l1(out_s, out01[0::2], flow_d, inb, mg)
            loss = loss + a.w_eq * l_eq
            terms.append(("eq", l_eq))
        if a.w_tex > 0:
            l_tex = (texture_amount(out01) - texture_amount(t01)).abs()[inner].mean()
            loss = loss + a.w_tex * l_tex
            terms.append(("tex", l_tex))
        if a.w_carry > 0:
            # source: the student's own 2-step output for frame t of each pair; target: a frame 1..carry_gap later,
            # optionally through one intermediate carried frame (as in the steady state of the stream)
            with torch.no_grad():
                src = m.decode_u8(x0[0::2].detach())
            chains = []
            for (d, k, t) in items:
                u = t + random.randint(1, min(a.carry_gap, nfr - 1 - t))
                mid = random.randrange(t + 1, u) if u - t >= 2 and random.random() < a.carry_rollout else None
                chains.append((d, k, t, mid, u))
            hop = [(i, c) for i, c in enumerate(chains) if c[3] is not None]
            if hop:
                with torch.no_grad():
                    st = [carry_start(src[i], c[0]["frames"][c[2]], c[0]["frames"][c[3]]) for i, c in hop]
                    il_m = torch.from_numpy(np.stack([c[0]["img_lat"][c[3]] for _, c in hop])).to(dev)
                    out_m = m.decode_u8(carry_step([g for g, _ in st], [w for _, w in st], il_m,
                                                   torch.stack([emb[2 * i] for i, _ in hop])))
                src = list(src)
                for (i, _), o in zip(hop, out_m):
                    src[i] = o
            st = [carry_start(src[i], d["frames"][t if mid is None else mid], d["frames"][u])
                  for i, (d, k, t, mid, u) in enumerate(chains)]
            il_c = torch.from_numpy(np.stack([d["img_lat"][u] for d, _, _, _, u in chains])).to(dev)
            x0c = carry_step([g for g, _ in st], [w for _, w in st], il_c, emb[0::2])
            tlat_c = torch.from_numpy(np.stack([d[f"tlat_{k}"][u] for d, k, _, _, u in chains])).to(dev)
            timg_c = torch.from_numpy(np.stack([d[f"timg_{k}"][u, Y:Y + S, X:X + S] for d, k, _, _, u in chains]))
            timg_c = timg_c.to(dev).permute(0, 3, 1, 2).float() / 127.5 - 1
            with torch.autocast("cuda", dtype=torch.bfloat16):
                l_clat = F.smooth_l1_loss(x0c, tlat_c, beta=0.1)
                dec_c = decode_crop(x0c, cy, cx)
            l_clp = lp(dec_c[inner].clamp(-1, 1), timg_c[inner]).mean()
            # agreement with the carried output where the flow holds (its own temporal term)
            g01 = torch.from_numpy(np.stack([g[Y:Y + S, X:X + S] for g, _ in st])).to(dev).permute(0, 3, 1, 2).float() / 255
            wc = torch.from_numpy(np.stack([w[Y:Y + S, X:X + S] for _, w in st])).to(dev)[:, None]
            err = ((dec_c.clamp(-1, 1) + 1) / 2 - g01).abs().mean(1, keepdim=True)[inner]
            l_ct = (err * wc[inner]).sum() / wc[inner].sum().clamp_min(1.0)
            loss = loss + a.w_carry * (l_clat + a.w_lp * l_clp + a.lam * l_ct)
            terms += [("c_huber", l_clat), ("c_lpips", l_clp), ("c_temp", l_ct)]
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
            dmd += f"eq {r['eq'] * 255:.2f}  " if a.w_eq > 0 else ""
            dmd += f"tex {r['tex'] * 255:.2f}  " if a.w_tex > 0 else ""
            dmd += (f"carry huber {r['c_huber']:.4f} lpips {r['c_lpips']:.4f} temp {r['c_temp'] * 255:.2f}  "
                    if a.w_carry > 0 else "")
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
                                                     w_dmd=a.w_dmd, w_eq=a.w_eq, eq_shift=a.eq_shift,
                                                     w_tex=a.w_tex, w_carry=a.w_carry,
                                                     carry_sigma=a.carry_sigma if a.w_carry > 0 else None, init=a.init))
    print(f"[train] saved {run / 'lora.safetensors'} after {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
