"""A training pack at another size (640) for a student resized from 768 with transformer blocks removed: SA-V windows,
Klein's targets, and broad instructions self-distilled from the full student.

--mode klein: every window of --src (pack_klein.py's 768 pack, train_student.py format) at --size: frames resized
(INTER_AREA), Farneback flows recomputed, DC-AE latents, and per key the window holds: the instruction embedding at
--size and the target, Klein's own 768 frames (<klein>/<window>_<key>.npz) resized and passed through DC-AE (tlat,
timg), or for the keys VIBE's teacher made in --src (meta vibe_keys) the 768 pack's timg resized.
--mode self: adds --per_window keys of presets.BROAD_PROMPTS to every window (fixed rotation, each key on about the
same number of windows). Their targets are the full student --teacher (scripts/resize_student.py from the 768 one) at
--size, 2 steps from its own init noise: self-distillation, so a student trained on this pack with blocks removed keeps
what the full student does beyond Klein's 14 instructions. The held-out evaluation keys are not among them.
meta.json lists every key, so train_student.py samples the (window, key) pairs present. Windows are written to a
temporary name and renamed, so an interrupted run resumes where it stopped.

Usage: python scripts/resize_pack.py --mode klein --src data/teacher_sav_768_vnight --klein data/sav_klein \
           --out data/teacher_sav_640
       python scripts/resize_pack.py --mode self --src data/teacher_sav_768_vnight --out data/teacher_sav_640 \
           --teacher runs/klein_sav_vnight_cont_768/lora_640.safetensors
"""
import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import _paths  # noqa: F401
from vibe_stream import lora
from vibe_stream.metrics import farneback
from vibe_stream.presets import ALL_PROMPTS, BROAD_PROMPTS
from vibe_stream.vibe_model import VibeModel


def resize(frames, size):
    return np.stack([cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames])


def save(path, rec):
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, **rec)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["klein", "self"], required=True)
    ap.add_argument("--src", required=True, help="the 768 pack (pack_klein.py, repack_keys.py)")
    ap.add_argument("--klein", default=None, help="--mode klein: klein_teacher.py's output directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=640)
    ap.add_argument("--teacher", default=None, help="--mode self: the full student at --size (LoRA file)")
    ap.add_argument("--per_window", type=int, default=3)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src_meta = json.loads((Path(a.src) / "meta.json").read_text(encoding="utf-8"))
    for k in BROAD_PROMPTS:
        if k in src_meta["prompts"] or k in ALL_PROMPTS:
            raise SystemExit(f"broad key {k} clashes with an existing key")
    t0 = time.time()
    if a.mode == "klein":
        if not a.klein:
            ap.error("--mode klein needs --klein")
        vibe_keys = set(src_meta.get("vibe_keys", []))
        m = VibeModel(size=a.size, denoiser_on_gpu=False)   # the text encoder and the embedding path on the GPU
        m.vae.to(m.device)                                  # DC-AE too; the transformer blocks stay on the CPU
        files = sorted(Path(a.src).glob("w*.npz"))
        for n, w in enumerate(files):
            path = out / w.name
            if path.exists():
                continue
            z = np.load(w)
            keys = [f[5:] for f in z.files if f.startswith("tlat_")]
            fr = resize(z["frames"], a.size)
            rec = dict(clip=z["clip"], start=z["start"], stride=z["stride"], frames=fr,
                       flows=np.stack([farneback(fr[i], fr[i + 1]) for i in range(len(fr) - 1)]).astype(np.float16),
                       img_lat=m.encode(list(fr)).cpu().numpy())
            for k in keys:
                kf = Path(a.klein) / f"{w.stem}_{k}.npz"
                tgt = np.load(kf)["frames"] if k not in vibe_keys and kf.exists() else z[f"timg_{k}"]
                lat = m.encode(list(resize(tgt, a.size)))
                rec[f"emb_{k}"] = m.embed(ALL_PROMPTS[k], fr[0]).float().cpu().numpy()
                rec[f"tlat_{k}"] = lat.cpu().numpy()
                rec[f"timg_{k}"] = m.decode_u8(lat)
            save(path, rec)
            if n % 20 == 0:
                print(f"[pack] klein {w.stem} ({n + 1}/{len(files)}, {time.time() - t0:.0f} s): keys {keys}", flush=True)
        teacher = None
    else:
        if not a.teacher:
            ap.error("--mode self needs --teacher")
        state, tmeta = lora.load(a.teacher)
        if int(tmeta["size"]) != a.size:
            raise SystemExit(f"{a.teacher} is for {tmeta['size']}, not {a.size}")
        m = VibeModel(size=a.size, noise=state[lora.NOISE_KEY].float())
        print(f"[pack] teacher {a.teacher}: {lora.merge(m.tr, state)} layers merged, "
              f"{int(tmeta.get('student_steps', 2))} steps", flush=True)
        steps = int(tmeta.get("student_steps", 2))
        bk = list(BROAD_PROMPTS)
        files = [f for f in sorted(out.glob("w*.npz")) if not f.name.endswith(".tmp.npz")]
        for n, f in enumerate(files):
            ks = [bk[(n * a.per_window + j) % len(bk)] for j in range(a.per_window)]
            with np.load(f) as z:   # closed before the rename (Windows refuses to replace an open file)
                if all(f"tlat_{k}" in z.files for k in ks):
                    continue
                rec = {k: z[k] for k in z.files}
            il = torch.from_numpy(rec["img_lat"]).to(m.device)
            for k in ks:
                emb = m.embed(BROAD_PROMPTS[k], rec["frames"][0])
                with torch.inference_mode():
                    lat = m.student(il, emb, steps)
                rec[f"emb_{k}"] = emb.float().cpu().numpy()
                rec[f"tlat_{k}"] = lat.float().cpu().numpy()
                rec[f"timg_{k}"] = m.decode_u8(lat)
            save(f, rec)
            if n % 20 == 0:
                print(f"[pack] self {f.stem} ({n + 1}/{len(files)}, {time.time() - t0:.0f} s): keys {ks}", flush=True)
        teacher = a.teacher
    old = json.loads((out / "meta.json").read_text(encoding="utf-8")) if (out / "meta.json").exists() else {}
    meta = dict(src_meta, size=a.size, prompts={**src_meta["prompts"], **BROAD_PROMPTS},
                resized_from=f"{a.src} (768; frames and targets resized with INTER_AREA, Klein frames from "
                             f"{a.klein or old.get('klein')})",
                klein=a.klein or old.get("klein"), self_keys=list(BROAD_PROMPTS),
                self_teacher=teacher or old.get("self_teacher"),
                self_per_window=a.per_window if teacher else old.get("self_per_window"))
    meta["windows"] = len([f for f in out.glob("w*.npz") if not f.name.endswith(".tmp.npz")])
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"[pack] {a.mode} done in {time.time() - t0:.0f} s, {meta['windows']} windows", flush=True)


if __name__ == "__main__":
    main()
