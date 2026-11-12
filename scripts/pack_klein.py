"""Pack Klein teacher outputs (klein_teacher.py) into train_student.py's window format.

Per window, as make_teacher_data.py writes it: frames, the Farneback flows of the input frames (t -> t+1), their DC-AE
latents, and per instruction key the window was assigned: the VIBE instruction embedding of presets.ALL_PROMPTS[key]
with the window's first frame (what the student is prompted with), the DC-AE latents of Klein's frames (tlat) and their
decode (timg, the target the student can reach through DC-AE). Keys in --vibe_keys take VIBE's own --teacher_steps
teacher instead, from the shared noise, as make_teacher_data.py does. Windows hold only their own keys;
train_student.py samples the (window, key) pairs present. meta.json lists all keys and which teacher made which.

Usage: python scripts/pack_klein.py --windows data/sav_windows --klein data/sav_klein --out data/teacher_sav_768
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
from vibe_stream.metrics import farneback
from vibe_stream.presets import ALL_PROMPTS
from vibe_stream.vibe_model import VibeModel


def psnr(a, b):
    return 10 * np.log10(255 ** 2 / np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", required=True)
    ap.add_argument("--klein", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--vibe_keys", nargs="*", default=[])
    ap.add_argument("--teacher_steps", type=int, default=8, help="steps of VIBE's teacher for --vibe_keys")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    info = json.loads((Path(a.klein) / "assign.json").read_text())
    assign = info["assign"]
    keys = sorted({k for ks in assign.values() for k in ks}, key=list(ALL_PROMPTS).index)
    m = VibeModel(size=a.size)
    t0, rt = time.time(), []
    for n, w in enumerate(sorted(Path(a.windows).glob("w*.npz"))):
        path = out / w.name
        if path.exists():
            continue
        src = np.load(w)
        kfiles = {k: Path(a.klein) / f"{w.stem}_{k}.npz" for k in assign[w.stem] if k not in a.vibe_keys}
        if not all(f.exists() for f in kfiles.values()):
            print(f"[pack] {w.stem}: Klein outputs missing, skipped", flush=True)
            continue
        fr = src["frames"]
        flows = np.stack([farneback(fr[i], fr[i + 1]) for i in range(len(fr) - 1)]).astype(np.float16)
        with torch.inference_mode():
            rec = dict(clip=str(src["clip"]), start=int(src["start"]), stride=int(src["stride"]), frames=fr, flows=flows,
                       img_lat=m.encode(list(fr)).cpu().numpy())
            img_lat = torch.from_numpy(rec["img_lat"]).to(m.device)
            for k in assign[w.stem]:
                emb = m.embed(ALL_PROMPTS[k], fr[0])
                if k in kfiles:
                    kf = np.load(kfiles[k])["frames"]
                    lat = m.encode(list(kf))
                else:
                    lat = m.sample(img_lat, emb, a.teacher_steps)
                rec[f"emb_{k}"] = emb.float().cpu().numpy()
                rec[f"tlat_{k}"] = lat.cpu().numpy()
                rec[f"timg_{k}"] = m.decode_u8(lat)
                if k in kfiles:
                    rt.append(psnr(rec[f"timg_{k}"], kf))
        np.savez(path, **rec)
        if n % 20 == 0 and rt:
            print(f"[pack] {w.stem} ({time.time() - t0:.0f} s); DC-AE round trip of Klein frames so far "
                  f"PSNR median {np.median(rt):.1f} dB, min {min(rt):.1f}", flush=True)
    nfr = int(np.load(next(out.glob("w*.npz")))["frames"].shape[0])
    (out / "meta.json").write_text(json.dumps(dict(
        size=a.size, prompts={k: ALL_PROMPTS[k] for k in keys}, teacher="FLUX.2-klein-4B, 4 steps, seed 42, per frame",
        klein_prompts=info["klein_prompts"], vibe_keys=a.vibe_keys, teacher_steps=a.teacher_steps, nfr=nfr,
        stride="per window (windows.json)", windows=len(list(out.glob("w*.npz"))), source="SA-V (CC BY 4.0)",
        noise="cpu fp32 randn seed 42")), encoding="utf-8")
    print(f"[pack] done in {time.time() - t0:.0f} s; round trip PSNR median {np.median(rt) if rt else 0:.1f} dB", flush=True)


if __name__ == "__main__":
    main()
