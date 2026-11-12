"""Copy a packed training set (pack_klein.py) with the targets of some instruction keys made by VIBE's own teacher.

For each window of --src holding one of --keys, tlat_<key> / timg_<key> are remade from the window's img_lat and its
stored emb_<key> with VIBE's --teacher_steps teacher from the shared noise (as pack_klein.py --vibe_keys does); the
other entries are kept. Windows without those keys are hard-linked, not copied. meta.json gets the keys added to
vibe_keys. Used for night: Klein redraws indoor scenes as streets at night, VIBE keeps the scene.

Usage: python scripts/repack_keys.py --src data/teacher_sav_768 --out data/teacher_sav_768_vnight --keys night
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import _paths  # noqa: F401
from vibe_stream.vibe_model import VibeModel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keys", nargs="+", required=True)
    ap.add_argument("--size", type=int, default=768)
    ap.add_argument("--teacher_steps", type=int, default=8)
    a = ap.parse_args()
    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = VibeModel(size=a.size)
    t0, n_new = time.time(), 0
    for w in sorted(src.glob("w*.npz")):
        path = out / w.name
        if path.exists():
            continue
        with np.load(w) as z:
            todo = [k for k in a.keys if f"tlat_{k}" in z.files]
            if not todo:
                os.link(w, path)
                continue
            rec = {f: z[f] for f in z.files}
        img_lat = torch.from_numpy(rec["img_lat"]).to(m.device)
        with torch.inference_mode():
            for k in todo:
                emb = torch.from_numpy(rec[f"emb_{k}"]).to(m.device, torch.bfloat16)   # as train_student.py
                lat = m.sample(img_lat, emb, a.teacher_steps)
                rec[f"tlat_{k}"] = lat.cpu().numpy()
                rec[f"timg_{k}"] = m.decode_u8(lat)
        np.savez(path, **rec)
        n_new += 1
        if n_new % 10 == 1:
            print(f"[repack] {w.stem} {todo} ({time.time() - t0:.0f} s)", flush=True)
    meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
    meta["vibe_keys"] = sorted(set(meta.get("vibe_keys", [])) | set(a.keys))
    meta["teacher_steps"] = a.teacher_steps
    (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    print(f"[repack] done: {n_new} windows remade in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
