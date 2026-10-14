"""Fetch the weights this repo runs on into models/ (gitignored).

  VIBE (CFG-distilled)  huggingface.co/iitolstykh/VIBE-Image-Edit-DistilledCFG -> models/VIBE-Image-Edit-DistilledCFG
                        (about 8.9 GB; the model card asks to follow Sana's licence)
  taesana               github.com/madebyollin/taesd (MIT)                     -> models/taesana

Usage: python scripts/download_weights.py [--vibe] [--taesana]   (both when neither is given)
"""
import argparse
import urllib.request

import _paths  # noqa: F401
from vibe_stream.taesana import DEFAULT_DIR as TAE_DIR
from vibe_stream.vibe_model import DEFAULT_WEIGHTS

VIBE_REPO_ID = "iitolstykh/VIBE-Image-Edit-DistilledCFG"
TAE_URL = "https://raw.githubusercontent.com/madebyollin/taesd/main/{}"
TAE_FILES = ("taesana_encoder.pth", "taesana_decoder.pth")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--vibe", action="store_true")
    p.add_argument("--taesana", action="store_true")
    a = p.parse_args()
    both = not (a.vibe or a.taesana)
    if a.taesana or both:
        TAE_DIR.mkdir(parents=True, exist_ok=True)
        for name in TAE_FILES:
            dst = TAE_DIR / name
            if dst.exists():
                print(f"[taesana] {dst} exists")
                continue
            urllib.request.urlretrieve(TAE_URL.format(name), dst)
            print(f"[taesana] {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    if a.vibe or both:
        from huggingface_hub import snapshot_download
        path = snapshot_download(VIBE_REPO_ID, local_dir=str(DEFAULT_WEIGHTS))
        print(f"[vibe] {path}")


if __name__ == "__main__":
    main()
