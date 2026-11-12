"""Copy a student for another resolution, e.g. for the face pass (vibe_stream.face): the metadata's size set to --size
and the init noise centre-cropped (or tiled, if the new grid is larger) to the new latent grid. The LoRA itself is
unchanged; the student was fitted at its own size, and how well it carries over is a matter of trying (a 768 student
at 256 draws faces well from crops of 2.6 face widths).

Usage: python scripts/resize_student.py runs/multi_carry_s05_lut_2step_768/lora.safetensors \
           runs/multi_carry_s05_lut_2step_768/lora_256.safetensors --size 256
"""
import argparse

import _paths  # noqa: F401
from vibe_stream import lora


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--size", type=int, required=True, help="a multiple of 32")
    a = ap.parse_args()
    if a.size % 32:
        ap.error("--size must be a multiple of 32")
    state, meta = lora.load(a.src)
    n = state[lora.NOISE_KEY]
    g = a.size // 32
    h, w = n.shape[-2:]
    if g > h or g > w:
        n = n.repeat(1, 1, -(-g // h), -(-g // w))
        h, w = n.shape[-2:]
    y, x = (h - g) // 2, (w - g) // 2
    state[lora.NOISE_KEY] = n[..., y:y + g, x:x + g].contiguous()
    lora.save(a.dst, state, dict(meta, size=a.size, resized_from=a.src))
    print(f"[resize] {a.src} -> {a.dst}: size {a.size}, noise {tuple(state[lora.NOISE_KEY].shape)}")


if __name__ == "__main__":
    main()
