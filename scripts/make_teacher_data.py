"""Teacher data for the student: VIBE's K-step outputs on short windows of consecutive frames.

For every clip in --clips_dir and every window start, writes <out>/w###.npz with the frames, the Farneback flows of
the input frames (t -> t+1), their DC-AE latents, and per prompt the clip embedding (from the window's first frame)
and the teacher's latents and decoded images. All teacher runs start from the same init noise (CPU float32 draw,
seed 42), as the student will at run time.

Before that it checks, on one frame, that this repo's sampler reproduces VIBE's own pipeline call and that the
first-order k-step formula matches VIBE's scheduler for k = 1, 2.

Usage: python scripts/make_teacher_data.py --out data/teacher_768      (47 windows: about 1 h on an RTX 3090 Ti)
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import _paths  # noqa: F401
from vibe_stream.metrics import farneback, read_video_frames
from vibe_stream.presets import PROMPTS, TRAIN_CLIPS_DIR
from vibe_stream.vibe_model import ROOT, VibeModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--clips_dir", default=str(ROOT / TRAIN_CLIPS_DIR))
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--teacher_steps", type=int, default=8)
    p.add_argument("--window", type=int, default=10, help="frames per window")
    p.add_argument("--starts", type=int, nargs="+", default=[0, 40], help="window start frames in each clip")
    a = p.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    m = VibeModel(size=a.size)
    clips = sorted(Path(a.clips_dir).glob("*.mp4"))
    if not clips:
        raise SystemExit(f"no clips in {a.clips_dir}")

    f0 = read_video_frames(clips[0], a.size, 1)[0]
    emb = m.embed(PROMPTS["oil"], f0)
    with torch.inference_mode():
        il = m.encode([f0])
        # the pipeline casts given latents to bf16: feed both the same rounded noise for this check
        noise = m.noise
        m.noise = noise.to(torch.bfloat16).float()
        ref = m.pipe(conditioning_image=Image.fromarray(f0), prompt_embeds=emb, latents=m.noise.clone(), height=a.size,
                     width=a.size, num_inference_steps=a.teacher_steps, guidance_scale=0.0, image_guidance_scale=0.0,
                     output_type="latent", use_resolution_binning=False).images
        mine = m.sample(il, emb, a.teacher_steps)
        m.noise = noise
        print(f"[check] own sampler vs pipeline, {a.teacher_steps} steps: max |diff| {(ref.float() - mine).abs().max().item():.4f} "
              f"(latent std {ref.float().std().item():.3f})", flush=True)
        for k in (1, 2):
            d = (m.sample(il, emb, k) - m.student(il, emb, k)).abs().max().item()
            print(f"[check] {k}-step first-order formula vs scheduler: max |diff| {d:.5f}; timesteps "
                  f"{[int(t) for t in m.schedules[k][0]]} sigmas {[round(v, 4) for v in m.schedules[k][1]]}", flush=True)

    t0 = time.perf_counter()
    n = 0
    for clip in clips:
        frames = read_video_frames(clip, a.size)
        for s in a.starts:
            if s + a.window > len(frames):
                continue
            path = out / f"w{n:03d}.npz"
            n += 1
            if path.exists():
                continue
            fr = np.stack(frames[s:s + a.window])
            flows = np.stack([farneback(fr[i], fr[i + 1]) for i in range(a.window - 1)]).astype(np.float16)
            img_lat = m.encode(list(fr))
            rec = dict(clip=clip.stem, start=s, frames=fr, flows=flows, img_lat=img_lat.cpu().numpy())
            for key, prompt in PROMPTS.items():
                e = m.embed(prompt, fr[0])
                lat = m.sample(img_lat, e, a.teacher_steps)
                rec[f"emb_{key}"] = e.float().cpu().numpy()
                rec[f"tlat_{key}"] = lat.cpu().numpy()
                rec[f"timg_{key}"] = m.decode_u8(lat)
            np.savez(path, **rec)
            print(f"[teacher] {path.name} {clip.stem}@{s}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    (out / "meta.json").write_text(json.dumps(dict(size=a.size, prompts=PROMPTS, teacher_steps=a.teacher_steps,
                                                   nfr=a.window, windows=n, noise="cpu fp32 randn seed 42")),
                                   encoding="utf-8")
    print(f"[teacher] {n} windows in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
