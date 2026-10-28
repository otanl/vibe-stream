"""Teacher data for the student: VIBE's K-step outputs on short windows of consecutive frames.

For every clip in --clips_dir and every window start, writes <out>/w###.npz with the frames, the Farneback flows of
the input frames (t -> t+1), their DC-AE latents, and per prompt the clip embedding (from the window's first frame)
and the teacher's latents and decoded images. All teacher runs start from the same init noise (CPU float32 draw,
seed 42), as the student will at run time.

Reference styles: --style_list names reference images; each window gets --styles_per_window of them (keys s0, s1,
...; every reference used about equally, fixed order), handed to the encoder next to the window's first frame
(`VibeModel.embed_with_reference`), with the path stored as ref_s<i>. --base_lora merges a fine-tune of the base
model (scripts/train_style_ft.py) into the teacher first. --reuse copies frames, flows, latents and the prompt keys
from an earlier run over the same windows instead of recomputing them (those keep that run's teacher); prompts that run
lacks are computed. --prompts takes any key of presets.ALL_PROMPTS. --stride takes every n-th frame, for windows with
n times the motion between consecutive frames.

Before that it checks, on one frame, that this repo's sampler reproduces VIBE's own pipeline call and that the
first-order k-step formula matches VIBE's scheduler for k = 1, 2.

Usage: python scripts/make_teacher_data.py --out data/teacher_768      (47 windows: about 1 h on an RTX 3090 Ti)
       python scripts/make_teacher_data.py --out data/teacher_768_multi --reuse data/teacher_768 --prompts oil anime \
           watercolor pencil ...        (the new prompts only are computed)
       python scripts/make_teacher_data.py --out data/teacher_768_refs --reuse data/teacher_768 \
           --style_list data/style_ft_768/student_refs.txt --base_lora runs/style_ft_b/lora.safetensors
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import _paths  # noqa: F401
from vibe_stream import lora
from vibe_stream.metrics import farneback, read_video_frames
from vibe_stream.presets import ALL_PROMPTS, PROMPTS, TRAIN_CLIPS_DIR
from vibe_stream.vibe_model import REFERENCE_INSTRUCTION, ROOT, VibeModel


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", required=True)
    p.add_argument("--clips_dir", default=str(ROOT / TRAIN_CLIPS_DIR))
    p.add_argument("--size", type=int, default=768)
    p.add_argument("--teacher_steps", type=int, default=8)
    p.add_argument("--window", type=int, default=10, help="frames per window")
    p.add_argument("--starts", type=int, nargs="+", default=[0, 40], help="window start frames in each clip")
    p.add_argument("--prompts", nargs="*", default=list(PROMPTS), help="keys of presets.ALL_PROMPTS")
    p.add_argument("--stride", type=int, default=1, help="every n-th frame (n times the motion per frame)")
    p.add_argument("--style_list", default=None, help="text file with one reference image path per line")
    p.add_argument("--styles_per_window", type=int, default=4)
    p.add_argument("--base_lora", default=None, help="fine-tune of the base model merged into the teacher")
    p.add_argument("--reuse", default=None, help="earlier teacher dir over the same windows to copy from")
    a = p.parse_args()
    if a.reuse and a.stride != json.loads((Path(a.reuse) / "meta.json").read_text(encoding="utf-8")).get("stride", 1):
        raise SystemExit(f"{a.reuse} was made at another --stride")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    refs = [Path(line.strip()) for line in Path(a.style_list).read_text(encoding="utf-8").splitlines()
            if line.strip()] if a.style_list else []
    m = VibeModel(size=a.size)
    if a.base_lora:
        lora.merge(m.tr, lora.load(a.base_lora)[0])
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
            if s + (a.window - 1) * a.stride + 1 > len(frames):
                continue
            path = out / f"w{n:03d}.npz"
            n += 1
            if path.exists():
                continue
            if a.reuse:
                old = np.load(Path(a.reuse) / path.name)
                if str(old["clip"]) != clip.stem or int(old["start"]) != s:
                    raise SystemExit(f"{a.reuse}/{path.name} is {old['clip']}@{old['start']}, not {clip.stem}@{s}")
                rec = {k: old[k] for k in ("clip", "start", "frames", "flows", "img_lat")}
                for key in a.prompts:
                    if f"tlat_{key}" in old:
                        rec.update({f"{x}_{key}": old[f"{x}_{key}"] for x in ("emb", "tlat", "timg")})
                fr, img_lat = rec["frames"], torch.from_numpy(rec["img_lat"]).to(m.device)
            else:
                fr = np.stack(frames[s:s + (a.window - 1) * a.stride + 1:a.stride])
                flows = np.stack([farneback(fr[i], fr[i + 1]) for i in range(a.window - 1)]).astype(np.float16)
                img_lat = m.encode(list(fr))
                rec = dict(clip=clip.stem, start=s, frames=fr, flows=flows, img_lat=img_lat.cpu().numpy())
            embs = {key: m.embed(ALL_PROMPTS[key], fr[0]) for key in a.prompts if f"tlat_{key}" not in rec}
            for i in range(a.styles_per_window if refs else 0):
                ref = refs[((n - 1) * a.styles_per_window + i) % len(refs)]
                embs[f"s{i}"] = m.embed_with_reference(REFERENCE_INSTRUCTION, fr[0], Image.open(ref).convert("RGB"))
                rec[f"ref_s{i}"] = str(ref)
            for key, e in embs.items():
                lat = m.sample(img_lat, e, a.teacher_steps)
                rec[f"emb_{key}"] = e.float().cpu().numpy()
                rec[f"tlat_{key}"] = lat.cpu().numpy()
                rec[f"timg_{key}"] = m.decode_u8(lat)
            np.savez(path, **rec)
            print(f"[teacher] {path.name} {clip.stem}@{s}  ({time.perf_counter() - t0:.0f} s)", flush=True)
    prompts = {key: ALL_PROMPTS[key] for key in a.prompts}
    prompts.update({f"s{i}": f"reference image to the encoder (ref_s{i} of each window): {REFERENCE_INSTRUCTION}"
                    for i in range(a.styles_per_window if refs else 0)})
    (out / "meta.json").write_text(json.dumps(dict(size=a.size, prompts=prompts, teacher_steps=a.teacher_steps,
                                                   nfr=a.window, stride=a.stride, windows=n, noise="cpu fp32 randn seed 42",
                                                   style_list=a.style_list, base_lora=a.base_lora, reuse=a.reuse)),
                                   encoding="utf-8")
    print(f"[teacher] {n} windows in {time.perf_counter() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
