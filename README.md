# vibe-stream

Video-rate streaming edits with [VIBE](https://github.com/ai-forever/vibe) (Sana 1.5 1.6B + Qwen3-VL-2B): a
few-step student trained with a temporal-consistency loss, madebyollin's tiny autoencoder `taesana` in place of
DC-AE, and VIBE's transformer as a TensorRT engine.

This is the successor line to [dreamlite-stream](https://github.com/otanl/dreamlite-stream) (arXiv:2606.05981),
whose DreamLite-mobile weights are CC BY-NC 4.0. Everything this repository runs on is published under Apache-2.0
or MIT (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)), and VIBE's still-image quality is not below
DreamLite-mobile's: styles are about equal and instruction edits are clearly better.

**Status: research code.** Two students are trained (2 steps, two style prompts, at 768 and at 640 px); the whole
streaming loop runs at 21.0 fps at 768 and 29.1 fps at 640 on an RTX 3090 Ti. The numbers below say what has been
measured and how.

## Demo

https://github.com/user-attachments/assets/26081288-3447-4734-902b-ddc4dcb9c8a9

Input | oil painting | anime on five DAVIS-2017 clips through the real-time path at 640 px on an RTX 3090 Ti: the
2-step student, taesana, and the transformer as a TensorRT engine (`scripts/make_demo.py`). Each output panel shows
the loop fps of its run: 27-29 fps on these short clips, 29.1 fps on the 480-frame loop in the speed table. Input
clips: [DAVIS-2017](https://davischallenge.org/), see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## How it works

VIBE alone flickers on video because it redraws structure freely on every frame: even with the noise and the
embedding fixed, its 8-step output has a warping error of 26.4 on the evaluation clips, against 17.3 for
DreamLite-mobile with its one-step LoRA (table below). The streaming setting fixes the two free inputs:

- one init-noise tensor shared by every frame (a CPU float32 draw, stored with each student), and
- one instruction embedding per clip, computed from the first frame (Qwen3-VL sees instruction and image together).

VIBE's K-step output is then a deterministic function of the frame. A rank-32 LoRA on the transformer is trained to
reproduce the 8-step output in k = 1 or 2 first-order steps (paired regression: Huber on the latent + LPIPS on
decoded crops) plus `lam` times a masked L1 between output t and output t+1 warped back by the optical flow of the
*input* frames. The temporal term uses the same flow convention as the reported warping error.

At run time per batch of 8 frames: taesana encode, k calls of the transformer (TensorRT engine, LoRA merged), taesana
decode, all on the GPU with uint8 frames in and out (`src/vibe_stream/stream.py`).

## Results

RTX 3090 Ti, Windows, torch 2.6.0+cu124, transformers 4.57.3, diffusers 0.37.1, TensorRT 10.16.

### Quality (768, DC-AE decode, PyTorch bf16)

`scripts/eval_student.py`: DAVIS `parkour`, `libby`, `blackswan` and `dance-twirl`, first 16 frames, prompts `oil`
and `anime` (mean of 8; the rewritten scripts reproduce the per-clip numbers of the original runs).
Warp = warping error (0-255, lower is steadier); LPIPS to VIBE's 8-step output; Sobel = texture; edge = agreement of
edges with the input; CLIP dir = CLIP directional similarity per prompt (how far the change from the input points
from "a photo" towards "an oil painting ..." / "an anime illustration ...", `presets.CLIP_CAPTIONS`; higher is a
stronger edit).

| Configuration | Warp | LPIPS to teacher | Sobel | Edge | CLIP dir oil / anime |
|---|---:|---:|---:|---:|---:|
| VIBE, 8 steps (teacher) | 26.4 | — | 10.9 | 0.35 | 0.248 / 0.217 |
| VIBE, 1 step, no student | 26.2 | 0.578 | 22.5 | 0.24 | 0.175 / 0.124 |
| VIBE, 2 steps, no student | 21.6 | 0.448 | 16.3 | 0.31 | 0.154 / 0.194 |
| 1-step student, lam 0 | 22.6 | 0.275 | 10.4 | 0.42 | 0.235 / 0.202 |
| 1-step student, lam 4 | 15.9 | 0.288 | 8.0 | 0.43 | 0.230 / 0.204 |
| 1-step student, lam 12 | 9.7 | 0.338 | 5.4 | 0.44 | 0.204 / 0.196 |
| **2-step student, lam 4** (`lam4_2step`) | 18.3 | **0.208** | 8.2 | 0.35 | **0.244 / 0.212** |
| *DreamLite-mobile, 1 step + LoRA (dreamlite-stream)* | *17.3* | | *7.7* | *0.71* | |

- The temporal term works: lam 4 brings the one-step student from 22.6 to 15.9. lam 12 reaches 9.7 by haze,
  desaturation and ghosting (Sobel 5.4).
- One step loses much of VIBE's edit strength on restructuring prompts (anime stays photographic). The 2-step
  student restores most of it (LPIPS 0.208, CLIP dir within 0.005 of the teacher) and is the one used for streaming.
- VIBE keeps less of the source structure than DreamLite (edge agreement 0.35 against 0.71): it repaints more
  freely, which is also why its steadiness lags at the same step count.
- With taesana for encode and decode instead of DC-AE, the lam-4 one-step student's output is LPIPS 0.134 from the
  DC-AE path and its warping error is unchanged (15.4 vs 15.9). taesana is crisp and slightly grainy.

**At 640 px** (`lam4_2step_640`: same recipe on 640 px teacher data, scored against VIBE's own 640 px 8-step output):

| Configuration (640) | Warp | LPIPS to teacher | Sobel | Edge | CLIP dir oil / anime |
|---|---:|---:|---:|---:|---:|
| VIBE, 8 steps (teacher) | 29.4 | — | 12.2 | 0.29 | 0.245 / 0.218 |
| 2-step student, lam 4 (`lam4_2step_640`) | 19.4 | 0.234 | 8.8 | 0.30 | 0.240 / 0.204 |

The student keeps the same distance to its teacher as at 768. VIBE itself repaints less at 640 on some clips (on
`parkour` its anime output keeps a plain sky where the 768 output paints clouds), which the student inherits.

### Speed (batch 8)

| Stage, ms per frame | 768 | 640 |
|---|---:|---:|
| Transformer, TensorRT fp16 engine, per step (bf16 PyTorch eager at 768: 36) | 20.5 | 15.2 |
| taesana encode + decode, PyTorch fp16 | 0.9 + 8.7 | |
| DC-AE encode + decode at 768, TensorRT fp16 (not used) | 39 + 58 | |
| **Whole loop: 2-step student + taesana, uint8 frames in and out** | **47.7 (21.0 fps)** | **34.4 (29.1 fps)** |

The last row is `scripts/stream_video.py` on DAVIS `blackswan` (50 frames cycled to 480, prompt `oil`): loop and
step-only fps are equal at both sizes (21.0 / 21.0 at 768, 29.1 / 29.1 at 640), i.e. the host side (pinned copies,
uint8 conversion) is fully hidden behind the GPU. The instruction embedding is computed once per clip. Output
warping error with taesana and the engine: 18.3 at 768, 18.1 at 640. Without TensorRT (`--student`, PyTorch bf16
with the LoRA merged) the 768 loop runs at 12.6 fps, warping error 18.4.

Step-only and loop fps are defined as in dreamlite-stream (`src/vibe_stream/stream.py`). For comparison,
dreamlite-stream's lean path with its one-step LoRA and TensorRT UNet runs at 25 fps loop at 768. The remaining cost is the transformer (2 steps); a one-step student would run at about 30 ms per
frame at 768 but loses edit strength (table above), and DMD does not win it back (Negative results).

## Setup

The code expects an upstream VIBE checkout next to it (or `VIBE_REPO` pointing at one) and the weights in `models/`:

```bash
git clone https://github.com/ai-forever/vibe ../vibe          # tested at 7f0f01f
python scripts/download_weights.py                           # VIBE DistilledCFG (8.9 GB) + taesana
```

Python 3.10 with torch (CUDA), transformers >= 4.57 (Qwen3-VL), diffusers, safetensors, huggingface_hub,
opencv-python, lpips, and for the engine tensorrt 10.x and onnx (see `requirements.txt`). VIBE's own dependencies
beyond these are not needed; when `loguru` is missing a silent stand-in is used.

Video clips are not tracked; see [assets/README.md](assets/README.md).

## Usage

```bash
# Stream a clip through the 2-step student with the TensorRT engine
python scripts/build_trt.py --student runs/lam4_2step/lora.safetensors \
    --engine out/trt/vibe_lam4_2step_768_b8.engine
python scripts/stream_video.py --input assets/davis_mp4/blackswan.mp4 --prompt oil \
    --engine out/trt/vibe_lam4_2step_768_b8.engine --output out/stream/blackswan_oil.mp4 --compare
# ... or with PyTorch only
python scripts/stream_video.py --input ... --student runs/lam4_2step/lora.safetensors
# Demo video: input | oil | anime for every clip in assets/davis_mp4 through the same path (H.264 if ffmpeg is on PATH)
python scripts/make_demo.py --engine out/trt/vibe_lam4_2step_640_b8.engine --out out/demo_640

# Train a student
python scripts/make_teacher_data.py --out data/teacher_768                   # 47 windows x 10 frames x 2 prompts
python scripts/train_student.py --data data/teacher_768 --run runs/lam4_2step \
    --student_steps 2 --lam 4 --grad_ckpt                                    # 1500 steps, ~56 min, 8 GB
python scripts/eval_student.py --out out/eval --runs runs/lam4_2step --base
# ... at 640: data in 6 min, training ~39 min at 1.55 s/step; build_trt.py and stream_video.py read the size
# from the student / engine
python scripts/make_teacher_data.py --out data/teacher_640 --size 640
python scripts/train_student.py --data data/teacher_640 --run runs/lam4_2step_640 --student_steps 2 --lam 4 --grad_ckpt
python scripts/eval_student.py --size 640 --out out/eval_640 --runs runs/lam4_2step_640
```

A student is tied to its resolution, its init noise (stored in the file) and, in practice, the prompts it was
trained on; other instructions run but come out weaker. A TensorRT engine is tied to one resolution, one batch size,
one TensorRT version and platform; `<engine>.json` records them and a mismatch is refused.

## Layout

- `src/vibe_stream/`
  - `vibe_model.py`: VIBE's pipeline driven step by step (embedding, DC-AE, velocity, teacher sampler, k-step student)
  - `lora.py`: student LoRA (attach, save, load, merge)
  - `temporal_loss.py`: flow warp and the masked temporal term
  - `taesana.py`: the tiny autoencoder
  - `trt_engine.py`: ONNX export, engine build, runtime wrapper
  - `stream.py`: batched, double-buffered streaming editor
  - `metrics.py`: video I/O, warping error, Sobel, edge agreement, LPIPS
- `scripts/`: one entry point per step above.

`models/`, `data/`, `runs/` and `out/` are gitignored; weights are never committed.

## Negative results

- **Distribution matching distillation (DMD) for the one-step student** (`train_student.py --w_dmd`, after
  Yin et al. 2024 and the 4-step DMD preview of SANA-Video 2.0). The hope was that a distribution-level loss would
  push the one-step student towards the teacher's look where paired regression settles near the input. Added on top
  of the lam-4 recipe (fake score model = a second rank-32 LoRA on the same frozen base, 2 updates per step;
  2.1 s/step, 17 GB), same evaluation as above:

  | One-step student | Warp | LPIPS to teacher | Sobel | Edge | CLIP dir oil / anime |
  |---|---:|---:|---:|---:|---:|
  | lam 4, no DMD | 15.9 | 0.288 | 8.0 | 0.43 | 0.230 / 0.204 |
  | + DMD, weight 1, sigma 0.02-0.98 | 16.7 | 0.287 | 8.2 | 0.43 | 0.232 / 0.206 |
  | + DMD, weight 3, sigma 0.02-0.98 | 17.6 | 0.290 | 8.4 | 0.44 | 0.235 / 0.206 |
  | + DMD, weight 10, sigma 0.4-0.98 | 21.4 | 0.327 | 9.6 | 0.44 | 0.231 / 0.203 |
  | *2-step student, lam 4* | *18.3* | *0.208* | *8.2* | *0.35* | *0.244 / 0.212* |
  | *VIBE, 8 steps (teacher)* | *26.4* | | *10.9* | *0.35* | *0.248 / 0.217* |

  DMD adds contrast and texture (Sobel) and flicker, but not the restyling a second step brings (sky, brush strokes
  on water); CLIP directional similarity does not move. One step's limit here is not the choice of loss.
- **Fine-tuning taesana's decoder towards VIBE's DC-AE** (sana-1.1). L1 + LPIPS + low-resolution MSE lowered LPIPS to
  the DC-AE decode (0.103) but visibly softened brush strokes (Sobel 6.2 vs DC-AE 8.0, released taesana 8.3). An
  adversarial version after TAESD's recipe (relativistic loss, replay buffer, R1/R2) was only slightly crisper at its
  best (Sobel 6.8) and collapsed with a weaker penalty. The released taesana is used as is.
- **Mirror-flip augmentation for encoder distillation** is wrong for DC-AE latents (they are not mirror-symmetric),
  and crop training misses DC-AE's global context: train an encoder on whole frames.

## Reference: SANA-Streaming

NVIDIA's [SANA-Streaming](https://arxiv.org/abs/2605.30409) is a dedicated streaming video editor: a 2B hybrid
linear/softmax DiT on a causal LTX-2 video VAE (8x temporal compression), 4 steps. It is not an option for this
repository's purpose: its model card says Apache-2.0, but the VAE comes from Lightricks' LTX-2 (LTX-2 Community
License: entities with at least $10M annual revenue need a paid licence, derivatives including distillations are
covered, use restrictions must be passed on) and the text encoder is Gemma-2-2B (Gemma terms).

Measured once for reference with the released BF16 inference path (NVlabs/Sana 0354b3d, torch 2.9.1, triton 3.5.1,
WSL) on the four evaluation clips, 49 frames, this repository's two prompts, seed 0:

- 768 x 768: DiT 36 ms per frame for all 4 steps (7 latent frames for 49 frames), VAE encode 70-78 ms and decode about
  445 ms per frame through the released chunk/tile wrapper, 19.7 GB peak. At its native 704 x 1280: DiT 53 ms per
  frame, decode 1.1 s per frame with memory full (24.1 GB). The paper's 24 fps (RTX 5090) uses fused kernels and
  quantisation not in the public script.
- First 16 frames, metrics as above: warp 17.1, Sobel 7.0, edge 0.45, CLIP dir 0.174 / 0.130. The first frame is
  strongly stylised; later frames smear on fast motion at both resolutions (these prompts may be outside its
  instruction distribution).

Temporal compression is what makes its DiT about 4x cheaper per frame and step than VIBE's (about 9 against 36 ms,
both bf16 eager), but it needs an editor trained on a video latent.

## Licence

Code: Apache-2.0 ([LICENSE](LICENSE)); the taesana architecture is MIT (see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)). Weights are not part of the repository; check the publishers'
terms before redistributing weights or anything derived from them, such as a trained student or an engine.
