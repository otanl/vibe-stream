# vibe-stream

Video-rate streaming edits with [VIBE](https://github.com/ai-forever/vibe) (Sana 1.5 1.6B + Qwen3-VL-2B): a
few-step student trained with a temporal-consistency loss, madebyollin's tiny autoencoder `taesana` in place of
DC-AE, and VIBE's transformer as a TensorRT engine.

This is the successor line to [dreamlite-stream](https://github.com/otanl/dreamlite-stream) (arXiv:2606.05981),
whose DreamLite-mobile weights are CC BY-NC 4.0. Everything this repository runs on is published under Apache-2.0
or MIT (see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)), and VIBE's still-image quality is not below
DreamLite-mobile's: styles are about equal and instruction edits are clearly better.

**Status: research code.** Two students are trained (2 steps, two style prompts, at 768 and at 640 px); the whole
streaming loop runs at 21.0 fps at 768 and 29.1 fps at 640 on an RTX 3090 Ti. A student trained to carry the
previous output over into the next batch (one transformer call per frame) runs at 47.8 fps at 640 with less flicker
and the same edit strength ("Flicker"). The numbers below say what has been measured and how.

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

The options of the next section, same run at 640 (TensorRT engine, loop / step-only fps):

| 640, 480-frame loop | Loop fps | Step-only fps |
|---|---:|---:|
| 2 steps per frame | 29.6 | 29.6 |
| + temporal filter 0.7 / 0.3 | 28.7 | 28.6 |
| carry-over (one transformer call per frame after the first batch) | 47.8 | 48.1 |
| carry-over + temporal filter | 45.2 | 45.7 |

The filter's optical flow runs on the host behind the GPU; its blend costs about 1 ms per frame on the GPU. The
carry-over's flows are hidden the same way.

### Flicker (640, streaming path)

`scripts/eval_flicker.py`: DAVIS `blackswan`, `parkour`, `dance-twirl`, `camel`, `goat` and `libby`, first 40 frames,
prompts `oil` and `anime`, through the streaming editor with taesana (PyTorch bf16). *Valid* is the frame-to-frame
change of the output after warping by the input's flow, counted only where the same warp reproduces the input
(`metrics.flicker`; the input itself scores about 2.5): what remains is change the edit added. *Low* / *high* split
it at a Gaussian of sigma 4 px. DreamLite-mobile's lean path scores about 12 for both prompts on the same clips
(measured on dreamlite-stream's 768 demo).

**Where it comes from.** Every frame starts from the same init noise, and the transformer works on 32-pixel latent
tokens, so the student's texture is anchored to the screen: content that moves by a few pixels gets repainted
instead of carried along. The shift probe crops a wider frame at horizontal offsets and compares each output with
the unshifted output moved by the same offset (0 = the output moves with the content):

| Offset (px) | 0.5 | 1 | 2 | 4 | 8 | 16 | 32 | 32, noise rolled 1 token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `lam4_2step_640` | 3.5 | 5.6 | 8.5 | 11.1 | 14.3 | 18.3 | 19.3 | 6.5 |
| `lam8_2step_640` | 2.7 | 4.0 | 6.3 | 8.6 | 11.1 | 13.8 | 14.8 | 4.8 |
| `lam4_eq4_2step_640` | 2.5 | 3.7 | 5.7 | 7.8 | 10.0 | 12.4 | 13.1 | 4.3 |
| `lam4_eq8_2step_640` | 2.1 | 2.7 | 4.2 | 5.9 | 7.6 | 9.0 | 9.8 | 3.1 |

On the released student a one-pixel shift already changes the output by 5.6; rolling the noise along with a
whole-token shift removes two thirds of the change at 32 px, so the shared noise is most of it. But motion between
frames is mostly a few pixels, and noise cannot follow sub-token motion (Negative results). What can be changed is
the student (the rows below it, described next) and the display.

**Students and the display filter.** `lam8`: the temporal weight doubled. `eq4`: lam 4 plus a shift-equivariance
term (`train_student.py --w_eq 4`): the first frame of a pair is also shifted by a random sub-token offset
(up to 16 px), and its output must equal the unshifted output moved by the same offset, an exact correspondence
instead of the input flow; `eq8` the same with weight 8. `lam8_tex8`: lam 8 plus a texture-amount term
(`--w_tex 8`): L1 between the local amplitude of detail of the output and of the teacher (energy above a sigma-2
blur, averaged over sigma 8 px), which asks for the teacher's amount of texture but not its placement.

*Filter*: `temporal_filter.TemporalFilter(0.7, 0.3)` on the decoded frames (`--temporal_filter 0.7 0.3`): the
output is split at a Gaussian of sigma 6 px and each band is blended with the previous shown frame warped by the
input's flow (DIS at 320 px), 0.7 of the previous frame in the low band and 0.3 in the high band, turned off where
the warp does not reproduce the input.

*Carry-over* (`carry.Carry`, `--carry`): after a keyframe batch (2 steps from noise), each batch skips step 1 and
starts step 2 from the previous batch's last output warped to each frame by the input's flow, mixed on the latent
grid with the frame itself where the flow does not hold (occlusions, new content), at step 2's noise level
(sigma 0.75). It is the division of labour between the steps carried over time: step 1 decides composition and
colour areas, and those come from the previous output instead of being decided afresh. One transformer call per
frame. `lam4_carry` is trained for it (`--w_carry 1`): the student's own output, warped to a frame 1-8 frames later
as at run time (and with probability 0.5 carried once more through a frame in between), is the start of a step that
gets the Huber and LPIPS terms against the teacher and lam times an L1 against the warped output; the usual pair
terms stay, so its 2-step path still serves keyframes. Keyframes come at the start and at cuts (detected from the
flow, `Carry.cut_below`); the runs below have no other keyframes.

*Plain* is the 2-step path without the filter.

| Student | Path / display | Valid oil / anime | Low / high (oil) | Warp oil | Sobel oil | CLIP dir oil / anime |
|---|---|---:|---:|---:|---:|---:|
| `lam4_2step_640` (released) | plain | 20.9 / 12.2 | 12.6 / 13.7 | 22.3 | 13.1 | 0.213 / 0.175 |
| | filter 0.7 / 0.3 | 14.0 / 8.1 | 7.3 / 10.3 | 15.9 | 11.2 | 0.208 / 0.165 |
| | filter 0.85 / 0.5 | 11.5 / 6.8 | 6.0 / 8.5 | 13.7 | 10.1 | 0.204 / 0.159 |
| `lam8_2step_640` | plain | 16.3 / 9.8 | 9.6 / 10.9 | 17.5 | 10.5 | 0.202 / 0.174 |
| | filter 0.7 / 0.3 | 11.0 / 6.6 | 5.6 / 8.2 | 12.6 | 8.9 | 0.198 / 0.165 |
| `lam4_eq4_2step_640` | plain | 14.8 / 8.9 | 8.7 / 9.9 | 15.9 | 9.6 | 0.196 / 0.173 |
| | filter 0.7 / 0.3 | 10.0 / 6.0 | 5.1 / 7.5 | 11.5 | 8.2 | 0.192 / 0.165 |
| `lam4_eq8_2step_640` | plain | 11.0 / 6.9 | 6.3 / 7.6 | 12.0 | 7.6 | 0.183 / 0.174 |
| | filter 0.7 / 0.3 | 7.5 / 4.7 | 3.7 / 5.8 | 8.8 | 6.4 | 0.181 / 0.166 |
| `lam8_tex8_2step_640` | plain | 20.3 / 11.6 | 11.2 / 14.4 | 21.8 | 15.0 | 0.215 / 0.175 |
| | filter 0.7 / 0.3 | 13.9 / 7.9 | 6.6 / 10.8 | 15.9 | 12.6 | 0.209 / 0.167 |
| `lam4_2step_640` (not trained for it) | carry-over | 12.8 / 8.8 | 6.8 / 9.4 | 15.4 | 8.8 | 0.192 / 0.166 |
| `lam4_carry_2step_640` | plain | 20.7 / 12.2 | 12.5 / 13.5 | 22.1 | 12.7 | 0.214 / 0.174 |
| | **carry-over** | **16.4 / 9.5** | 8.2 / 12.3 | 18.3 | **12.5** | **0.213 / 0.176** |
| | carry-over + filter 0.7 / 0.3 | 11.5 / 6.6 | 5.0 / 9.3 | 13.8 | 10.5 | 0.209 / 0.172 |
| `lam4_carry_s05_2step_640` | carry-over from sigma 0.5 | 14.4 / 8.1 | 6.8 / 11.2 | 16.5 | 11.9 | 0.209 / 0.179 |
| | carry-over from sigma 0.5 + filter 0.7 / 0.3 | 10.2 / 5.7 | 4.3 / 8.4 | 12.7 | 10.1 | 0.205 / 0.175 |
| `lam4_carry_s05_cl8_2step_640` | carry-over from sigma 0.5 | 13.6 / 7.8 | 6.7 / 10.3 | 15.6 | 10.9 | 0.205 / 0.178 |
| | carry-over from sigma 0.5 + filter 0.7 / 0.3 | 9.6 / 5.4 | 4.2 / 7.8 | 11.9 | 9.2 | 0.201 / 0.174 |

Against VIBE's 640 px 8-step output (`eval_student.py --size 640`, DC-AE path, as in the 640 table above):

| Student | Warp | LPIPS to teacher | Sobel | Edge | CLIP dir oil / anime |
|---|---:|---:|---:|---:|---:|
| `lam4_2step_640` | 19.4 | 0.234 | 8.8 | 0.30 | 0.240 / 0.204 |
| `lam8_2step_640` | 14.9 | 0.261 | 7.0 | 0.30 | 0.233 / 0.202 |
| `lam4_eq4_2step_640` | 13.4 | 0.274 | 6.5 | 0.29 | 0.227 / 0.201 |

- Both training changes buy steadiness with texture and edit strength, and at the same rate: the line from lam 4
  through lam 8, extended to eq4's warping error, predicts eq4's Sobel, LPIPS and CLIP oil in the teacher comparison
  to within 0.1, 0.004 and 0.004. The equivariance term does not move the trade-off, and it doubles the training
  time (3.0 s/step). Weight 8 goes further along the same line for oil (CLIP 0.183). Anime is the exception: its
  CLIP score stays at 0.174 down to eq8's 6.9, a cel-shaded picture having little texture to lose.
- The texture term does move it: with lam 8 it keeps the released student's flicker (20.3 against 20.9) with more
  texture (Sobel 15.0 against 13.1; the lam line predicts 12.8 at that flicker) and the same edit strength. It spends
  the stronger temporal weight on texture instead of steadiness.
- The display filter does better than lam or the equivariance term: on the released student it takes flicker from 20.9 to 14.0 (oil) and
  keeps Sobel 11.2 and CLIP 0.208, where training to about that level (eq4, 14.8) costs Sobel 9.6 and CLIP 0.196.
  Most of the gain is in the low band (12.6 to 7.3: the token-sized colour blotches); the light high-band blend
  keeps the strokes. A stronger setting (0.85 / 0.5) reaches lam 8's filtered flicker on the released student with
  more texture and oil edit strength (Sobel 10.1 against 8.9, CLIP 0.204 against 0.198), though anime loses a little
  (0.159 against 0.165). It works on any student and stacks with them. Single frames of the 0.7 / 0.3 output look like
  the plain ones with slightly softer strokes, with no double edges on the moving figures of these clips.
- The carry-over is the first change that buys steadiness without paying for it: the trained student's carry-over
  path has lam 8's flicker (16.4 / 9.5 against 16.3 / 9.8) with the released student's texture and edit strength
  (Sobel 12.5, CLIP 0.213 / 0.176), and it runs at 47.8 fps instead of 29.6 (Speed). Its 2-step path, used for
  keyframes, is unchanged (20.7 / 12.2, CLIP 0.214 / 0.174). Without the training the carry-over steadies more but
  loses style (CLIP 0.192 / 0.166, Sobel 8.8, as in Negative results): the trained student repaints more of what it
  is handed. What it carries is the low band (12.6 to 8.2); at sigma 0.75 the carried picture is mostly noise to the
  step, so the strokes are still repainted every frame (high band 13.7 to 12.3). With the filter on top: 11.5 / 6.6
  at Sobel 10.5 and CLIP 0.209 / 0.172, the best anime edit strength of any filtered row.
- It does not drift on long runs. `blackswan` and `goat` played back and forth to 480 frames, one keyframe at the
  start: the carry-over output's distance to the 2-step output of the same frame settles after about 100 frames
  (oil 21-25, anime 25-30) and stays there; in oil its Sobel and saturation match the 2-step path's throughout,
  and frame 470 is as sharp as frame 40. In anime it is more saturated than the 2-step path (about 60 against 40 in
  HSV saturation), greener foliage and water, which then stays put.
- Starting the carried step lower (`--carry_sigma 0.5`, `lam4_carry_s05`; the step then sees the carried picture at
  equal parts signal and noise) carries more: 14.4 / 8.1, for a little oil texture and edit strength (Sobel 11.9,
  CLIP 0.209) and a little more anime edit strength (0.179). Most of the extra is again in the low band; the strokes
  are still largely repainted (high band 11.2). With the filter it reaches 10.2 / 5.7 at CLIP 0.205 / 0.175, where
  eq4 with the filter has 10.0 / 6.0 at 0.192 / 0.165. The long-run check behaves as at sigma 0.75 (plateau after
  80 frames, nothing degrades). Where the flow fails, as on rippling water, the frame's own colours come through
  the mix: `blackswan`'s pond stays green in anime, where the 2-step path paints it pale blue.
- Raising the carried step's temporal weight to 8 so that it keeps the strokes too (`--carry_lam 8`, default lam;
  `lam4_carry_s05_cl8`) lowers the high band from 11.2 to 10.3, but by smoothing the strokes rather than holding
  them: Sobel 11.9 to 10.9 and CLIP oil 0.209 to 0.205 for 14.4 to 13.6. That is a worse exchange than the filter's
  (on `lam4_carry_s05` the filter takes 4.2 off the flicker for 1.9 of Sobel; the weight takes 0.9 for 1.0). In the
  long-run check its carry-over output on `blackswan` oil has less texture than the 2-step path (Sobel 9.4-9.9
  against 10.8-11.7, where `lam4_carry_s05` matched it), visibly smoother water; it does not drift either.
  `lam4_carry_s05` with the filter remains the best setting found.

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
# ... with the display-side temporal filter and / or the carry-over (any of these scripts, including make_demo.py);
# with an engine, --carry_sigma must repeat a non-default value the student was trained with
python scripts/stream_video.py --input ... --engine ... --temporal_filter 0.7 0.3 --carry
# Flicker of a student, plain and filtered, and the shift probe; --table compares result files
python scripts/eval_flicker.py --student runs/lam4_2step_640/lora.safetensors --out out/flicker/lam4_2step_640.json
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
# ... trained for the carry-over as well (~71 min at 2.8 s/step)
python scripts/train_student.py --data data/teacher_640 --run runs/lam4_carry_2step_640 --student_steps 2 --lam 4 \
    --w_carry 1 --grad_ckpt
python scripts/eval_flicker.py --student runs/lam4_carry_2step_640/lora.safetensors --carry \
    --out out/flicker/lam4_carry_2step_640_carry.json
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
  - `temporal_filter.py`: display-side two-band temporal filter
  - `carry.py`: carry-over of the previous output into the next batch's start
  - `metrics.py`: video I/O, warping error, flicker split, Sobel, edge agreement, LPIPS, CLIP scores
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
- **Init noise that moves with the content** (no retraining; `lam4_2step_640` engine, the six clips and metrics of
  "Flicker" above). A noise canvas on a grid eight times finer than the latent tokens holds the student's noise at
  offset 0, so that every offset by whole cells is again i.i.d. N(0, 1) per token; the window followed the global
  motion of the input (median flow, accumulated). Flicker went up instead of down (valid-pixel change oil 20.8 to
  23.1, anime 12.1 to 12.6). Sub-token offsets give every token new noise, and new noise per frame is a new picture.
  It is not that the student only knows its own noise: another fixed draw scores like it (20.4 / 11.2).
- **Run-time anchoring** (no retraining, same setting). Each variant cut flicker only by giving up style or texture:

  | Variant | Valid change oil / anime | CLIP dir oil / anime | Sobel oil |
  |---|---:|---:|---:|
  | 2 steps from the shared noise (as shipped) | 20.8 / 12.1 | 0.213 / 0.175 | 13.1 |
  | Step 2 only, from the input frame's latent re-noised to step 2's sigma | 9.4 / 6.8 | 0.157 / 0.128 | 7.1 |
  | Step 2 only, from the previous output warped by the input flow | 12.7 / 8.8 | 0.192 / 0.166 | 8.8 |
  | Both steps, step 1's clean estimate replaced by the warped previous output where the flow holds | 18.2 / 11.6 | 0.206 / 0.175 | 11.5 |
  | Full-band blend of the output with the warped previous output, 50/50 | 12.3 / 7.3 | 0.203 / 0.163 | 9.7 |

  The two-band display filter ("Flicker" above: 14.0 / 8.1, Sobel 11.2, CLIP 0.208 / 0.165) keeps more texture than
  the full-band blend for slightly more flicker. The third row, with a student trained for it, became the carry-over
  ("Flicker" above), which keeps the style.

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
