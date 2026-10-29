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
and about the same edit strength; with the display filter on top, the flicker is half the 2-step path's at 44.1 fps
at 640 and 31.1 fps at 768 ("Flicker"). One trained with a colour-mapped fill also keeps the 2-step path's colours,
at the same speed at 768 and a few percent slower at 640. With taesana as TensorRT engines, the transformer as an INT8
engine and its cross-attention K / V computed once per batch, that path (carry-over + filter) runs at 49.7 fps at 768
and 67.4 fps at 640 ("Faster"). A student trained further on 14 instructions and on windows with twice the motion
(`multi_carry_s05_lut_2step_768`, the default at 768 since) follows the 8-step teacher more closely, also on
instructions it never saw, no longer loses edit strength on the carry-over path, and with a stronger filter
(0.85 / 0.5) flickers less than the previous setting at the same speed ("Many instructions"). The numbers below say
what has been measured and how.

## Demo

https://github.com/user-attachments/assets/c6173d79-383f-47d2-982b-764050744838

Input | oil painting | anime on five DAVIS-2017 clips through the real-time path at 768 px on an RTX 3090 Ti: the
student trained for the carry-over with the colour-mapped fill (`lam4_carry_s05_lut_2step_768`) with the transformer
as an INT8 TensorRT engine, taesana as TensorRT engines, the carry-over and the temporal filter 0.7 / 0.3
(`scripts/make_demo.py --tae_engine ... --carry --temporal_filter 0.7 0.3`; see "Flicker" and Speed, "Faster"). Each
output panel shows the loop fps of its run: 48-50 fps on these short clips, 49.7 fps on the 480-frame loop. The
default at 768 is now `multi_carry_s05_lut_2step_768` with the filter at 0.85 / 0.5 ("Many instructions"); the videos
show the previous one.

The same student and engine on the plain 2-step path (no carry-over, no filter: about twice the flicker; 30-32 fps):

https://github.com/user-attachments/assets/49b755a7-33bc-41a8-9291-0d36e390c07c

Input clips: [DAVIS-2017](https://davischallenge.org/), see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

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
carry-over's flows are hidden the same way. The engine of `lam4_carry_s05_2step_640` (Flicker; `build_trt.py` copies
the student's `carry_sigma` 0.5 into the spec, which `StreamEditor` reads) runs the same loop at 29.6, 47.8 and
44.1 fps (2 steps, carry-over, carry-over + filter). Its output differs from PyTorch bf16 by 0.81 in mean absolute
value on the carried step (`blackswan`, oil, 40 frames) and by 1.16 on the 2-step path. At 768 the engine of
`lam4_carry_s05_2step_768` (velocity within 1.3-1.9 % of bf16 PyTorch, 21.3 ms per frame per step) runs it at 20.1,
32.4 and 31.1 fps, and that of `lam4_carry_s05_lut_2step_768` (colour-mapped fill) at 20.7, 32.4 and 31.1. At 640
the engine of `lam4_carry_s05_lut_2step_640`, timed alternately with that of `lam4_carry_s05_2step_640` (two rounds
each; the machine was busier than for the numbers above, and the latter ran 28.6, 43.2 and 41.2), runs at 27.8, 41.3
and 39.0: 3-5 % slower, of which 3 % already on the 2-step path, where the fill plays no part.

**Faster: taesana in TensorRT and an INT8 transformer.** An Nsight Systems profile of the 768 carry-over + filter
loop put over a third of the GPU time outside the transformer engine, most of it in taesana: in PyTorch its convolutions
accumulate in fp32 (half rate on GeForce Ampere) and every ReLU, residual add and upsample is a kernel of its own.
Four changes, each timed alternately against the one before (`stream_video.py`, `blackswan` cycled to 480 frames,
`oil`, the engines of `lam4_carry_s05_lut_2step_768` / `_640`, loop fps, two to five rounds each):

| Loop fps | 768, 2 steps | 768, carry-over + filter | 640, 2 steps | 640, carry-over + filter |
|---|---:|---:|---:|---:|
| taesana in PyTorch | 19.9 | 29.9 | 28.0 | 42.4 |
| taesana as TensorRT engines (`--tae_engine`) | 22.2 | 37.5 | 31.5 | 50.9 |
| + the filter's and the carry-over's warps batched | | 39.0 | | 52.8 |
| + INT8 transformer engine (`build_trt.py --int8`) | 29.9 | 46.3 | 39.9 | 63.6 |
| + cross-attention K / V once per call, not per frame | 31.0 | 49.7 | 43.0 | 67.4 |

- taesana as two strongly typed fp16 engines with the uint8 conversion and the layout changes inside
  (`scripts/build_trt_tae.py`, `vibe_stream.trt_tae`): encode 0.63 to 0.27 and decode 8.73 to 3.68 ms per frame at
  768 (0.50 to 0.21 and 6.02 to 2.43 at 640), decode 55 dB PSNR from PyTorch's; warping error, Sobel and edge
  agreement of the output unchanged.
- The temporal filter warps the whole batch's occlusion weights at once and both bands in one call per frame, the
  carry-over its previous output and input in one call (`temporal_loss.flow_grid`): identical output, the filter
  from 16.5 to 8.3 ms per batch of 8 at 768.
- The INT8 engine (`vibe_stream.trt_int8`, NVIDIA ModelOpt; weights per channel, activations per tensor, static)
  runs a step in 15.8 ms per frame at 768 against 21.2-22.0 for fp16 timed alternately (640: 12.2), and is 1.73 GB
  against 2.98. Its velocity is within 2.8-4.9 % of bf16 PyTorch (fp16 engine: 1.2-1.7 %). Through the streaming
  editor on the four evaluation clips x `oil` / `anime`, its output is at LPIPS 0.015 (2 steps) and 0.008
  (carry-over + filter) from the fp16 engine's, with CLIP direction (0.208 / 0.198 against 0.209 / 0.198), flicker
  (16.41 / 7.13 against 16.41 / 7.14) and warping error unchanged; side by side the frames are indistinguishable.
  At 640: LPIPS 0.014 / 0.008, CLIP direction 0.201 / 0.195 for both, flicker 17.32 / 7.15 against 17.29 / 7.13.
  The reference-style student `refs_carry_s05_lut_2step_768` (Results, "Other instructions and style from a
  reference image") is calibrated on its reference prompts with 40 of its training references handed to the encoder
  (`--calib_refs`; the prompts' text would be meaningless to it). Velocity within 2.6-4.7 % of bf16, 16.0 ms per
  frame per step. On the four clips x `oil` / `anime` and x four held-out references (image way): LPIPS to the fp16
  engine 0.013 / 0.008 (2 steps / carry-over + filter) for the instructions and 0.020 / 0.012 for the references;
  CLIP direction 0.211 / 0.200 against 0.210 / 0.199, CLIP similarity gain to the reference 0.149 / 0.116 against
  0.151 / 0.116, flicker and warping error unchanged. Loop fps with Van Gogh's *Starry Night* as the reference (same
  setup as the table, two rounds, before the cross-attention change below): image way 21.7-22.8 to 27.9-29.6 (2 steps) and 36.3-38.2 to 44.6-47.2 (carry-over
  + filter); `--style_way both` runs as fast as the image way (36.3-38.2 to 44.5-44.6).
  What it took: ModelOpt's SmoothQuant smooths Linear layers only, and the feed-forward's 1x1 convolutions carry the
  worst activation outliers (`conv_point` quantized alone: 12-19 % velocity error; all layers: 13-20 %), so
  `trt_int8` smooths those too, folding the scale into the depthwise convolution before them where it can; the
  depthwise convolutions, the cross-attention key / value projections and the embeddings stay fp16; Q/DQ are fp16
  (with ModelOpt's fp32 default the layers between them ran in fp32). Calibration: 75 transformer calls as the
  streaming editor makes them, from 10 training clips. Writing the 1x1 convolutions as Linear layers, to avoid
  TensorRT's interleaved INT8 layout and its permutes, was slower and less accurate.
- Every frame of a batch carries the same instruction, yet each of the 20 cross-attention layers projected the
  224-token context to K / V once per frame, in fp16 even in the INT8 engine. Engines are now exported with K / V
  projected from the first row only and the batch's queries attending to them as one sequence
  (`trt_engine.SharedContextAttnProcessor`). Exact: the PyTorch fp16 velocity's error to bf16 is unchanged to four
  digits. A step takes 18.7-19.1 instead of 21.3-21.5 ms per frame at 768 with the fp16 engine (640: 14.0 instead of
  16.0-16.1) and 14.8-15.0 instead of 15.7-16.1 with INT8 (640: 11.1-11.2 instead of 12.0-12.1), timed alternately.
  In the same loop runs the INT8 engines without it ran 29.1-29.5, 47.3-47.5, 39.4-39.9 and 61.3-64.1 fps. The new
  INT8 engine's velocity is within 2.8 / 5.2 % of bf16, and its output is at LPIPS 0.016 / 0.008 from the fp16
  engine's (before: 0.015 / 0.008), CLIP direction 0.209 / 0.198 and flicker 16.52 / 7.18 against 16.41 / 7.14.
  Engines built before keep working (`EngineSpec.shared_context` false); one built with it reads only the first row
  of its context input.

After these, the transformer is 80 % of the loop's GPU time at 768. The profile also answered whether lifting
GeForce Ampere's half-rate fp32 accumulation (e.g. with the gpu-unlock tool) would help: the engines already pick
fp16-accumulating kernels, so it would have added 8-10 % to the loop before taesana moved to TensorRT and next to
nothing after.

**Delay with a live source.** Offline, a batch's host work (the DIS flows of the filter and the carry-over, about
15 ms per frame at 768) overlaps the previous batch's GPU work. A live batch can only be staged once its last frame
has arrived, and staged the same way that host work sat between the last frame and the GPU: 117 + 169 ms per batch
of 8 at 768, more than the 267 ms a 30 fps camera leaves, so the delay grew without bound at every batch size.
`vibe_stream.live.LiveEditor` computes each frame's flows on a thread as the frame arrives (`TemporalFilter.host_flow`,
`Carry.host_flow`) and hands the batch to the GPU as soon as it is complete; with the batch-8 engines its output is
identical to `StreamEditor.run`'s bit for bit. Delay from a frame's hand-over to its edited copy back on the host,
`blackswan` handed over at 30 fps in wall-clock time (`scripts/measure_latency.py`), INT8 engines of
`lam4_carry_s05_lut_2step_768` / `_640` with carry-over + filter 0.7 / 0.3 (the default student's engines are the same
network at the same speed, "Many instructions"), mean / 95th percentile, two rounds:

| Batch | 768: delay | 768: offline loop fps | 640: delay | 640: offline loop fps |
|---|---:|---:|---:|---:|
| 8 | 298 / 415 ms | 50.0 | 250 / 368 ms | 63.7 |
| 4 | 150 / 201 ms | 46.3 | 128 / 178 ms | 53.5 |
| 2 | 80 / 98 ms | 35.8 | 68 / 87 ms | 40.8 |
| 1 | falls behind | 26 | | |

A frame waits for the rest of its batch, up to (batch - 1) / 30 s, and then for the batch itself, so batch 2 is the
shortest delay that still keeps up with 30 fps, and batch 4 keeps more headroom (46 fps at 768) for a busier GPU.
Smaller batches carry the previous output over a shorter distance: warping error 9.93 / 9.57 / 9.57 at 768 and
9.52 / 9.13 / 9.21 at 640 (batch 8 / 4 / 2), Sobel a little lower (8.98 / 8.90 / 8.82 and 9.21 / 8.98 / 8.95).
The INT8 engines for batch 4 and 2 are not bit-reproducible from run to run (TensorRT's kernels for their smaller
matrix products sum in a varying order: now and then one call differs by up to about 0.1 in velocity, which fades
over the following batches); the batch-8 engines are. Checking this turned up a second source of run-to-run
variation, now removed: the colour map of the LUT fill was summed with atomic adds (`Carry`'s `fit_colour_map`).

### Flicker (streaming path)

`scripts/eval_flicker.py`: DAVIS `blackswan`, `parkour`, `dance-twirl`, `camel`, `goat` and `libby`, first 40 frames,
prompts `oil` and `anime`, through the streaming editor with taesana (PyTorch bf16), at 640 except for the last table
of the section. *Valid* is the frame-to-frame change of the output after warping by the input's flow, counted only
where the same warp reproduces the input (`metrics.flicker`; the input itself scores about 2.5): what remains is
change the edit added. *Low* / *high* split it at a Gaussian of sigma 4 px. DreamLite-mobile's lean path scores
about 12 for both prompts on the same clips (measured on dreamlite-stream's 768 demo).

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
flow, `Carry.cut_below`); the runs below have no other keyframes. With `--carry_fill lut` what fills in where the
flow does not hold is the frame passed through a colour map fitted at the keyframe instead of the frame itself (last
point below).

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
| `lam4_carry_s035_2step_640` | carry-over from sigma 0.35 | 13.4 / 7.7 | 6.5 / 10.3 | 15.6 | 11.4 | 0.205 / 0.180 |
| | carry-over from sigma 0.35 + filter 0.7 / 0.3 | 9.5 / 5.4 | 4.1 / 7.8 | 12.0 | 9.7 | 0.201 / 0.175 |
| `lam4_carry_s05_cl8_2step_640` | carry-over from sigma 0.5 | 13.6 / 7.8 | 6.7 / 10.3 | 15.6 | 10.9 | 0.205 / 0.178 |
| | carry-over from sigma 0.5 + filter 0.7 / 0.3 | 9.6 / 5.4 | 4.2 / 7.8 | 11.9 | 9.2 | 0.201 / 0.174 |
| `lam4_carry_s05_lut_2step_640` | carry-over from sigma 0.5, colour-mapped fill | 13.8 / 7.5 | 6.5 / 10.7 | 15.8 | 11.3 | 0.210 / 0.179 |
| | the same + filter 0.7 / 0.3 | **9.8 / 5.3** | 4.1 / 8.1 | 12.2 | 9.6 | **0.207 / 0.174** |

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
  80 frames, nothing degrades). In anime, `blackswan`'s pond is green (the source's colour) by frame 40 and stays
  so, where the 2-step path paints it pale blue. This is not the flow failing on the pond: the flow check passes as
  much of the pond as of the rest of the picture (mean w 0.55 in the lower half of the frame, 0.53 overall), and a
  looser check (`eval_flicker.py --carry_tau 24`, 48, against 12) leaves the pond green while costing edit strength
  (CLIP 0.208 / 0.175 and 0.201 / 0.170) without changing flicker (14.6 / 8.0, 14.8 / 7.9); at 48 the moving figure
  in `parkour` leaves ghosts. The colour comes in through the mix itself (last point).
- Lower still (`--carry_sigma 0.35`) the gain halves: 13.4 / 7.7, with Sobel 11.9 to 11.4 and CLIP oil 0.209 to
  0.205 (anime 0.180). Texture goes at about the rate the filter spends it, oil edit strength faster, so the sigma
  series has levelled off at 0.5. The long-run check behaves as before.
- Raising the carried step's temporal weight to 8 so that it keeps the strokes too (`--carry_lam 8`, default lam;
  `lam4_carry_s05_cl8`) lowers the high band from 11.2 to 10.3, but by smoothing the strokes rather than holding
  them: Sobel 11.9 to 10.9 and CLIP oil 0.209 to 0.205 for 14.4 to 13.6. That is a worse exchange than the filter's
  (on `lam4_carry_s05` the filter takes 4.2 off the flicker for 1.9 of Sobel; the weight takes 0.9 for 1.0). In the
  long-run check its carry-over output on `blackswan` oil has less texture than the 2-step path (Sobel 9.4-9.9
  against 10.8-11.7, where `lam4_carry_s05` matched it), visibly smoother water; it does not drift either.
- The source's colour comes in through the mix. With w about 0.5 everywhere, half of every carried start is the
  frame's own latent, and one step from sigma 0.5 does not repaint large colour areas: where the 2-step path
  recolours (the pond, `goat`'s hillside in anime), the carried output moves towards the source over the first
  batches and then carries that colour on. `Carry(fill="lut")` (`--carry_fill lut`) fills with the frame passed
  through a colour map instead: fitted on each keyframe batch, for each input colour (16³ bins, frames blurred by
  4 px) the mean change the 2-step path made to it, shrunk towards the overall change where a colour is rare.
  Applied at run time to `lam4_carry_s05` it fixes the saturation but not the colour: the low-band distance to the
  2-step output of the same frames (blur 16 px, six clips, after the keyframe batch) goes from 12.5 to 12.9, and oil
  is oversaturated (`libby` 138 against the 2-step path's 124). Trained with it (`lam4_carry_s05_lut`; the map is
  fitted on the student's own 2-step output of the chain's first frame) it works: distance 8.5, `blackswan` anime
  27 to 12 (the pond pale green instead of green), saturation within 2 of the 2-step path's, and slightly less
  flicker (13.8 / 7.5, with the filter 9.8 / 5.3) at the same edit strength (CLIP 0.210 / 0.179), for some texture
  (Sobel 11.9 to 11.3). In the long-run check `blackswan` anime settles at 19-20 from the 2-step output instead of
  36-37, `goat` anime at 14 instead of 23-27, and in oil the saturation now matches the 2-step path's (113 against
  111, where `lam4_carry_s05` had 123); nothing drifts. `lam4_carry_s05_lut` with the filter is the best setting
  found.

**At 768.** The released 768 student `lam4_2step`, and `lam4_carry_s05_2step_768` and `lam4_carry_s05_lut_2step_768`
trained as `lam4_carry_s05` and `lam4_carry_s05_lut` at 640 (84 and 90 min at 3.3 and 3.6 s/step):

| Student | Path / display | Valid oil / anime | Low / high (oil) | Warp oil | Sobel oil | CLIP dir oil / anime |
|---|---|---:|---:|---:|---:|---:|
| `lam4_2step` (released) | plain | 21.4 / 10.3 | 13.1 / 13.8 | 23.4 | 13.0 | 0.232 / 0.181 |
| | filter 0.7 / 0.3 | 14.5 / 6.9 | 7.7 / 10.5 | 17.0 | 11.1 | 0.219 / 0.175 |
| `lam4_carry_s05_2step_768` | carry-over from sigma 0.5 | 14.9 / 7.8 | 7.0 / 11.6 | 17.7 | 11.9 | 0.214 / 0.187 |
| | carry-over from sigma 0.5 + filter 0.7 / 0.3 | 10.8 / 5.6 | 4.6 / 8.9 | 13.9 | 10.1 | 0.205 / 0.183 |
| `lam4_carry_s05_lut_2step_768` | carry-over from sigma 0.5, colour-mapped fill | 13.9 / 6.7 | 6.6 / 10.8 | 16.7 | 11.0 | 0.216 / 0.184 |
| | the same + filter 0.7 / 0.3 | **10.1 / 4.9** | 4.3 / 8.2 | 13.1 | 9.3 | **0.206 / 0.179** |

- The carry-over does at 768 what it does at 640: with the filter it halves the flicker (21.4 / 10.3 to 10.8 / 5.6)
  and the loop runs at 31.1 fps instead of 20.1 (Speed). Oil pays more edit strength than at 640: the 2-step path's
  CLIP 0.232, higher than at 640, comes down to 0.214 with the carry-over and 0.205 with the filter, where the 640
  rows end too. Anime gains (0.181 to 0.187).
- The long-run check behaves as at 640: plateau after 80 frames, oil Sobel equal to the 2-step path's throughout. The
  anime colour difference is larger. The 768 2-step path paints `goat` paler than the 640 one (HSV saturation 17-21
  against 36), the carry-over output keeps 47-50, and its distance to the 2-step output settles at 38-41 (640: 23-27).
- The colour-mapped fill works as at 640 (`lam4_carry_s05_lut_2step_768`): low-band colour distance 13.5 to 8.9,
  saturation excess +4.8 to +1.1. In the long-run check `goat` anime settles at 13 from the 2-step output instead of
  38-41, with the 2-step path's pale colours (saturation 14-18 against 15-19), and `blackswan` anime at 17 instead
  of 35-37. Flicker goes down a little (with the filter 10.1 / 4.9), edit strength stays (CLIP 0.206 / 0.179 with
  the filter), texture goes down a little (Sobel 10.1 to 9.3 with the filter), and the loop speed is unchanged
  (Speed).

**Brush strokes and resolution (oil).** Oil strokes come out finer the higher the resolution, and seen as a whole the
frame reads less like a painting (DreamLite does the same). A probe on the 8-step teacher at 768 (`blackswan`,
`goat`, `parkour`, `camel`, frame 40, judged by eye; a mid / high band-energy ratio did not separate the variants)
says VIBE paints strokes of a roughly fixed size in pixels. Three prompts asking for larger strokes (thick impasto,
broad visible brushstrokes, palette knife) and an input softened to 384 or 256 px and scaled back change neither the
stroke size nor much else. Only the working resolution does: the 640 output scaled up to 768 has bolder strokes and
colours, and the 512 one bolder still, but it repaints the scene (an orange camel, a red face). In the demo, though,
most of the brushwork goes with the carry-over. In a 2 x 2 oil comparison on the engine of
`lam4_carry_s05_lut_2step_768` (2 steps, + filter, carry-over, carry-over + filter) the 2-step path keeps the
teacher's strokes (dabs on the water, strokes on the walls), the carried output keeps the edges and smooths the
surfaces, and the filter changes little either way. On video the 2-step path's flicker weighs more than the lost
strokes, so carry-over + filter stays the setting.

### Other instructions and style from a reference image

**Prompts the student never saw.** `lam4_carry_s05_lut_2step_768` was distilled on the two style prompts only. On its
2-step path (the four evaluation clips x 8 frames, one embedding per clip) it keeps up with the 8-step teacher on new
ones: CLIP directional similarity 0.86-1.24 of the teacher's on seven styles (watercolour, pencil sketch, ukiyo-e,
cyberpunk, pixel art, stained glass, claymation) and 0.96-1.00 on three global edits (winter, night, sunset), at LPIPS
0.13-0.37 to the teacher, where the base model run as a 2-step student, blurred and washed out, is at 0.18-0.66. The
LoRA has learned to sample in two steps more than it has learned two styles. It falls short where the teacher
rebuilds the scene: "built from LEGO bricks" (0.80: the teacher turns a person into a minifigure, the student
recolours the shirt), cyberpunk (0.86: no neon signs), claymation and pencil sketch (the teacher drops the background,
the student keeps it, which is where its ratios above 1 come from). On the streaming path (carry-over + filter, five
clips) the flicker of the new prompts falls to 0.36-0.64 of the 2-step path's, as for oil and anime (0.46 / 0.47),
and ends at or below oil's.

**Style from a reference image** (`stream_video.py --style_image`, `make_demo.py --style_images`). Three ways compared
on nine public-domain works, by the CLIP image similarity to the reference gained over the input (teacher / student,
three clips): VIBE's own Qwen3-VL writes an instruction from the reference without naming artist or work
(`VibeModel.style_prompt`), 0.081 / 0.087; the reference goes to the encoder next to the frame, "transfer this to
the artistic style of the second image" (`VibeModel.embed_with_reference`), 0.141 / 0.120; the artist and title as
the prompt, 0.101 / 0.095. The second scores highest partly by copying content: Klimt's couple, Mucha's model,
Beardsley's figure, Hokusai's wave and a poster's lettering appear in the picture (as ghosts in the student). Cutting
the reference into shuffled 4 x 4 or 8 x 8 tiles first removes the content and most of the style with it (student
0.097 / 0.080; the teacher paints the tile seams), and adding the tiles to the written instruction changes little.
The options use the written instruction. It is made once per clip, so the loop runs as fast as with any prompt (31
fps at 768 with carry-over + filter). It works for styles with a clear medium or palette (Van Gogh, Hokusai, Mucha,
Beardsley, Kandinsky) and not where VIBE cannot paint the style over the kept layout: Klimt's ornament comes out as a
gold tint, and Monet and Seurat barely register, with the artist named too.

On the streaming path, styles carried by texture fade after the keyframe batch: the reference gain of Van Gogh,
Kandinsky and Klimt drops by a quarter to a third once the carry-over starts and then stays flat, while palette
styles (Hokusai's blue, Mucha's sepia, Beardsley's black and white) strengthen. Oil and anime, which the carried step
was trained on, do not fade. A keyframe every 4 batches brings the later batches most of the way back (Van Gogh
0.061 to 0.096; 2-step path 0.115) at 27.8 fps for 16-22 % more flicker, but the style visibly pulses with the
keyframes; every 2 batches adds little more. Starting the carried step from sigma 0.6 instead of the trained 0.5
holds a little more (0.076) for 6-10 % more flicker, without pulsing. Training the carried step on more instructions
removes the fading for written instructions ("Many instructions" below); the reference styles have not been measured
with that student.

**Fine-tuning the base model for the reference image.** To keep the image way's style without its content, VIBE
itself was fine-tuned on its own selected outputs. Data (`make_style_data.py`): 2000 Open Images photos x 3 of 484
public-domain references (475 from the Art Institute of Chicago, the nine above), 8-step candidates under the
written instruction and under the reference image, two seeds each (4.3 h at 768); `select_style_targets.py` keeps
the candidate with the highest reference gain among those that keep the photo's edges (edge agreement >= 0.25) and
content (CLIP similarity to the photo 0.62-0.95), change it (LPIPS >= 0.25) and move towards the reference (gain >=
0.05): 2947 of 6000 pairs. `train_style_ft.py`: a rank-32 LoRA on the transformer, flow matching, conditioned as at
run time (instruction "transfer this to the artistic style of the second image" + reference image), 3000 steps (28
min), 25 references held out (the nine famous works and 16 others). `eval_style_ft.py` on the 298 held-out pairs,
8 steps, base -> fine-tuned: share of outputs passing all four criteria 0.24 -> 0.34 (famous works 0.16 -> 0.34),
edge agreement 0.28 -> 0.40, reference gain 0.085 -> 0.066 (the copied content is gone: Beardsley's and Mucha's
figures, Hokusai's wave in place of the subject); plain instructions barely move (LPIPS 0.12-0.21 to the base
model's output, CLIP direction for `oil` 0.220 -> 0.209). The streaming student was then trained again with four
references per window from the fine-tuned teacher besides `oil` and `anime` (`make_teacher_data.py --style_list
--base_lora`, `train_student.py --ref_min_edge 0.25 --init` the LUT carry student, 1500 steps, 87 min):
`refs_carry_s05_lut_2step_768`, as fast as before, with `oil` / `anime` kept (CLIP direction 0.209 / 0.176, filtered
flicker 13.7 / 7.9 against 0.206 / 0.179 and 13.1 / 7.6). On six held-out famous works through the streaming path,
the image way now gains about as much as before with fewer leaks, the written instruction gains at least as much on
five of six, and both together (`--style_way both`: the written instruction with the reference image next to the
frame) gain most, 0.075 against 0.071 (image) and 0.065 (instruction), and look the most balanced: Kandinsky as
vivid as with the instruction, Hokusai's subject kept. The embedding has a fixed length, so the way costs nothing per
frame. A second round of the same loop on the fine-tuned model's outputs did not help (Negative results).

### Many instructions

`multi_carry_s05_lut_2step_768`, the default at 768: `lam4_carry_s05_lut_2step_768` trained further on 14
instructions instead of two (`oil`, `anime` and the first twelve of `presets.MORE_PROMPTS`: nine styles and three
global edits) and on windows of every second frame besides the consecutive ones (`make_teacher_data.py --stride 2`:
twice the motion between frames, where the carry-over lost edit strength). Teacher data: 47 + 45 windows of 10 frames
(44 + 51 min, 29 GB; above 8 GB of windows `train_student.py` reads the teacher images from disk at each use), then
2500 steps from the LUT student (2.8 h at 4.0 s/step). Four instructions are kept out of training
(`presets.HELD_OUT`: gouache, crayon, autumn, fog).

`scripts/eval_stream.py`: the four evaluation clips played forward then backward (96 frames, no cut), seven of the
trained instructions (oil, anime, watercolour, pencil sketch, ukiyo-e, winter, night) and the four held out, INT8
engines and taesana engines at 768. *Worst batch*: the CLIP direction of the batch of 8 frames that scored lowest in
a clip, averaged over clips and instructions; it shows where the carry-over fades in fast motion. *LPIPS to teacher*:
the 2-step path against the 8-step teacher on frames 0-28, where the teacher's CLIP direction is 0.199 / 0.167.
Values for trained / held-out instructions:

| Student | Path / display | CLIP dir | Worst batch | Valid | Warp | Sobel | LPIPS to teacher |
|---|---|---:|---:|---:|---:|---:|---:|
| `lam4_carry_s05_lut_2step_768` | 2 steps | 0.189 / 0.174 | 0.159 / 0.146 | 12.82 / 10.87 | 15.25 / 13.73 | 7.52 / 5.75 | 0.266 / 0.231 |
| | carry-over + filter 0.7 / 0.3 | 0.180 / 0.162 | 0.146 / 0.130 | 5.76 / 5.10 | 8.75 / 8.21 | 5.70 / 4.74 | |
| | carry-over + filter 0.85 / 0.5 | 0.177 / 0.161 | 0.144 / 0.130 | **4.91 / 4.35** | 7.96 / 7.52 | 5.21 / 4.34 | |
| `multi_carry_s05_lut_2step_768` | 2 steps | 0.187 / 0.169 | 0.158 / 0.142 | 11.94 / 10.37 | 14.04 / 12.94 | 6.46 / 4.98 | **0.229 / 0.196** |
| | carry-over + filter 0.7 / 0.3 | **0.190 / 0.168** | **0.159 / 0.134** | 6.43 / 5.26 | 9.38 / 8.30 | 6.19 / 4.49 | |
| | carry-over + filter 0.85 / 0.5 (default) | 0.188 / 0.167 | 0.158 / 0.134 | 5.49 / 4.51 | 8.50 / 7.60 | 5.66 / 4.13 | |

- The 2-step path comes closer to the teacher, on held-out instructions as much as on trained ones (LPIPS 0.266 to
  0.229 and 0.231 to 0.196), most for pencil sketch (0.279 to 0.203), night (0.265 to 0.187) and crayon (0.294 to
  0.234). Its CLIP direction does not move on average; where it drops, the old student had gone past the teacher
  (pencil sketch 0.217 and crayon 0.187 against the teacher's 0.178 and 0.151: it keeps the background the teacher
  drops, see "Prompts the student never saw").
- The carry-over no longer fades. With the filter at 0.7 / 0.3 the carry-over path scores as high as the 2-step path
  (0.190 against 0.187; before 0.180 against 0.189), and the worst batch rises from 0.146 to 0.159. In `parkour`, the
  fastest clip, it rises for 9 of 11 instructions (mean 0.109 to 0.126; pencil sketch 0.11 to 0.18). Ukiyo-e and
  night, which faded most, gain most (0.130 to 0.148 and 0.120 to 0.152).
- More of the style is kept, and more of it moves: at the same filter the new student flickers 12 % more on trained
  instructions (6.43 against 5.76; held out 3 %), in both bands, with 9 % more Sobel. The filter costs little edit
  strength (0.190 to 0.188 from 0.7 / 0.3 to 0.85 / 0.5), so at 0.85 / 0.5 the new student flickers less than the old
  one at 0.7 / 0.3 (5.49 against 5.76, held out 4.51 against 5.10) with more edit strength (0.188 against 0.180,
  worst batch 0.158 against 0.146) and the same texture (Sobel 5.66 against 5.70). That is the default. The old
  student at 0.85 / 0.5 flickers least (4.91) with the least edit strength and texture. Strengthening the low band
  alone (`--filters 0.85 0.3`) barely helps (6.33): the extra flicker is in the strokes. On the frames of largest
  motion in `parkour` and `dance-twirl`, 0.85 / 0.5 shows no double edges, and the old student is visibly paler there
  than the new one.
- Same network, same speed: both INT8 engines run at 14.9 ms per frame per step at batch 8 (build-time bench), and
  the engines for batch 4 and 2 (live input, "Delay with a live source") quantize at least as well as the one for batch 8
  (velocity error 3.1 / 5.2-5.3 % against 3.1 / 6.2 %). INT8 against fp16 on 18 clip x instruction pairs of `parkour`
  and `libby`, carry-over + filter 0.7 / 0.3 (LPIPS on the 2-step path): CLIP direction
  0.177 / 0.178, flicker 6.59 / 6.61, LPIPS to teacher 0.219 / 0.217.

## Setup

The code expects an upstream VIBE checkout next to it (or `VIBE_REPO` pointing at one) and the weights in `models/`:

```bash
git clone https://github.com/ai-forever/vibe ../vibe          # tested at 7f0f01f
python scripts/download_weights.py                           # VIBE DistilledCFG (8.9 GB) + taesana
```

Python 3.10 with torch (CUDA), transformers >= 4.57 (Qwen3-VL), diffusers, safetensors, huggingface_hub,
opencv-python, lpips, and for the engine tensorrt 10.x and onnx (see `requirements.txt`). VIBE's own dependencies
beyond these are not needed; when `loguru` is missing a silent stand-in is used. The INT8 engine also needs
`nvidia-modelopt` (tested with 0.39.0; later releases require a newer torch than the one TensorRT 10.16 was used with
here, 2.6).

Video clips are not tracked; see [assets/README.md](assets/README.md).

## Usage

```bash
# Stream a clip through the 2-step student with the TensorRT engine
python scripts/build_trt.py --student runs/lam4_2step/lora.safetensors \
    --engine out/trt/vibe_lam4_2step_768_b8.engine
python scripts/stream_video.py --input assets/davis_mp4/blackswan.mp4 --prompt oil \
    --engine out/trt/vibe_lam4_2step_768_b8.engine --output out/stream/blackswan_oil.mp4 --compare
# ... faster: taesana as TensorRT engines (once per size and batch) and the transformer in INT8
python scripts/build_trt_tae.py --size 768 --prefix out/trt/taesana_768_b8
python scripts/build_trt.py --student runs/lam4_2step/lora.safetensors \
    --engine out/trt/vibe_lam4_2step_768_b8_int8.engine --int8
python scripts/stream_video.py --input ... --engine out/trt/vibe_lam4_2step_768_b8_int8.engine \
    --tae_engine out/trt/taesana_768_b8
# ... the default at 768: the multi-instruction student (trained below), carry-over and the filter at 0.85 / 0.5
python scripts/build_trt.py --student runs/multi_carry_s05_lut_2step_768/lora.safetensors \
    --engine out/trt/vibe_multi_carry_s05_lut_2step_768_b8_int8.engine --int8
python scripts/stream_video.py --input ... --engine out/trt/vibe_multi_carry_s05_lut_2step_768_b8_int8.engine \
    --tae_engine out/trt/taesana_768_b8 --carry --temporal_filter 0.85 0.5
# ... delay with a live source (frames handed over at 30 fps), here with engines for batch 2
python scripts/build_trt_tae.py --size 768 --batch 2 --prefix out/trt/taesana_768_b2
python scripts/build_trt.py --student runs/multi_carry_s05_lut_2step_768/lora.safetensors \
    --engine out/trt/vibe_multi_carry_s05_lut_2step_768_b2_int8.engine --int8 --batch 2 --calib_clips 25
python scripts/measure_latency.py --engine out/trt/vibe_multi_carry_s05_lut_2step_768_b2_int8.engine \
    --tae_engine out/trt/taesana_768_b2 --carry --temporal_filter 0.85 0.5
# ... or with PyTorch only
python scripts/stream_video.py --input ... --student runs/lam4_2step/lora.safetensors
# ... with the display-side temporal filter and / or the carry-over (any of these scripts, including make_demo.py);
# a student trained for another start or fill of the carried step (--carry_sigma, --carry_fill) records it, and so
# does its engine
python scripts/stream_video.py --input ... --engine ... --temporal_filter 0.7 0.3 --carry
# Style from a reference image: VIBE's Qwen3-VL turns it into the instruction, once per clip
python scripts/stream_video.py --input ... --engine ... --temporal_filter 0.7 0.3 --carry --style_image ref.jpg
python scripts/make_demo.py --engine ... --carry --temporal_filter 0.7 0.3 --style_images a.jpg b.jpg --out out/style
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
# ... trained for the carry-over as well, carried step from sigma 0.5 with the colour-mapped fill (~76 min at
# 3.0 s/step; at 768 with data/teacher_768, ~90 min)
python scripts/train_student.py --data data/teacher_640 --run runs/lam4_carry_s05_lut_2step_640 --student_steps 2 \
    --lam 4 --w_carry 1 --carry_sigma 0.5 --carry_fill lut --grad_ckpt
python scripts/eval_flicker.py --student runs/lam4_carry_s05_lut_2step_640/lora.safetensors --carry \
    --out out/flicker/lam4_carry_s05_lut_2step_640_carry.json
# ... the default at 768: 14 instructions on the usual windows (oil / anime copied from data/teacher_768) and on
# windows of every second frame, then 2500 steps from the 768 LUT carry student (~2.8 h at 4.0 s/step)
P="oil anime watercolor pencil ukiyoe impressionist comic charcoal pastel stainedglass popart winter night sunset"
python scripts/make_teacher_data.py --out data/teacher_768_multi --reuse data/teacher_768 --prompts $P    # 44 min
python scripts/make_teacher_data.py --out data/teacher_768_multi_s2 --stride 2 --prompts $P             # 51 min
python scripts/train_student.py --data data/teacher_768_multi data/teacher_768_multi_s2 \
    --run runs/multi_carry_s05_lut_2step_768 --student_steps 2 --lam 4 --w_carry 1 --carry_sigma 0.5 \
    --carry_fill lut --grad_ckpt --steps 2500 --init runs/lam4_carry_s05_lut_2step_768/lora.safetensors
# Streaming-path quality over trained and held-out instructions, worst batch included; --table compares result files
python scripts/eval_stream.py --engine out/trt/vibe_multi_carry_s05_lut_2step_768_b8_int8.engine \
    --filters 0.7 0.3 0.85 0.5 --teacher --out out/eval_stream/multi_carry_s05_lut_768_int8.json

# Fine-tuning data for style from a reference image: public-domain artworks from the Art Institute
# of Chicago API (CC0 images), Open Images validation photos listed as CC BY 2.0 (attribution kept in images.jsonl),
# 8-step candidates under two ways of conditioning (~2.6 s per pair at 768), then one target per pair or none
python scripts/fetch_style_refs.py --out data/style_refs_aic
python scripts/fetch_content_images.py --out data/content_openimages --count 2000
python scripts/make_style_data.py --content data/content_openimages --refs data/style_refs_aic --out data/style_ft_768
python scripts/select_style_targets.py --data data/style_ft_768
# ... fine-tune VIBE on the selected targets, score it on the held-out references, retrain the streaming student
# with references from the fine-tuned teacher, stream with instruction + reference image
python scripts/train_style_ft.py --data data/style_ft_768 --run runs/style_ft_b --cond B
python scripts/eval_style_ft.py --data data/style_ft_768 --lora runs/style_ft_b/lora.safetensors --out out/style_ft/b --ways B C
python scripts/make_teacher_data.py --out data/teacher_768_refs --style_list data/style_ft_768/student_refs.txt \
    --base_lora runs/style_ft_b/lora.safetensors --reuse data/teacher_768
python scripts/train_student.py --data data/teacher_768_refs --run runs/refs_carry_s05_lut_2step_768 --student_steps 2 \
    --lam 4 --w_carry 1 --carry_sigma 0.5 --carry_fill lut --grad_ckpt --steps 1500 --ref_min_edge 0.25 \
    --init runs/lam4_carry_s05_lut_2step_768/lora.safetensors
python scripts/stream_video.py --input ... --engine out/trt/vibe_refs_carry_s05_lut_2step_768_b8.engine \
    --style_image ref.jpg --style_way both --carry --temporal_filter 0.7 0.3
# ... its INT8 engine: the reference prompts are calibrated on reference images (--calib_refs, by default the list above)
python scripts/build_trt.py --student runs/refs_carry_s05_lut_2step_768/lora.safetensors \
    --engine out/trt/vibe_refs_carry_s05_lut_2step_768_b8_int8.engine --int8
```

A student is tied to its resolution and its init noise (stored in the file). Other instructions than the ones it was
trained on work for styles and global edits, less for edits that rebuild the scene; the two-instruction students'
carried step holds texture styles less well than their own, the multi-instruction student's holds written
instructions as well as its 2-step path (Results, "Other instructions and style from a reference image" and "Many
instructions"). A TensorRT
engine is tied to one resolution, one batch size, one TensorRT version and platform; `<engine>.json` records them and
a mismatch is refused.

## Layout

- `src/vibe_stream/`
  - `vibe_model.py`: VIBE's pipeline driven step by step (embedding, DC-AE, velocity, teacher sampler, k-step student)
  - `lora.py`: student LoRA (attach, save, load, merge)
  - `temporal_loss.py`: flow warp and the masked temporal term
  - `taesana.py`: the tiny autoencoder
  - `trt_engine.py`: ONNX export, engine build, runtime wrapper
  - `trt_int8.py`: INT8 quantization of the transformer for its engine (ModelOpt)
  - `trt_tae.py`: taesana as TensorRT engines
  - `stream.py`: batched, double-buffered streaming editor
  - `live.py`: the same for frames arriving one at a time, host flows computed as they arrive
  - `temporal_filter.py`: display-side two-band temporal filter
  - `carry.py`: carry-over of the previous output into the next batch's start
  - `state_attn.py`: self-attention state carried from frame to frame (experimental, Negative results)
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
- **A second round of the base-model fine-tune on its own outputs.** Candidates from the fine-tuned model
  (`make_style_data.py --base_lora --ways B C`, 5139 pairs; C = written instruction + reference image), selected
  together with the first round's (`select_style_targets.py --extra`: 3949 pairs, 2493 of them new), fine-tuned
  from the base again on the same held-out references and scored against the first round with the same pairs and
  noise:

  | Held-out pairs, 8 steps | Round 1 (B) | Round 2, B + C | Round 2, B |
  |---|---:|---:|---:|
  | Passing all criteria, reference image (famous works) | 0.34 (0.34) | 0.37 (0.41) | 0.37 (0.38) |
  | Edge agreement | 0.40 | 0.47 | 0.46 |
  | Reference gain (median) | 0.066 | 0.057 | 0.058 |
  | CLIP direction, plain `oil` instruction (base 0.220) | 0.209 | 0.170 | 0.206 |

  The second round keeps the photo's structure better and leaks less, but paints paler: its reference gain is above
  the first round's on only a third of the pairs and its contrast lower on three quarters. Training on the reference
  image alone (last column) removes the drift of the plain instruction, so that came from the B + C conditioning;
  the paleness comes from the data. The new candidates have lower luminance contrast than the first round's targets
  for the same pairs (73 %) at the same CLIP reference gain (median 0.098 against 0.094), which the CLIP criterion
  does not see. The first round stays.
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
- **Self-attention state carried from frame to frame** (`train_student.py --state_carry`, `state_attn.py`). VIBE's
  self-attention is ReLU linear attention: a frame's keys and values reduce to one (d+1) x d matrix per head,
  S = [V; 1] K^T, so frame t can read S_t + alpha S'_t-1, S' accumulated over the earlier frames: a recurrent memory
  at no extra tokens, with no positional embedding in the way. Without retraining (2-step path of
  `multi_carry_s05_lut_2step_768`, two clips x two instructions, state carried across batches per denoising step),
  alpha 0.9 cut flicker by 21 % with no lag (edge agreement with the frame two back unchanged) and slightly softer
  backgrounds in pans. Trained: 800 steps from that student with the carried step on runs of 4 consecutive frames and
  a temporal term between them (`--carry_seq 4`, which the state needs to learn from), per-block alpha learned from
  0.5 (it ended at 0.34-0.68), and a control without the state. Scored in PyTorch on the streaming path (the four
  evaluation clips played forward then backward, 96 frames; oil, anime, ukiyo-e and night trained, gouache and autumn
  held out; taesana engines):

  | Student | 2 steps: valid | Sobel | Carry-over + filter 0.85 / 0.5: CLIP dir | Worst batch | Valid | Sobel |
  |---|---:|---:|---:|---:|---:|---:|
  | `multi_carry_s05_lut_2step_768` | 13.20 | 7.01 | 0.167 | 0.135 | 5.66 | 5.65 |
  | + runs of 4, no state (control) | 13.25 | 6.97 | 0.161 | 0.131 | 4.89 | 4.91 |
  | + runs of 4, state carry | **11.43** | 6.96 | 0.162 | 0.131 | 4.85 | 4.94 |
  | the same, state carry off | 13.47 | 6.99 | 0.162 | 0.131 | 4.94 | 4.96 |

  The state helps the 2-step path (14 % less flicker than the control at the same texture and edit strength) but not
  the carry-over path that ships, where consecutive frames are already tied together: there the runs of 4 lower
  flicker, texture and a little edit strength together, with or without the state, as a stronger filter does. Not adopted; in an engine the
  state would also have to be an input and an output of every call.
- **A student in the Wan2.2 video latent space** (probe scripts not in this repository). A video VAE ties four frames
  into one latent: the Wan2.2 VAE (16 x 16 x 4, 48 channels, Apache-2.0), with madebyollin's taew2_2_super (MIT) as
  the real-time codec of the same latent space (1.1 + 3.5 ms per frame at 768). So consistency could come from the
  latent space, and the transformer would run once per four frames. The 768 teacher data (92 windows of 9 frames,
  14 instructions) were re-encoded into normalised Wan latents, and two students were trained from
  `multi_carry_s05_lut_2step_768` (Huber on the latents, LPIPS on taew2_2_super-decoded crops, the temporal term):
  - *key frame + renderer* (3000 steps): VIBE, unchanged, on one key frame per latent (its DC-AE latent, which a live
    stream has), and a small convolutional net that renders the latent's Wan tokens from the edited key frame, the
    source key frame and the source's Wan tokens.
  - *Wan tokens* (2000 steps): VIBE editing Wan tokens directly (2 x 2 latent pixels per token), with the patch
    embedding and output projection replaced and initialised from ridge fits between the two latent spaces.

  Two training-free references bound what the latent space gives: the starting student's 2-step frames through
  taew2_2_super, and the same student run on key frames only, each edit held for its latent with the source's motion
  added in latent space. Streaming evaluation as in the previous entry, filter 0.85 / 0.5:

  | Output | CLIP dir | Worst batch | Valid | Sobel |
  |---|---:|---:|---:|---:|
  | `multi_carry_s05_lut_2step_768`, 2 steps | 0.170 | 0.142 | 13.20 | 7.01 |
  | the same, carry-over + filter (shipped) | 0.167 | 0.135 | 5.66 | 5.65 |
  | its 2-step frames through taew2_2_super | 0.162 | 0.130 | 10.55 | 4.11 |
  | the same + filter | 0.148 | 0.115 | 6.21 | 3.81 |
  | key frames only, held, + source motion | 0.141 | 0.103 | 14.84 | 3.86 |
  | Student, key frame + renderer | 0.120 | 0.091 | 12.03 | 6.37 |
  | the same + filter | 0.115 | 0.085 | 7.20 | 5.30 |
  | Student, Wan tokens | 0.132 | 0.098 | 10.38 | 6.21 |
  | the same + filter | 0.124 | 0.088 | 6.36 | 5.21 |

  Neither the students nor the references reach the shipped carry-over + filter path.
  - The students keep the photo's structure and lose 22-29 % of the edit strength: the sketch and cel-shading
    restyles mostly do not happen.
  - The latent space smooths by itself: the 2-step frames come back with 20 % less flicker, but they lose 41 % of
    their Sobel and 5 % of edit strength. Even with the filter they stay behind carry-over + filter on all three
    measures.
  - Part of the softness is the tiny codec. Source clips through taew2_2_super keep 61-67 % of their Sobel
    (27-30 dB); the full Wan2.2 VAE reaches 32 dB but decodes at 180 ms per frame.
  - Part is the latent itself. The teacher's frames come back from the Wan2.2 VAE at 23-25 dB: their texture
    changes from frame to frame cannot be held in a 4-frame latent and are averaged, so a student trained on those
    latents learns averaged targets.

  A temporally compressed latent needs a generator whose textures already agree from frame to frame, and an image
  teacher does not provide one. taew2_2's encoder also looks three frames ahead, which adds 100 ms in a live stream.

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
