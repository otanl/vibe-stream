# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Language

Converse with the user in Japanese (会話は日本語で). Everything written into the repo — code, comments, docstrings,
commit messages, README — stays in English.

## What this is

Research code for video-rate streaming edits with VIBE (Sana 1.5 1.6B DiT + Qwen3-VL-2B): a few-step student LoRA
trained with a temporal-consistency loss, taesana in place of DC-AE, and the transformer as a TensorRT engine. It is
the licence-clean successor of the sibling repo `../dreamlite-stream` (DreamLite weights are CC BY-NC); metric and
throughput definitions are kept identical to that repo so numbers compare. No CI, linter or pytest; correctness is
checked by running scripts and reading what they print.

## Environment

- Upstream VIBE is a sibling checkout `../vibe` (ai-forever/vibe at 7f0f01f), or `VIBE_REPO`.
  `vibe_model.add_vibe_to_path()` puts it on sys.path and stubs loguru if missing.
- Weights in `models/` (`scripts/download_weights.py`). On this machine `models/`, `data/`, `runs/` and `out/` are
  junctions to `D:\work_offload\vibe-stream\` because F: is nearly full; put large artefacts there too.
- Interpreter: `F:/work/dreamlite/.venv/Scripts/python.exe` (Python 3.10, torch 2.6.0+cu124, transformers 4.57.3,
  diffusers 0.37.1, TensorRT 10.16, lpips). Set `PYTHONUTF8=1` (sources contain non-ASCII; the console is cp932).
- Scripts insert `src/` themselves (`scripts/_paths.py`); run them from the repo root.
- Check `nvidia-smi` before GPU work. A TensorRT build next to a process that still holds the models has overflowed
  24 GB: `build_trt.py` runs export, build and bench in separate processes for that reason.

## Architecture

- `vibe_model.VibeModel` drives VIBE's pipeline step by step instead of `pipeline.__call__` (which snaps 768 to the
  512 aspect bin and cannot reuse embeddings). `embed` = Qwen3-VL over (instruction, frame) -> meta tokens -> edit
  heads; the reference frame's DC-AE latent is concatenated to the noise along channels.
- Streaming setting: one init-noise tensor for every frame (CPU float32 draw, seed 42; float32 because seeded bf16
  draws differ between torch builds) and one embedding per clip from its first frame. The teacher's output is then a
  deterministic function of the frame, so the student is trained by paired regression.
- Student = rank-32 LoRA on all attention projections and the GLUMBConv 1x1 convs (`lora.TARGET_PATTERN`),
  k first-order steps on VIBE's own k-step schedule (`flow_step`, `student_schedule`; exact for k <= 2). A student
  file stores `init_noise`, `student_steps`, `size` and `prompts`; it is tied to all of them.
- Temporal term (`temporal_loss.py`): flow on the INPUT frames t -> t+1, warp output t+1 back, photometric
  occlusion mask; the same convention as `metrics.warping_error`.
- Optional DMD term (`train_student.py --w_dmd`): the teacher is the base weights and the fake score model a second
  LoRA on the same frozen base (`lora.add_aux`; `lora.use` selects "student", "aux" or None). The selection is read
  at call time, so a checkpointed backward must run with the selection of its forward: the student's backward runs
  with "student" active, the fake model's update inside `use(..., "aux")`. It did not help (README, Negative
  results).
- Edit strength is measured by CLIP directional similarity (`metrics.CLIPScore`, captions in
  `presets.CLIP_CAPTIONS`); LPIPS to the teacher alone cannot tell a weaker edit from a different one.
- TensorRT (`trt_engine.py`): LoRA merged, `caption_projection` + `caption_norm` kept outside the engine and
  computed once per clip in fp32 (`VibeModel.project_embedding`), the rest fp16 strongly typed. The engine is fixed
  to one size, batch, TensorRT version and platform; `<engine>.json` records them, `<engine>.noise.safetensors`
  holds the student's noise.
- `stream.StreamEditor.run` is double-buffered through pinned host memory. Two throughput figures: step-only (CUDA
  events around `edit`) and loop (steady-state wall time per batch, host uint8 in and out; video decoding excluded).
  Quote loop fps when judging a change and say which one a number is.

## Rules

- Weights, teacher data, engines and videos are never committed (`models/`, `data/`, `runs/`, `out/`, `assets/*`
  are gitignored).
- Evaluate students against the base model's K-step output; build references before attaching wrappers or after
  detaching them.
