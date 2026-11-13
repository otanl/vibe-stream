"""Stream a video file through a student: taesana encode -> k steps -> taesana decode, batched, on the GPU.

The instruction embedding is computed once from the first frame. Prints loop fps and step-only fps (defined in
vibe_stream.stream), the embedding time, peak VRAM, and the warping error, Sobel and edge agreement of the output.

Usage:
  python scripts/stream_video.py --input assets/davis_mp4/blackswan.mp4 --output out/stream/blackswan_oil.mp4 \
      --engine out/trt/vibe_lam4_2step_768_b8.engine --prompt oil
  python scripts/stream_video.py ... --student runs/lam4_2step/lora.safetensors      (PyTorch, no TensorRT)
  --loop_frames 480 repeats the clip for a sustained throughput figure; --compare writes input | output.
"""
import argparse
import json
from itertools import cycle, islice
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from PIL import Image

import _paths  # noqa: F401
from vibe_stream import lora
from vibe_stream.carry import Carry
from vibe_stream.metrics import edge_agreement, read_video_frames, sobel, video_fps, warping_error, write_video
from vibe_stream.presets import PROMPTS
from vibe_stream.stream import StreamEditor
from vibe_stream.taesana import TinyAutoencoder
from vibe_stream.temporal_filter import TemporalFilter
from vibe_stream.trt_tae import TRTTinyAutoencoder
from vibe_stream.vibe_model import REFERENCE_INSTRUCTION, ROOT, VibeModel, student_model_args


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--prompt", default="oil", help=f"an instruction, or a preset key: {', '.join(PROMPTS)}")
    p.add_argument("--style_image", default=None,
                   help="instead of --prompt: an image whose style VIBE's Qwen3-VL turns into the instruction")
    p.add_argument("--style_way", choices=["text", "image", "both"], default="text",
                   help="with --style_image: the instruction written from it, the image itself next to the frame, or "
                        "both (as make_demo.py)")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--engine", help="TensorRT engine from scripts/build_trt.py")
    g.add_argument("--student", help="student LoRA file (PyTorch bf16 path)")
    p.add_argument("--tae_engine", default=None,
                   help="taesana as TensorRT engines: the prefix given to scripts/build_trt_tae.py")
    p.add_argument("--batch", type=int, default=8, help="PyTorch path only; an engine fixes its batch")
    p.add_argument("--max_frames", type=int, default=None)
    p.add_argument("--loop_frames", type=int, default=None, help="cycle the clip to this many frames")
    p.add_argument("--warmup_batches", type=int, default=2)
    p.add_argument("--compare", action="store_true", help="write input | output side by side")
    p.add_argument("--temporal_filter", type=float, nargs=2, default=None, metavar=("LOW", "HIGH"),
                   help="display-side temporal filter with these blend weights for the low and high band, e.g. 0.7 0.3")
    p.add_argument("--carry", action="store_true",
                   help="start batches after a keyframe from the previous output (vibe_stream.carry)")
    p.add_argument("--keyframe_every", type=int, default=0, help="with --carry: a keyframe every N batches (0: first only)")
    p.add_argument("--carry_sigma", type=float, default=None,
                   help="with --carry: noise level of the carried start "
                        "(default: the student's, as recorded in it or in the engine's spec)")
    p.add_argument("--carry_fill", choices=["frame", "lut"], default=None,
                   help="with --carry: fill where the flow fails with the frame or the frame through the keyframe's "
                        "colour map (default: the student's, as recorded in it or in the engine's spec)")
    p.add_argument("--low_vram", action="store_true",
                   help="with --engine: keep VIBE's transformer blocks and DC-AE on the CPU, and bring the text encoder "
                        "to the GPU only while it embeds the instruction")
    p.add_argument("--te_bits", type=int, choices=[8, 4], default=None,
                   help="Qwen3-VL's language model HQQ-quantized, kept on the GPU (changes the instruction embedding; "
                        "needs the hqq package)")
    p.add_argument("--te_vision_bits", action="store_true", help="with --te_bits: the vision tower quantized too")
    p.add_argument("--swap_edit_head", action="store_true",
                   help="with --low_vram: VIBE's edit head on the GPU only while an instruction is embedded")
    p.add_argument("--share_scratch", action="store_true",
                   help="with --engine and --tae_engine: one TensorRT scratch buffer for the engines that run in turn")
    p.add_argument("--tae_dec_engine", default=None,
                   help="with --tae_engine: the decoder from engines of a smaller batch (less scratch memory)")
    p.add_argument("--reembed_async", action="store_true",
                   help="with --carry: at a cut the new embedding is made on the side and the stream does not wait")
    p.add_argument("--fast_post", action="store_true",
                   help="the filter's and the carry-over's smooth parts at reduced resolution (temporal_filter)")
    p.add_argument("--remove_blocks", type=lambda s: [int(x) for x in s.split(",")], default=None,
                   help="with --student: transformer blocks it was trained without (default: as its metadata says)")
    p.add_argument("--face_engine", default=None,
                   help="face pass (vibe_stream.face): transformer engine at the crop size, same batch as the stream's; \"main\": the stream's engine at its other size (build_trt.py --extra_sizes)")
    p.add_argument("--face_tae_engine", default=None, help="with --face_engine: taesana engines at the crop size")
    p.add_argument("--face_model", default=str(ROOT / "models/face_detection_yunet_2023mar.onnx"),
                   help="with --face_engine: YuNet face detector (OpenCV model zoo)")
    p.add_argument("--packed_engine", default=None,
                   help="with --face_engine main and batch 1: the face crop rides along in the frame's transformer call "
                        "(build_trt_packed.py, vibe_stream.packed)")
    p.add_argument("--face_every", type=int, default=1,
                   help="with --packed_engine: edit the crop on every n-th frame, its last edit warped in between")
    p.add_argument("--out_json", default=None)
    a = p.parse_args()
    prompt = PROMPTS.get(a.prompt, a.prompt)

    if a.engine:
        from vibe_stream.trt_engine import TRTTransformer
        engine = TRTTransformer(a.engine)
        size, trained, removed = engine.spec.size, engine.spec.prompts, ()
    else:
        engine = None
        meta = lora.load(a.student)[1]
        size, trained = int(meta.get("size", 768)), meta.get("prompts", {})
        removed = a.remove_blocks if a.remove_blocks is not None else student_model_args(meta)["remove_blocks"]
    frames = read_video_frames(a.input, size, a.max_frames)
    print(f"[input] {a.input}: {len(frames)} frames at {size}", flush=True)
    if a.low_vram and not a.engine:
        p.error("--low_vram needs --engine")
    vibe = VibeModel(size=size, keep_text_encoder=not a.low_vram or a.te_bits is not None,
                     denoiser_on_gpu=not a.low_vram, text_encoder_bits=a.te_bits, quantize_vision=a.te_vision_bits,
                     edit_head_on_gpu=not a.swap_edit_head, remove_blocks=removed)
    reference = None
    if a.style_image:
        img = Image.open(a.style_image).convert("RGB")
        if a.style_way == "image":
            prompt = REFERENCE_INSTRUCTION
        else:
            prompt = vibe.style_prompt(img)
            print(f"[style] {a.style_image}: {prompt}", flush=True)
        reference = img if a.style_way != "text" else None
    if trained and prompt not in trained.values():
        print(f"[note] the student was trained on {list(trained.values())}; other instructions may come out weaker")
    tae = TRTTinyAutoencoder(a.tae_engine, dec_prefix=a.tae_dec_engine) if a.tae_engine else TinyAutoencoder()
    tf = TemporalFilter(*a.temporal_filter, fast=a.fast_post) if a.temporal_filter else None
    face = None
    if a.face_engine:
        from vibe_stream.face import FacePass
        from vibe_stream.trt_engine import TRTTransformer
        if a.face_engine == "main":     # the stream's own engine at its other size (build_trt.py --extra_sizes)
            other = [s for s in (engine.spec.profiles or []) if s != engine.spec.size] if engine is not None else []
            if not other:
                p.error("--face_engine main needs an --engine built with --extra_sizes")
            face_engine = TRTTransformer(a.engine, size=other[0], share=engine)
        else:
            face_engine = TRTTransformer(a.face_engine)
        face = FacePass(vibe, face_engine,
                        TRTTinyAutoencoder(a.face_tae_engine) if a.face_tae_engine else TinyAutoencoder(), a.face_model)
    kw = {}
    if a.packed_engine:
        from vibe_stream.packed import PackedEngine, PackedStreamEditor
        kw = dict(packed=PackedEngine(a.packed_engine), face_every=a.face_every)
    ed = (PackedStreamEditor if kw else StreamEditor)(
        vibe, tae, engine=engine, student=a.student, batch_size=a.batch, temporal_filter=tf,
        carry=Carry(a.keyframe_every, sigma=a.carry_sigma, fill=a.carry_fill, fast=a.fast_post) if a.carry else None,
        reembed_async=a.reembed_async, face=face, **kw)
    scratch = None
    if a.share_scratch:
        from vibe_stream.trt_engine import share_scratch
        runners = [engine, tae.enc, tae.dec]
        if face is not None:
            runners += [face.ed.engine] + ([face.ed.tae.enc, face.ed.tae.dec] if hasattr(face.ed.tae, "enc") else [])
        scratch = share_scratch(*runners)
        print(f"[run] one TensorRT scratch buffer of {scratch.numel() / 2**30:.2f} GB for {len(runners)} contexts",
              flush=True)
    embed_ms = ed.set_prompt(prompt, frames[0], reference)
    torch.cuda.reset_peak_memory_stats()
    run_frames = list(islice(cycle(frames), a.loop_frames)) if a.loop_frames else frames
    outs, tm = ed.run(run_frames, a.warmup_batches)
    outs = outs[:len(frames)]
    backend = f"TensorRT {engine.spec.tensorrt} {engine.spec.precision}" if engine else "PyTorch bf16"
    if a.tae_engine:
        backend += ", taesana TensorRT"
    if tf is not None:
        backend += f", temporal filter {tf.low:g}/{tf.high:g}"
    if ed.carry is not None:
        backend += (f", carry-over ({ed.carry.keyframes} keyframe batches of {-(-len(run_frames) // ed.batch_size)}, "
                    f"fill {ed.carry_fill()})")
    if face is not None:
        backend += (f", face pass at {face.size} ({len(face.embed_ms)} face embeddings"
                    + (f", {mean(face.embed_ms):.0f} ms each)" if face.embed_ms else ")"))
    print(f"[run] {backend}, {ed.steps}-step student, batch {ed.batch_size}, {len(run_frames)} frames", flush=True)
    print(f"[run] loop      {tm.loop_fps:6.1f} fps  ({tm.frames} frames in {tm.loop_s:.2f} s, steady state)")
    print(f"[run] step-only {tm.step_fps:6.1f} fps  ({mean(tm.step_ms) / ed.batch_size:.1f} ms per frame)")
    free, total = torch.cuda.mem_get_info()
    print(f"[run] embedding {embed_ms:.0f} ms once per clip; peak VRAM {torch.cuda.max_memory_allocated() / 2**30:.1f} GB "
          f"(allocated by torch; the engine's own memory is not included); in use on the device after the run "
          f"{(total - free) / 2**30:.2f} GB (all processes)")
    res = dict(input=a.input, prompt=prompt, backend=backend, steps=ed.steps, batch=ed.batch_size, size=size,
               frames=len(run_frames), loop_fps=tm.loop_fps, step_fps=tm.step_fps, embed_ms=embed_ms,
               device_in_use_gb=(total - free) / 2**30,
               warp=warping_error(frames, outs), sobel=sobel(outs), sobel_input=sobel(frames),
               edge=mean(edge_agreement(x, y) for x, y in zip(frames[::4], outs[::4])))
    print(f"[quality] warp {res['warp']:.2f}  sobel {res['sobel']:.2f} (input {res['sobel_input']:.2f})  edge {res['edge']:.3f}")
    if a.output:
        vid = [np.concatenate([f, o], axis=1) for f, o in zip(frames, outs)] if a.compare else outs
        write_video(a.output, vid, video_fps(a.input))
        print(f"[out] {a.output}")
    if a.out_json:
        Path(a.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out_json).write_text(json.dumps(res, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
