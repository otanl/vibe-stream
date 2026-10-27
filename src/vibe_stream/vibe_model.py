"""VIBE's editing model, driven step by step.

VIBE (ai-forever/vibe; weights iitolstykh/VIBE-Image-Edit-DistilledCFG) edits with a Sana 1.5 1.6B linear-attention
DiT. The instruction and the image go through Qwen3-VL-2B together; learned meta tokens read its last hidden state and
VIBE's edit heads turn them into the DiT's cross-attention context (the "embedding" below). The reference image enters
as its DC-AE f32c32 latent, concatenated to the noisy latent along channels.

This module bypasses `pipeline.__call__` so that the embedding can be computed once and reused, latents can be fed
directly, and stages can be timed separately. The streaming setting is that of dreamlite-stream's lean path: one
init-noise tensor shared by every frame and one embedding per clip, from its first frame. The K-step output is then
a deterministic function of the frame, which is what lets a 1- or 2-step student be distilled by paired regression.
"""

from __future__ import annotations

import copy
import os
import sys
import types
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models" / "VIBE-Image-Edit-DistilledCFG"
LATENT_CHANNELS = 32
NOISE_SEED = 42
# Asked of VIBE's own Qwen3-VL to turn a style reference image into an instruction (`VibeModel.style_prompt`).
STYLE_QUESTION = ("Write one instruction for an image editing model that restyles any photo into the artistic style of "
                  "this artwork. Describe only the style (medium, technique, brushwork or linework, colour palette, "
                  "texture), not the subject or the scene, and do not name the artist or the work. Start with "
                  "'transfer this to' and use at most 40 words.")
# The instruction that goes with a reference image handed to the encoder next to the frame (`embed_with_reference`).
REFERENCE_INSTRUCTION = "transfer this to the artistic style of the second image"


def add_vibe_to_path() -> None:
    """Make the upstream `vibe` package importable: $VIBE_REPO, else a sibling checkout `../vibe`.

    Upstream logs through loguru. When it is not installed a silent stand-in is registered; nothing here reads its
    output.
    """
    repo = Path(os.environ.get("VIBE_REPO", ROOT.parent / "vibe"))
    if not (repo / "vibe" / "__init__.py").exists():
        raise FileNotFoundError(f"no upstream VIBE checkout at {repo}: clone https://github.com/ai-forever/vibe there "
                                "or set VIBE_REPO")
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    try:
        import loguru  # noqa: F401
    except ImportError:
        shim = types.ModuleType("loguru")
        shim.logger = types.SimpleNamespace(**{k: (lambda *a, **kw: None)
                                               for k in ("info", "warning", "debug", "error", "exception")})
        sys.modules["loguru"] = shim


def make_init_noise(size: int, seed: int = NOISE_SEED) -> torch.Tensor:
    """The shared init-noise tensor, (1, 32, size / 32, size / 32), drawn on the CPU in float32.

    A float32 CPU draw is the same under every torch build, a seeded bf16 draw is not. A student file stores the
    tensor it was trained on in any case (`lora.NOISE_KEY`).
    """
    lat = size // 32
    return torch.randn((1, LATENT_CHANNELS, lat, lat), generator=torch.Generator("cpu").manual_seed(seed),
                       dtype=torch.float32)


def flow_step(x: torch.Tensor, v: torch.Tensor, s0: float, s1: float) -> torch.Tensor:
    """One first-order (Euler) step from sigma s0 to s1 for the flow velocity v = eps - x0, x = (1 - s) x0 + s eps.

    Same as re-noising the predicted x0 = x - s0 v with the predicted eps = x + (1 - s0) v at level s1.
    """
    return x + (s1 - s0) * v


def student_schedule(scheduler, k: int, device) -> Tuple[List[torch.Tensor], List[float]]:
    """VIBE's own timesteps and sigmas for k steps (DPM-Solver++ multistep, flow prediction, flow shift 3).

    For k <= 2 every step of that scheduler is first order, so `flow_step` along these sigmas reproduces it up to the
    bf16 rounding of the transformer input. k = 1: t 999, sigma 0.9997. k = 2: t 999 / 749, sigmas 0.9997 / 0.7496.
    """
    s = copy.deepcopy(scheduler)
    s.set_timesteps(k, device=device)
    return list(s.timesteps), [float(v) for v in s.sigmas]


class VibeModel:
    """VIBE's pipeline plus the pieces this repo calls directly.

    `size` is the working resolution (square, a multiple of 32). VIBE's own call would snap it to an aspect-ratio bin
    (768 goes to the 512 bin unless `use_resolution_binning=False`).
    """

    def __init__(self, weights=DEFAULT_WEIGHTS, size: int = 768, device: str = "cuda", keep_text_encoder: bool = True,
                 noise: torch.Tensor | None = None):
        add_vibe_to_path()
        from vibe.generative_pipeline import VIBESanaEditingPipeline
        # TF32 for fp32 matmuls (process-wide). Qwen3-VL's rotary angles are an fp32 matmul, so this changes the
        # embedding and, through it, VIBE's output; every reported number and the released student used it.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())   # parameters report cuda:N
        self.size = size
        self.pipe = VIBESanaEditingPipeline.from_pretrained(str(weights), torch_dtype=torch.bfloat16).to(self.device)
        # __call__ hands precomputed embeddings to Sana's input check without the attention mask that check insists
        # on; the mask plays no part in VIBE's denoiser, so the check is skipped in that case.
        check = self.pipe.check_inputs
        self.pipe.check_inputs = lambda *a, **kw: None if kw.get("prompt_embeds") is not None else check(*a, **kw)
        self.tr = self.pipe.transformer
        self.vae = self.pipe.vae
        self.sf = self.vae.config.scaling_factor
        self.tscale = getattr(self.tr.config, "timestep_scale", 1.0)
        self.noise = (make_init_noise(size) if noise is None else noise).to(self.device, torch.float32)
        self.schedules: Dict[int, Tuple[List[torch.Tensor], List[float]]] = {
            k: student_schedule(self.pipe.scheduler, k, self.device) for k in (1, 2)}
        self._proj32 = None
        if not keep_text_encoder:
            self.offload_text_encoder()

    # ---------------------------------------------------------------- conditioning
    def offload_text_encoder(self) -> None:
        self.pipe.text_encoder.to("cpu")
        torch.cuda.empty_cache()

    @torch.inference_mode()
    def embed(self, prompt: str, frame: np.ndarray) -> torch.Tensor:
        """The instruction embedding (1, L, D), bf16, for one RGB uint8 frame. Brings the text encoder back to the
        GPU for the call if it was offloaded."""
        te = self.pipe.text_encoder
        offloaded = next(te.parameters()).device != self.device
        if offloaded:
            te.to(self.device)
        try:
            emb, _ = self.pipe._get_editing_embeddings(edit_prompt=prompt, conditioning_image=Image.fromarray(frame))
        finally:
            if offloaded:
                self.offload_text_encoder()
        return emb.contiguous()

    @torch.inference_mode()
    def embed_with_reference(self, prompt: str, frame: np.ndarray, reference: np.ndarray | Image.Image) -> torch.Tensor:
        """Like `embed`, with a second image after the frame in the same user turn, e.g. "transfer this to the artistic
        style of the second image". VIBE was trained with one image; the second one brings its content along as well
        as its style. Does not bring an offloaded text encoder back."""
        pipe = self.pipe
        f = Image.fromarray(frame)
        ref = reference if isinstance(reference, Image.Image) else Image.fromarray(reference)
        content = [{"type": "image", "image": f}, {"type": "image", "image": ref},
                   {"type": "text", "text": pipe.edit_query_template.format(instruction=prompt.strip())}]
        text = pipe.tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                  add_generation_prompt=False)
        x = pipe.tokenizer(text=[text], images=[f, ref], padding=True, return_tensors="pt",
                           min_pixels=pipe.min_pixels, max_pixels=pipe.max_pixels).data
        mi = pipe.prepare_initial_input_embeddings(**{k: v.to(self.device) for k, v in x.items()})
        hidden = pipe.text_encoder.language_model(input_ids=None, **mi).last_hidden_state
        meta = pipe.transformer.get_hidden_states_for_meta_tokens(hidden, mi["attention_mask"])
        return pipe.transformer.forward_edit_heads(meta.shape[0], meta)[0].contiguous()

    @torch.inference_mode()
    def style_prompt(self, image: np.ndarray | Image.Image, max_new_tokens: int = 80) -> str:
        """An instruction in the style of a reference image, written by VIBE's own Qwen3-VL (greedy, STYLE_QUESTION).

        Handing the reference to the encoder next to the frame instead brings its content along (figures, lettering),
        and cutting it into shuffled tiles first removes the content but most of the style too. Like `embed`, brings
        the text encoder back to the GPU for the call if it was offloaded.
        """
        ref = image if isinstance(image, Image.Image) else Image.fromarray(image)
        proc, te = self.pipe.tokenizer, self.pipe.text_encoder
        msg = [{"role": "user", "content": [{"type": "image", "image": ref}, {"type": "text", "text": STYLE_QUESTION}]}]
        text = proc.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        x = proc(text=[text], images=[ref], return_tensors="pt", min_pixels=self.pipe.min_pixels,
                 max_pixels=self.pipe.max_pixels).data
        offloaded = next(te.parameters()).device != self.device
        if offloaded:
            te.to(self.device)
        try:
            x = {k: v.to(self.device) for k, v in x.items()}
            ids = te.generate(**x, max_new_tokens=max_new_tokens, do_sample=False)
        finally:
            if offloaded:
                self.offload_text_encoder()
        out = proc.batch_decode(ids[:, x["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return " ".join(out.split())

    @torch.no_grad()
    def project_embedding(self, emb: torch.Tensor) -> torch.Tensor:
        """caption_projection + caption_norm in fp32, (B, L, heads * head_dim).

        This is the context input of the TensorRT engine: the projection stays outside the engine and runs once per
        clip, as dreamlite-stream does with DreamLite's prompt projection (whose RMSNorm overflowed fp16).
        """
        if self._proj32 is None:
            self._proj32 = (copy.deepcopy(self.tr.caption_projection).float(), copy.deepcopy(self.tr.caption_norm).float())
        proj, norm = self._proj32
        hidden = self.tr.config.num_attention_heads * self.tr.config.attention_head_dim
        return norm(proj(emb.float()).view(emb.shape[0], -1, hidden))

    # ---------------------------------------------------------------- DC-AE
    @torch.inference_mode()
    def encode(self, frames: Sequence[np.ndarray], chunk: int = 10) -> torch.Tensor:
        """DC-AE latents (N, 32, size / 32, size / 32) of RGB uint8 frames, scaled, fp32."""
        out = []
        for i in range(0, len(frames), chunk):
            x = self.pipe.image_processor.preprocess([Image.fromarray(f) for f in frames[i:i + chunk]],
                                                     height=self.size, width=self.size)
            out.append((self.vae.encode(x.to(self.device, torch.bfloat16)).latent * self.sf).float())
        return torch.cat(out)

    @torch.inference_mode()
    def decode_u8(self, lat: torch.Tensor, chunk: int = 8) -> np.ndarray:
        """DC-AE decode of scaled latents to RGB uint8 (N, H, W, 3) on the CPU."""
        out = []
        for i in range(0, len(lat), chunk):
            x = self.vae.decode((lat[i:i + chunk] / self.sf).to(torch.bfloat16)).sample.float().clamp(-1, 1)
            out.append(((x + 1) * 127.5).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy())
        return np.concatenate(out)

    # ---------------------------------------------------------------- denoising
    def velocity(self, x: torch.Tensor, img_lat: torch.Tensor, emb: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The transformer's flow velocity for noisy latents x given the reference latents and a raw embedding."""
        b = x.shape[0]
        return self.tr(hidden_states=torch.cat([x, img_lat], 1).to(torch.bfloat16),
                       encoder_hidden_states=emb.expand(b, -1, -1), timestep=t.expand(b) * self.tscale,
                       return_dict=False, t2i_samples=[False] * b)[0].float()

    @torch.inference_mode()
    def sample(self, img_lat: torch.Tensor, emb: torch.Tensor, steps: int, chunk: int = 10) -> torch.Tensor:
        """VIBE's own scheduler for `steps` steps from the shared noise: the teacher."""
        out = []
        for i in range(0, len(img_lat), chunk):
            il = img_lat[i:i + chunk]
            sch = copy.deepcopy(self.pipe.scheduler)
            sch.set_timesteps(steps, device=self.device)
            x = self.noise.expand(len(il), -1, -1, -1).clone()
            for t in sch.timesteps:
                x = sch.step(self.velocity(x, il, emb, t), t, x, return_dict=False)[0]
            out.append(x)
        return torch.cat(out)

    def student(self, img_lat: torch.Tensor, emb: torch.Tensor, k: int = 2) -> torch.Tensor:
        """k first-order flow steps on VIBE's k-step schedule from the shared noise. Differentiable."""
        ts, sg = self.schedules[k]
        x = self.noise.expand(len(img_lat), -1, -1, -1)
        for t, s0, s1 in zip(ts, sg[:-1], sg[1:]):
            x = flow_step(x, self.velocity(x, img_lat, emb, t), s0, s1)
        return x
