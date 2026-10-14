"""taesana: madebyollin's tiny autoencoder for the DC-AE f32c32 latent space, used in place of DC-AE.

The architecture below is re-typed from madebyollin/taesd `taesd.py` (MIT License, Copyright (c) 2023 Ollin Boer
Bohan; see THIRD_PARTY_NOTICES.md); the weights `taesana_encoder.pth` / `taesana_decoder.pth` come from the same
repository (`scripts/download_weights.py`). They were trained against DC-AE sana-1.0 and work on VIBE's sana-1.1 as
they are. Images are in [0, 1]; latents are in the scaled (diffusion) space, the same as `VibeModel.encode`.

Measured at 768, batch 8, fp16 on an RTX 3090 Ti: encode 0.9 + decode 8.7 ms per frame, against 97 ms for DC-AE as
a TensorRT engine. Reconstruction of real photos 27.9 dB (DC-AE itself 28.8).
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .vibe_model import ROOT

DEFAULT_DIR = ROOT / "models" / "taesana"


def conv(n_in, n_out, **kw):
    return nn.Conv2d(n_in, n_out, 3, padding=1, **kw)


class Clamp(nn.Module):
    def forward(self, x):
        return torch.tanh(x / 3) * 3


class Block(nn.Module):
    def __init__(self, n_in, n_out):
        super().__init__()
        self.conv = nn.Sequential(conv(n_in, n_out), nn.ReLU(), conv(n_out, n_out), nn.ReLU(), conv(n_out, n_out))
        self.skip = nn.Conv2d(n_in, n_out, 1, bias=False) if n_in != n_out else nn.Identity()
        self.fuse = nn.ReLU()

    def forward(self, x):
        return self.fuse(self.conv(x) + self.skip(x))


def F32Encoder(lc=32, ic=3):
    return nn.Sequential(
        conv(ic, 32, stride=2), nn.ReLU(inplace=True), conv(32, 64, stride=2), nn.ReLU(inplace=True), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, 64, stride=2, bias=False), Block(64, 64), Block(64, 64), Block(64, 64),
        conv(64, lc))


def F32Decoder(lc=32, ic=3):
    return nn.Sequential(
        Clamp(), conv(lc, 256), nn.ReLU(),
        Block(256, 256), Block(256, 256), Block(256, 256), nn.Upsample(scale_factor=2), conv(256, 128, bias=False),
        Block(128, 128), Block(128, 128), Block(128, 128), nn.Upsample(scale_factor=2), conv(128, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), Block(64, 64), Block(64, 64), nn.Upsample(scale_factor=2), conv(64, 64, bias=False),
        Block(64, 64), conv(64, ic))


def _load(net: nn.Module, path: Path) -> nn.Module:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run scripts/download_weights.py --taesana")
    net.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return net.eval().requires_grad_(False)


class TinyAutoencoder:
    """taesana encoder + decoder in fp16 / channels_last on the GPU, uint8 frames in and out."""

    def __init__(self, weights_dir=DEFAULT_DIR, device: str = "cuda"):
        d = Path(weights_dir)
        self.device = torch.device(device)
        self.enc = _load(F32Encoder(), d / "taesana_encoder.pth").to(self.device, torch.float16,
                                                                    memory_format=torch.channels_last)
        self.dec = _load(F32Decoder(), d / "taesana_decoder.pth").to(self.device, torch.float16,
                                                                    memory_format=torch.channels_last)

    @torch.inference_mode()
    def encode_u8(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, H, W, 3) uint8 on the GPU -> (B, 32, H / 32, W / 32) fp16 scaled latents."""
        x = frames.permute(0, 3, 1, 2).to(torch.float16).div_(255).contiguous(memory_format=torch.channels_last)
        return self.enc(x).contiguous()

    @torch.inference_mode()
    def decode_u8(self, lat: torch.Tensor) -> torch.Tensor:
        """Scaled latents -> (B, H, W, 3) uint8 on the GPU."""
        x = self.dec(lat.to(torch.float16).contiguous(memory_format=torch.channels_last))
        return x.clamp_(0, 1).mul_(255).round_().to(torch.uint8).permute(0, 2, 3, 1).contiguous()
