"""Student LoRA: low-rank deltas on VIBE's transformer that make k first-order steps reproduce its K-step output.

Adapted from dreamlite-stream's `onestep_lora.py` (same author, Apache-2.0). In the streaming setting every frame uses
the same init noise and the clip's embedding is fixed, so the teacher's output is a deterministic function of the
frame and distillation is paired regression (`scripts/train_student.py`).

At run time the deltas are merged into the weights (`merge`), so the merged transformer costs what the base one costs
and an ONNX export sees an ordinary model.

DMD training needs two more models next to the student: the teacher (the base weights) and a "fake score" model that
tracks the student's outputs. Both share the frozen base here: `add_aux` gives every wrapper a second, independent
delta and `use` selects which delta runs ("student", "aux", or None for the base alone). The selection is read at
call time, so under gradient checkpointing a backward pass must run with the same selection as its forward.

File format: safetensors with `<module path>.lora_A` (rank, in) and `<module path>.lora_B` (out, rank) per adapted
layer, `init_noise` (1, 32, h, w), and string metadata (JSON-encoded values) describing how it was trained, among
them `student_steps`, `size` and `prompts`. alpha equals rank, so the delta is B @ A.
"""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

NOISE_KEY = "init_noise"   # the shared init-noise tensor a student was fitted to

# Every attention projection (attn1 is linear self-attention, attn2 cross-attention) and both 1x1 convolutions of
# the GLUMBConv feed-forward, in all 20 blocks: 200 layers, 36.6 M parameters at rank 32.
TARGET_PATTERN = re.compile(
    r"transformer_blocks\.\d+\.(attn[12]\.(to_q|to_k|to_v|to_out\.0)|ff\.(conv_inverted|conv_point))$")


def _low_rank_pair(out_dim: int, in_dim: int, rank: int) -> Tuple[nn.Parameter, nn.Parameter]:
    a = nn.Parameter(torch.zeros(rank, in_dim))
    nn.init.kaiming_uniform_(a, a=5 ** 0.5)
    return a, nn.Parameter(torch.zeros(out_dim, rank))


class LoRAWrapper(nn.Module):
    """A Linear or 1x1 Conv2d plus a trainable low-rank residual (training only). B starts at zero.

    (A, B) is the student's delta; `aux_A` / `aux_B` an optional second one (`add_aux`). `active` picks the delta that
    runs: "student" (default), "aux", or None for the host layer alone.
    """

    def __init__(self, host: nn.Module, rank: int):
        super().__init__()
        self.host = host
        self.rank = rank
        for p in self.host.parameters():
            p.requires_grad = False
        self.A, self.B = _low_rank_pair(host.weight.shape[0], host.weight.shape[1], rank)
        self.aux_A: Optional[nn.Parameter] = None
        self.aux_B: Optional[nn.Parameter] = None
        self.active: Optional[str] = "student"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.host(x)
        if self.active is None:
            return h
        a, b = (self.A, self.B) if self.active == "student" else (self.aux_A, self.aux_B)
        # Under autocast the low-rank ops run in the autocast dtype; an fp32 copy of the activation would only cost
        # memory.
        xa = x if torch.is_autocast_enabled() else x.to(a.dtype)
        if isinstance(self.host, nn.Linear):
            r = F.linear(F.linear(xa, a), b)
        else:
            r = F.conv2d(F.conv2d(xa, a[:, :, None, None]), b[:, :, None, None])
        return h + r.to(h.dtype)


def _targets(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    out = []
    for name, mod in model.named_modules():
        if not TARGET_PATTERN.search(name):
            continue
        if isinstance(mod, nn.Linear) or (isinstance(mod, nn.Conv2d) and mod.kernel_size == (1, 1) and mod.groups == 1):
            out.append((name, mod))
    return out


def attach(model: nn.Module, rank: int) -> Dict[str, LoRAWrapper]:
    """Wrap every target layer in a LoRAWrapper and freeze everything else. Returns {module path: wrapper}."""
    wrappers: Dict[str, LoRAWrapper] = {}
    for name, mod in _targets(model):
        parent, attr = name.rsplit(".", 1)
        w = LoRAWrapper(mod, rank).to(mod.weight.device)
        setattr(model.get_submodule(parent), attr, w)
        wrappers[name] = w
    if not wrappers:
        raise ValueError("no layers matched the LoRA target pattern")
    for p in model.parameters():
        p.requires_grad_(False)
    for w in wrappers.values():
        w.A.requires_grad_(True)
        w.B.requires_grad_(True)
    return wrappers


def add_aux(wrappers: Dict[str, LoRAWrapper], rank: int) -> List[nn.Parameter]:
    """Give every wrapper a second trainable delta (B at zero, so it starts as the base model). Returns its
    parameters."""
    params: List[nn.Parameter] = []
    for w in wrappers.values():
        a, b = _low_rank_pair(w.host.weight.shape[0], w.host.weight.shape[1], rank)
        w.aux_A = nn.Parameter(a.data.to(w.A.device))
        w.aux_B = nn.Parameter(b.data.to(w.A.device))
        params += [w.aux_A, w.aux_B]
    return params


@contextmanager
def use(wrappers: Dict[str, LoRAWrapper], which: Optional[str]):
    """Run the block with the given delta active in every wrapper ("student", "aux" or None), then restore."""
    before = [w.active for w in wrappers.values()]
    for w in wrappers.values():
        w.active = which
    try:
        yield
    finally:
        for w, b in zip(wrappers.values(), before):
            w.active = b


def detach(model: nn.Module, wrappers: Dict[str, LoRAWrapper]) -> None:
    """Undo `attach`: put the original layers back."""
    for name, w in wrappers.items():
        parent, attr = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), attr, w.host)


def load_into(wrappers: Dict[str, LoRAWrapper], state: Dict[str, torch.Tensor]) -> None:
    with torch.no_grad():
        for name, w in wrappers.items():
            w.A.copy_(state[f"{name}.lora_A"])
            w.B.copy_(state[f"{name}.lora_B"])


def state_from_wrappers(wrappers: Dict[str, LoRAWrapper]) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, w in wrappers.items():
        state[f"{name}.lora_A"] = w.A.detach().float().cpu().contiguous()
        state[f"{name}.lora_B"] = w.B.detach().float().cpu().contiguous()
    return state


def save(path, state: Dict[str, torch.Tensor], meta: dict) -> None:
    from safetensors.torch import save_file
    save_file(state, str(path), metadata={k: json.dumps(v) for k, v in meta.items()})


def load(path) -> Tuple[Dict[str, torch.Tensor], dict]:
    from safetensors import safe_open
    state: Dict[str, torch.Tensor] = {}
    with safe_open(str(path), framework="pt") as f:
        raw = f.metadata() or {}
        for k in f.keys():
            state[k] = f.get_tensor(k)
    meta = {}
    for k, v in raw.items():
        try:
            meta[k] = json.loads(v)
        except (TypeError, ValueError):
            meta[k] = v
    return state, meta


def layer_names(state: Dict[str, torch.Tensor]) -> List[str]:
    return sorted(k[: -len(".lora_A")] for k in state if k.endswith(".lora_A"))


def merge(model: nn.Module, state: Dict[str, torch.Tensor]) -> int:
    """Add the deltas to the weights in place (summed in fp32, rounded once to the weight dtype). Returns the number
    of layers merged."""
    names = layer_names(state)
    if not names:
        raise ValueError("LoRA file holds no '<layer>.lora_A' entries")
    with torch.no_grad():
        for name in names:
            w = model.get_submodule(name).weight
            a = state[f"{name}.lora_A"].to(w.device, torch.float32)
            b = state[f"{name}.lora_B"].to(w.device, torch.float32)
            w.copy_((w.float() + (b @ a).reshape(w.shape)).to(w.dtype))
    return len(names)
