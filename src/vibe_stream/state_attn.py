"""Linear self-attention state carried from frame to frame (experimental).

VIBE's self-attention (Sana's ReLU linear attention, `SanaLinearAttnProcessor2_0`) reduces a frame's keys and values
to one (d+1) x d matrix per head, S = [V; 1] K^T, which every query reads out. `StateCarry` replaces the processor of
each block's `attn1` with one that lets frame t read S_t + alpha * S'_{t-1}, S' accumulated over the earlier frames of
its sequence, so a token can also draw on the previous frames' tokens with similar features. alpha is per block,
learnable (a sigmoid of `logit`); alpha = 0 is the unchanged model.

A batch holds `n` sequences of `group` consecutive frames each, in time order. The group is looked up by batch size
(`groups`; other sizes use `default_group`, None meaning the whole batch), not set per call, because gradient
checkpointing reruns the forward during backward. With `slot` set, the last frame's state is kept per (block, slot)
and starts the next call with the same slot (the stream: one slot per denoising step), and `reset` clears it; with
`slot` None nothing is kept.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

ALPHA_KEY = "state_carry_alpha"   # per-block alpha in a student file


class StateCarry:
    def __init__(self, transformer, alpha: float | torch.Tensor = 0.5, learn: bool = False):
        blocks = transformer.transformer_blocks
        a = torch.as_tensor(alpha, dtype=torch.float32).expand(len(blocks)).clamp(1e-4, 1 - 1e-4)
        device = next(transformer.parameters()).device
        self.logit = torch.nn.Parameter(torch.log(a / (1 - a)).to(device), requires_grad=learn)
        self.groups: Dict[int, int] = {}
        self.default_group: Optional[int] = None      # for batch sizes not in `groups`; None: the whole batch
        self.slot: Optional[object] = None
        self.prev: Dict[tuple, torch.Tensor] = {}
        for i, block in enumerate(blocks):
            block.attn1.set_processor(_Processor(self, i))

    def alphas(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)

    def reset(self) -> None:
        self.prev = {}


class _Processor:
    def __init__(self, carry: StateCarry, index: int):
        self.carry, self.index = carry, index

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None):
        c = self.carry
        dtype = hidden_states.dtype
        q, k, v = attn.to_q(hidden_states), attn.to_k(hidden_states), attn.to_v(hidden_states)
        if attn.norm_q is not None:
            q = attn.norm_q(q)
        if attn.norm_k is not None:
            k = attn.norm_k(k)
        q = F.relu(q.transpose(1, 2).unflatten(1, (attn.heads, -1))).float()                    # B H d N
        k = F.relu(k.transpose(1, 2).unflatten(1, (attn.heads, -1)).transpose(2, 3)).float()    # B H N d
        v = F.pad(v.transpose(1, 2).unflatten(1, (attn.heads, -1)).float(), (0, 0, 0, 1), value=1.0)  # B H d+1 N
        s = torch.matmul(v, k)                                                                   # B H d+1 d
        b = s.shape[0]
        g = c.groups.get(b, c.default_group or b)
        if b % g:
            raise ValueError(f"batch {b} is not a whole number of sequences of {g}")
        alpha = torch.sigmoid(c.logit[self.index])
        key = (self.index, c.slot)
        seq = s.unflatten(0, (b // g, g))
        prev = c.prev.get(key) if c.slot is not None else None
        acc = []
        for j in range(g):
            cur = seq[:, j] if prev is None else seq[:, j] + alpha * prev
            acc.append(cur)
            prev = cur
        if c.slot is not None:
            c.prev[key] = prev[-1:].detach()
        s = torch.stack(acc, 1).flatten(0, 1)
        out = torch.matmul(s, q)
        out = (out[:, :, :-1] / (out[:, :, -1:] + 1e-15)).flatten(1, 2).transpose(1, 2).to(dtype)
        return attn.to_out[1](attn.to_out[0](out))
