"""Alignment adapter: per-teacher-head single-layer Transformer.

Each preserved teacher head gets one :class:`AlignmentAdapter`. It consumes the
value-mixed output of a paired native student head (``[B, L, head_dim]``),
projects it to ``adapter_dim``, runs one pre-LN Transformer block whose
**single-head self-attention softmax map is the predicted attention map**
``Â(d, t)`` — aligned directly against the cached teacher attention ``A(d, t)``.

Parameter count with the defaults (``input_dim=24``, ``adapter_dim=64``):
input_proj 24*64+64 = 1,600; attention q/k/v/o 4*(64*64+64) = 16,640;
FFN 64*128+128 + 128*64+64 = 16,576; two LayerNorms 256; total ≈ 35k trainable
parameters per adapter (a few hundred KB in fp32).
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F


class SingleHeadSelfAttention(nn.Module):
    """Single-head scaled-dot-product self-attention with q/k/v/o projections.

    forward returns ``(out, attn_probs)`` where ``attn_probs`` is ``[B, L, L]``
    (no head dimension). Padded key positions are filled with ``-inf`` before
    the softmax, so rows over real queries remain valid distributions.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.embed_dim)
        if key_padding_mask is not None:
            # key_padding_mask: [B, L] bool, True = padded key position.
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
        attn_probs = F.softmax(scores, dim=-1)
        # Fully-padded rows (should not occur for real queries) come out as NaN
        # from softmax(-inf); zero them so they stay finite.
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        out = self.o_proj(torch.matmul(attn_probs, v))
        return out, attn_probs


class AlignmentAdapter(nn.Module):
    """Input projection + one pre-LN Transformer layer per teacher head.

    ``forward(x, key_padding_mask)`` with ``x`` = ``[B, L, input_dim]`` returns
    ``{"pred_map": [B, L, L], "hidden": [B, L, adapter_dim]}`` — ``pred_map`` is
    the softmax attention map of the internal single-head attention and is the
    quantity aligned against the cached teacher map.
    """

    def __init__(
        self,
        input_dim: int = 24,
        adapter_dim: int = 64,
        ffn_multiplier: int = 2,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, adapter_dim)
        self.ln1 = nn.LayerNorm(adapter_dim)
        self.self_attn = SingleHeadSelfAttention(adapter_dim)
        self.ln2 = nn.LayerNorm(adapter_dim)
        self.ffn = nn.Sequential(
            nn.Linear(adapter_dim, ffn_multiplier * adapter_dim),
            nn.GELU(),
            nn.Linear(ffn_multiplier * adapter_dim, adapter_dim),
        )

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        h = self.input_proj(x)
        attn_out, attn_probs = self.self_attn(self.ln1(h), key_padding_mask)
        h = h + attn_out
        h = h + self.ffn(self.ln2(h))
        return {"pred_map": attn_probs, "hidden": h}


class LinearAlignmentAdapter(nn.Module):
    """Parameter-controlled linear/QK projection adapter producing one attention head.

    Projects the native head output to ``adapter_dim`` with separate Q/K
    projections and computes ``softmax(Q @ K^T / sqrt(adapter_dim))``.  This is
    the lightweight linear alternative to :class:`AlignmentAdapter`.
    """

    def __init__(self, input_dim: int = 24, adapter_dim: int = 64) -> None:
        super().__init__()
        self.adapter_dim = adapter_dim
        self.q_proj = nn.Linear(input_dim, adapter_dim, bias=False)
        self.k_proj = nn.Linear(input_dim, adapter_dim, bias=False)

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        q = self.q_proj(x)  # [B, L, adapter_dim]
        k = self.k_proj(x)  # [B, L, adapter_dim]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.adapter_dim)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
        attn_probs = F.softmax(scores, dim=-1)
        attn_probs = torch.nan_to_num(attn_probs, nan=0.0)
        return {"pred_map": attn_probs, "hidden": q}
