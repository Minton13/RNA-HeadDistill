"""Pairwise contact-map probe head (secondary structure)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class PairProbe(nn.Module):
    """Lightweight bilinear pairwise decoder.

    Computes ``logits[i, j] = H_i @ W @ H_j + b`` and optionally symmetrizes
    the output.  Parameter count is ``D * D + 1``, which is ~230k for the
    RiNALMo-micro hidden size.

    Parameters
    ----------
    input_dim:
        Per-token feature dimension.
    symmetric:
        If ``True`` (default), symmetrize logits as ``(logits + logits.T) / 2``.
    """

    def __init__(self, input_dim: int, symmetric: bool = True) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.symmetric = symmetric
        self.W = nn.Parameter(torch.randn(input_dim, input_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``[B, L, L]`` logits for ``[B, L, D]`` features."""
        # More stable than einsum for large L: scale W to keep initial logits small.
        logits = torch.einsum("bid,de,bje->bij", features, self.W / math.sqrt(self.input_dim), features)
        logits = logits + self.bias
        if self.symmetric:
            logits = (logits + logits.transpose(-1, -2)) * 0.5
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
