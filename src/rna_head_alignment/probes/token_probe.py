"""Per-token regression probe head (chemical reactivity)."""

from __future__ import annotations

import torch
import torch.nn as nn


class TokenProbe(nn.Module):
    """Two-layer MLP applied independently to each token position.

    Parameters
    ----------
    input_dim:
        Per-token feature dimension.
    hidden_dim:
        Hidden width (default 128).
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Return ``[B, L]`` predictions for ``[B, L, D]`` features."""
        return self.net(features).squeeze(-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
