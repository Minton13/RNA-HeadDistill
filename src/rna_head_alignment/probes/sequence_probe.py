"""Sequence-level classification probe head."""

from __future__ import annotations

import torch
import torch.nn as nn


class SequenceProbe(nn.Module):
    """Small MLP classifier over a pooled per-sequence feature vector.

    Parameters
    ----------
    input_dim:
        Dimension of the per-token feature vector.
    num_classes:
        Number of classes.  ``2`` means binary classification with a single
        logit and BCE loss.
    hidden_dim:
        Hidden width of the MLP (default 256).
    dropout:
        Dropout probability.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim

        out_dim = 1 if num_classes == 2 else num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward.

        Parameters
        ----------
        features:
            ``[B, L, D]`` per-token features.
        mask:
            ``[B, L]`` bool/int attention mask.  If ``None``, mean-pool over all
            positions.

        Returns
        -------
        ``[B]`` logits for binary classification or ``[B, C]`` for multiclass.
        """
        if mask is None:
            pooled = features.mean(dim=1)
        else:
            mask_f = mask.unsqueeze(-1).float()
            pooled = (features * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp_min(1.0)
        logits = self.net(pooled)
        if self.num_classes == 2:
            return logits.squeeze(-1)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
