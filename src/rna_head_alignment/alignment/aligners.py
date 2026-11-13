from __future__ import annotations

import torch
from torch import nn


class PerHeadAttentionAligner(nn.Module):
    """Independent affine map for one selected student attention head."""

    def __init__(self) -> None:
        super().__init__()
        self.log_scale = nn.Parameter(torch.zeros(()))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, attn: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(attn * self.log_scale.exp() + self.bias)


class LayerHeadAlignerBank(nn.Module):
    def __init__(self, mapping: dict[str, list[int]]) -> None:
        super().__init__()
        self.aligners = nn.ModuleDict()
        for layer_name, heads in mapping.items():
            for head in heads:
                self.aligners[f"{layer_name}_H{head}"] = PerHeadAttentionAligner()

    def forward_one(self, layer_name: str, head: int, attn: torch.Tensor) -> torch.Tensor:
        return self.aligners[f"{layer_name}_H{head}"](attn)
