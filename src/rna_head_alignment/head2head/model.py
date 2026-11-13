"""Stage 4B head-to-head student: RiNALMo-micro backbone + 15 alignment adapters.

One adapter per preserved teacher target head. Adapter keys follow the bank-yaml
aligner naming convention ``'L{route}_H{k}'`` (e.g. ``'L6_H0'``), 5 adapters per
route at layers 6/8/12 → 15 adapters total.

Channel ordering contract
-------------------------
``pred_L6[:, t]`` corresponds to ``adapter_specs[6][t]``. Callers must ensure
this order matches the target head manifest and the slim shard's
``teacher_head_ids_L6`` list, so that ``pred_L6[:, t]`` aligns against
``teacher_heads_L6[:, t]``.
"""

from __future__ import annotations

import torch
from torch import nn

from .alignment_transformer import AlignmentAdapter, LinearAlignmentAdapter
from .student_backbone import DEFAULT_ROUTES, DEFAULT_STUDENT_HEADS, RiNALMoStudentBackbone


def default_adapter_specs(
    routes: tuple[int, ...] = DEFAULT_ROUTES,
    student_heads: dict[int, list[int]] | None = None,
) -> dict[int, list[tuple[str, int]]]:
    """route -> list of (adapter_key, paired student head index)."""
    heads = student_heads or DEFAULT_STUDENT_HEADS
    return {
        route: [(f"L{route}_H{k}", head) for k, head in enumerate(heads[route])]
        for route in routes
    }


class RiNALMoHeadToHeadStudent(nn.Module):
    """RiNALMo-micro student with per-teacher-head alignment adapters.

    Parameters
    ----------
    backbone:
        A :class:`RiNALMoStudentBackbone` (real weights or tiny test config).
    adapter_specs:
        route (1-based) -> ordered list of ``(adapter_key, student_head_index)``.
        Defaults to :func:`default_adapter_specs` (L6/L8/L12 × heads 0-4).
    adapter_dim:
        Hidden width of every adapter.
    ffn_multiplier:
        FFN hidden-size multiplier for every adapter (default 2).  Only used
        when ``adapter_type`` is ``"transformer"``.
    adapter_type:
        ``"transformer"`` for :class:`AlignmentAdapter`, ``"linear"`` for
        :class:`LinearAlignmentAdapter`.
    """

    def __init__(
        self,
        backbone: RiNALMoStudentBackbone,
        adapter_specs: dict[int, list[tuple[str, int]]] | None = None,
        adapter_dim: int = 64,
        ffn_multiplier: int = 2,
        adapter_type: str = "transformer",
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.adapter_specs = (
            adapter_specs
            if adapter_specs is not None
            else default_adapter_specs(
                routes=backbone.routes, student_heads=backbone.student_heads
            )
        )
        self.adapter_type = adapter_type
        self.adapters = nn.ModuleDict()
        for route, specs in self.adapter_specs.items():
            for key, _head_idx in specs:
                if adapter_type == "linear":
                    self.adapters[key] = LinearAlignmentAdapter(
                        input_dim=backbone.head_dim,
                        adapter_dim=adapter_dim,
                    )
                else:
                    self.adapters[key] = AlignmentAdapter(
                        input_dim=backbone.head_dim,
                        adapter_dim=adapter_dim,
                        ffn_multiplier=ffn_multiplier,
                    )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_backbone: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Returns ``{'pred_L6': [B, T6, L, L], 'pred_L8': ..., 'pred_L12': ...}``.

        ``pred_L{r}[:, t]`` is the predicted map of the t-th adapter in
        ``adapter_specs[r]``. Channel order must match the caller's teacher head
        ordering (see module docstring).
        """
        exits = self.backbone(input_ids, attention_mask)
        key_padding_mask = attention_mask == 0
        out: dict[str, torch.Tensor] = {}
        for route, specs in self.adapter_specs.items():
            head_outputs = exits[route]["head_outputs"]  # [B, K, L, head_dim]
            preds = []
            for k, (key, _head_idx) in enumerate(specs):
                preds.append(self.adapters[key](head_outputs[:, k], key_padding_mask)["pred_map"])
            out[f"pred_L{route}"] = torch.stack(preds, dim=1)
        if return_backbone:
            out["backbone"] = exits  # type: ignore[assignment]
        return out

    # ------------------------------------------------------------------ utils

    def count_parameters(self) -> dict[str, int]:
        """Trainable-parameter counts: backbone, adapters, total."""
        backbone = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        adapters = sum(p.numel() for p in self.adapters.parameters() if p.requires_grad)
        return {"backbone": backbone, "adapters": adapters, "total": backbone + adapters}

    def route_parameter_groups(self) -> dict:
        """Parameter groups for optimizer construction / diagnostics.

        Returns ``{'L6': [...], 'L8': [...], 'L12': [...], 'backbone':
        {layer_idx: [params]}}`` where ``layer_idx`` is the 0-based backbone
        layer index and covers embeddings (key ``-1``), encoder layers, and the
        encoder final norm / pooler (key ``num_layers``).
        """
        groups: dict = {}
        for route, specs in self.adapter_specs.items():
            params: list[torch.nn.Parameter] = []
            for key, _ in specs:
                params.extend(self.adapters[key].parameters())
            groups[f"L{route}"] = params

        by_layer: dict[int, list[torch.nn.Parameter]] = {}
        for name, param in self.backbone.model.named_parameters():
            if name.startswith("encoder.layer."):
                layer_idx = int(name.split(".")[2])
            elif name.startswith("embeddings"):
                layer_idx = -1
            else:  # encoder.layer_norm, pooler, ...
                layer_idx = self.backbone.num_layers
            by_layer.setdefault(layer_idx, []).append(param)
        groups["backbone"] = by_layer
        return groups
