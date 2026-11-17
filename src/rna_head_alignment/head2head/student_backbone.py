"""RiNALMo-micro student backbone for Stage 4B head-to-head alignment.

Wraps the locally staged RiNALMo-micro HF checkpoint (``assets/models/rinalmo-micro``)
and exposes, in a **single** forward pass, the three alignment exits at layers
6/8/12 (1-based; 0-based indices 5/7/11):

- ``hidden``        [B, L, 480]  — the layer output hidden states at each route.
- ``head_outputs``  [B, K, L, 24] — the value-mixed per-head outputs
  ``attn_probs[h] @ V[h]`` of the paired native student heads, taken *before*
  the attention output projection.

Per-head extraction — introspection notes (multimolecule 0.0.x, transformers 5.7)
------------------------------------------------------------------------------
``RiNALMoModel`` (``multimolecule.models.rinalmo.modeling_rinalmo``) has:

- ``encoder.layer.{i}.attention.self.{query,key,value}`` — bias-free
  ``nn.Linear(hidden, hidden)``;
- ``RiNALMoAttention.forward`` applies ``attention.layer_norm`` first and feeds
  the *normed* hidden states into ``RiNALMoSelfAttention``;
- ``RiNALMoSelfAttention.forward`` ends with
  ``context_layer = attn_probs @ value_layer``  ([B, H, L, head_dim]),
  then ``transpose(1, 2).contiguous().view(B, L, H * head_dim)`` and returns it
  as ``outputs[0]`` — i.e. the concatenated per-head value-mixed outputs before
  ``attention.output.dense``.

Therefore a plain **forward hook on ``encoder.layer.{i}.attention.self``**
captures ``outputs[0]`` which we reshape to ``[B, L, H, head_dim]`` and slice the
paired heads. This is the exact tensor the model itself computes (rotary,
masking, scaling all applied by the model), and gradients flow through it —
unlike ``output_attentions=True`` which is designed for inspection. The hidden
states are captured with a forward hook on ``encoder.layer.{i}``. Older
MultiMolecule versions return a tuple (``outputs[0]``); newer versions return
the hidden-state tensor directly.

The local checkpoint stores weights under a ``model.`` wrapper prefix while
``AutoModel`` instantiates ``RiNALMoModel`` whose parameters are unprefixed; we
reuse ``_load_raw_state_dict`` / ``_strip_wrapper_prefix`` from
``rna_head_alignment.models.loaders`` to remap them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn

from rna_head_alignment.models.loaders import _load_raw_state_dict, _strip_wrapper_prefix

# 1-based route layer -> paired native student heads (0-based head indices).
# Default until the head-pair manifest (parquet) is wired in; callers may pass
# an explicit mapping to override.
DEFAULT_STUDENT_HEADS: dict[int, list[int]] = {
    6: [0, 1, 2, 3, 4],
    8: [0, 1, 2, 3, 4],
    12: [0, 1, 2, 3, 4],
}

DEFAULT_ROUTES: tuple[int, ...] = (6, 8, 12)  # 1-based

DEFAULT_MODEL_DIR = "assets/models/rinalmo-micro"

# Tiny test config. NOTE: the spec's draft (4 heads / head_dim 12) cannot host
# the 5 paired native heads required per route, so the tiny default uses 6
# heads (head_dim 8, still even for rotary) on the same 48-dim / 12-layer trunk.
_TINY_CONFIG_DEFAULTS: dict[str, Any] = {
    "hidden_size": 48,
    "num_attention_heads": 6,
    "num_hidden_layers": 12,
    "intermediate_size": 96,
    "attention_dropout": 0.0,
    "hidden_dropout": 0.0,
    "token_dropout": False,
}


class RiNALMoStudentBackbone(nn.Module):
    """Trainable RiNALMo-micro wrapper exposing the layer 6/8/12 alignment exits.

    Parameters
    ----------
    model_dir:
        Local HF snapshot directory of rinalmo-micro.
    routes:
        1-based layer indices of the alignment exits.
    student_heads:
        Mapping route layer (1-based) -> list of paired native head indices
        (0-based). Defaults to :data:`DEFAULT_STUDENT_HEADS`.
    config_overrides:
        If given, the model is built from config (random weights) with these
        overrides applied on top of the base rinalmo-micro config — used for
        tiny CPU test models. Real weights are *not* loaded in this mode.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        routes: tuple[int, ...] = DEFAULT_ROUTES,
        student_heads: dict[int, list[int]] | None = None,
        config_overrides: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        import multimolecule  # noqa: F401  (registers the rinalmo model type)
        from transformers import AutoConfig, AutoModel

        self.model_dir = Path(model_dir)
        self.routes = tuple(routes)
        self.student_heads = {
            int(k): list(v) for k, v in (student_heads or DEFAULT_STUDENT_HEADS).items()
        }

        config = AutoConfig.from_pretrained(self.model_dir, local_files_only=True)
        if config_overrides:
            for key, value in config_overrides.items():
                setattr(config, key, value)
        self.num_layers = int(config.num_hidden_layers)
        self.num_heads = int(config.num_attention_heads)
        self.hidden_size = int(config.hidden_size)
        self.head_dim = self.hidden_size // self.num_heads

        for route in self.routes:
            if not 1 <= route <= self.num_layers:
                raise ValueError(f"route layer {route} out of range 1..{self.num_layers}")
            for head in self.student_heads.get(route, []):
                if not 0 <= head < self.num_heads:
                    raise ValueError(f"student head {head} out of range 0..{self.num_heads - 1}")

        self.model = AutoModel.from_config(config, trust_remote_code=True)
        if not config_overrides:
            state = _load_raw_state_dict(self.model_dir)
            if state is None:
                raise RuntimeError(f"no checkpoint found under {self.model_dir}")
            state = _strip_wrapper_prefix(state, self.model)
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing and len(missing) > 0.05 * sum(1 for _ in self.model.parameters()):
                raise RuntimeError(
                    f"checkpoint mismatch: {len(missing)} missing keys, {len(unexpected)} unexpected"
                )

        # Captured activations, filled by hooks on every forward pass.
        self._captured: dict[int, dict[str, torch.Tensor]] = {}
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        for route in self.routes:
            layer = self.model.encoder.layer[route - 1]
            self._hooks.append(
                layer.register_forward_hook(self._make_layer_hook(route))
            )
            self._hooks.append(
                layer.attention.self.register_forward_hook(self._make_self_attn_hook(route))
            )

    # ------------------------------------------------------------------ hooks

    def _make_layer_hook(self, route: int):
        def hook(module: nn.Module, args: tuple, output: torch.Tensor | tuple) -> None:
            slot = self._captured.setdefault(route, {})
            # Preserve the batch dimension for both MultiMolecule layer APIs.
            slot["hidden"] = output if isinstance(output, torch.Tensor) else output[0]

        return hook

    def _make_self_attn_hook(self, route: int):
        def hook(module: nn.Module, args: tuple, output: tuple) -> None:
            slot = self._captured.setdefault(route, {})
            # RiNALMoSelfAttention outputs[0]: context [B, L, H * head_dim],
            # i.e. concat over heads of attn_probs[h] @ V[h] (see module docstring).
            slot["head_context"] = output[0]

        return hook

    # --------------------------------------------------------------- forward

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_all_heads: bool = False,
    ) -> dict[int, dict[str, torch.Tensor]]:
        """One forward pass over all 12 layers; returns the route exits.

        Parameters
        ----------
        return_all_heads:
            If ``True``, each route dict also contains ``all_head_outputs``
            with shape ``[B, H, L, head_dim]`` for *every* native attention head
            at that layer (used by the data-driven head-pair search). The
            default ``head_outputs`` (paired heads only) is always returned.

        Returns
        -------
        dict mapping route layer (1-based int) -> {"hidden": [B, L, D],
        "head_outputs": [B, K, L, head_dim]} and optionally
        "all_head_outputs": [B, H, L, head_dim].
        """
        self._captured = {}
        self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
        )
        out: dict[int, dict[str, torch.Tensor]] = {}
        for route in self.routes:
            slot = self._captured[route]
            context = slot["head_context"]  # [B, L, H * head_dim]
            bsz, seq_len = context.shape[0], context.shape[1]
            per_head = context.view(bsz, seq_len, self.num_heads, self.head_dim)
            heads = self.student_heads[route]
            selected = per_head[:, :, heads, :].permute(0, 2, 1, 3).contiguous()
            route_out: dict[str, torch.Tensor] = {
                "hidden": slot["hidden"],
                "head_outputs": selected,
            }
            if return_all_heads:
                route_out["all_head_outputs"] = per_head.permute(
                    0, 2, 1, 3
                ).contiguous()
            out[route] = route_out
        return out

    # -------------------------------------------------------------- utilities

    def tokenizer(self):
        """Load the local rinalmo-micro tokenizer (offline)."""
        import multimolecule  # noqa: F401
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            self.model_dir, local_files_only=True, use_fast=False
        )

    @classmethod
    def from_config(
        cls,
        config_overrides: dict[str, Any] | None = None,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        routes: tuple[int, ...] = DEFAULT_ROUTES,
        student_heads: dict[int, list[int]] | None = None,
    ) -> "RiNALMoStudentBackbone":
        """Build a randomly-initialized backbone from config (for tests).

        ``config_overrides`` are applied on top of the staged rinalmo-micro
        config; missing tiny-model keys fall back to
        ``{"hidden_size": 48, "num_attention_heads": 6, "num_hidden_layers": 12,
        "intermediate_size": 96, ...}`` (head_dim 8; 6 heads so the 5 paired
        heads per route fit — see the note above ``_TINY_CONFIG_DEFAULTS``).
        """
        overrides = dict(_TINY_CONFIG_DEFAULTS)
        overrides.update(config_overrides or {})
        return cls(
            model_dir=model_dir,
            routes=routes,
            student_heads=student_heads,
            config_overrides=overrides,
        )
