from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class TokenSelection:
    positions: torch.Tensor
    token_count: int
    target_length: int
    method: str


def normalize_sequence(sequence: str, model_type: str = "RNA") -> str:
    seq = "".join(ch for ch in sequence.upper() if ch in "ACGTUN")
    return seq.replace("T", "U") if model_type.upper() == "RNA" else seq.replace("U", "T")


def non_special_positions(
    attention_mask: torch.Tensor | None,
    special_tokens_mask: torch.Tensor | None,
    input_length: int,
) -> torch.Tensor:
    if attention_mask is None:
        keep = torch.ones(input_length, dtype=torch.bool)
    else:
        keep = attention_mask.detach().cpu().bool().flatten()
    if special_tokens_mask is not None:
        keep = keep & ~special_tokens_mask.detach().cpu().bool().flatten()
    positions = torch.nonzero(keep, as_tuple=False).flatten()
    if positions.numel() == 0:
        positions = torch.arange(input_length)
    return positions


def select_token_positions(
    encoded: dict[str, torch.Tensor],
    target_length: int,
) -> TokenSelection:
    input_ids = encoded["input_ids"]
    if input_ids.ndim == 2:
        input_length = int(input_ids.shape[1])
    else:
        input_length = int(input_ids.shape[0])
    attention_mask = encoded.get("attention_mask")
    special_tokens_mask = encoded.get("special_tokens_mask")
    if attention_mask is not None and attention_mask.ndim == 2:
        attention_mask = attention_mask[0]
    if special_tokens_mask is not None and special_tokens_mask.ndim == 2:
        special_tokens_mask = special_tokens_mask[0]
    positions = non_special_positions(attention_mask, special_tokens_mask, input_length)
    method = "special_tokens_mask" if special_tokens_mask is not None else "attention_mask_fallback"
    return TokenSelection(
        positions=positions,
        token_count=int(positions.numel()),
        target_length=target_length,
        method=method,
    )


def _interpolate_square(attn: torch.Tensor, target_length: int) -> torch.Tensor:
    if attn.shape[-1] == target_length:
        return attn
    flat = attn.reshape(-1, 1, attn.shape[-2], attn.shape[-1])
    resized = F.interpolate(flat, size=(target_length, target_length), mode="bilinear", align_corners=False)
    out = resized.reshape(*attn.shape[:-2], target_length, target_length)
    out = out.clamp_min(0)
    denom = out.sum(dim=-1, keepdim=True).clamp_min(1e-8)
    return out / denom


def attention_to_nucleotide(
    attentions: tuple[torch.Tensor, ...] | list[torch.Tensor] | torch.Tensor,
    selection: TokenSelection,
    selected_heads: list[dict[str, Any]] | None = None,
) -> torch.Tensor:
    # Keep operations on the original device (usually GPU) to avoid slow PCI
    # copies of the full attention tensor. The caller is responsible for moving
    # the small selected-head result to CPU if needed.
    #
    # To avoid a large OOM spike on models with many layers (rinalmo_giga etc.)
    # we process tuple/list inputs layer-by-layer instead of stacking all layers
    # into one contiguous tensor.  When selected_heads is provided we only keep
    # those heads, returning [K, L, L] instead of [layers, H, L, L].
    if selected_heads is not None:
        processed: list[torch.Tensor] = []
        for h in selected_heads:
            layer_idx = int(h["layer"])
            head_idx = int(h["head"])
            if isinstance(attentions, torch.Tensor):
                # [layers, B, H, T, T]
                a = attentions[layer_idx, 0, head_idx]
            else:
                # tuple/list of [B, H, T, T]
                a = attentions[layer_idx][0, head_idx]
            pos = selection.positions.to(a.device)
            a = a.index_select(-2, pos).index_select(-1, pos)
            a = a.float()
            denom = a.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            a = a / denom
            a = _interpolate_square(a, selection.target_length)
            processed.append(a)
        return torch.stack(processed, dim=0)

    # Full attention path (used by extract_attention_bundle without head filter)
    if isinstance(attentions, torch.Tensor):
        stacked = attentions.detach()
        if stacked.ndim != 5:
            raise ValueError(f"Expected attentions [layers,B,H,T,T], got {tuple(stacked.shape)}")
        attn = stacked[:, 0]
        pos = selection.positions.to(attn.device)
        attn = attn.index_select(-2, pos).index_select(-1, pos)
        # Renormalize over the selected positions so each query position is a
        # proper probability distribution over the nucleotide-level target.
        attn = attn.float()
        denom = attn.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attn = attn / denom
        return _interpolate_square(attn, selection.target_length)

    # Tuple/list path without head filter: each element is [B, H, T, T].
    processed = []
    for a in attentions:
        a = a.detach()
        if a.ndim != 4:
            raise ValueError(f"Expected attention layer [B,H,T,T], got {tuple(a.shape)}")
        attn_layer = a[0]
        pos = selection.positions.to(attn_layer.device)
        attn_layer = attn_layer.index_select(-2, pos).index_select(-1, pos)
        attn_layer = attn_layer.float()
        denom = attn_layer.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        attn_layer = attn_layer / denom
        attn_layer = _interpolate_square(attn_layer, selection.target_length)
        processed.append(attn_layer)
    return torch.stack(processed, dim=0)


def _interpolate_hidden(hidden: torch.Tensor, target_length: int) -> torch.Tensor:
    if hidden.shape[-2] == target_length:
        return hidden
    layers, tokens, dim = hidden.shape
    x = hidden.permute(0, 2, 1)
    x = F.interpolate(x, size=target_length, mode="linear", align_corners=False)
    return x.permute(0, 2, 1).reshape(layers, target_length, dim)


def hidden_to_nucleotide(
    hidden_states: tuple[torch.Tensor, ...] | list[torch.Tensor] | torch.Tensor,
    selection: TokenSelection,
) -> torch.Tensor:
    if isinstance(hidden_states, torch.Tensor):
        stacked = hidden_states.detach()
    else:
        stacked = torch.stack([h.detach() for h in hidden_states], dim=0)
    if stacked.ndim != 4:
        raise ValueError(f"Expected hidden states [layers,B,T,D], got {tuple(stacked.shape)}")
    hidden = stacked[:, 0]
    pos = selection.positions.to(hidden.device)
    hidden = hidden.index_select(-2, pos).float()
    return _interpolate_hidden(hidden, selection.target_length)
