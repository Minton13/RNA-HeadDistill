from __future__ import annotations

from dataclasses import dataclass
import gc
from typing import Any

import torch

from .registry import ModelSpec
from .token_mapping import (
    attention_to_nucleotide,
    hidden_to_nucleotide,
    select_token_positions,
)


@dataclass
class AttentionBundle:
    model: str
    model_type: str
    seq_id: str
    sequence: str
    token_mask: torch.Tensor
    hidden_states: torch.Tensor | None
    attentions: torch.Tensor | None
    metadata: dict[str, Any]

    @property
    def length(self) -> int:
        return len(self.sequence)


def _is_codon_tokenizer(tokenizer: Any) -> bool:
    """Return True if the tokenizer requires the sequence length to be a multiple of 3."""
    # Some tokenizers expose the flag directly; otherwise probe with a 4-nt sequence.
    if getattr(tokenizer, "codon", False):
        return True
    try:
        tokenizer(
            "ACGU",
            return_tensors="pt",
            return_attention_mask=True,
            return_special_tokens_mask=True,
        )
        return False
    except ValueError as exc:
        if "multiple of 3" in str(exc).lower():
            return True
        raise


def _pad_to_codon_multiple(sequence: str, pad_char: str = "N") -> tuple[str, int]:
    """Pad ``sequence`` so its length is divisible by 3.

    Returns ``(padded_sequence, original_length)``.
    """
    original_length = len(sequence)
    pad_len = (-original_length) % 3
    return sequence + pad_char * pad_len, original_length


def prepare_tokenizer_inputs(tokenizer: Any, sequence: str, max_len: int | None = None) -> dict[str, torch.Tensor]:
    kwargs = {
        "return_tensors": "pt",
        "return_attention_mask": True,
        "return_special_tokens_mask": True,
        "truncation": bool(max_len),
    }
    if max_len:
        kwargs["max_length"] = max_len
    if _is_codon_tokenizer(tokenizer):
        sequence, _ = _pad_to_codon_multiple(sequence)
    try:
        encoded = tokenizer(sequence, **kwargs)
    except TypeError:
        kwargs.pop("return_special_tokens_mask", None)
        encoded = tokenizer(sequence, **kwargs)
    if "special_tokens_mask" in encoded and "input_ids" in encoded:
        ids = encoded["input_ids"]
        mask = encoded["special_tokens_mask"]
        if ids.numel() != mask.numel():
            special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
            encoded["special_tokens_mask"] = torch.zeros_like(ids)
            for special_id in special_ids:
                encoded["special_tokens_mask"] |= (ids == int(special_id)).long()
    if "special_tokens_mask" not in encoded and "input_ids" in encoded:
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        if special_ids:
            ids = encoded["input_ids"]
            encoded["special_tokens_mask"] = torch.zeros_like(ids)
            for special_id in special_ids:
                encoded["special_tokens_mask"] |= (ids == int(special_id)).long()
    return encoded


@torch.no_grad()
def extract_attention_bundle(
    spec: ModelSpec,
    tokenizer: Any,
    model: torch.nn.Module,
    sequence: str,
    seq_id: str = "seq0",
    max_len: int | None = None,
    device: str = "cpu",
    selected_heads: list[dict[str, Any]] | None = None,
) -> AttentionBundle:
    seq = spec.normalize_sequence(sequence)
    if max_len:
        seq = seq[:max_len]
    encoded = prepare_tokenizer_inputs(tokenizer, seq, max_len=max_len)
    selection = select_token_positions(encoded, len(seq))
    model_input_keys = {"input_ids", "attention_mask", "token_type_ids", "position_ids", "inputs_embeds"}
    encoded_device = {
        k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor) and k in model_input_keys
    }
    try:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(spec.name == "rinalmo_mega")):
            outputs = model(
                **encoded_device,
                output_hidden_states=True,
                output_attentions=True,
                return_dict=True,
            )
    except TypeError:
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(spec.name == "rinalmo_mega")):
            outputs = model(**encoded_device, return_dict=True)
    hidden = getattr(outputs, "hidden_states", None)
    attn = getattr(outputs, "attentions", None)
    hidden_nt = hidden_to_nucleotide(hidden, selection) if hidden is not None else None
    attn_nt = attention_to_nucleotide(attn, selection, selected_heads=selected_heads) if attn is not None else None
    if attn_nt is not None and attn_nt.shape[-1] != len(seq):
        raise ValueError(
            f"{spec.name}: attention length {attn_nt.shape[-1]} does not match sequence length {len(seq)}"
        )
    if hidden_nt is not None and hidden_nt.shape[-2] != len(seq):
        raise ValueError(
            f"{spec.name}: hidden length {hidden_nt.shape[-2]} does not match sequence length {len(seq)}"
        )
    return AttentionBundle(
        model=spec.name,
        model_type=spec.model_type,
        seq_id=seq_id,
        sequence=seq,
        token_mask=torch.ones(len(seq), dtype=torch.bool),
        hidden_states=hidden_nt,
        attentions=attn_nt,
        metadata={
            "token_count": selection.token_count,
            "mapping_method": selection.method,
            "has_hidden": hidden_nt is not None,
            "has_attentions": attn_nt is not None,
        },
    )


def _ensure_special_tokens_mask(encoded: dict[str, torch.Tensor], tokenizer: Any) -> dict[str, torch.Tensor]:
    if "input_ids" not in encoded:
        return encoded
    ids = encoded["input_ids"]
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
    if "special_tokens_mask" in encoded:
        mask = encoded["special_tokens_mask"]
        if mask.shape == ids.shape:
            return encoded
    encoded = dict(encoded)
    encoded["special_tokens_mask"] = torch.zeros_like(ids)
    for special_id in special_ids:
        encoded["special_tokens_mask"] |= (ids == int(special_id)).long()
    return encoded


@torch.no_grad()
def extract_attention_bundle_batch(
    spec: ModelSpec,
    tokenizer: Any,
    model: torch.nn.Module,
    items: list[dict[str, Any]],
    max_len: int | None = None,
    device: str = "cpu",
    max_tokens_per_batch: int = 4096,
    output_hidden_states: bool = True,
    output_attentions: bool = True,
    selected_heads: list[dict[str, Any]] | None = None,
) -> list[AttentionBundle]:
    """Extract attention bundles for a batch of sequences.

    Sequences are grouped by length to keep padding small and GPU utilization high.
    """
    if not items:
        return []

    # Normalize and truncate sequences, keep original index for reordering.
    is_codon = _is_codon_tokenizer(tokenizer)
    indexed = []
    for idx, item in enumerate(items):
        seq = spec.normalize_sequence(item["sequence"])
        if max_len:
            seq = seq[:max_len]
        original_length = len(seq)
        if is_codon:
            seq, original_length = _pad_to_codon_multiple(seq)
        indexed.append(
            {
                "idx": idx,
                "seq_id": item["seq_id"],
                "sequence": seq,
                "original_length": original_length,
                "length": original_length,
            }
        )

    # Group by exact length to avoid padding influencing attention values.
    # Padding alters the attention maps for some models (e.g. nucleotide-transformer-v2-500m),
    # so we only batch sequences with identical nucleotide length.  Each group is
    # also capped by max_tokens_per_batch to keep GPU memory bounded.
    ordered = sorted(indexed, key=lambda x: x["length"])
    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    current_tokens = 0
    for item in ordered:
        if current:
            same_len = item["length"] == current[0]["length"]
            fits = current_tokens + item["length"] <= max_tokens_per_batch
            if not same_len or not fits:
                groups.append(current)
                current = [item]
                current_tokens = item["length"]
            else:
                current.append(item)
                current_tokens += item["length"]
        else:
            current = [item]
            current_tokens = item["length"]
    if current:
        groups.append(current)

    results: list[AttentionBundle | None] = [None] * len(items)
    model_input_keys = {"input_ids", "attention_mask", "token_type_ids", "position_ids", "inputs_embeds"}

    for group in groups:
        seqs = [item["sequence"] for item in group]
        encoded = tokenizer(
            seqs,
            return_tensors="pt",
            return_attention_mask=True,
            return_special_tokens_mask=True,
            padding=True,
            truncation=bool(max_len),
            max_length=max_len if max_len else None,
        )
        encoded = _ensure_special_tokens_mask(encoded, tokenizer)
        encoded_device = {
            k: v.to(device) for k, v in encoded.items() if isinstance(v, torch.Tensor) and k in model_input_keys
        }
        try:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(spec.name == "rinalmo_mega")):
                outputs = model(
                    **encoded_device,
                    output_hidden_states=output_hidden_states,
                    output_attentions=output_attentions,
                    return_dict=True,
                )
        except TypeError:
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(spec.name == "rinalmo_mega")):
                outputs = model(**encoded_device, return_dict=True)
        hidden = getattr(outputs, "hidden_states", None) if output_hidden_states else None
        attn = getattr(outputs, "attentions", None) if output_attentions else None

        for j, item in enumerate(group):
            seq = item["sequence"]
            single_encoded = {k: v[j : j + 1] for k, v in encoded.items() if isinstance(v, torch.Tensor)}
            selection = select_token_positions(single_encoded, item["original_length"])
            if hidden is not None:
                hidden_stacked = hidden if isinstance(hidden, torch.Tensor) else torch.stack(list(hidden), dim=0)
                hidden_nt = hidden_to_nucleotide(hidden_stacked[:, j : j + 1], selection)
            else:
                hidden_nt = None
            if attn is not None:
                if isinstance(attn, torch.Tensor):
                    attn_input = attn[:, j : j + 1]
                else:
                    attn_input = [a[j : j + 1] for a in attn]
                attn_nt = attention_to_nucleotide(attn_input, selection, selected_heads=selected_heads)
            else:
                attn_nt = None
            target_len = item["original_length"]
            if attn_nt is not None and attn_nt.shape[-1] != target_len:
                raise ValueError(
                    f"{spec.name}: attention length {attn_nt.shape[-1]} does not match sequence length {target_len}"
                )
            if hidden_nt is not None and hidden_nt.shape[-2] != target_len:
                raise ValueError(
                    f"{spec.name}: hidden length {hidden_nt.shape[-2]} does not match sequence length {target_len}"
                )
            results[item["idx"]] = AttentionBundle(
                model=spec.name,
                model_type=spec.model_type,
                seq_id=item["seq_id"],
                sequence=item["sequence"][: item["original_length"]],
                token_mask=torch.ones(item["original_length"], dtype=torch.bool),
                hidden_states=hidden_nt,
                attentions=attn_nt,
                metadata={
                    "token_count": selection.token_count,
                    "mapping_method": selection.method,
                    "has_hidden": hidden_nt is not None,
                    "has_attentions": attn_nt is not None,
                },
            )
        # Aggressive cleanup after each group to prevent cross-group memory
        # accumulation, which is the main cause of OOM on long-sequence shards.
        del outputs, hidden, attn, encoded, encoded_device
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return [r for r in results if r is not None]
