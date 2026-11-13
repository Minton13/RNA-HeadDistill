from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

from rna_head_alignment.io.paths import get_device, load_yaml
from rna_head_alignment.models.attention_extractors import prepare_tokenizer_inputs
from rna_head_alignment.models.loaders import load_frozen_model
from rna_head_alignment.models.registry import get_model_spec
from rna_head_alignment.models.token_mapping import select_token_positions


@dataclass
class StudentAttentionConfig:
    layer: int
    heads: list[int]
    display_layer: int | None = None


class RiNALMoMicroStudent:
    def __init__(self, models_yaml: str = "configs/models.yaml", device: str = "auto") -> None:
        self.spec = get_model_spec(models_yaml, "rinalmo_micro")
        self.device = get_device(device)
        tokenizer, model = load_frozen_model(self.spec, self.device)
        for p in model.parameters():
            p.requires_grad_(True)
        self.tokenizer = tokenizer
        self.model = model

    def extract_selected_attentions(
        self,
        sequence: str,
        selected: dict[str, StudentAttentionConfig],
        max_len: int = 512,
    ) -> dict[str, torch.Tensor]:
        seq = self.spec.normalize_sequence(sequence)[:max_len]
        encoded = prepare_tokenizer_inputs(self.tokenizer, seq, max_len=max_len)
        selection = select_token_positions(encoded, len(seq))
        model_input_keys = {"input_ids", "attention_mask", "token_type_ids", "position_ids", "inputs_embeds"}
        encoded_device = {
            k: v.to(self.device)
            for k, v in encoded.items()
            if isinstance(v, torch.Tensor) and k in model_input_keys
        }
        outputs = self.model(
            **encoded_device,
            output_attentions=True,
            output_hidden_states=False,
            return_dict=True,
        )
        if outputs.attentions is None:
            raise RuntimeError("Student model returned no attentions")
        attn = torch.stack(outputs.attentions, dim=0)[:, 0]
        n_layers, n_heads = int(attn.shape[0]), int(attn.shape[1])
        pos = selection.positions.to(attn.device)
        attn = attn.index_select(-2, pos).index_select(-1, pos)
        if attn.shape[-1] != len(seq):
            flat = attn.reshape(-1, 1, attn.shape[-2], attn.shape[-1])
            attn = F.interpolate(flat, size=(len(seq), len(seq)), mode="bilinear", align_corners=False)
            attn = attn.reshape(n_layers, n_heads, len(seq), len(seq))
        out = {}
        for name, cfg in selected.items():
            if cfg.layer >= n_layers:
                raise IndexError(
                    f"{name}: requested 0-based layer index {cfg.layer}, but student returned {n_layers} layers"
                )
            out[name] = attn[cfg.layer, cfg.heads]
        return out


def load_student_head_config(path: str = "configs/student_alignment.yaml") -> dict[str, StudentAttentionConfig]:
    cfg = load_yaml(path)
    out = {}
    for bank_name, item in cfg.get("student", {}).get("selected_heads", {}).items():
        heads = item.get("heads")
        if not isinstance(heads, list):
            heads = list(range(int(item.get("n_heads", 1))))
        display_layer = int(item.get("layer", item.get("layer_index", 0)))
        if "layer_index" in item:
            layer_index = int(item["layer_index"])
        else:
            layer_index = int(item["layer"])
        out[bank_name] = StudentAttentionConfig(
            layer=layer_index,
            heads=[int(h) for h in heads],
            display_layer=display_layer,
        )
    return out
