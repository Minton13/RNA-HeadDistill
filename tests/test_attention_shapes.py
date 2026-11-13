from types import SimpleNamespace

import torch

from rna_head_alignment.models.attention_extractors import extract_attention_bundle
from rna_head_alignment.models.registry import ModelSpec


class FakeTokenizer:
    def __call__(self, sequence, **kwargs):
        # BOS + nucleotides + EOS
        L = len(sequence) + 2
        return {
            "input_ids": torch.arange(L).unsqueeze(0),
            "attention_mask": torch.ones(1, L, dtype=torch.long),
            "special_tokens_mask": torch.tensor([[1] + [0] * len(sequence) + [1]], dtype=torch.long),
        }


class FakeModel(torch.nn.Module):
    def forward(self, **kwargs):
        T = kwargs["input_ids"].shape[1]
        h = torch.randn(1, T, 8)
        attn = torch.softmax(torch.randn(1, 2, T, T), dim=-1)
        return SimpleNamespace(hidden_states=(h, h), attentions=(attn, attn))


def test_attention_extraction_removes_special_tokens():
    spec = ModelSpec(name="fake", model_type="RNA", local_dir=".", attention_teacher=True)
    bundle = extract_attention_bundle(spec, FakeTokenizer(), FakeModel(), "AUGCU", device="cpu")
    assert bundle.attentions.shape == (2, 2, 5, 5)
    assert bundle.hidden_states.shape == (2, 5, 8)
    assert bundle.token_mask.shape == (5,)
