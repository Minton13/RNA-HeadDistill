"""Unit tests for the MLM utilities (small-student S0/S1 arms)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rna_head_alignment.head2head.mlm import (  # noqa: E402
    RiNALMoMLMHead,
    mask_tokens,
    mlm_loss_from_logits,
)
from rna_head_alignment.head2head.student_backbone import (  # noqa: E402
    RiNALMoStudentBackbone,
)


def _batch(bsz: int = 2, len1: int = 12, len2: int = 8):
    torch.manual_seed(0)
    max_tok = len1 + 2  # bos/eos
    input_ids = torch.randint(6, 26, (bsz, max_tok))
    attention_mask = torch.zeros(bsz, max_tok, dtype=torch.long)
    attention_mask[0, : len1 + 2] = 1
    attention_mask[1, : len2 + 2] = 1
    return input_ids, attention_mask


def test_mask_tokens_rate_and_labels():
    input_ids, attention_mask = _batch()
    orig = input_ids.clone()
    gen = torch.Generator().manual_seed(7)
    masked, labels = mask_tokens(input_ids, attention_mask, mlm_prob=0.15, generator=gen)

    # Input unchanged.
    assert torch.equal(input_ids, orig)
    selected = labels != -100
    # Only eligible (attention_mask == 1) positions are labelled.
    assert attention_mask.bool()[selected].all()
    assert not ((attention_mask == 0) & selected).any().item()
    # Labels carry the original token ids.
    assert torch.equal(labels[selected], input_ids[selected])
    # Masked ids differ from original only at selected positions.
    assert torch.equal(masked[~selected], input_ids[~selected])


def test_mask_tokens_rate_statistical():
    torch.manual_seed(0)
    input_ids = torch.randint(6, 26, (64, 128))
    attention_mask = torch.ones(64, 128, dtype=torch.long)
    gen = torch.Generator().manual_seed(123)
    masked, labels = mask_tokens(input_ids, attention_mask, mlm_prob=0.15, generator=gen)
    frac = (labels != -100).float().mean().item()
    assert 0.12 < frac < 0.18
    mask_frac = (masked == 4).float().mean().item()
    # ~80% of selected become [MASK]
    assert 0.09 < mask_frac < 0.135


def test_mlm_head_tied_and_loss_backward(tiny_rinalmo_model_dir):
    backbone = RiNALMoStudentBackbone.from_config(model_dir=tiny_rinalmo_model_dir)
    head = RiNALMoMLMHead(backbone)
    # Weight tying.
    assert head.decoder.weight is backbone.model.embeddings.word_embeddings.weight

    input_ids, attention_mask = _batch()
    gen = torch.Generator().manual_seed(3)
    masked, labels = mask_tokens(input_ids, attention_mask, generator=gen)
    out = backbone.model(input_ids=masked, attention_mask=attention_mask, return_dict=True)
    logits = head(out.last_hidden_state)
    assert logits.shape == (2, input_ids.shape[1], 26)
    loss = mlm_loss_from_logits(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert backbone.model.embeddings.word_embeddings.weight.grad is not None


def test_mlm_loss_empty_selection():
    logits = torch.randn(2, 5, 26)
    labels = torch.full((2, 5), -100)
    loss = mlm_loss_from_logits(logits, labels)
    assert float(loss) == 0.0
