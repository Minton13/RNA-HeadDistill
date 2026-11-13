"""Masked-language-model (MLM) utilities for the from-scratch small student.

The small-student control experiment trains a randomly-initialised RiNALMo
architecture with MLM (S0) and then with MLM + head-to-head alignment (S1).
This module provides the MLM head (tied to the backbone token embeddings) and
the 80/10/10 masking used by both arms.

Token layout of the staged rinalmo-micro tokenizer (vocab 26):

* 0 pad, 1 bos, 2 eos, 3 unk, 4 mask, 5 null — special, never predicted
* 6..25 — regular nucleotide tokens (random-replacement pool)

The MLM forward pass must use a **separate** masked copy of the batch: the
alignment teacher targets were computed on unmasked sequences, so the
alignment branch always sees the original ``input_ids``.
"""

from __future__ import annotations

import torch
from torch import nn

MASK_TOKEN_ID = 4
FIRST_REGULAR_TOKEN_ID = 6  # ids 0..5 are specials in the rinalmo vocab


class RiNALMoMLMHead(nn.Module):
    """Minimal MLM head: LayerNorm -> tied linear decoder over the vocab.

    The decoder weight is tied to ``backbone.model.embeddings.word_embeddings``
    (the RiNALMo config sets ``tie_word_embeddings: true``).
    """

    def __init__(self, backbone: nn.Module, eps: float = 1e-5) -> None:
        super().__init__()
        word_emb = backbone.model.embeddings.word_embeddings
        hidden = word_emb.weight.shape[1]
        self.layer_norm = nn.LayerNorm(hidden, eps=eps)
        self.decoder = nn.Linear(hidden, word_emb.weight.shape[0], bias=True)
        self.decoder.weight = word_emb.weight  # weight tying
        nn.init.zeros_(self.decoder.bias)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.layer_norm(hidden))


def mask_tokens(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mlm_prob: float = 0.15,
    mask_token_id: int = MASK_TOKEN_ID,
    first_regular_id: int = FIRST_REGULAR_TOKEN_ID,
    vocab_size: int = 26,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """80/10/10 BERT-style masking.

    Returns ``(masked_ids, labels)`` where ``labels`` is ``-100`` everywhere
    except the selected positions. Only positions with ``attention_mask == 1``
    are eligible. ``input_ids`` is not modified.
    """
    device = input_ids.device
    labels = torch.full_like(input_ids, -100)
    eligible = attention_mask.bool()
    prob = torch.rand(input_ids.shape, generator=generator, device=device)
    selected = (prob < mlm_prob) & eligible

    masked_ids = input_ids.clone()

    # 80% -> [MASK]
    r = torch.rand(input_ids.shape, generator=generator, device=device)
    to_mask = selected & (r < 0.8)
    masked_ids[to_mask] = mask_token_id

    # 10% -> random regular token
    to_random = selected & (r >= 0.8) & (r < 0.9)
    n_regular = vocab_size - first_regular_id
    rand_ids = torch.randint(
        first_regular_id, first_regular_id + n_regular, input_ids.shape,
        generator=generator, device=device,
    )
    masked_ids[to_random] = rand_ids[to_random]

    # 10% -> unchanged
    labels[selected] = input_ids[selected]
    return masked_ids, labels


def mlm_loss_from_logits(
    logits: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """Cross-entropy over masked positions; NaN-safe when nothing is masked."""
    if not (labels != -100).any():
        return logits.new_zeros(())
    return nn.functional.cross_entropy(
        logits.view(-1, logits.shape[-1]).float(), labels.view(-1), ignore_index=-100
    )
