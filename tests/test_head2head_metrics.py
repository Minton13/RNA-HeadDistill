"""Unit tests for Stage 4B head-to-head evaluation metrics."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rna_head_alignment.head2head.metrics import (
    compute_head_metrics,
    masked_cosine_similarity,
    masked_mae,
    masked_mse,
    masked_pearson_r,
    masked_spearman_rho,
    rowwise_kl,
    top_k_overlap,
)


def _row_normalized(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(*shape, generator=g)
    return x / x.sum(dim=-1, keepdim=True)


def test_masked_mse_ignores_padding():
    lengths = torch.tensor([3, 2])
    pred = _row_normalized((2, 4, 4), seed=1)
    target = _row_normalized((2, 4, 4), seed=2)
    base = masked_mse(pred, target, _pair_mask(lengths, 4))

    junk_pred = pred.clone()
    junk_target = target.clone()
    for b, l in enumerate(lengths.tolist()):
        junk_pred[b, l:, :] = torch.randn(4 - l, 4) * 5
        junk_pred[b, :, l:] = torch.randn(4, 4 - l) * 5
        junk_target[b, l:, :] = torch.randn(4 - l, 4) * 5
        junk_target[b, :, l:] = torch.randn(4, 4 - l) * 5
    perturbed = masked_mse(junk_pred, junk_target, _pair_mask(lengths, 4))
    assert perturbed.item() == pytest.approx(base.item(), abs=1e-6)


def test_perfect_prediction_metrics():
    lengths = torch.tensor([5, 4])
    target = _row_normalized((2, 6, 6), seed=3)
    metrics = compute_head_metrics(
        target.unsqueeze(1),
        target.unsqueeze(1),
        lengths,
        metrics=["mse", "mae", "cosine", "rowwise_kl"],
    )
    assert metrics["mse"].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["mae"].item() == pytest.approx(0.0, abs=1e-6)
    assert metrics["cosine"].item() == pytest.approx(1.0, abs=1e-5)
    assert metrics["rowwise_kl"].item() == pytest.approx(0.0, abs=1e-5)


def test_per_sequence_shapes():
    B, T, L = 3, 4, 8
    lengths = torch.tensor([6, 5, 4])
    pred = _row_normalized((B, T, L, L), seed=4)
    target = _row_normalized((B, T, L, L), seed=5)
    metrics = compute_head_metrics(pred, target, lengths, include_per_sequence=True)
    assert metrics["mse"].shape == (T,)
    assert metrics["mse_per_seq"].shape == (B, T)


def test_top_overlap_deterministic_and_bounded():
    lengths = torch.tensor([4, 3])
    pred = _row_normalized((2, 4, 4), seed=6)
    target = pred.clone()
    overlap = top_k_overlap(pred, target, lengths, "L")
    assert 0.0 <= overlap.item() <= 1.0
    assert overlap.item() == pytest.approx(1.0, abs=1e-5)


def test_correlation_perfect_rank():
    lengths = torch.tensor([5])
    pred = torch.arange(25, dtype=torch.float32).view(1, 5, 5)
    target = pred.clone()
    mask = _pair_mask(lengths, 5)
    assert masked_pearson_r(pred, target, mask).item() == pytest.approx(1.0, abs=1e-5)
    assert masked_spearman_rho(pred, target, mask).item() == pytest.approx(1.0, abs=1e-5)


def _pair_mask(lengths: torch.Tensor, size: int) -> torch.Tensor:
    idx = torch.arange(size, device=lengths.device)
    valid = idx.unsqueeze(0) < lengths.unsqueeze(1)
    return valid.unsqueeze(2) & valid.unsqueeze(1)
