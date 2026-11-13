"""Unit tests for the head-to-head alignment losses."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rna_head_alignment.head2head.losses import (
    head_alignment_loss,
    route_alignment_loss,
    total_route_loss,
)


def _row_normalized(shape: tuple[int, ...], seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(*shape, generator=g)
    return x / x.sum(dim=-1, keepdim=True)


def test_perfect_prediction_gives_zero_loss():
    lengths = torch.tensor([5, 3])
    teacher = _row_normalized((2, 6, 6), seed=1)
    total, comps = head_alignment_loss(teacher.clone(), teacher.clone(), lengths)
    assert total.item() == pytest.approx(0.0, abs=1e-5)
    for key in ("kl", "mse", "cos", "total"):
        assert key in comps
    assert comps["kl"].item() >= -1e-6
    assert comps["mse"].item() == pytest.approx(0.0, abs=1e-7)


def test_padding_region_is_ignored():
    lengths = torch.tensor([4, 2])
    pred = _row_normalized((2, 6, 6), seed=2)
    teacher = _row_normalized((2, 6, 6), seed=3)
    base, _ = head_alignment_loss(pred, teacher, lengths)

    g = torch.Generator().manual_seed(99)
    junk_pred = pred.clone()
    junk_teacher = teacher.clone()
    for b, l in enumerate(lengths.tolist()):
        # finite garbage (including negatives) outside [:L, :L]
        junk_pred[b, l:, :] = torch.randn(6 - l, 6, generator=g) * 5
        junk_pred[b, :, l:] = torch.randn(6, 6 - l, generator=g) * 5
        junk_teacher[b, l:, :] = torch.randn(6 - l, 6, generator=g) * 5
        junk_teacher[b, :, l:] = torch.randn(6, 6 - l, generator=g) * 5
    perturbed, _ = head_alignment_loss(junk_pred, junk_teacher, lengths)
    assert torch.isfinite(perturbed)
    assert perturbed.item() == pytest.approx(base.item(), abs=1e-6)


def test_kl_nonnegative_for_row_normalized_inputs():
    lengths = torch.tensor([6, 4])
    pred = _row_normalized((2, 6, 6), seed=4)
    teacher = _row_normalized((2, 6, 6), seed=5)
    _, comps = head_alignment_loss(pred, teacher, lengths)
    assert comps["kl"].item() >= -1e-6


def test_route_loss_components_and_weighting():
    lengths = torch.tensor([5, 5])
    pred = _row_normalized((2, 3, 5, 5), seed=6)
    teacher = _row_normalized((2, 3, 5, 5), seed=7)
    route_loss, per_head = route_alignment_loss(pred, teacher, lengths)
    assert len(per_head) == 3
    for comps in per_head:
        for key in ("kl", "mse", "cos", "total"):
            assert key in comps
            assert torch.isfinite(comps[key])
    expected = torch.stack([c["total"] for c in per_head]).mean()
    assert route_loss.item() == pytest.approx(expected.item(), abs=1e-6)

    weighted, _ = route_alignment_loss(pred, teacher, lengths, head_weights=[1.0, 0.0, 0.0])
    assert weighted.item() == pytest.approx(per_head[0]["total"].item(), abs=1e-6)


def test_total_route_loss():
    losses = {"L6": torch.tensor(2.0), "L8": torch.tensor(3.0), "L12": torch.tensor(5.0)}
    assert total_route_loss(losses).item() == pytest.approx(10.0)
    weighted = total_route_loss(losses, {"L6": 0.5, "L8": 2.0, "L12": 0.0})
    assert weighted.item() == pytest.approx(1.0 + 6.0 + 0.0)
