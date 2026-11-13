"""Lightweight unit tests for the data-driven head-pair search."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")
pytest.importorskip("scipy")

import torch

from rna_head_alignment.head2head.pair_search import (
    compute_student_head_outputs,
    score_head_pairs,
    solve_pairing,
)


def _make_slim_shard(B: int = 2, T: int = 2, L: int = 4, d: int = 8):
    """Build a deterministic slim shard and student outputs.

    Teacher head 0 is eye-like; teacher head 1 is uniform.  Student head 1 is
    constructed to have an eye-like cosine map, so it should be the best match
    for teacher head 0.
    """
    A = torch.zeros(B, T, L, L)
    for b in range(B):
        A[b, 0] = torch.eye(L) * 0.8 + 0.05
        A[b, 1] = torch.ones(L, L) / L

    S = torch.randn(B, 3, L, d)
    for b in range(B):
        for i in range(L):
            S[b, 1, i] = 0.0
            S[b, 1, i, i % d] = 1.0

    lengths = torch.tensor([L, max(1, L - 1)])
    slim = {
        "teacher_heads_L6": A,
        "teacher_head_ids_L6": ["t0", "t1"],
        "lengths": lengths,
    }
    return slim, S


def test_score_head_pairs_columns_and_shape():
    slim, S = _make_slim_shard()
    scores = score_head_pairs(
        slim,
        S,
        route_depth=6,
        metrics=["mse", "cosine", "pearson", "spearman"],
        spearman_sample_size=16,
    )
    assert set(scores.columns) == {
        "teacher_idx",
        "teacher_head_key",
        "student_head",
        "mse",
        "cosine",
        "pearson",
        "spearman",
    }
    assert len(scores) == 2 * 3  # 2 teacher heads x 3 student heads


def test_best_student_head_matches_teacher_zero():
    slim, S = _make_slim_shard()
    scores = score_head_pairs(
        slim,
        S,
        route_depth=6,
        metrics=["mse", "cosine", "pearson", "spearman"],
        spearman_sample_size=16,
    )
    t0_scores = scores[scores["teacher_idx"] == 0]
    best_by_mse = t0_scores.loc[t0_scores["mse"].idxmin(), "student_head"]
    best_by_cosine = t0_scores.loc[t0_scores["cosine"].idxmax(), "student_head"]
    assert best_by_mse == 1
    assert best_by_cosine == 1


def test_solve_pairing_bipartite_assigns_all_teachers():
    slim, S = _make_slim_shard()
    scores = score_head_pairs(
        slim,
        S,
        route_depth=6,
        metrics=["mse", "cosine", "pearson", "spearman"],
        spearman_sample_size=16,
    )
    solved = solve_pairing(scores, method="bipartite")
    assert len(solved) == 2
    assert solved["pairing_method"].iloc[0] == "bipartite"
    assert solved["matching_metric"].iloc[0] == "combined"
    # Bipartite matching should not assign the same student head twice here.
    assert solved["student_head"].nunique() == len(solved)


def test_solve_pairing_top1_has_one_row_per_teacher():
    slim, S = _make_slim_shard()
    scores = score_head_pairs(
        slim,
        S,
        route_depth=6,
        metrics=["mse", "cosine"],
        spearman_sample_size=16,
    )
    solved = solve_pairing(scores, method="top1")
    assert len(solved) == 2
    assert solved["pairing_method"].iloc[0] == "top1"


def test_compute_student_head_outputs_shape(tiny_rinalmo_model_dir):
    pytest.importorskip("transformers")
    pytest.importorskip("multimolecule")
    from rna_head_alignment.head2head.student_backbone import RiNALMoStudentBackbone

    torch.manual_seed(0)
    backbone = RiNALMoStudentBackbone.from_config(model_dir=tiny_rinalmo_model_dir)
    backbone.eval()
    sequences = ["AUGCUU", "AUGC"]
    lengths = [6, 4]
    outputs = compute_student_head_outputs(backbone, sequences, lengths, route_depth=6)
    assert outputs.ndim == 4
    assert outputs.shape[0] == 2
    assert outputs.shape[1] == backbone.num_heads
    assert outputs.shape[2] == max(lengths)
    assert outputs.shape[3] == backbone.head_dim
