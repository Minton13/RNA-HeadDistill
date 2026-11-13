"""Unit tests for Stage 4B head-to-head training helpers (no heavy training)."""

from __future__ import annotations

import pandas as pd
import pytest
import torch

from rna_head_alignment.head2head.head_manifest import make_head_key
from rna_head_alignment.head2head.train_ddp import (
    _build_adapter_specs,
    _lambdas_from_config,
    _resolve_shard_paths,
)


def _make_slim_shard(path, shard_id: str, n_heads: int = 5, lmax: int = 16):
    shard = {
        "schema_version": "slim-1",
        "shard_id": shard_id,
        "seq_ids": [f"{shard_id}_seq{i}" for i in range(2)],
        "sequences": ["ACGUACGUAC", "GGGAAACCC"],
        "lengths": torch.tensor([10, 9], dtype=torch.int32),
        "metadata": {"routes": ["L6", "L8", "L12"]},
    }
    for depth in (6, 8, 12):
        t = torch.rand(2, n_heads, lmax, lmax)
        t = t / t.sum(dim=-1, keepdim=True)
        shard[f"teacher_heads_L{depth}"] = t.half()
        shard[f"teacher_head_ids_L{depth}"] = [
            make_head_key(f"model_{i}", depth, i, f"L{depth}", f"task{depth}")
            for i in range(n_heads)
        ]
    torch.save(shard, path)


def test_resolve_shard_paths_all(tmp_path):
    root = tmp_path / "slim"
    root.mkdir()
    for idx in (0, 5, 12):
        _make_slim_shard(root / f"shard_{idx:05d}.pt", f"{idx:05d}")
    paths = _resolve_shard_paths(root, "all", None, None)
    assert [int(p.stem.split("_")[1]) for p in paths] == [0, 5, 12]


def test_resolve_shard_paths_list(tmp_path):
    root = tmp_path / "slim"
    root.mkdir()
    for idx in (0, 5, 12):
        _make_slim_shard(root / f"shard_{idx:05d}.pt", f"{idx:05d}")
    paths = _resolve_shard_paths(root, [5, 12], None, None)
    assert [int(p.stem.split("_")[1]) for p in paths] == [5, 12]


def test_resolve_shard_paths_stride(tmp_path):
    root = tmp_path / "slim"
    root.mkdir()
    for idx in range(10):
        _make_slim_shard(root / f"shard_{idx:05d}.pt", f"{idx:05d}")
    paths = _resolve_shard_paths(root, [], 3, 2)
    assert [int(p.stem.split("_")[1]) for p in paths] == [0, 2, 4]


def test_build_adapter_specs_matches_slim_order(tmp_path):
    teacher_head_ids = {
        6: [
            make_head_key("m2", 6, 2, "L6", "t6"),
            make_head_key("m0", 6, 0, "L6", "t6"),
            make_head_key("m1", 6, 1, "L6", "t6"),
        ],
    }
    rows = []
    for i, head_key in enumerate(teacher_head_ids[6]):
        parts = head_key.split(":")
        rows.append(
            {
                "task": "t6",
                "route_depth": 6,
                "teacher_model": parts[0],
                "teacher_layer": int(parts[1][1:]),
                "teacher_head": int(parts[2][1:]),
                "teacher_cache_channel": i,
                "student_model": "rinalmo_micro",
                "student_layer": 6,
                "student_head": 4 - i,
                "matching_metric": None,
                "matching_score": float("nan"),
                "pairing_method": "manual_seed",
                "keep": True,
                "adapter_key": f"L6_H{4 - i}",
            }
        )
    # shuffle rows to simulate manifest order different from slim order
    df = pd.DataFrame(rows).sample(frac=1, random_state=7)
    path = tmp_path / "pair_manifest.parquet"
    df.to_parquet(path, index=False)

    adapter_specs, student_heads = _build_adapter_specs(path, teacher_head_ids)
    assert adapter_specs[6] == [
        ("L6_H4", 4),
        ("L6_H3", 3),
        ("L6_H2", 2),
    ]
    assert student_heads[6] == [4, 3, 2]


def test_lambdas_from_config():
    assert _lambdas_from_config({"kl": 2.0, "mse": 3.0, "cos": 0.5}) == (2.0, 3.0, 0.5)
    assert _lambdas_from_config({"kl": 1.0}) == (1.0, 1.0, 0.1)
