"""Sequence/teacher-length consistency checks on slim shards."""

from __future__ import annotations

import pytest
import torch

from rna_head_alignment.head2head.head_manifest import ROUTE_DEPTHS
from rna_head_alignment.head2head.slim_cache import build_slim_shard


@pytest.fixture
def slim_shards(fake_teacher_cache, tmp_path):
    out_root = tmp_path / "slim"
    paths = {}
    for idx in (0, 1):
        report = build_slim_shard(
            idx, fake_teacher_cache.cache_root, out_root, fake_teacher_cache.manifest
        )
        assert report["ok"], report["error"]
        paths[idx] = out_root / f"shard_{idx:05d}.pt"
    return paths


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_lengths_leq_lmax(slim_shards, fake_teacher_cache):
    for path in slim_shards.values():
        slim = _load(path)
        lmax = slim["metadata"]["Lmax"]
        assert lmax == fake_teacher_cache.lmax
        assert int(slim["lengths"].max()) <= lmax
        assert int(slim["lengths"].min()) >= 1


def test_attention_shape_matches_batch_and_lmax(slim_shards):
    for path in slim_shards.values():
        slim = _load(path)
        batch = len(slim["seq_ids"])
        lmax = slim["metadata"]["Lmax"]
        for depth in ROUTE_DEPTHS:
            tensor = slim[f"teacher_heads_L{depth}"]
            n_heads = len(slim[f"teacher_head_ids_L{depth}"])
            assert tensor.shape == (batch, n_heads, lmax, lmax), (
                path,
                depth,
                tuple(tensor.shape),
            )


def test_seq_metadata_lengths_consistent(slim_shards):
    for path in slim_shards.values():
        slim = _load(path)
        n = len(slim["seq_ids"])
        assert len(slim["sequences"]) == n
        assert slim["lengths"].shape == (n,)


def test_no_duplicate_seq_ids(slim_shards):
    for path in slim_shards.values():
        slim = _load(path)
        ids = slim["seq_ids"]
        assert len(set(ids)) == len(ids), f"duplicate seq_ids in {path}"
