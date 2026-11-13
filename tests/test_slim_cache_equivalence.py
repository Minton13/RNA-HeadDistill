"""Tests for build_slim_shard + verify_slim_equivalence on a fake mini cache."""

from __future__ import annotations

import torch

from rna_head_alignment.head2head.slim_cache import (
    SLIM_SCHEMA_VERSION,
    build_slim_shard,
    verify_slim_equivalence,
)


def test_build_and_verify_full_shard(fake_teacher_cache, tmp_path):
    out_root = tmp_path / "slim"
    report = build_slim_shard(
        0, fake_teacher_cache.cache_root, out_root, fake_teacher_cache.manifest
    )
    assert report["ok"], report["error"]
    assert report["n_seqs"] == 6
    assert report["n_dropped"] == 0
    assert report["route_heads"] == {"L6": 1, "L8": 1, "L12": 1}

    slim = torch.load(out_root / "shard_00000.pt", map_location="cpu", weights_only=False)
    assert slim["schema_version"] == SLIM_SCHEMA_VERSION
    assert slim["shard_id"] == "00000"

    results = verify_slim_equivalence(
        0, fake_teacher_cache.cache_root, out_root / "shard_00000.pt",
        fake_teacher_cache.manifest,
    )
    assert len(results) == 3
    assert all(r["equal"] for r in results), results


def test_dropped_seqs_recorded_never_silent(fake_teacher_cache, tmp_path):
    out_root = tmp_path / "slim"
    report = build_slim_shard(
        1, fake_teacher_cache.cache_root, out_root, fake_teacher_cache.manifest
    )
    assert report["ok"], report["error"]
    assert report["n_dropped"] == 2
    assert sorted(report["dropped_seq_ids"]) == fake_teacher_cache.dropped
    assert report["n_seqs"] == 4
    assert any("dropped" in w for w in report["warnings"])

    slim = torch.load(out_root / "shard_00001.pt", map_location="cpu", weights_only=False)
    assert slim["seq_ids"] == ["t0", "t2", "t4", "t5"]  # reference order kept
    assert slim["metadata"]["dropped_seq_ids"] == fake_teacher_cache.dropped or \
        sorted(slim["metadata"]["dropped_seq_ids"]) == fake_teacher_cache.dropped
    assert slim["metadata"]["n_dropped"] == 2

    results = verify_slim_equivalence(
        1, fake_teacher_cache.cache_root, out_root / "shard_00001.pt",
        fake_teacher_cache.manifest,
    )
    assert all(r["equal"] for r in results), results


def test_slim_stores_metadata_once_and_no_valid_mask(fake_teacher_cache, tmp_path):
    out_root = tmp_path / "slim"
    build_slim_shard(0, fake_teacher_cache.cache_root, out_root, fake_teacher_cache.manifest)
    slim = torch.load(out_root / "shard_00000.pt", map_location="cpu", weights_only=False)

    assert "valid_mask" not in slim
    # exactly one copy of seq metadata
    assert sum(1 for k in slim if k == "seq_ids") == 1
    assert isinstance(slim["seq_ids"], list) and isinstance(slim["sequences"], list)
    assert slim["lengths"].dtype == torch.int32
    # attention groups are fp16 with one channel per target head
    for depth, n in ((6, 1), (8, 1), (12, 1)):
        tensor = slim[f"teacher_heads_L{depth}"]
        assert tensor.dtype == torch.float16
        assert tensor.shape[1] == n
        assert len(slim[f"teacher_head_ids_L{depth}"]) == n


def test_head_key_mismatch_raises_in_strict_mode(fake_teacher_cache, tmp_path):
    manifest = fake_teacher_cache.manifest.copy()
    manifest.loc[0, "head_key"] = "alpha:L1:H0:L6:does_not_exist"
    try:
        build_slim_shard(
            0, fake_teacher_cache.cache_root, tmp_path / "slim", manifest, strict=True
        )
    except KeyError:
        pass
    else:
        raise AssertionError("strict mode should raise on unknown head_key")


def test_head_key_mismatch_skipped_in_nonstrict_mode(fake_teacher_cache, tmp_path):
    manifest = fake_teacher_cache.manifest.copy()
    manifest.loc[0, "head_key"] = "alpha:L1:H0:L6:does_not_exist"
    report = build_slim_shard(
        0, fake_teacher_cache.cache_root, tmp_path / "slim", manifest, strict=False
    )
    assert report["ok"], report["error"]
    assert "alpha:L1:H0:L6:does_not_exist" in report["skipped_heads"]
    assert report["route_heads"]["L6"] == 0
    assert report["warnings"]
