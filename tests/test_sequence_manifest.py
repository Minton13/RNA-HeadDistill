"""Tests for the Stage 4B sequence manifest builder."""

from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest
import torch

from rna_head_alignment.head2head.sequence_manifest import build_sequence_manifest


EXPECTED_COLUMNS = [
    "seq_id",
    "sequence",
    "length",
    "original_dataset",
    "original_shard_id",
    "sequence_hash",
    "available_teacher_routes",
]


@pytest.fixture
def fake_slim_cache(tmp_path):
    """Two-shard slim cache with deterministic metadata and one head per route."""
    cache_root = tmp_path / "slim"
    cache_root.mkdir(parents=True, exist_ok=True)

    shards = [
        {
            "shard_id": "00000",
            "seq_ids": ["s0", "s1"],
            "sequences": ["ACGUACGU", "UGCAUGCA"],
            "lengths": torch.tensor([8, 8], dtype=torch.int32),
            "teacher_head_ids_L6": ["alpha:L1:H0:L6:task"],
            "teacher_heads_L6": torch.zeros(2, 1, 8, 8, dtype=torch.float16),
            "teacher_head_ids_L8": ["beta:L2:H1:L8:task"],
            "teacher_heads_L8": torch.zeros(2, 1, 8, 8, dtype=torch.float16),
            "teacher_head_ids_L12": ["gamma:L3:H2:L12:task"],
            "teacher_heads_L12": torch.zeros(2, 1, 8, 8, dtype=torch.float16),
        },
        {
            "shard_id": "00001",
            "seq_ids": ["s2"],
            "sequences": ["AAAACCCCUUUU"],
            "lengths": torch.tensor([12], dtype=torch.int32),
            "teacher_head_ids_L6": ["alpha:L1:H0:L6:task"],
            "teacher_heads_L6": torch.zeros(1, 1, 12, 12, dtype=torch.float16),
            "teacher_head_ids_L8": ["beta:L2:H1:L8:task"],
            "teacher_heads_L8": torch.zeros(1, 1, 12, 12, dtype=torch.float16),
            "teacher_head_ids_L12": ["gamma:L3:H2:L12:task"],
            "teacher_heads_L12": torch.zeros(1, 1, 12, 12, dtype=torch.float16),
        },
    ]
    for shard in shards:
        torch.save(shard, cache_root / f"shard_{shard['shard_id']}.pt")

    source_path = tmp_path / "source.jsonl"
    records = [
        {"seq_id": "s0", "sequence": "ACGUACGU", "length": 8, "source": "rinalmo_pretrain_10pct"},
        {"seq_id": "s1", "sequence": "UGCAUGCA", "length": 8, "source": "rinalmo_pretrain_10pct"},
        {"seq_id": "s2", "sequence": "AAAACCCCUUUU", "length": 12, "source": "rinalmo_pretrain_10pct"},
    ]
    source_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records) + "\n",
        encoding="utf-8",
    )

    return {
        "cache_root": cache_root,
        "source_path": source_path,
        "records": records,
    }


def test_manifest_columns_and_order(fake_slim_cache, tmp_path):
    out = tmp_path / "sequence_manifest.parquet"
    report = build_sequence_manifest(
        fake_slim_cache["source_path"],
        fake_slim_cache["cache_root"],
        out,
    )

    df = pd.read_parquet(out)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert report["n_output_rows"] == 3
    assert report["n_collisions"] == 0


def test_sequence_hash_is_stable_sha256(fake_slim_cache, tmp_path):
    out = tmp_path / "sequence_manifest.parquet"
    build_sequence_manifest(
        fake_slim_cache["source_path"],
        fake_slim_cache["cache_root"],
        out,
    )
    df = pd.read_parquet(out)

    for _, row in df.iterrows():
        expected = hashlib.sha256(row["sequence"].upper().encode("utf-8")).hexdigest()
        assert row["sequence_hash"] == expected


def test_length_matches_sequence_and_teacher_attention(fake_slim_cache, tmp_path):
    out = tmp_path / "sequence_manifest.parquet"
    report = build_sequence_manifest(
        fake_slim_cache["source_path"],
        fake_slim_cache["cache_root"],
        out,
    )

    df = pd.read_parquet(out)
    for _, row in df.iterrows():
        assert row["length"] == len(row["sequence"])

    length_warnings = [w for w in report["warnings"] if "teacher attention L" in w]
    source_len_warnings = [w for w in report["warnings"] if "source length" in w]
    assert not length_warnings
    assert not source_len_warnings


def test_no_collisions_for_unique_sequences(fake_slim_cache, tmp_path):
    out = tmp_path / "sequence_manifest.parquet"
    report = build_sequence_manifest(
        fake_slim_cache["source_path"],
        fake_slim_cache["cache_root"],
        out,
    )
    assert report["n_collisions"] == 0
    assert not report["collisions"]


def test_available_routes_and_shard_ids(fake_slim_cache, tmp_path):
    out = tmp_path / "sequence_manifest.parquet"
    build_sequence_manifest(
        fake_slim_cache["source_path"],
        fake_slim_cache["cache_root"],
        out,
    )
    df = pd.read_parquet(out)

    for _, row in df.iterrows():
        assert list(row["available_teacher_routes"]) == [6, 8, 12]

    s0 = df[df["seq_id"] == "s0"].iloc[0]
    assert list(s0["original_shard_id"]) == ["00000"]

    s2 = df[df["seq_id"] == "s2"].iloc[0]
    assert list(s2["original_shard_id"]) == ["00001"]


def test_collision_detected_for_duplicate_sequence(tmp_path):
    cache_root = tmp_path / "slim"
    cache_root.mkdir()
    torch.save(
        {
            "shard_id": "00000",
            "seq_ids": ["a1", "a2"],
            "sequences": ["ACGU", "ACGU"],
            "lengths": torch.tensor([4, 4], dtype=torch.int32),
            "teacher_head_ids_L6": ["m:L1:H0:L6:t"],
            "teacher_heads_L6": torch.zeros(2, 1, 4, 4, dtype=torch.float16),
        },
        cache_root / "shard_00000.pt",
    )
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"seq_id": "a1", "sequence": "ACGU", "length": 4, "source": "src"})
        + "\n"
        + json.dumps({"seq_id": "a2", "sequence": "ACGU", "length": 4, "source": "src"})
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "manifest.parquet"
    report = build_sequence_manifest(source, cache_root, out)

    assert report["n_collisions"] == 1
    assert {"a1", "a2"} == set(report["collisions"][0]["seq_ids"])


def test_dna_thymine_warning(tmp_path):
    cache_root = tmp_path / "slim"
    cache_root.mkdir()
    torch.save(
        {
            "shard_id": "00000",
            "seq_ids": ["t1"],
            "sequences": ["ACGT"],
            "lengths": torch.tensor([4], dtype=torch.int32),
            "teacher_head_ids_L6": ["m:L1:H0:L6:t"],
            "teacher_heads_L6": torch.zeros(1, 1, 4, 4, dtype=torch.float16),
        },
        cache_root / "shard_00000.pt",
    )
    source = tmp_path / "source.jsonl"
    source.write_text(
        json.dumps({"seq_id": "t1", "sequence": "ACGT", "length": 4, "source": "src"})
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "manifest.parquet"
    report = build_sequence_manifest(source, cache_root, out)

    assert any("thymine" in w for w in report["warnings"])
