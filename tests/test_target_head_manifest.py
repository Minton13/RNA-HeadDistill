"""Tests for build_target_head_manifest against the real teacher bank config."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from rna_head_alignment.head2head.head_manifest import (
    TARGET_MANIFEST_COLUMNS,
    build_target_head_manifest,
    compute_model_channel_map,
)

BANK_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "teacher_bank_selected.yaml"
)

HEAD_KEY_RE = re.compile(r"^[a-z0-9_]+:L\d+:H\d+:L(6|8|12):[a-zA-Z0-9_]+$")

# Known channel counts per model (from the teacher cache, K per model).
EXPECTED_K = {
    "rnafm": 1,
    "mrnafm": 1,
    "utrlm_mrl": 1,
    "moderngena_base": 1,
    "rinalmo_mega": 1,
    "nucleotide_transformer_v2_500m": 1,
    "rinalmo_giga": 2,
    "splicebert": 2,
    "plantrna_fm": 5,
}


@pytest.fixture(scope="module")
def manifest():
    return build_target_head_manifest(BANK_PATH)


def test_bank_exists():
    assert BANK_PATH.exists(), f"missing bank config: {BANK_PATH}"


def test_fifteen_rows_five_per_route(manifest):
    assert len(manifest) == 15
    counts = manifest.groupby("route_depth").size().to_dict()
    assert counts == {6: 5, 8: 5, 12: 5}


def test_columns_complete(manifest):
    assert list(manifest.columns) == TARGET_MANIFEST_COLUMNS
    for col in ("task", "teacher_model", "head_key"):
        assert manifest[col].notna().all()


def test_head_key_format(manifest):
    for _, row in manifest.iterrows():
        assert HEAD_KEY_RE.match(row["head_key"]), row["head_key"]
        expected = (
            f"{row['teacher_model']}:L{row['teacher_layer']}:H{row['teacher_head']}"
            f":L{row['route_depth']}:{row['task']}"
        )
        assert row["head_key"] == expected


def test_channel_map_matches_known_k():
    channel_map = compute_model_channel_map(BANK_PATH)
    assert {m: len(keys) for m, keys in channel_map.items()} == EXPECTED_K


def test_cache_channel_consistent_with_channel_map(manifest):
    channel_map = compute_model_channel_map(BANK_PATH)
    for _, row in manifest.iterrows():
        keys = channel_map[row["teacher_model"]]
        assert keys[row["cache_channel"]] == row["head_key"]
    # plantrna_fm must span 5 distinct channels 0..4
    pf = manifest[manifest["teacher_model"] == "plantrna_fm"]
    assert sorted(pf["cache_channel"]) == [0, 1, 2, 3, 4]
    # single-channel models are always channel 0
    for model in ("rnafm", "mrnafm", "utrlm_mrl", "moderngena_base"):
        sub = manifest[manifest["teacher_model"] == model]
        assert (sub["cache_channel"] == 0).all()


def test_ranks_unique_within_route(manifest):
    for depth, sub in manifest.groupby("route_depth"):
        assert sorted(sub["rank"]) == [1, 2, 3, 4, 5]


def test_defaults_keep_and_loss_weight(manifest):
    assert manifest["keep"].all()
    assert (manifest["loss_weight"] == 1.0).all()
    assert set(manifest["student_route_layer"]) == {6, 8, 12}
    assert (
        manifest["student_route_layer"] == manifest["route_depth"]
    ).all()
