"""Tests for build_default_head_pair_manifest against the real bank config."""

from __future__ import annotations

from pathlib import Path

import pytest

from rna_head_alignment.head2head.head_manifest import (
    PAIR_MANIFEST_COLUMNS,
    build_default_head_pair_manifest,
    build_target_head_manifest,
)

BANK_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "teacher_bank_selected.yaml"
)


@pytest.fixture(scope="module")
def pair_manifest():
    return build_default_head_pair_manifest(BANK_PATH)


def test_fifteen_rows(pair_manifest):
    assert len(pair_manifest) == 15
    assert list(pair_manifest.columns) == PAIR_MANIFEST_COLUMNS


def test_student_heads_0_to_4_per_route(pair_manifest):
    assert set(pair_manifest["student_model"]) == {"rinalmo_micro"}
    for depth in (6, 8, 12):
        sub = pair_manifest[pair_manifest["route_depth"] == depth]
        assert sorted(sub["student_head"]) == [0, 1, 2, 3, 4]
        assert set(sub["student_layer"]) == {depth}


def test_adapter_keys_unique(pair_manifest):
    assert pair_manifest["adapter_key"].is_unique
    assert pair_manifest["adapter_key"].str.match(r"^L(6|8|12)_H[0-4]$").all()


def test_each_teacher_head_paired_exactly_once(pair_manifest):
    target = build_target_head_manifest(BANK_PATH)
    target_keys = set(
        zip(
            target["teacher_model"],
            target["teacher_layer"],
            target["teacher_head"],
        )
    )
    pair_keys = list(
        zip(
            pair_manifest["teacher_model"],
            pair_manifest["teacher_layer"],
            pair_manifest["teacher_head"],
        )
    )
    # same 15 teacher heads, each appearing exactly once in the pairs
    assert set(pair_keys) == target_keys
    assert len(set(pair_keys)) == len(pair_keys) == 15


def test_manual_seed_defaults(pair_manifest):
    assert set(pair_manifest["pairing_method"]) == {"manual_seed"}
    assert pair_manifest["keep"].all()
    assert pair_manifest["matching_score"].isna().all()
    assert pair_manifest["matching_metric"].isna().all()


def test_teacher_channel_matches_target_manifest(pair_manifest):
    target = build_target_head_manifest(BANK_PATH)
    channel_of = {
        (r["teacher_model"], r["teacher_layer"], r["teacher_head"]): r["cache_channel"]
        for _, r in target.iterrows()
    }
    for _, row in pair_manifest.iterrows():
        key = (row["teacher_model"], row["teacher_layer"], row["teacher_head"])
        assert row["teacher_cache_channel"] == channel_of[key]
