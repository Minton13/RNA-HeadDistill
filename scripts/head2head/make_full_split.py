#!/usr/bin/env python
"""Create the Stage 4B train/val/test split for the bucketed full slim cache.

Run after the merge step, e.g.:

    PYTHONPATH=src python scripts/head2head/make_full_split.py \
        --sequence-manifest outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/sequence_manifest.parquet \
        --slim-cache-manifest outputs/teacher_attention_cache_slim/stage4b_head2head/full/manifests/slim_cache_manifest.parquet \
        --out-root outputs/teacher_attention_cache_slim/stage4b_head2head/full \
        --max-position 1024 \
        --seed 42

Stratification is by length bin and available teacher routes; grouping is by
``sequence_hash`` to prevent identical sequences from leaking across splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from safetensors import safe_open


def _fmt_pct(n: int, total: int) -> str:
    return f"{n:,} ({n/total*100:.2f}%)" if total else f"{n:,}"


def _collect_trainable_indices(slim_manifest_df: pd.DataFrame) -> set[int]:
    """Read sequence_manifest_indices from every bucket file."""
    indices: set[int] = set()
    for file_path in slim_manifest_df["file_path"]:
        with safe_open(str(file_path), framework="pt") as f:
            idx = f.get_tensor("sequence_manifest_indices")
            indices.update(int(x) for x in idx.tolist())
    return indices


def _build_strata(seq_df: pd.DataFrame, n_length_bins: int) -> pd.DataFrame:
    """Add length_bin and route_signature columns used for stratification."""
    seq_df = seq_df.copy()
    seq_df["length_bin"] = pd.qcut(
        seq_df["length"],
        q=n_length_bins,
        labels=False,
        duplicates="drop",
    )

    def _route_sig(routes: object) -> str:
        if routes is None:
            return "none"
        if isinstance(routes, np.ndarray):
            routes = routes.tolist()
        return ",".join(str(x) for x in sorted(routes))

    seq_df["route_signature"] = seq_df["available_teacher_routes"].apply(_route_sig)
    seq_df["stratum"] = (
        seq_df["length_bin"].astype(str) + "_" + seq_df["route_signature"]
    )
    return seq_df


def _assign_splits_by_hash(
    hash_df: pd.DataFrame,
    train_frac: float,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, str]:
    """Assign each sequence_hash to train/val/test stratified by stratum."""
    rng = np.random.default_rng(seed)
    split_by_hash: dict[str, str] = {}

    for stratum, group in hash_df.groupby("stratum"):
        hashes = group["sequence_hash"].tolist()
        rng.shuffle(hashes)
        n = len(hashes)
        n_val = int(round(n * val_frac))
        n_test = int(round(n * test_frac))
        # Ensure no negative train count.
        if n_val + n_test >= n:
            n_train = n
            n_val = 0
            n_test = 0
        else:
            n_train = n - n_val - n_test

        for i, h in enumerate(hashes):
            if i < n_train:
                split_by_hash[h] = "train"
            elif i < n_train + n_val:
                split_by_hash[h] = "val"
            else:
                split_by_hash[h] = "test"

    return split_by_hash


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence-manifest",
        default="outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/sequence_manifest.parquet",
        help="sequence_manifest.parquet",
    )
    parser.add_argument(
        "--slim-cache-manifest",
        default="outputs/teacher_attention_cache_slim/stage4b_head2head/full/manifests/slim_cache_manifest.parquet",
        help="slim_cache_manifest.parquet from merge_full_slim_manifest.py",
    )
    parser.add_argument(
        "--out-root",
        default="outputs/teacher_attention_cache_slim/stage4b_head2head/full",
        help="output root; index/ and reports/ subdirs are created",
    )
    parser.add_argument(
        "--max-position",
        type=int,
        default=1024,
        help="student max_position_embeddings; longer sequences are held out",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.98,
        help="fraction of hashes assigned to train",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.01,
        help="fraction of hashes assigned to validation",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.01,
        help="fraction of hashes assigned to test",
    )
    parser.add_argument(
        "--length-bins",
        type=int,
        default=20,
        help="number of length quantile bins for stratification",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for hash-level split",
    )
    args = parser.parse_args()

    if abs(args.train_frac + args.val_frac + args.test_frac - 1.0) > 1e-6:
        raise SystemExit("train/val/test fractions must sum to 1.0")

    out_root = Path(args.out_root)
    index_dir = out_root / "index"
    reports_dir = out_root / "reports"
    index_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    seq_df = pd.read_parquet(args.sequence_manifest)
    slim_manifest = pd.read_parquet(args.slim_cache_manifest)

    # Long-sequence holdout.
    holdout_df = seq_df[seq_df["length"] > args.max_position].copy()
    holdout_path = index_dir / "long_sequence_holdout.parquet"
    holdout_df.to_parquet(holdout_path, index=False)

    # Trainable sequences must be present in the slim cache and short enough.
    trainable_indices = _collect_trainable_indices(slim_manifest)
    short_mask = seq_df["length"] <= args.max_position
    present_mask = pd.Series(seq_df.index.isin(trainable_indices), index=seq_df.index)
    trainable_df = seq_df[short_mask & present_mask].copy()

    # Stratify and split by sequence_hash.
    trainable_df = _build_strata(trainable_df, n_length_bins=args.length_bins)

    # Determine a stable stratum for each sequence_hash.
    hash_df = (
        trainable_df.groupby("sequence_hash")
        .agg(
            n_seqs=("seq_id", "count"),
            length_bin=("length_bin", lambda s: int(s.mode().iloc[0]) if len(s.mode()) else int(s.iloc[0])),
            route_signature=("route_signature", lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0]),
        )
        .reset_index()
    )
    hash_df["stratum"] = hash_df["length_bin"].astype(str) + "_" + hash_df["route_signature"]

    split_by_hash = _assign_splits_by_hash(
        hash_df,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed,
    )

    trainable_df["split"] = trainable_df["sequence_hash"].map(split_by_hash)

    train_df = trainable_df[trainable_df["split"] == "train"].copy()
    val_df = trainable_df[trainable_df["split"] == "val"].copy()
    test_df = trainable_df[trainable_df["split"] == "test"].copy()

    train_path = index_dir / "train.parquet"
    val_path = index_dir / "val.parquet"
    test_path = index_dir / "test.parquet"

    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)

    # ---- Report. ------------------------------------------------------------
    total = len(seq_df)
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    n_holdout = len(holdout_df)
    n_trainable = n_train + n_val + n_test

    lines = [
        "# Stage 4B full split report",
        "",
        "## Overview",
        "",
        f"- total sequences in sequence manifest: {total:,}",
        f"- trainable sequences (length \u003c= {args.max_position} and in slim cache): {n_trainable:,}",
        f"- train: {_fmt_pct(n_train, total)}",
        f"- val: {_fmt_pct(n_val, total)}",
        f"- test: {_fmt_pct(n_test, total)}",
        f"- long-sequence holdout: {_fmt_pct(n_holdout, total)}",
        f"- unique sequence hashes in trainable set: {len(hash_df):,}",
        f"- random seed: {args.seed}",
        "",
        "## Outputs",
        "",
        f"- train: `{train_path}`",
        f"- val: `{val_path}`",
        f"- test: `{test_path}`",
        f"- long holdout: `{holdout_path}`",
        "",
        "## Split by length bin",
        "",
        "| length_bin | train | val | test | total |",
        "|---|---|---|---|---|",
    ]
    bin_summary = (
        trainable_df.groupby("length_bin")["split"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for _, r in bin_summary.iterrows():
        lines.append(
            f"| {int(r['length_bin'])} | {int(r.get('train', 0)):,} | "
            f"{int(r.get('val', 0)):,} | {int(r.get('test', 0)):,} | "
            f"{int(r.get('train', 0) + r.get('val', 0) + r.get('test', 0)):,} |"
        )

    lines.extend(["", "## Split by route signature", "", "| route_signature | train | val | test | total |", "|---|---|---|---|---|"])
    route_summary = (
        trainable_df.groupby("route_signature")["split"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )
    for _, r in route_summary.iterrows():
        lines.append(
            f"| {r['route_signature']} | {int(r.get('train', 0)):,} | "
            f"{int(r.get('val', 0)):,} | {int(r.get('test', 0)):,} | "
            f"{int(r.get('train', 0) + r.get('val', 0) + r.get('test', 0)):,} |"
        )

    report_path = reports_dir / "full_split_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"train: {n_train:,} rows -> {train_path}")
    print(f"val: {n_val:,} rows -> {val_path}")
    print(f"test: {n_test:,} rows -> {test_path}")
    print(f"long holdout: {n_holdout:,} rows -> {holdout_path}")
    print(f"report: {report_path}")


if __name__ == "__main__":
    main()
