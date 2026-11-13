#!/usr/bin/env python
"""Build the Stage 4B bucketed full slim teacher attention cache.

Run from the project root with the package importable, e.g.:

    PYTHONPATH=src python scripts/head2head/04_build_full_slim_cache.py \
        --shard-idx 0 \
        --cache-root outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads \
        --target-manifest outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/target_head_manifest.parquet \
        --sequence-manifest outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/sequence_manifest.parquet \
        --out-root outputs/teacher_attention_cache_slim/stage4b_head2head/full \
        --max-position 1024

Select shards with ``--shard-idx`` or ``--shard-range``.  Each invocation
writes one partial manifest row under ``<out-root>/partials/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rna_head_alignment.head2head.full_slim_cache import (
    build_bucketed_slim_shard,
    write_partial_manifest,
)
from rna_head_alignment.head2head.head_manifest import ROUTE_DEPTHS, load_manifest


def _parse_shards(shard_idx: int | None, shard_range: str | None) -> list[int]:
    if shard_idx is not None:
        return [shard_idx]
    if shard_range:
        start, _, end = shard_range.partition(":")
        return list(range(int(start), int(end)))
    raise SystemExit("provide --shard-idx or --shard-range")


def _parse_buckets(buckets: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in buckets.split(",") if x.strip())


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n} B"


def _skipped_report(out_root: Path, shard_idx: int) -> dict[str, object]:
    """Synthetic report when all bucket files already exist and --resume is set."""
    shard_id = f"{shard_idx:05d}"
    return {
        "shard_idx": shard_idx,
        "shard_id": shard_id,
        "ok": True,
        "error": None,
        "n_seqs": 0,
        "n_trainable": 0,
        "n_holdout": 0,
        "n_available_any_head": 0,
        "source_bytes": 0,
        "out_bytes": 0,
        "models_used": [],
        "missing_models": [],
        "skipped_heads": [],
        "bucket_files": [],
        "bucket_sizes": [],
        "bucket_seq_counts": [],
        "checksums": {},
        "head_coverage": {},
        "route_coverage": {},
        "validation_metrics": {},
        "warnings": ["skipped (--resume): all bucket files already exist"],
        "build_time_sec": 0.0,
    }


def _all_bucket_files_exist(
    shard_idx: int,
    out_root: Path,
    buckets: tuple[int, ...],
    sequence_manifest: pd.DataFrame,
    max_position: int,
) -> bool:
    """Return True if every non-empty bucket file for the shard already exists."""
    from rna_head_alignment.head2head.full_slim_cache import _bucket_for_length

    shard_id = f"{shard_idx:05d}"

    def _shard_matches(ids: object) -> bool:
        if ids is None:
            return False
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
        return shard_id in {str(x) for x in ids}

    mask = sequence_manifest["original_shard_id"].apply(_shard_matches)
    ref_lengths = sequence_manifest.loc[mask, "length"].astype(int).tolist()
    needed: set[int] = set()
    for L in ref_lengths:
        b, holdout = _bucket_for_length(L, buckets, max_position)
        if not holdout:
            needed.add(b)
    for b in needed:
        path = out_root / "buckets" / f"shard_{shard_id}_bucket_{b}.safetensors"
        if not path.exists() or path.stat().st_size == 0:
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-idx",
        type=int,
        default=None,
        help="single shard index to build",
    )
    parser.add_argument(
        "--shard-range",
        default=None,
        help="a:b (end exclusive) range of shard indices",
    )
    parser.add_argument(
        "--cache-root",
        default="outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads",
        help="original teacher attention cache root",
    )
    parser.add_argument(
        "--target-manifest",
        default="outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/target_head_manifest.parquet",
        help="target_head_manifest.parquet",
    )
    parser.add_argument(
        "--sequence-manifest",
        default="outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/sequence_manifest.parquet",
        help="sequence_manifest.parquet",
    )
    parser.add_argument(
        "--out-root",
        default="outputs/teacher_attention_cache_slim/stage4b_head2head/full",
        help="output root for bucket files and partial manifests",
    )
    parser.add_argument(
        "--buckets",
        default="64,128,256,512,1024",
        help="comma-separated bucket upper bounds",
    )
    parser.add_argument(
        "--max-position",
        type=int,
        default=1024,
        help="student max_position_embeddings; longer sequences are held out",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip shards whose bucket files already exist",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="skip missing head_keys / model shards instead of failing",
    )
    parser.add_argument(
        "--validation-samples",
        type=int,
        default=5,
        help="number of sequences to sample per model for fp16 validation",
    )
    args = parser.parse_args()

    indices = _parse_shards(args.shard_idx, args.shard_range)
    buckets = _parse_buckets(args.buckets)
    out_root = Path(args.out_root)
    partial_dir = out_root / "partials"
    partial_dir.mkdir(parents=True, exist_ok=True)

    target_df = load_manifest(args.target_manifest)
    sequence_df = pd.read_parquet(args.sequence_manifest)

    strict = not args.no_strict
    n_ok = 0
    n_fail = 0

    for idx in indices:
        # A second resume check: if every expected bucket file exists, skip.
        if args.resume and _all_bucket_files_exist(
            idx, out_root, buckets, sequence_df, args.max_position
        ):
            print(f"[full-slim] shard {idx:05d}: all buckets exist, skipping", flush=True)
            report = _skipped_report(out_root, idx)
        else:
            print(f"[full-slim] shard {idx:05d}: building ...", flush=True)
            report = build_bucketed_slim_shard(
                shard_idx=idx,
                cache_root=args.cache_root,
                target_manifest=target_df,
                sequence_manifest=sequence_df,
                out_root=out_root,
                buckets=buckets,
                max_position=args.max_position,
                strict=strict,
                resume=args.resume,
                validation_sample_count=args.validation_samples,
            )

        write_partial_manifest(report, partial_dir)

        if report["ok"]:
            n_ok += 1
            status = "OK"
        else:
            n_fail += 1
            status = f"FAILED: {report['error']}"

        print(
            f"[full-slim] shard {idx:05d}: {status}, "
            f"seqs={report['n_seqs']}, trainable={report['n_trainable']}, "
            f"holdout={report['n_holdout']}, "
            f"out={_fmt_bytes(report['out_bytes'])}, "
            f"buckets={len(report['bucket_files'])}",
            flush=True,
        )
        for warn in report.get("warnings", []):
            print(f"[full-slim]   warning: {warn}", flush=True)

    print(
        f"[full-slim] done: {len(indices)} shards ({n_ok} ok, {n_fail} failed)"
    )
    if n_fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
