#!/usr/bin/env python
"""Merge Stage 4B bucketed full slim cache partial manifests.

Run after the array job has finished, e.g.:

    PYTHONPATH=src python scripts/head2head/merge_full_slim_manifest.py \
        --out-root outputs/teacher_attention_cache_slim/stage4b_head2head/full

Writes:

- ``full/manifests/slim_cache_manifest.parquet`` — one row per bucket file.
- ``full/manifests/slim_cache_checksums.jsonl`` — one JSON object per bucket.
- ``full/reports/full_slim_cache_report.md`` — human-readable summary.
- ``full/reports/full_slim_cache_failures.parquet`` — failed shard records.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from rna_head_alignment.head2head.head_manifest import ROUTE_DEPTHS


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if n < 1024 or unit == "TiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n} B"


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.1f}m"
    return f"{minutes/60:.1f}h"


def _maybe_load_json(value: object) -> object:
    if isinstance(value, str):
        return json.loads(value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-root",
        default="outputs/teacher_attention_cache_slim/stage4b_head2head/full",
        help="root containing partials/ and buckets/",
    )
    parser.add_argument(
        "--partial-dir",
        default=None,
        help="override partial manifest directory (default: out-root/partials)",
    )
    args = parser.parse_args()

    out_root = Path(args.out_root)
    partial_dir = Path(args.partial_dir) if args.partial_dir else out_root / "partials"

    if not partial_dir.exists():
        raise SystemExit(f"partial directory not found: {partial_dir}")

    partial_paths = sorted(partial_dir.glob("shard_*_partial.parquet"))
    if not partial_paths:
        raise SystemExit(f"no partial manifests found in {partial_dir}")

    dfs = [pd.read_parquet(p) for p in partial_paths]
    df = pd.concat(dfs, ignore_index=True)

    # ---- Write failures. ----------------------------------------------------
    failures = df[~df["ok"]].copy()
    failures_dir = out_root / "reports"
    failures_dir.mkdir(parents=True, exist_ok=True)
    failures_path = failures_dir / "full_slim_cache_failures.parquet"
    failures.to_parquet(failures_path, index=False)

    # ---- Build per-bucket manifest and checksum list. -----------------------
    manifest_rows: list[dict[str, object]] = []
    checksum_rows: list[dict[str, object]] = []

    for _, row in df[df["ok"]].iterrows():
        shard_id = str(row["shard_id"])
        shard_idx = int(row["shard_idx"])
        bucket_files = _maybe_load_json(row["bucket_files"])
        bucket_sizes = _maybe_load_json(row["bucket_sizes"])
        bucket_counts = _maybe_load_json(row["bucket_seq_counts"])
        checksums = _maybe_load_json(row["checksums"])

        if not isinstance(bucket_files, list):
            continue
        for bf, bs, bc in zip(bucket_files, bucket_sizes, bucket_counts):
            file_path = Path("buckets") / str(bf)
            md5 = (checksums or {}).get(str(bf))
            manifest_rows.append(
                {
                    "shard_idx": shard_idx,
                    "shard_id": shard_id,
                    "bucket_size": int(bs),
                    "n_seqs": int(bc),
                    "file_path": str(file_path),
                    "file_name": str(bf),
                    "md5": md5,
                }
            )
            checksum_rows.append(
                {
                    "shard_id": shard_id,
                    "bucket_size": int(bs),
                    "file": str(bf),
                    "file_path": str(file_path),
                    "md5": md5,
                }
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifests_dir = out_root / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifests_dir / "slim_cache_manifest.parquet"
    manifest_df.to_parquet(manifest_path, index=False)

    checksums_path = manifests_dir / "slim_cache_checksums.jsonl"
    with checksums_path.open("w", encoding="utf-8") as fh:
        for entry in checksum_rows:
            fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # ---- Aggregate coverage and sizes. --------------------------------------
    planned_shards = 1033
    success_count = int(df["ok"].sum())
    failed_count = len(df) - success_count

    total_trainable = int(df[df["ok"]]["n_trainable"].sum())
    total_holdout = int(df[df["ok"]]["n_holdout"].sum())
    total_available_any = int(df[df["ok"]]["n_available_any_head"].sum())

    original_bytes = int(df[df["ok"]]["source_bytes"].sum())
    slim_bytes = int(df[df["ok"]]["out_bytes"].sum())
    reduction = (original_bytes / slim_bytes) if slim_bytes else float("nan")

    total_build_time = float(df[df["ok"]]["build_time_sec"].sum())
    throughput = (total_trainable / total_build_time) if total_build_time > 0 else 0.0

    missing_head_records = sum(
        len(_maybe_load_json(x)) for x in df["skipped_heads"]
    )

    # Per-route / per-head coverage.
    route_coverage: dict[str, int] = {f"L{d}": 0 for d in ROUTE_DEPTHS}
    head_coverage: dict[str, int] = {}
    for _, row in df[df["ok"]].iterrows():
        rc = _maybe_load_json(row["route_coverage"])
        if isinstance(rc, dict):
            for k, v in rc.items():
                if k in route_coverage:
                    route_coverage[k] += int(v)
        hc = _maybe_load_json(row["head_coverage"])
        if isinstance(hc, dict):
            for k, v in hc.items():
                head_coverage[k] = head_coverage.get(k, 0) + int(v)

    # ---- Markdown report. ---------------------------------------------------
    lines: list[str] = [
        "# Stage 4B bucketed full slim cache report",
        "",
        "## Overview",
        "",
        f"- planned shards: {planned_shards}",
        f"- processed shards: {len(df)}",
        f"- successful shards: {success_count}",
        f"- failed shards: {failed_count}",
        f"- trainable sequences (length <= max_position): {total_trainable:,}",
        f"- sequences with any available teacher head: {total_available_any:,}",
        f"- long-sequence holdout count: {total_holdout:,}",
        f"- missing/skipped teacher-head records: {missing_head_records}",
        f"- original cache size (models x shards): {_fmt_bytes(original_bytes)}",
        f"- full slim cache size: {_fmt_bytes(slim_bytes)}",
        f"- reduction factor: {reduction:.2f}x",
        f"- total build time: {_fmt_duration(total_build_time)}",
        f"- throughput: {throughput:.1f} sequences/sec",
        "",
        "## Per-route coverage",
        "",
        "| route | available sequences | trainable sequences | coverage |",
        "|---|---|---|---|",
    ]
    for depth in ROUTE_DEPTHS:
        key = f"L{depth}"
        avail = route_coverage.get(key, 0)
        cov = (avail / total_trainable * 100) if total_trainable else 0.0
        lines.append(
            f"| {key} | {avail:,} | {total_trainable:,} | {cov:.2f}% |"
        )

    lines.extend(["", "## Per-head coverage", "", "| head_key | route | available | coverage |", "|---|---|---|---|"])
    # Sort by route depth then head key.
    head_items = sorted(head_coverage.items(), key=lambda kv: (kv[0].split(":")[-2] if ":" in kv[0] else "", kv[0]))
    for head_key, avail in head_items:
        route = head_key.split(":")[-2] if ":" in head_key else "?"
        cov = (avail / total_trainable * 100) if total_trainable else 0.0
        lines.append(f"| `{head_key}` | {route} | {avail:,} | {cov:.2f}% |")

    lines.extend(["", "## Bucket file summary", "", "| bucket_size | files | total sequences |", "|---|---|---|"])
    if len(manifest_df):
        bucket_summary = manifest_df.groupby("bucket_size").agg(files=("file_name", "count"), n_seqs=("n_seqs", "sum")).reset_index()
        for _, r in bucket_summary.iterrows():
            lines.append(f"| {int(r['bucket_size'])} | {int(r['files']):,} | {int(r['n_seqs']):,} |")
    else:
        lines.append("| - | 0 | 0 |")

    if failed_count:
        lines.extend(["", "## Failures", ""])
        for _, row in failures.iterrows():
            lines.append(f"- shard {row['shard_id']}: {row['error']}")

    lines.extend(["", "## Outputs", ""])
    lines.append(f"- slim cache manifest: `{manifest_path}`")
    lines.append(f"- checksums: `{checksums_path}`")
    lines.append(f"- failures: `{failures_path}`")

    report_path = failures_dir / "full_slim_cache_report.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"merged partials: {len(partial_paths)}")
    print(f"successful shards: {success_count}/{len(df)}")
    print(f"trainable sequences: {total_trainable:,}")
    print(f"long holdout: {total_holdout:,}")
    print(f"original size: {_fmt_bytes(original_bytes)}")
    print(f"slim size: {_fmt_bytes(slim_bytes)}")
    print(f"reduction: {reduction:.2f}x")
    print(f"manifest: {manifest_path}")
    print(f"checksums: {checksums_path}")
    print(f"report: {report_path}")
    print(f"failures: {failures_path}")

    if failed_count:
        sys.exit(1)


if __name__ == "__main__":
    main()
