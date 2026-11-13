#!/usr/bin/env python
"""Build the Stage 4B slim teacher attention cache.

Run from the project root with the package importable, e.g.:

    PYTHONPATH=src python scripts/head2head/02_build_slim_cache.py \
        --cache-root outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads \
        --manifest outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/target_head_manifest.parquet \
        --out-root outputs/teacher_attention_cache_slim/stage4b_head2head \
        --shards 0,516,936,1032 \
        --report outputs/teacher_attention_cache_slim/stage4b_head2head/reports/slim_cache_size_report.md

Select shards with --shards 0,516,936,1032 or --shard-range 0:1033 (end
exclusive). Writes one slim shard per index plus a cumulative size report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rna_head_alignment.head2head.head_manifest import load_manifest
from rna_head_alignment.head2head.slim_cache import (
    ROUTE_DEPTHS,
    build_slim_shard,
)


def _parse_shards(shards: str | None, shard_range: str | None) -> list[int]:
    if shards:
        return sorted({int(s) for s in shards.split(",") if s.strip()})
    if shard_range:
        start, _, end = shard_range.partition(":")
        return list(range(int(start), int(end)))
    raise SystemExit("provide --shards or --shard-range")


def _fmt_bytes(n: int) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if n < 1024 or unit == "GiB":
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} {unit}"
        n /= 1024
    return f"{n} B"


def _skipped_report(out_path: Path) -> dict:
    """Synthetic report used when --resume skips an existing shard."""
    return {
        "shard_id": out_path.stem.split("_", 1)[1],
        "ok": True,
        "error": None,
        "out_path": str(out_path),
        "out_bytes": out_path.stat().st_size,
        "source_bytes": 0,
        "models_used": [],
        "n_seqs": 0,
        "n_dropped": 0,
        "dropped_seq_ids": [],
        "warnings": ["skipped (--resume): output already exists"],
        "route_heads": {},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--manifest", required=True, help="target_head_manifest.parquet")
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--shards", default=None, help="comma list, e.g. 0,516,936,1032")
    parser.add_argument("--shard-range", default=None, help="a:b, end exclusive")
    parser.add_argument("--report", required=True, help="Markdown size report path")
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="skip missing head_keys / model shards instead of failing",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="skip shards whose output file already exists",
    )
    args = parser.parse_args()

    indices = _parse_shards(args.shards, args.shard_range)
    manifest = load_manifest(args.manifest)
    strict = not args.no_strict

    reports = []
    out_root = Path(args.out_root)
    for idx in indices:
        out_path = out_root / f"shard_{idx:05d}.pt"
        if args.resume and out_path.exists() and out_path.stat().st_size > 0:
            print(f"[slim] shard {idx:05d} already exists, skipping", flush=True)
            rep = _skipped_report(out_path)
        else:
            print(f"[slim] building shard {idx:05d} ...", flush=True)
            rep = build_slim_shard(
                idx, args.cache_root, args.out_root, manifest, strict=strict
            )
        status = "OK" if rep["ok"] else f"FAILED: {rep['error']}"
        print(
            f"[slim] shard {idx:05d}: {status}, seqs={rep['n_seqs']}, "
            f"dropped={rep['n_dropped']}, out={_fmt_bytes(rep['out_bytes'])}",
            flush=True,
        )
        for warn in rep["warnings"]:
            print(f"[slim]   warning: {warn}", flush=True)
        reports.append(rep)

    n_ok = sum(1 for r in reports if r["ok"])
    total_src = sum(r["source_bytes"] for r in reports if r["ok"])
    total_out = sum(r["out_bytes"] for r in reports if r["ok"])
    reduction = (total_src / total_out) if total_out else float("nan")

    lines = ["# Slim teacher cache size report (Stage 4B)", ""]
    lines.append(f"- shards processed: {len(reports)} ({n_ok} ok, {len(reports) - n_ok} failed)")
    lines.append(f"- original source bytes (9 model shards each): {_fmt_bytes(total_src)}")
    lines.append(f"- slim bytes: {_fmt_bytes(total_out)}")
    lines.append(f"- reduction factor: {reduction:.2f}x")
    lines.append("")
    lines.append("**Honesty note**: all attention data for the 15 preserved "
                 "teacher heads is kept bit-for-bit (fp16). The size reduction "
                 "comes almost entirely from de-duplicating the per-model "
                 "metadata (seq_ids / sequences / lengths / valid_mask were "
                 "stored 9 times; the slim cache stores them once) and from "
                 "dropping valid_mask, not from compressing attention maps.")
    lines.append("")
    lines.append("## Per-shard detail")
    lines.append("")
    lines.append("| shard | ok | seqs | dropped | heads L6/L8/L12 | source | slim | warnings |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in reports:
        rh = r["route_heads"]
        heads = "/".join(str(rh.get(f"L{d}", 0)) for d in ROUTE_DEPTHS)
        lines.append(
            f"| {r['shard_id']} | {'yes' if r['ok'] else 'NO'} | {r['n_seqs']} "
            f"| {r['n_dropped']} | {heads} | {_fmt_bytes(r['source_bytes'])} "
            f"| {_fmt_bytes(r['out_bytes'])} | {len(r['warnings'])} |"
        )
    dropped_reports = [r for r in reports if r["dropped_seq_ids"]]
    if dropped_reports:
        lines.append("")
        lines.append("## Dropped seq_ids (never silently discarded)")
        lines.append("")
        for r in dropped_reports:
            lines.append(f"- shard {r['shard_id']}: {len(r['dropped_seq_ids'])} dropped "
                         f"(first 5: {r['dropped_seq_ids'][:5]})")
    failed = [r for r in reports if not r["ok"]]
    if failed:
        lines.append("")
        lines.append("## Failures")
        lines.append("")
        for r in failed:
            lines.append(f"- shard {r['shard_id']}: {r['error']}")

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[slim] report written to {report_path}")

    if n_ok != len(reports):
        sys.exit(1)


if __name__ == "__main__":
    main()
