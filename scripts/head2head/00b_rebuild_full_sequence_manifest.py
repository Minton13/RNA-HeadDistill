#!/usr/bin/env python
"""Rebuild the Stage 4B sequence manifest directly from the authoritative source.

The previous manifest was built from a small smoke slim cache and only covered
four shards.  This script reads the canonical source JSONL
(``outputs/task_samples/rinalmo_pretrain_10pct_len_sorted.jsonl``), assigns each
sequence to its original teacher-cache shard by line index, and writes a full
sequence manifest covering all 2,114,289 sequences.

Route availability is optionally refined by scanning the original teacher cache
for missing model shards/records.  If ``--scan-cache`` is omitted, all routes are
marked available; the bucketed slim-cache files record exact per-head
availability for partially-missing model shards (e.g. ``rinalmo_mega`` shard
936).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch


REQUIRED_SOURCE_FIELDS = {"seq_id", "sequence", "length", "source"}

DEFAULT_ROUTE_MODELS = {
    6: ["splicebert", "plantrna_fm", "nucleotide_transformer_v2_500m", "rinalmo_mega", "rnafm"],
    8: ["plantrna_fm", "utrlm_mrl", "splicebert"],
    12: ["rinalmo_giga", "plantrna_fm", "moderngena_base", "mrnafm"],
}

SHARD_SIZE = 2048


def _sequence_hash(sequence: str) -> str:
    return hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest()


def _read_source(source_jsonl: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    warnings: list[str] = []
    with source_jsonl.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            missing = REQUIRED_SOURCE_FIELDS - set(rec.keys())
            if missing:
                raise ValueError(f"{source_jsonl}:{line_no} missing fields {sorted(missing)}")
            seq_id = rec["seq_id"]
            sequence = str(rec["sequence"])
            length = int(rec["length"])
            if len(sequence) != length:
                warnings.append(
                    f"{source_jsonl}:{line_no}: seq_id {seq_id} length={length} "
                    f"!= len(sequence)={len(sequence)}"
                )
            records.append(
                {
                    "seq_id": seq_id,
                    "sequence": sequence,
                    "length": length,
                    "original_dataset": rec["source"],
                    "line_index": len(records),
                }
            )
    return records, warnings


def _discover_models(cache_root: Path) -> list[str]:
    return sorted(p.name.split("=", 1)[1] for p in cache_root.glob("model=*") if p.is_dir())


def _scan_missing_indices(
    cache_root: Path,
    models: list[str],
    n_seqs: int,
) -> dict[str, set[int]]:
    """Return, for each model, the set of canonical line indices that are missing."""
    n_shards = (n_seqs + SHARD_SIZE - 1) // SHARD_SIZE
    missing_by_model: dict[str, set[int]] = {m: set() for m in models}

    for model in models:
        model_dir = cache_root / f"model={model}"
        for shard_idx in range(n_shards):
            path = model_dir / f"shard_{shard_idx:05d}.pt"
            if not path.exists():
                start = shard_idx * SHARD_SIZE
                end = min((shard_idx + 1) * SHARD_SIZE, n_seqs)
                missing_by_model[model].update(range(start, end))
                continue
            try:
                shard = torch.load(path, map_location="cpu", weights_only=False)
                seq_ids = list(shard.get("seq_ids", []))
                start = shard_idx * SHARD_SIZE
                end = min((shard_idx + 1) * SHARD_SIZE, n_seqs)
                expected_n = end - start
                if len(seq_ids) < expected_n:
                    # Build a quick set of present seq_ids and compare with the
                    # canonical range derived from the source order below.
                    # We defer exact per-index matching to a second pass after
                    # reading the source, but for now record that this shard is
                    # incomplete; the exact indices will be filled in by the caller.
                    missing_by_model[model].add((shard_idx, len(seq_ids), expected_n))
            except Exception as exc:  # noqa: BLE001
                missing_by_model[model].add(f"load_error:{shard_idx}:{exc}")

    return missing_by_model


def _resolve_missing_indices(
    cache_root: Path,
    models: list[str],
    seq_id_to_index: dict[str, int],
) -> dict[str, set[int]]:
    """Walk original cache shards and record which canonical indices are absent."""
    missing_by_model: dict[str, set[int]] = {m: set() for m in models}

    for model in models:
        model_dir = cache_root / f"model={model}"
        for path in sorted(model_dir.glob("shard_*.pt")):
            try:
                shard = torch.load(path, map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001
                missing_by_model[model].add(f"load_error:{path.name}:{exc}")
                continue
            seq_ids = list(shard.get("seq_ids", []))
            present = {seq_id_to_index.get(s) for s in seq_ids}
            present.discard(None)
            shard_idx = int(path.stem.split("_", 1)[1])
            start = shard_idx * SHARD_SIZE
            end = start + len(seq_ids) + 32  # allow a small slack for known gaps
            end = min(end, len(seq_id_to_index))
            # Determine the actual span of this shard from the present indices.
            if present:
                actual_end = max(present) + 1
                actual_start = min(present)
            else:
                actual_start, actual_end = start, start
            for idx in range(actual_start, actual_end):
                if idx not in present:
                    missing_by_model[model].add(idx)

    return missing_by_model


def _available_routes(missing: set[int], line_index: int) -> list[int]:
    routes: list[int] = []
    for depth, models in DEFAULT_ROUTE_MODELS.items():
        # Route is considered available for this sequence if none of its required
        # models mark the index as missing.
        if not any(line_index in missing.get(m, set()) for m in models):
            routes.append(depth)
    return sorted(routes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-jsonl",
        default="outputs/task_samples/rinalmo_pretrain_10pct_len_sorted.jsonl",
        help="authoritative source JSONL",
    )
    parser.add_argument(
        "--cache-root",
        default="outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads",
        help="original teacher attention cache root (used with --scan-cache)",
    )
    parser.add_argument(
        "--target-manifest",
        default="outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/sequence_manifest.parquet",
        help="output parquet path",
    )
    parser.add_argument(
        "--scan-cache",
        action="store_true",
        help="scan original cache to compute exact per-sequence route availability",
    )
    parser.add_argument(
        "--report",
        default="outputs/teacher_attention_cache_slim/stage4b_head2head/reports/full_sequence_manifest_report.json",
        help="JSON report path",
    )
    args = parser.parse_args()

    source_jsonl = Path(args.source_jsonl)
    target_manifest = Path(args.target_manifest)
    target_manifest.parent.mkdir(parents=True, exist_ok=True)

    records, warnings = _read_source(source_jsonl)
    n_seqs = len(records)

    missing_by_model: dict[str, set[int]] = {}
    if args.scan_cache:
        cache_root = Path(args.cache_root)
        models = _discover_models(cache_root)
        seq_id_to_index = {r["seq_id"]: i for i, r in enumerate(records)}
        missing_by_model = _resolve_missing_indices(cache_root, models, seq_id_to_index)

    rows = []
    for line_index, rec in enumerate(records):
        shard_idx = line_index // SHARD_SIZE
        original_shard_id = f"{shard_idx:05d}"
        routes = _available_routes(missing_by_model, line_index) if args.scan_cache else [6, 8, 12]
        rows.append(
            {
                "seq_id": rec["seq_id"],
                "sequence": rec["sequence"],
                "length": rec["length"],
                "original_dataset": rec["original_dataset"],
                "original_shard_id": [original_shard_id],
                "sequence_hash": _sequence_hash(rec["sequence"]),
                "available_teacher_routes": routes,
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "seq_id",
            "sequence",
            "length",
            "original_dataset",
            "original_shard_id",
            "sequence_hash",
            "available_teacher_routes",
        ],
    )
    df.to_parquet(target_manifest, index=False)

    report = {
        "n_source": n_seqs,
        "n_output_rows": len(df),
        "n_shards": (n_seqs + SHARD_SIZE - 1) // SHARD_SIZE,
        "scan_cache": args.scan_cache,
        "missing_by_model": {m: len(s) for m, s in missing_by_model.items()} if args.scan_cache else {},
        "warnings": warnings,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"sequence manifest: {len(df):,} rows -> {target_manifest}")
    print(f"shards covered: {report['n_shards']}")
    print(f"report: {args.report}")
    if warnings:
        print(f"warnings: {len(warnings)}")


if __name__ == "__main__":
    main()
