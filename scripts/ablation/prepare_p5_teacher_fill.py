#!/usr/bin/env python
"""Prepare inputs for filling the P5 downstream exact-teacher cache.

Runs on the management node (CPU only, parquet/jsonl I/O):

1. Builds one length-sorted, de-duplicated sequence JSONL per route depth
   (L6/L8/L12) from the Stage 5 downstream parquets.
2. Writes the Slurm job CSVs for the per-model cache array and the merge
   array.

The downstream teacher cache must be keyed by downstream ``seq_id``; the
existing route cache only covers the pretraining corpus (see
``outputs/probes/frozen/p5_cache_gap_report.md``).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FILL_DIR = ROOT / "outputs" / "downstream_teacher_fill"
DATA_ROOT = ROOT / "data" / "downstream_stage5"

ROUTE_TO_TASKS: dict[int, list[str]] = {
    6: ["splice_donor_seqcls", "splice_acceptor_seqcls", "rbp_binding_seqcls", "ncRNA_family_seqcls"],
    8: ["chemical_reactivity_token"],
    12: ["secondary_structure_pair"],
}

SHARD_SIZE = 2048
CACHE_GROUP_SHARDS = 64  # shards per (route, model) array task
BUCKETS = (64, 128, 256, 512)

# Canonical head order per route, copied verbatim from the existing route
# cache ``teacher_head_ids_L*`` metadata.
ROUTE_MODELS: dict[int, dict[str, list[tuple[int, int]]]] = {
    6: {
        "nucleotide_transformer_v2_500m": [(4, 3)],
        "plantrna_fm": [(8, 18)],
        "rinalmo_mega": [(18, 5)],
        "rnafm": [(4, 3)],
        "splicebert": [(5, 15)],
    },
    8: {
        "plantrna_fm": [(9, 9), (9, 7), (10, 7)],
        "splicebert": [(2, 7)],
        "utrlm_mrl": [(2, 10)],
    },
    12: {
        "moderngena_base": [(12, 11)],
        "mrnafm": [(4, 9)],
        "plantrna_fm": [(8, 2)],
        "rinalmo_giga": [(31, 1), (31, 13)],
    },
}


def bucket_for_length(length: int) -> int:
    for b in BUCKETS:
        if length <= b:
            return b
    raise ValueError(f"sequence length {length} exceeds max bucket {BUCKETS[-1]}")


def collect_route_sequences(depth: int) -> list[dict]:
    seen: dict[str, str] = {}
    for task in ROUTE_TO_TASKS[depth]:
        for split in ("train", "val", "test"):
            df = pd.read_parquet(
                DATA_ROOT / task / f"{split}.parquet", columns=["seq_id", "sequence"]
            )
            for seq_id, seq in zip(df["seq_id"].astype(str), df["sequence"].astype(str)):
                if seq_id in seen:
                    continue
                norm = seq.upper().replace("T", "U")
                if norm:
                    seen[seq_id] = norm
    rows = [
        {"seq_id": sid, "sequence": seq, "length": len(seq)}
        for sid, seq in seen.items()
    ]
    rows.sort(key=lambda r: (r["length"], r["seq_id"]))
    return rows


def main() -> None:
    FILL_DIR.mkdir(parents=True, exist_ok=True)
    stats: dict[str, object] = {}
    cache_jobs: list[dict] = []
    merge_jobs: list[dict] = []

    for depth in (6, 8, 12):
        rows = collect_route_sequences(depth)
        jsonl_path = FILL_DIR / f"sequences_L{depth}.jsonl"
        with jsonl_path.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")

        n_shards = (len(rows) + SHARD_SIZE - 1) // SHARD_SIZE
        bucket_counts: dict[int, int] = {}
        for i in range(n_shards):
            shard_rows = rows[i * SHARD_SIZE : (i + 1) * SHARD_SIZE]
            b = bucket_for_length(max(r["length"] for r in shard_rows))
            bucket_counts[b] = bucket_counts.get(b, 0) + 1
        stats[f"L{depth}"] = {
            "n_sequences": len(rows),
            "n_shards": n_shards,
            "shard_bucket_counts": bucket_counts,
            "jsonl": str(jsonl_path),
        }
        print(f"[prep] L{depth}: {len(rows):,} seqs -> {n_shards} shards, buckets={bucket_counts}", flush=True)

        groups = [
            (start, min(start + CACHE_GROUP_SHARDS, n_shards))
            for start in range(0, n_shards, CACHE_GROUP_SHARDS)
        ]
        for model in ROUTE_MODELS[depth]:
            for start, end in groups:
                cache_jobs.append({
                    "route_depth": depth,
                    "model": model,
                    "shard_start": start,
                    "shard_end": end,  # exclusive
                })
        for start, end in groups:
            merge_jobs.append({
                "route_depth": depth,
                "shard_start": start,
                "shard_end": end,  # exclusive
            })

    with (FILL_DIR / "p5_cache_jobs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["route_depth", "model", "shard_start", "shard_end"])
        writer.writeheader()
        writer.writerows(cache_jobs)
    with (FILL_DIR / "p5_merge_jobs.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["route_depth", "shard_start", "shard_end"])
        writer.writeheader()
        writer.writerows(merge_jobs)
    with (FILL_DIR / "prep_stats.json").open("w", encoding="utf-8") as fh:
        json.dump(stats, fh, indent=2)

    print(f"[prep] cache jobs: {len(cache_jobs)} -> p5_cache_jobs.csv", flush=True)
    print(f"[prep] merge jobs: {len(merge_jobs)} -> p5_merge_jobs.csv", flush=True)


if __name__ == "__main__":
    main()
