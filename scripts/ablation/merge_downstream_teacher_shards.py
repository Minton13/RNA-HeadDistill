#!/usr/bin/env python
"""Merge per-model partials into final downstream teacher-cache shards.

CPU-only.  Each Slurm array task handles one (route, shard range) row from
``outputs/downstream_teacher_fill/p5_merge_jobs.csv`` and writes shards in the
``downstream-teacher-cache-v1`` schema expected by
:class:`rna_head_alignment.probes.feature_extractor.TeacherCacheIndex`:

    outputs/downstream_teacher_head_cache/.route_cache/L{depth}_downstream/
        shard_d{i:05d}_bucket_{B}.pt
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache_downstream_teacher_heads import (  # noqa: E402
    BUCKETS,
    FILL_DIR,
    MAX_LEN,
    ROUTE_HEADS,
    SHARD_SIZE,
    bucket_for_length,
    read_jsonl_slice,
    route_head_ids,
)

SLIM_SCHEMA_VERSION = "downstream-teacher-cache-v1"
OUT_ROOT = _PROJECT_ROOT / "outputs" / "downstream_teacher_head_cache"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route-depth", type=int, required=True, choices=(6, 8, 12))
    p.add_argument("--shard-start", type=int, required=True)
    p.add_argument("--shard-end", type=int, required=True, help="exclusive")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    depth = args.route_depth
    heads = ROUTE_HEADS[depth]
    models = sorted({m for m, _, _ in heads})
    head_ids = route_head_ids(depth)
    sequences_path = FILL_DIR / f"sequences_L{depth}.jsonl"
    out_dir = OUT_ROOT / ".route_cache" / f"L{depth}_downstream"
    out_dir.mkdir(parents=True, exist_ok=True)

    n_fail = 0
    for shard_idx in range(args.shard_start, args.shard_end):
        rows = read_jsonl_slice(
            sequences_path, shard_idx * SHARD_SIZE, (shard_idx + 1) * SHARD_SIZE
        )
        if not rows:
            continue
        bucket = bucket_for_length(min(max(r["length"] for r in rows), MAX_LEN))
        out_path = out_dir / f"shard_d{shard_idx:05d}_bucket_{bucket}.pt"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[p5-merge] L{depth} shard {shard_idx:05d} exists, skip", flush=True)
            continue

        n = len(rows)
        k = len(heads)
        attn = torch.zeros((n, k, bucket, bucket), dtype=torch.float16)
        avail = torch.zeros((n, k), dtype=torch.bool)
        ok = True
        for model in models:
            part_path = (
                FILL_DIR / "partials" / f"L{depth}" / f"model={model}" / f"shard_{shard_idx:05d}.pt"
            )
            if not part_path.exists():
                print(f"[p5-merge] MISSING partial {part_path}", flush=True)
                ok = False
                break
            part = torch.load(part_path, map_location="cpu", weights_only=False)
            if list(part["seq_ids"]) != [r["seq_id"] for r in rows]:
                print(f"[p5-merge] seq_id mismatch in {part_path}", flush=True)
                ok = False
                break
            if int(part["bucket"]) != bucket:
                print(f"[p5-merge] bucket mismatch in {part_path}", flush=True)
                ok = False
                break
            slots = list(part["head_slots"])
            attn[:, slots] = part["attention"]
            avail[:, slots] = part["availability"]
            del part
            gc.collect()
        if not ok:
            n_fail += 1
            continue

        payload = {
            "schema_version": SLIM_SCHEMA_VERSION,
            "shard_id": f"shard_d{shard_idx:05d}",
            "seq_ids": [r["seq_id"] for r in rows],
            "sequences": [r["sequence"] for r in rows],
            "lengths": torch.tensor(
                [min(int(r["length"]), MAX_LEN) for r in rows], dtype=torch.int32
            ),
            f"teacher_heads_L{depth}": attn,
            f"teacher_head_ids_L{depth}": head_ids,
            "head_availability": avail,
            "metadata": {
                "source": "cache_downstream_teacher_heads.py + merge_downstream_teacher_shards.py",
                "route_depth": depth,
                "route_key": f"L{depth}",
                "n_heads": k,
                "bucket_size": bucket,
                "n_seqs": n,
                "built_at": datetime.now(timezone.utc).isoformat(),
            },
        }
        torch.save(payload, out_path)
        n_avail = int(avail.all(dim=1).sum())
        print(
            f"[p5-merge] L{depth} shard {shard_idx:05d}: n={n} bucket={bucket} "
            f"full_avail={n_avail}/{n} -> {out_path.name}",
            flush=True,
        )
        del payload, attn, avail
        gc.collect()

    print(
        f"[p5-merge] L{depth} shards {args.shard_start}-{args.shard_end} done, "
        f"failures={n_fail}",
        flush=True,
    )
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
