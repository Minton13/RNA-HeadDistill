#!/usr/bin/env python
"""Link merged downstream teacher shards into task dirs and build index.json.

CPU-only.  One array task per route depth (6/8/12).  For every downstream
task routed at this depth:

1. Symlinks ``.route_cache/L{depth}_downstream/shard_d*.pt`` into
   ``outputs/downstream_teacher_head_cache/{task}/``.
2. Writes ``index.json`` mapping downstream ``seq_id -> shard filename``
   (only the downstream shards; the legacy pretraining-corpus shards carry
   no downstream seq_ids and are intentionally excluded).
3. Validates coverage against the Stage 5 parquets (train/val/test) and
   prints a per-task coverage summary as JSON lines for the gap report.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cache_downstream_teacher_heads import FILL_DIR, SHARD_SIZE  # noqa: E402

OUT_ROOT = _PROJECT_ROOT / "outputs" / "downstream_teacher_head_cache"
DATA_ROOT = _PROJECT_ROOT / "data" / "downstream_stage5"

ROUTE_TO_TASKS: dict[int, list[str]] = {
    6: ["splice_donor_seqcls", "splice_acceptor_seqcls", "rbp_binding_seqcls", "ncRNA_family_seqcls"],
    8: ["chemical_reactivity_token"],
    12: ["secondary_structure_pair"],
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route-depth", type=int, required=True, choices=(6, 8, 12))
    return p.parse_args()


def main() -> int:
    args = parse_args()
    depth = args.route_depth
    route_dir = OUT_ROOT / ".route_cache" / f"L{depth}_downstream"
    shards = sorted(route_dir.glob("shard_d*.pt"))
    if not shards:
        print(f"[p5-finalize] no downstream shards under {route_dir}", flush=True)
        return 1

    # Map shard filename -> seq_ids from the deterministic jsonl slice.
    sequences_path = FILL_DIR / f"sequences_L{depth}.jsonl"
    all_rows: list[str] = []
    with sequences_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            all_rows.append(json.loads(line)["seq_id"])

    seq_to_file: dict[str, str] = {}
    for shard in shards:
        # shard_d00012_bucket_64.pt -> 12
        idx = int(shard.stem.split("_")[1][1:])
        for sid in all_rows[idx * SHARD_SIZE : (idx + 1) * SHARD_SIZE]:
            seq_to_file[sid] = shard.name
    print(f"[p5-finalize] L{depth}: {len(shards)} shards, {len(seq_to_file):,} seq_ids", flush=True)

    status = 0
    for task in ROUTE_TO_TASKS[depth]:
        task_dir = OUT_ROOT / task
        task_dir.mkdir(parents=True, exist_ok=True)
        n_links = 0
        for shard in shards:
            link = task_dir / shard.name
            if link.is_symlink() or link.exists():
                link.unlink()
            try:
                os.symlink(shard.resolve(), link)
            except OSError:
                import shutil
                shutil.copy2(shard, link)
            n_links += 1

        wanted: set[str] = set()
        for split in ("train", "val", "test"):
            df = pd.read_parquet(DATA_ROOT / task / f"{split}.parquet", columns=["seq_id"])
            wanted |= set(df["seq_id"].astype(str))
        missing = sorted(wanted - set(seq_to_file))
        coverage = {
            "task": task,
            "route_depth": depth,
            "n_links": n_links,
            "task_seq_ids": len(wanted),
            "covered": len(wanted) - len(missing),
            "missing": len(missing),
            "missing_examples": missing[:10],
        }
        print("COVERAGE " + json.dumps(coverage), flush=True)
        if missing:
            status = 1
            continue
        # Only rewrite the index when coverage is complete.
        index = {sid: seq_to_file[sid] for sid in wanted}
        with (task_dir / "index.json").open("w", encoding="utf-8") as fh:
            json.dump(index, fh)
        print(f"[p5-finalize] {task}: index.json written ({len(index):,} entries)", flush=True)

    return status


if __name__ == "__main__":
    sys.exit(main())
