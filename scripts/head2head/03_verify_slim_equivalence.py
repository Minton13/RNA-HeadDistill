#!/usr/bin/env python
"""Verify slim-cache equivalence against the original teacher cache.

Bitwise (fp16, torch.equal) per-head comparison of each slim shard against
the original 9 model shards. Exits non-zero if any head fails.

Run from the project root with the package importable, e.g.:

    PYTHONPATH=src python scripts/head2head/03_verify_slim_equivalence.py \
        --cache-root outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads \
        --slim-root outputs/teacher_attention_cache_slim/stage4b_head2head \
        --manifest outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/target_head_manifest.parquet \
        --shards 0,516,936,1032
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rna_head_alignment.head2head.head_manifest import load_manifest
from rna_head_alignment.head2head.slim_cache import verify_slim_equivalence


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--slim-root", required=True)
    parser.add_argument("--manifest", required=True, help="target_head_manifest.parquet")
    parser.add_argument("--shards", required=True, help="comma list, e.g. 0,516,936,1032")
    args = parser.parse_args()

    indices = sorted({int(s) for s in args.shards.split(",") if s.strip()})
    manifest = load_manifest(args.manifest)
    slim_root = Path(args.slim_root)

    any_fail = False
    rows = []
    for idx in indices:
        slim_path = slim_root / f"shard_{idx:05d}.pt"
        if not slim_path.exists():
            print(f"[verify] shard {idx:05d}: MISSING slim file {slim_path}")
            any_fail = True
            continue
        results = verify_slim_equivalence(idx, args.cache_root, slim_path, manifest)
        n_pass = sum(1 for r in results if r["equal"])
        print(f"[verify] shard {idx:05d}: {n_pass}/{len(results)} heads equal")
        for r in results:
            rows.append(r)
            if not r["equal"]:
                any_fail = True
                print(
                    f"  FAIL {r['head_key']} (orig ch={r['channel_in_orig']}, "
                    f"slim ch={r['channel_in_slim']}): {r['note']}"
                )

    print()
    print("| shard | head_key | route | orig ch | slim ch | equal | note |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['shard_id']} | `{r['head_key']}` | L{r['route_depth']} "
            f"| {r['channel_in_orig']} | {r['channel_in_slim']} "
            f"| {'PASS' if r['equal'] else 'FAIL'} | {r['note']} |"
        )

    if any_fail:
        print("\n[verify] RESULT: FAIL")
        sys.exit(1)
    print("\n[verify] RESULT: all heads bitwise equal")


if __name__ == "__main__":
    main()
