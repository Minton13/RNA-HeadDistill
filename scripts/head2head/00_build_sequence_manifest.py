#!/usr/bin/env python
"""Build the Stage 4B sequence manifest.

Run from the project root with the package importable, e.g.:

    PYTHONPATH=src python scripts/head2head/00_build_sequence_manifest.py \
        --source-jsonl outputs/task_samples/rinalmo_pretrain_10pct_len_sorted.jsonl \
        --slim-cache-root outputs/teacher_attention_cache_slim/stage4b_head2head \
        --target-manifest outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/sequence_manifest.parquet \
        --report outputs/teacher_attention_cache_slim/stage4b_head2head/reports/sequence_manifest_report.json

The manifest joins the authoritative sequence source with the slim cache
metadata (``seq_ids`` / ``sequences`` / ``lengths``) and records which
teacher-attention routes are available for each sequence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rna_head_alignment.head2head.sequence_manifest import build_sequence_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-jsonl",
        required=True,
        help="authoritative JSONL with seq_id, sequence, length, source",
    )
    parser.add_argument(
        "--slim-cache-root",
        required=True,
        help="directory containing shard_*.pt slim cache files",
    )
    parser.add_argument(
        "--target-manifest",
        required=True,
        help="output parquet path for the sequence manifest",
    )
    parser.add_argument(
        "--report",
        default=None,
        help="optional JSON path to write the build report",
    )
    args = parser.parse_args()

    report = build_sequence_manifest(
        source_jsonl=args.source_jsonl,
        slim_cache_root=args.slim_cache_root,
        target_manifest=args.target_manifest,
    )

    print(f"sequence manifest: {report['n_output_rows']} rows -> {args.target_manifest}")
    print(
        f"source seqs: {report['n_source']}, "
        f"slim cache seqs: {report['n_slim_seqs']}, "
        f"collisions: {report['n_collisions']}"
    )
    if report["warnings"]:
        print(f"warnings: {len(report['warnings'])}")
        for w in report["warnings"]:
            print(f"  - {w}")

    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"report written to {args.report}")


if __name__ == "__main__":
    main()
