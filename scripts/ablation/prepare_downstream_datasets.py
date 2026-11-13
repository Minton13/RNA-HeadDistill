#!/usr/bin/env python
"""Prepare all Stage 5 downstream datasets with decontamination and split reporting.

Example
-------
    python scripts/ablation/prepare_downstream_datasets.py \
        --config configs/ablation/downstream_data.yaml \
        --tasks splice_donor_seqcls,rbp_binding_seqcls
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rna_head_alignment.downstream.prepare import (
    build_task_dataset,
    build_summary,
    decontaminate,
    load_decontamination_hashes,
    prepare_secondary_structure_streaming,
    write_decontamination_report,
    write_task_splits,
)
from rna_head_alignment.io.paths import load_yaml, resolve_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare downstream datasets for Stage 5 probes.")
    parser.add_argument("--config", required=True, help="Path to downstream_data.yaml.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task subset.")
    parser.add_argument("--skip-decontamination", action="store_true", help="Skip hash-based decontamination.")
    parser.add_argument("--dry-run", action="store_true", help="Build data but do not write parquet files.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    output_root = resolve_path(cfg.get("output_root", "data/downstream_stage5"))
    report_dir = resolve_path(cfg.get("report_dir", "outputs/ablation/downstream"))
    split_cfg = cfg.get("split", {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42})
    task_cfgs = cfg.get("tasks", {})

    requested = args.tasks.split(",") if args.tasks else list(task_cfgs.keys())
    requested = [t.strip() for t in requested if t.strip()]
    missing = [t for t in requested if t not in task_cfgs]
    if missing:
        raise ValueError(f"Unknown tasks in --tasks: {missing}. Supported: {list(task_cfgs)}")

    if not args.skip_decontamination:
        manifest_path = cfg.get("sequence_manifest")
        if not manifest_path:
            raise ValueError("config missing 'sequence_manifest' for decontamination")
        print(f"[prepare] loading decontamination hashes from {manifest_path}", flush=True)
        hashes = load_decontamination_hashes(manifest_path)
        print(f"[prepare] loaded {len(hashes):,} Stage 4B sequence hashes", flush=True)
    else:
        hashes = set()

    summaries: list[dict[str, Any]] = []

    for task_name in requested:
        print(f"[prepare] building {task_name} ...", flush=True)
        task_cfg = task_cfgs[task_name]
        task_type = task_cfg.get("task_type", "sequence_classification")

        if task_type in {"pairwise", "rna3d_contact"}:
            if args.dry_run:
                print(f"[prepare] {task_name}: dry-run streaming skipped", flush=True)
                continue
            summary = prepare_secondary_structure_streaming(
                task_name, task_cfg, split_cfg, hashes, output_root
            )
            print(
                f"[prepare] {task_name}: removed {summary['removed']:,} "
                f"({summary['contamination_rate']:.4%}) -> {summary['remaining']:,} rows "
                f"train={summary['split_counts']['train']:,} "
                f"val={summary['split_counts']['val']:,} test={summary['split_counts']['test']:,}",
                flush=True,
            )
            summaries.append(summary)
            continue

        df = build_task_dataset(task_name, task_cfg, split_cfg)
        print(f"[prepare] {task_name}: {len(df):,} raw rows", flush=True)

        if args.skip_decontamination:
            decon_summary = {
                "total_before": int(len(df)),
                "removed": 0,
                "remaining": int(len(df)),
                "contamination_rate": 0.0,
                "removed_by_source": {},
            }
        else:
            df, decon_summary = decontaminate(df, hashes)
            print(
                f"[prepare] {task_name}: removed {decon_summary['removed']:,} "
                f"({decon_summary['contamination_rate']:.4%}) -> {decon_summary['remaining']:,} rows",
                flush=True,
            )

        if args.dry_run:
            print(f"[prepare] {task_name}: dry-run; not writing parquet", flush=True)
            summary = build_summary(task_name, decon_summary, {"train": 0, "val": 0, "test": 0}, df)
            summaries.append(summary)
            continue

        if df.empty:
            print(f"[prepare] WARNING {task_name}: no rows remaining after decontamination", flush=True)
            continue

        split_counts = write_task_splits(df, output_root, task_name)
        print(
            f"[prepare] {task_name}: wrote splits train={split_counts['train']:,} "
            f"val={split_counts['val']:,} test={split_counts['test']:,}",
            flush=True,
        )

        summary = build_summary(task_name, decon_summary, split_counts, df)
        summaries.append(summary)

    write_decontamination_report(summaries, report_dir)
    print(f"[prepare] wrote decontamination report to {report_dir / 'decontamination_report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
