#!/usr/bin/env python
"""Smoke-test the eval-only ablation path on a tiny sliced val index.

Builds a 64-row slice of the val split, points a copy of the run config at it,
and runs ``evaluate_ablation_student``.  Writes to a throwaway directory.

Usage:
    PYTHONPATH=src python scripts/ablation/smoke_eval_only.py \
        --run-dir outputs/ablation/alignment/A2_head2head_top3/seed_17
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rna_head_alignment.ablation.ablation_runner import evaluate_ablation_student
from rna_head_alignment.io.paths import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--n-rows", type=int, default=64)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    config = load_yaml(run_dir / "config_resolved.yaml")

    smoke_root = Path("outputs/ablation/alignment/_smoke_eval_only")
    smoke_root.mkdir(parents=True, exist_ok=True)

    val_idx = pd.read_parquet(config["val_split_index"])
    slice_path = smoke_root / f"val_slice{args.n_rows}.parquet"
    val_idx.head(args.n_rows).to_parquet(slice_path, index=False)

    config["val_split_index"] = str(slice_path)
    config["output_root"] = str(smoke_root)
    config["run_name"] = config.get("run_name", run_dir.name) + "_smoke"

    checkpoint = run_dir / "checkpoints" / "best_val.pt"
    if not checkpoint.exists():
        checkpoint = run_dir / "checkpoints" / "latest.pt"

    report = evaluate_ablation_student(
        config=config,
        checkpoint_path=checkpoint,
        output_dir=config["output_root"],
        splits=["val"],
    )
    summary = report["splits"]["val"]["summary"]
    df = pd.read_parquet(summary)
    student_rows = df[df["baseline"] == "student"]
    print(json.dumps({
        "summary_rows": len(df),
        "student_rows": len(student_rows),
        "cosine_mean": float(student_rows[student_rows["metric"] == "cosine"]["mean"].mean()),
    }, indent=2))
    print("[smoke-eval-only] OK")


if __name__ == "__main__":
    main()
