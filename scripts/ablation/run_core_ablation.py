#!/usr/bin/env python
"""CLI for a single core ablation experiment (A0-A8).

Example:
    PYTHONPATH=src python scripts/ablation/run_core_ablation.py \
        --experiment-id A1_single_teacher_top1 --seed 17

Array use (inside Slurm):
    python scripts/ablation/run_core_ablation.py \
        --experiment-id "$EXPERIMENT_ID" --seed "$SEED"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rna_head_alignment.ablation.ablation_runner import run_ablation_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="configs/ablation/experiment_registry.yaml",
        help="Path to experiment_registry.yaml",
    )
    parser.add_argument(
        "--experiment-id",
        required=True,
        help="Experiment key in the registry (e.g. A3_head2head_top5_full)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Random seed for this run",
    )
    parser.add_argument(
        "--base-config",
        default=None,
        help="Optional base config YAML with shared overrides",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Override the per-seed run directory",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip post-training evaluation",
    )
    parser.add_argument(
        "--eval-splits",
        default=None,
        help="Comma-separated list of splits to evaluate (default val,test)",
    )
    args = parser.parse_args()

    eval_splits = [s.strip() for s in args.eval_splits.split(",")] if args.eval_splits else None

    summary = run_ablation_experiment(
        registry_path=args.registry,
        experiment_id=args.experiment_id,
        seed=args.seed,
        base_config_path=args.base_config,
        run_dir_override=args.run_dir,
        skip_eval=args.skip_eval,
        eval_splits=eval_splits,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
