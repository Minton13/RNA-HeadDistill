#!/usr/bin/env python
"""Alignment replay ablation: aggregate F2 fine-tuning runs across lambda values.

Reads per-run outputs under ``outputs/probes/finetune`` for feature mode ``F2``
(fine-tune with alignment replay) and writes:
  - outputs/probes/finetune/alignment_replay_ablation.parquet
  - outputs/probes/finetune/replay_ablation_report.md

If ``--run`` is passed, the script launches ``run_route_finetune.py`` for every
(task, seed, lambda_align) combination before summarizing.  Without ``--run``
it assumes the route-finetune array has already populated the outputs.

Example:
    PYTHONPATH=src python scripts/ablation/run_alignment_replay_ablation.py
    PYTHONPATH=src python scripts/ablation/run_alignment_replay_ablation.py --run
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rna_head_alignment.io.paths import resolve_path
from rna_head_alignment.probes.task_config import TASK_TO_ROUTE_DEPTH


TASKS = [
    "splice_donor_seqcls",
    "splice_acceptor_seqcls",
    "rbp_binding_seqcls",
    "ncRNA_family_seqcls",
    "chemical_reactivity_token",
    "secondary_structure_pair",
]
SEEDS = [17, 41, 89]
LAMBDAS = [0.0, 0.01, 0.1, 0.5]


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _read_eval(run_dir: Path, split: str) -> dict[str, Any]:
    path = run_dir / f"eval_{split}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "training_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def collect_runs(output_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    if not output_root.exists():
        return pd.DataFrame(rows)

    for task_dir in sorted(output_root.iterdir()):
        if not task_dir.is_dir() or task_dir.name not in TASKS:
            continue
        f2_dir = task_dir / "F2"
        if not f2_dir.exists():
            continue
        for seed_dir in sorted(f2_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = int(seed_dir.name.split("_")[1])
            for lambda_dir in sorted(seed_dir.iterdir()):
                if not lambda_dir.is_dir() or not lambda_dir.name.startswith("lambda_"):
                    continue
                lam = float(lambda_dir.name.split("_")[1])
                manifest = _read_manifest(lambda_dir)
                summary = _read_summary(lambda_dir)
                base = {
                    "task": manifest.get("task", task_dir.name),
                    "route_depth": TASK_TO_ROUTE_DEPTH.get(task_dir.name),
                    "seed": seed,
                    "lambda_align": lam,
                    "run_dir": str(lambda_dir),
                    "best_val_metric": summary.get("best_val_metric"),
                    "best_step": summary.get("best_step"),
                    "trainable_params": summary.get("trainable_params"),
                    "wall_time_sec": summary.get("wall_time_sec"),
                }
                for split in ("train", "val", "test"):
                    metrics = _read_eval(lambda_dir, split)
                    for metric_name, value in metrics.items():
                        base[f"{split}_{metric_name}"] = value
                pre = summary.get("pre_alignment", {}) or {}
                post = summary.get("post_alignment", {}) or {}
                for k, v in pre.items():
                    base[f"pre_{k}"] = v
                for k, v in post.items():
                    base[f"post_{k}"] = v
                rows.append(base)

    return pd.DataFrame(rows)


def _metric_for_task(task: str) -> str:
    from rna_head_alignment.probes.task_config import TASK_TO_TYPE

    task_type = TASK_TO_TYPE[task]
    if task_type == "sequence_classification":
        return "auroc"
    if task_type == "token_regression":
        return "pearson"
    return "top_l_precision"


def build_ablation_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    records: list[dict[str, Any]] = []
    for task in sorted(df["task"].unique()):
        sub = df[df["task"] == task]
        metric = _metric_for_task(task)
        for lam in sorted(sub["lambda_align"].unique()):
            lam_sub = sub[sub["lambda_align"] == lam]
            val_col = f"val_{metric}"
            pre_cols = [c for c in lam_sub.columns if c.startswith("pre_")]
            post_cols = [c for c in lam_sub.columns if c.startswith("post_")]
            record = {
                "task": task,
                "lambda_align": lam,
                "n_seeds": int(lam_sub["seed"].nunique()),
                f"val_{metric}_mean": float(lam_sub[val_col].mean()) if val_col in lam_sub else float("nan"),
                f"val_{metric}_std": float(lam_sub[val_col].std()) if val_col in lam_sub else float("nan"),
            }
            for c in pre_cols:
                record[f"{c}_mean"] = float(lam_sub[c].mean())
                record[f"{c}_std"] = float(lam_sub[c].std())
            for c in post_cols:
                record[f"{c}_mean"] = float(lam_sub[c].mean())
                record[f"{c}_std"] = float(lam_sub[c].std())
            records.append(record)

    return pd.DataFrame(records)


def _format_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No F2 replay runs found. Run the route-finetune array first._"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            else:
                vals.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def launch_runs(output_root: Path, mixed_precision: bool) -> None:
    """Launch all F2 runs sequentially (intended for local smoke tests only)."""
    for task in TASKS:
        for seed in SEEDS:
            for lam in LAMBDAS:
                cmd = [
                    sys.executable,
                    "scripts/ablation/run_route_finetune.py",
                    "--task", task,
                    "--feature-mode", "F2",
                    "--seed", str(seed),
                    "--lambda-align", str(lam),
                    "--output-root", str(output_root),
                ]
                if mixed_precision:
                    cmd.append("--mixed-precision")
                print("[launch]", " ".join(cmd))
                subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/probes/finetune")
    parser.add_argument("--run", action="store_true", help="Launch F2 runs before summarizing")
    parser.add_argument("--mixed-precision", action="store_true")
    args = parser.parse_args()

    output_root = Path(resolve_path(args.output_root))

    if args.run:
        launch_runs(output_root, args.mixed_precision)

    df = collect_runs(output_root)
    ablation = build_ablation_table(df)

    output_root.mkdir(parents=True, exist_ok=True)
    parquet_path = output_root / "alignment_replay_ablation.parquet"
    md_path = output_root / "replay_ablation_report.md"

    ablation.to_parquet(parquet_path, index=False)

    # Compute best lambda per task by validation metric.
    best_rows: list[str] = []
    if not ablation.empty:
        for task in sorted(ablation["task"].unique()):
            sub = ablation[ablation["task"] == task]
            metric_col = [c for c in sub.columns if c.startswith("val_") and c.endswith("_mean")][0]
            best = sub.loc[sub[metric_col].idxmax()]
            best_rows.append(
                f"- **{task}**: best lambda_align={best['lambda_align']} "
                f"({metric_col}={best[metric_col]:.4f})"
            )

    lines = [
        "# Alignment Replay Ablation Report",
        "",
        "This report compares route-specific fine-tuning with alignment replay "
        "across `lambda_align` values.  Training alternates 4 downstream batches "
        "with 1 alignment-replay batch drawn from a deterministic 50k-sequence "
        "stratified subset of the Stage 4B slim cache.",
        "",
        "## Summary: best lambda per task",
        "",
    ]
    lines.extend(best_rows if best_rows else ["_No data._"])
    lines.extend([
        "",
        "## Per-task, per-lambda results",
        "",
        _format_markdown(ablation),
        "",
        "## Method notes",
        "",
        "- `lambda_align = 0` corresponds to F1 (no replay), because the replay loss "
        "is multiplied by zero even though the replay batches are still sampled.",
        "- Teacher-head alignment before/after fine-tuning is measured on a 1,024-"
        "sequence sample from the replay subset using MSE, row-wise KL, and cosine "
        "similarity averaged over routes L6/L8/L12.",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[replay-ablation] wrote {parquet_path}")
    print(f"[replay-ablation] wrote {md_path}")


if __name__ == "__main__":
    main()
