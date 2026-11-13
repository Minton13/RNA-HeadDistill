#!/usr/bin/env python
"""Summarize multi-task joint model experiments.

Reads per-run outputs under ``outputs/probes/multitask`` and writes:
  - outputs/probes/multitask/multitask_summary.parquet
  - outputs/probes/multitask/multitask_summary.md

Example:
    PYTHONPATH=src python scripts/ablation/summarize_multitask.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rna_head_alignment.io.paths import resolve_path
from rna_head_alignment.probes.task_config import TASK_TO_ROUTE_DEPTH, TASK_TO_TYPE


def _read_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "manifest.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _read_eval(run_dir: Path, task: str, split: str) -> dict[str, Any]:
    path = run_dir / task / f"eval_{split}.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "training_summary.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_for_task(task: str) -> str:
    task_type = TASK_TO_TYPE[task]
    if task_type == "sequence_classification":
        return "auroc"
    if task_type == "token_regression":
        return "pearson"
    return "top_l_precision"


def summarize_multitask(output_root: str | Path = "outputs/probes/multitask") -> pd.DataFrame:
    root = Path(resolve_path(output_root))
    rows: list[dict[str, Any]] = []

    if not root.exists():
        return pd.DataFrame(rows)

    for mode_dir in sorted(root.iterdir()):
        if not mode_dir.is_dir():
            continue
        for seed_dir in sorted(mode_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            seed = int(seed_dir.name.split("_")[1])
            manifest = _read_manifest(seed_dir)
            summary = _read_summary(seed_dir)
            for task in TASK_TO_ROUTE_DEPTH:
                metric = _metric_for_task(task)
                for split in ("train", "val", "test"):
                    metrics = _read_eval(seed_dir, task, split)
                    rows.append({
                        "mode": manifest.get("mode", mode_dir.name),
                        "seed": seed,
                        "task": task,
                        "route_depth": TASK_TO_ROUTE_DEPTH[task],
                        "split": split,
                        "metric_name": metric,
                        "metric_value": metrics.get(metric),
                        "best_mean_val_primary": summary.get("best_mean_val_primary"),
                        "trainable_params": summary.get("trainable_params"),
                        "wall_time_sec": summary.get("wall_time_sec"),
                        "run_dir": str(seed_dir),
                    })

    return pd.DataFrame(rows)


def _format_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No multi-task runs found._"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="outputs/probes/multitask")
    args = parser.parse_args()

    df = summarize_multitask(args.output_root)
    out_root = Path(resolve_path(args.output_root))
    out_root.mkdir(parents=True, exist_ok=True)

    parquet_path = out_root / "multitask_summary.parquet"
    md_path = out_root / "multitask_summary.md"

    df.to_parquet(parquet_path, index=False)

    # Aggregate mean val metric per mode/task.
    agg_rows: list[dict[str, Any]] = []
    if not df.empty:
        val_df = df[df["split"] == "val"].copy()
        for (mode, task), sub in val_df.groupby(["mode", "task"]):
            metric_name = sub["metric_name"].iloc[0]
            agg_rows.append({
                "mode": mode,
                "task": task,
                "metric_name": metric_name,
                "mean": float(sub["metric_value"].mean()),
                "std": float(sub["metric_value"].std()),
                "n_seeds": int(sub["seed"].nunique()),
            })
    agg = pd.DataFrame(agg_rows)

    lines = [
        "# Multi-Task Joint Model Summary",
        "",
        "## Per-run, per-task results",
        "",
        _format_markdown(df),
        "",
        "## Aggregate validation performance",
        "",
        _format_markdown(agg),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[summarize] wrote {parquet_path}")
    print(f"[summarize] wrote {md_path}")


if __name__ == "__main__":
    main()
