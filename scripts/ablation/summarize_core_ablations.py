#!/usr/bin/env python
"""Summarize all core alignment ablation runs (A0-A8) into a single table.

Reads per-seed run directories under outputs/ablation/alignment and writes:
  - outputs/ablation/alignment/core_ablation_summary.parquet
  - outputs/ablation/alignment/core_ablation_summary.md

Example:
    PYTHONPATH=src python scripts/ablation/summarize_core_ablations.py \
        --registry configs/ablation/experiment_registry.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rna_head_alignment.io.paths import load_yaml, resolve_path


def _read_train_metrics(run_dir: Path) -> dict[str, Any]:
    """Best/last train and validation metrics from CSV logs."""
    out: dict[str, Any] = {}
    train_csv = run_dir / "train_metrics.csv"
    if train_csv.exists():
        df = pd.read_csv(train_csv)
        if not df.empty:
            out["final_train_loss_total"] = float(df["loss/total"].iloc[-1])
            out["final_step"] = int(df["step"].iloc[-1])
            if "peak_vram_gb" in df.columns:
                out["peak_vram_gb"] = float(df["peak_vram_gb"].max())
            if "wall_time_sec" in df.columns:
                out["total_wall_time_sec"] = float(df["wall_time_sec"].iloc[-1])
            if "data_wait_time_sec" in df.columns:
                out["mean_data_wait_time_sec"] = float(df["data_wait_time_sec"].mean())
    val_csv = run_dir / "val_metrics.csv"
    if val_csv.exists():
        df = pd.read_csv(val_csv)
        if not df.empty:
            out["best_val_loss_total"] = float(df["loss/total"].min())
            out["final_val_loss_total"] = float(df["loss/total"].iloc[-1])
            if "wall_time_sec" in df.columns:
                out["val_wall_time_at_best_sec"] = float(
                    df.loc[df["loss/total"].idxmin(), "wall_time_sec"]
                )
    return out


def _read_eval_summary(run_dir: Path, run_name: str, split: str = "val") -> dict[str, Any]:
    """Read aggregated eval summary parquet and return macro student metrics."""
    # Eval outputs are written to run_dir.parent / run_name / split.
    summary_path = run_dir.parent / run_name / split / "summary.parquet"
    if not summary_path.exists():
        return {}
    df = pd.read_parquet(summary_path)
    student_df = df[df["baseline"] == "student"]
    out: dict[str, Any] = {}
    for metric in student_df["metric"].unique():
        vals = student_df[student_df["metric"] == metric]["mean"]
        if not vals.empty:
            out[f"{split}_{metric}_mean"] = float(vals.mean())
    return out


def _read_vanilla_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "vanilla_manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _read_reuse_marker(run_dir: Path) -> dict[str, Any]:
    marker = run_dir / "REUSED_FROM"
    if not marker.exists():
        return {}
    return {"reused_from": marker.read_text(encoding="utf-8").strip().splitlines()[0]}


def summarize_core_ablations(
    registry_path: str | Path,
    output_root: str | Path = "outputs/ablation/alignment",
) -> pd.DataFrame:
    """Collect summary rows for every registered core ablation (experiment, seed)."""
    registry = load_yaml(registry_path)
    rows: list[dict[str, Any]] = []

    for experiment_id, exp_cfg in registry.get("experiments", {}).items():
        if exp_cfg.get("category") != "core_ablation":
            continue
        output_root_exp = Path(resolve_path(exp_cfg.get("output_root", f"{output_root}/{experiment_id}")))
        for seed in exp_cfg.get("seeds", registry.get("seeds", [])):
            run_dir = output_root_exp / f"seed_{seed}"
            run_name = f"{experiment_id}_seed{seed}"
            row: dict[str, Any] = {
                "experiment_id": experiment_id,
                "seed": seed,
                "scientific_question": exp_cfg.get("scientific_question", ""),
                "run_dir": str(run_dir),
                "run_name": run_name,
                "status": "missing",
            }
            if not run_dir.exists():
                rows.append(row)
                continue

            # Determine status from markers.
            vanilla = _read_vanilla_manifest(run_dir)
            reused = _read_reuse_marker(run_dir)
            if vanilla:
                row["status"] = "vanilla_manifest"
            elif reused:
                row["status"] = "reused"
                row["reused_from"] = reused["reused_from"]
            else:
                row["status"] = "trained"

            row.update(_read_train_metrics(run_dir))
            row.update(_read_eval_summary(run_dir, run_name, "val"))
            row.update(_read_eval_summary(run_dir, run_name, "test"))
            rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        default="configs/ablation/experiment_registry.yaml",
        help="Experiment registry YAML",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/ablation/alignment",
        help="Root directory for ablation outputs",
    )
    args = parser.parse_args()

    df = summarize_core_ablations(args.registry, args.output_root)
    out_root = Path(resolve_path(args.output_root))
    out_root.mkdir(parents=True, exist_ok=True)

    parquet_path = out_root / "core_ablation_summary.parquet"
    md_path = out_root / "core_ablation_summary.md"

    df.to_parquet(parquet_path, index=False)

    def _format_markdown(df: pd.DataFrame) -> str:
        if df.empty:
            return "_No ablation runs found._"
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
                    vals.append(str(v))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# Core Alignment Ablation Summary (A0-A8)",
        "",
        f"- experiments: {df['experiment_id'].nunique()}",
        f"- seeds per experiment: {df.groupby('experiment_id').size().unique().tolist()}",
        f"- rows: {len(df)}",
        "",
        _format_markdown(df),
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"[summarize] wrote {parquet_path}")
    print(f"[summarize] wrote {md_path}")


if __name__ == "__main__":
    main()
