#!/usr/bin/env python
"""Summarize supplementary ablations into per-type parquet and markdown tables.

Reads run directories under outputs/ablation/alignment/{data_scaling,
loss_ablation, pairing_stability, teacher_count_ablation} and writes:
  - outputs/ablation/alignment/{type}.parquet
  - outputs/ablation/alignment/{type}.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rna_head_alignment.io.paths import resolve_path

ABLATION_TYPES = [
    "data_scaling",
    "loss_ablation",
    "pairing_stability",
    "teacher_count_ablation",
]


def _read_train_metrics(run_dir: Path) -> dict[str, Any]:
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
    val_csv = run_dir / "val_metrics.csv"
    if val_csv.exists():
        df = pd.read_csv(val_csv)
        if not df.empty:
            out["best_val_loss_total"] = float(df["loss/total"].min())
            out["final_val_loss_total"] = float(df["loss/total"].iloc[-1])
    return out


def _read_eval_summary(run_dir: Path, run_name: str, split: str = "val") -> dict[str, Any]:
    summary_path = Path(run_dir).parent / run_name / split / "summary.parquet"
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


def _read_config(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "config_resolved.yaml"
    if not path.exists():
        return {}
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def summarize_one_type(ablation_type: str, root: Path) -> pd.DataFrame:
    type_root = root / ablation_type
    rows: list[dict[str, Any]] = []
    if not type_root.exists():
        return pd.DataFrame(rows)
    for run_dir in sorted(type_root.glob("*_seed*")):
        if not run_dir.is_dir():
            continue
        cfg = _read_config(run_dir)
        run_name = cfg.get("run_name", run_dir.name)
        # Infer variant and seed from directory name.
        name = run_dir.name
        seed = None
        if "_seed" in name:
            try:
                seed = int(name.split("_seed")[-1])
            except ValueError:
                pass
        variant = name.replace("_seed" + str(seed), "") if seed is not None else name
        row: dict[str, Any] = {
            "ablation_type": ablation_type,
            "variant": variant,
            "seed": seed,
            "run_dir": str(run_dir),
            "run_name": run_name,
            "status": "trained" if (run_dir / "train_metrics.csv").exists() else "missing",
        }
        row.update(_read_train_metrics(run_dir))
        row.update(_read_eval_summary(run_dir, run_name, "val"))
        row.update(_read_eval_summary(run_dir, run_name, "test"))
        rows.append(row)
    return pd.DataFrame(rows)


def _format_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No runs found._"
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="outputs/ablation/alignment",
        help="Root directory containing supplementary ablation outputs",
    )
    args = parser.parse_args()
    root = Path(resolve_path(args.output_root))
    root.mkdir(parents=True, exist_ok=True)

    for ablation_type in ABLATION_TYPES:
        df = summarize_one_type(ablation_type, root)
        parquet_path = root / f"{ablation_type}.parquet"
        md_path = root / f"{ablation_type}.md"
        df.to_parquet(parquet_path, index=False)
        lines = [
            f"# Supplementary Ablation Summary: {ablation_type}",
            "",
            f"- rows: {len(df)}",
            "",
            _format_markdown(df),
            "",
        ]
        md_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"[summarize] {ablation_type}: {parquet_path}, {md_path}")


if __name__ == "__main__":
    main()
