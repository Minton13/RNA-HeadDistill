#!/usr/bin/env python
"""Summarize all frozen downstream probe runs into a single table.

Reads per-run ``eval_*.json`` files under ``outputs/probes/frozen`` and writes:
  - outputs/probes/frozen/frozen_probe_summary.parquet
  - outputs/probes/frozen/frozen_probe_summary.md

Example:
    PYTHONPATH=src python scripts/ablation/summarize_frozen_probes.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from rna_head_alignment.io.paths import resolve_path


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


def summarize_frozen_probes(
    output_root: str | Path = "outputs/probes/frozen",
) -> pd.DataFrame:
    root = Path(resolve_path(output_root))
    rows: list[dict[str, Any]] = []

    if not root.exists():
        return pd.DataFrame(rows)

    for task_dir in sorted(root.iterdir()):
        if not task_dir.is_dir():
            continue
        for mode_dir in sorted(task_dir.iterdir()):
            if not mode_dir.is_dir():
                continue
            for seed_dir in sorted(mode_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                manifest = _read_manifest(seed_dir)
                summary = _read_summary(seed_dir)
                base = {
                    "task": manifest.get("task", task_dir.name),
                    "feature_mode": manifest.get("feature_mode", mode_dir.name),
                    "seed": manifest.get("seed", int(seed_dir.name.split("_")[1])),
                    "task_type": manifest.get("task_type"),
                    "route_depth": manifest.get("route_depth"),
                    "run_dir": str(seed_dir),
                    "best_val_metric": summary.get("best_val_metric"),
                    "best_lr": summary.get("best_cfg", {}).get("lr"),
                    "best_weight_decay": summary.get("best_cfg", {}).get("weight_decay"),
                    "trainable_params": summary.get("trainable_params"),
                    "wall_time_sec": summary.get("wall_time_sec"),
                }
                for split in ("train", "val", "test"):
                    metrics = _read_eval(seed_dir, split)
                    for metric_name, value in metrics.items():
                        base[f"{split}_{metric_name}"] = value
                rows.append(base)

    return pd.DataFrame(rows)


def _format_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_No frozen probe runs found._"
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
        default="outputs/probes/frozen",
        help="Root directory of frozen probe runs",
    )
    args = parser.parse_args()

    df = summarize_frozen_probes(args.output_root)
    out_root = Path(resolve_path(args.output_root))
    out_root.mkdir(parents=True, exist_ok=True)

    parquet_path = out_root / "frozen_probe_summary.parquet"
    md_path = out_root / "frozen_probe_summary.md"

    df.to_parquet(parquet_path, index=False)

    lines = [
        "# Frozen Probe Summary (P0-P6)",
        "",
        f"- tasks: {df['task'].nunique() if not df.empty else 0}",
        f"- feature modes: {df['feature_mode'].nunique() if not df.empty else 0}",
        f"- seeds: {sorted(df['seed'].unique().tolist()) if not df.empty else []}",
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
