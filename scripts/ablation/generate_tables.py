#!/usr/bin/env python
"""Generate markdown tables for the Stage 4B/5 ablation reports.

Reads summary parquet files and writes one markdown table per report under
``outputs/ablation/reports/tables/``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rna_head_alignment.io.paths import resolve_path


def _read_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _format_value(v: Any) -> str:
    if pd.isna(v):
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def df_to_markdown(df: pd.DataFrame, title: str, max_rows: int = 200) -> str:
    if df is None or df.empty:
        return f"## {title}\n\n_No data available._\n"
    df = df.head(max_rows)
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    lines = [f"## {title}", "", header, sep]
    for _, row in df.iterrows():
        vals = [_format_value(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/ablation/reports/tables")
    args = parser.parse_args()

    out_dir = Path(resolve_path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    sources: dict[str, tuple[str, str]] = {
        "alignment_ablation.md": ("Alignment Ablation", "outputs/ablation/alignment/core_ablation_summary.parquet"),
        "data_scaling.md": ("Data Scaling", "outputs/ablation/alignment/data_scaling_summary.parquet"),
        "systems_benchmark.md": ("Systems Benchmark", "outputs/systems/asynchronous_distillation/systems_benchmark_summary.parquet"),
        "frozen_probe.md": ("Frozen Probe Summary", "outputs/probes/frozen/frozen_probe_summary.parquet"),
        "route_finetune.md": ("Route Fine-Tuning", "outputs/probes/finetune/route_finetune_summary.parquet"),
        "replay_ablation.md": ("Alignment Replay Ablation", "outputs/probes/finetune/alignment_replay_ablation.parquet"),
        "multitask.md": ("Multi-Task Joint Model", "outputs/probes/multitask/multitask_summary.parquet"),
        "head_contribution.md": ("Head Contribution", "outputs/probes/frozen/head_contribution_summary.parquet"),
        "length_robustness.md": ("Length Robustness", "outputs/ablation/alignment/length_robustness.parquet"),
    }

    for filename, (title, path) in sources.items():
        df = _read_parquet(Path(resolve_path(path)))
        md = df_to_markdown(df, title)
        (out_dir / filename).write_text(md, encoding="utf-8")
        print(f"[tables] wrote {out_dir / filename}")


if __name__ == "__main__":
    main()
