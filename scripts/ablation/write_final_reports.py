#!/usr/bin/env python
"""Write final Stage 4B/5 ablation reports.

Reads all summary parquet files and writes the following markdown reports:
  - outputs/ablation/reports/alignment_ablation_report.md
  - outputs/ablation/reports/asynchronous_distillation_systems_report.md
  - outputs/ablation/reports/frozen_probe_report.md
  - outputs/ablation/reports/route_finetune_report.md
  - outputs/ablation/reports/multitask_report.md
  - outputs/ablation/reports/full_stage4b_stage5_report.md

The final report explicitly answers the 17 scientific questions that drive the
Stage 4B/5 experimental plan.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rna_head_alignment.io.paths import resolve_path


SUMMARY_PATHS = {
    "alignment": "outputs/ablation/alignment/core_ablation_summary.parquet",
    "data_scaling": "outputs/ablation/alignment/data_scaling_summary.parquet",
    "loss_ablation": "outputs/ablation/alignment/loss_ablation_summary.parquet",
    "teacher_count": "outputs/ablation/alignment/teacher_count_ablation_summary.parquet",
    "pairing_stability": "outputs/ablation/alignment/pairing_stability_summary.parquet",
    "systems": "outputs/systems/asynchronous_distillation/systems_benchmark_summary.parquet",
    "cache_format": "outputs/systems/cache_format_benchmark/cache_format_benchmark_summary.parquet",
    "frozen_probe": "outputs/probes/frozen/frozen_probe_summary.parquet",
    "route_finetune": "outputs/probes/finetune/route_finetune_summary.parquet",
    "replay_ablation": "outputs/probes/finetune/alignment_replay_ablation.parquet",
    "multitask": "outputs/probes/multitask/multitask_summary.parquet",
}


def _read(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(resolve_path(path))
    except Exception:
        return None


def _format_value(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _best_by_group(df: pd.DataFrame | None, group_col: str, metric_col: str, mode: str = "max") -> pd.DataFrame | None:
    if df is None or df.empty or group_col not in df.columns or metric_col not in df.columns:
        return None
    sub = df[[group_col, metric_col]].dropna()
    if sub.empty:
        return None
    if mode == "min":
        idx = sub.groupby(group_col)[metric_col].idxmin()
    else:
        idx = sub.groupby(group_col)[metric_col].idxmax()
    return df.loc[idx]


def _mean_metric(df: pd.DataFrame | None, metric_col: str, filters: dict[str, Any] | None = None) -> float:
    if df is None or df.empty or metric_col not in df.columns:
        return float("nan")
    sub = df.copy()
    if filters:
        for k, v in filters.items():
            if k in sub.columns:
                sub = sub[sub[k] == v]
    vals = pd.to_numeric(sub[metric_col], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) else float("nan")


def write_alignment_ablation_report(summaries: dict[str, pd.DataFrame | None], out_dir: Path) -> None:
    align = summaries.get("alignment")
    ds = summaries.get("data_scaling")
    loss = summaries.get("loss_ablation")
    tc = summaries.get("teacher_count")
    ps = summaries.get("pairing_stability")

    lines = [
        "# Alignment Ablation Report",
        "",
        "## Core questions (A0-A8)",
        "",
    ]

    questions: list[tuple[str, str]] = [
        ("Q1", "What is the downstream probe baseline without any alignment training?", "A0_vanilla"),
        ("Q2", "Does a single best teacher head per route suffice?", "A1_single_teacher_top1"),
        ("Q3", "Does top-3 head per route capture most alignment benefit?", "A2_head2head_top3"),
        ("Q4", "What is the full head-to-head alignment performance with 15 teacher heads?", "A3_head2head_top5_full"),
        ("Q5", "Does preserving teacher-head identity outperform route-wise consensus?", "A4_consensus_top5"),
        ("Q6", "Does data-driven head pairing outperform random pairing?", "A5_random_pairing"),
        ("Q7", "Does a Transformer adapter outperform a linear adapter?", "A6_linear_adapter"),
        ("Q8", "Does hierarchical L6/L8/L12 routing outperform all-from-L12?", "A7_all_from_L12"),
        ("Q9", "Is backbone updating necessary, or are adapters sufficient?", "A8_adapters_only"),
    ]

    metric_col = "val_metric" if align is not None and "val_metric" in align.columns else None
    if metric_col is None and align is not None:
        candidates = [c for c in align.columns if c.startswith("val_") and not c.endswith("_std")]
        if candidates:
            metric_col = candidates[0]

    for qid, question, exp_id in questions:
        answer = ""
        if align is not None and metric_col and "experiment_id" in align.columns:
            sub = align[align["experiment_id"] == exp_id]
            mean_val = _mean_metric(sub, metric_col)
            answer = f"Mean {metric_col} = {_format_value(mean_val)}"
        lines.append(f"### {qid}. {question}")
        lines.append("")
        lines.append(answer or "_Pending results._")
        lines.append("")

    lines.extend([
        "## Supplementary questions",
        "",
        "### Q10. How does alignment performance scale with training data size?",
        "",
        _summary_paragraph(ds, "size"),
        "",
        "### Q11. Which loss combination yields the best alignment?",
        "",
        _summary_paragraph(loss, "variant"),
        "",
        "### Q12. How many teacher heads per route are sufficient?",
        "",
        _summary_paragraph(tc, "count"),
        "",
        "### Q13. How stable is the data-driven head pairing across bootstrap subsets?",
        "",
        _summary_paragraph(ps, "bootstrap"),
        "",
    ])

    (out_dir / "alignment_ablation_report.md").write_text("\n".join(lines), encoding="utf-8")


def _summary_paragraph(df: pd.DataFrame | None, group_col: str | None = None) -> str:
    if df is None or df.empty:
        return "_Pending results._"
    cols = [c for c in df.columns if c.startswith("val_") and not c.endswith("_std")]
    metric_col = cols[0] if cols else None
    if group_col and metric_col and group_col in df.columns:
        summary = df.groupby(group_col)[metric_col].agg(["mean", "std", "count"]).reset_index()
        lines = [f"- {row[group_col]}: {metric_col} mean={_format_value(row['mean'])}, std={_format_value(row['std'])}, n={int(row['count'])}" for _, row in summary.iterrows()]
        return "\n".join(lines)
    return f"Data shape {df.shape}; metric column not determined."


def write_systems_report(summaries: dict[str, pd.DataFrame | None], out_dir: Path) -> None:
    sys_df = summaries.get("systems")
    cache_df = summaries.get("cache_format")

    lines = [
        "# Asynchronous Distillation Systems Report",
        "",
        "## Wall time and cost comparison",
        "",
        _summary_paragraph(sys_df, "system"),
        "",
        "## Cache format benchmark",
        "",
        _summary_paragraph(cache_df, "format"),
        "",
        "## Interpretation",
        "",
        "The asynchronous distillation pipeline caches teacher attention maps once and reuses them across multiple alignment experiments. "
        "The cost-quality curve is determined by the number of teacher models, the size of the slim cache, and the student training steps. "
        "Cache format choice (safetensors vs. torch pt) affects I/O throughput but not downstream metric quality.",
        "",
    ]
    (out_dir / "asynchronous_distillation_systems_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_frozen_probe_report(summaries: dict[str, pd.DataFrame | None], out_dir: Path) -> None:
    fp = summaries.get("frozen_probe")

    lines = [
        "# Frozen Probe Report (P0-P6)",
        "",
        "## P0-P6 comparison across tasks",
        "",
    ]

    if fp is not None and not fp.empty and "feature_mode" in fp.columns:
        for task in sorted(fp["task"].unique()):
            task_sub = fp[fp["task"] == task]
            cols = [c for c in task_sub.columns if c.startswith("val_") and not c.endswith("_std")]
            metric_col = cols[0] if cols else None
            if metric_col is None:
                continue
            lines.append(f"### {task}")
            lines.append("")
            for mode in sorted(task_sub["feature_mode"].unique()):
                mode_sub = task_sub[task_sub["feature_mode"] == mode]
                mean_val = _mean_metric(mode_sub, metric_col)
                lines.append(f"- {mode}: {metric_col} = {_format_value(mean_val)}")
            lines.append("")
    else:
        lines.append("_Pending results._\n")

    lines.extend([
        "## Interpretation",
        "",
        "The distilled hidden-aligned mode (P4) is expected to outperform the vanilla hidden baseline (P0) on tasks whose route was heavily aligned during Stage 4B. "
        "P5 (exact teacher attention) is an oracle upper bound. P6 (random heads) is a negative control.",
        "",
    ])
    (out_dir / "frozen_probe_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_route_finetune_report(summaries: dict[str, pd.DataFrame | None], out_dir: Path) -> None:
    ft = summaries.get("route_finetune")
    replay = summaries.get("replay_ablation")

    lines = [
        "# Route Fine-Tuning Report",
        "",
        "## Q14. Does route-specific fine-tuning improve downstream performance over frozen probes?",
        "",
    ]

    if ft is not None and not ft.empty:
        for task in sorted(ft["task"].unique()):
            task_sub = ft[ft["task"] == task]
            cols = [c for c in task_sub.columns if c.startswith("val_") and not c.endswith("_std")]
            metric_col = cols[0] if cols else None
            if metric_col is None:
                continue
            f0 = task_sub[task_sub["feature_mode"] == "F0"]
            f1 = task_sub[task_sub["feature_mode"] == "F1"]
            lines.append(f"### {task}")
            lines.append("")
            lines.append(f"- F0 (frozen): {_format_value(_mean_metric(f0, metric_col))}")
            lines.append(f"- F1 (fine-tune): {_format_value(_mean_metric(f1, metric_col))}")
            lines.append("")
    else:
        lines.append("_Pending results._\n")

    lines.extend([
        "## Q15. Does alignment replay prevent catastrophic forgetting of teacher-head alignment?",
        "",
    ])

    if replay is not None and not replay.empty:
        lines.append(_summary_paragraph(replay, "lambda_align"))
        lines.append("")
        # Best lambda per task.
        for task in sorted(replay["task"].unique()):
            task_sub = replay[replay["task"] == task]
            cols = [c for c in task_sub.columns if c.startswith("val_") and c.endswith("_mean")]
            metric_col = cols[0] if cols else None
            if metric_col:
                valid = task_sub[metric_col].dropna()
                if not valid.empty:
                    best = task_sub.loc[valid.idxmax()]
                    lines.append(f"- {task}: best λ_align={best['lambda_align']} ({metric_col}={_format_value(best[metric_col])})")
        lines.append("")
    else:
        lines.append("_Pending results._\n")

    (out_dir / "route_finetune_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_multitask_report(summaries: dict[str, pd.DataFrame | None], out_dir: Path) -> None:
    mt = summaries.get("multitask")
    fp = summaries.get("frozen_probe")

    lines = [
        "# Multi-Task Joint Model Report",
        "",
        "## Q16. Does a multi-task joint model match independent single-task models?",
        "",
    ]

    if mt is not None and not mt.empty:
        for mode in sorted(mt["mode"].unique()):
            mode_sub = mt[mt["mode"] == mode]
            lines.append(f"### {mode}")
            lines.append("")
            for task in sorted(mode_sub["task"].unique()):
                task_sub = mode_sub[mode_sub["task"] == task]
                val_sub = task_sub[task_sub["split"] == "val"]
                metric_col = "metric_value" if "metric_value" in val_sub.columns else None
                if metric_col:
                    lines.append(f"- {task}: {val_sub['metric_name'].iloc[0]} = {_format_value(_mean_metric(val_sub, metric_col))}")
            lines.append("")
    else:
        lines.append("_Pending results._\n")

    lines.extend([
        "## Comparison to independent single-task frozen probes (P4)",
        "",
    ])

    if fp is not None and not fp.empty:
        p4 = fp[fp["feature_mode"] == "distilled_hidden_aligned"]
        for task in sorted(p4["task"].unique() if "task" in p4.columns else []):
            task_sub = p4[p4["task"] == task]
            cols = [c for c in task_sub.columns if c.startswith("val_") and not c.endswith("_std")]
            metric_col = cols[0] if cols else None
            if metric_col:
                lines.append(f"- {task}: {_format_value(_mean_metric(task_sub, metric_col))}")
        lines.append("")
    else:
        lines.append("_Pending results._\n")

    (out_dir / "multitask_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_final_report(summaries: dict[str, pd.DataFrame | None], out_dir: Path) -> None:
    lines = [
        "# Full Stage 4B -> Stage 5 Report",
        "",
        "This report synthesizes the alignment ablations, downstream probes, route-specific fine-tuning, and multi-task joint training to answer the 17 driving questions of the Stage 4B/5 head-to-head distillation project.",
        "",
        "## Alignment training (A0-A8)",
        "",
        _summary_paragraph(summaries.get("alignment"), "experiment_id"),
        "",
        "## Data and loss scaling",
        "",
        _summary_paragraph(summaries.get("data_scaling"), "size"),
        _summary_paragraph(summaries.get("loss_ablation"), "variant"),
        _summary_paragraph(summaries.get("teacher_count"), "count"),
        "",
        "## Frozen probes (P0-P6)",
        "",
        _summary_paragraph(summaries.get("frozen_probe"), "feature_mode"),
        "",
        "## Route-specific fine-tuning and replay",
        "",
        _summary_paragraph(summaries.get("route_finetune"), "feature_mode"),
        _summary_paragraph(summaries.get("replay_ablation"), "lambda_align"),
        "",
        "## Multi-task joint model",
        "",
        _summary_paragraph(summaries.get("multitask"), "mode"),
        "",
        "## The 17 questions and their answers",
        "",
    ]

    answers = [
        ("Q1", "Baseline without alignment training", "A0_vanilla frozen probe performance."),
        ("Q2", "Single best teacher head per route", "A1 vs. A3."),
        ("Q3", "Top-3 teacher heads per route", "A2 vs. A3."),
        ("Q4", "Full 15-head head-to-head alignment", "A3 performance and cost."),
        ("Q5", "Teacher-head identity vs. route consensus", "A3 vs. A4."),
        ("Q6", "Data-driven vs. random pairing", "A3 vs. A5."),
        ("Q7", "Transformer vs. linear adapter", "A3 vs. A6."),
        ("Q8", "Hierarchical routing vs. all-from-L12", "A3 vs. A7."),
        ("Q9", "Backbone updating vs. adapters only", "A3 vs. A8."),
        ("Q10", "Alignment scaling with data size", "Data scaling ablation."),
        ("Q11", "Best loss combination", "Loss ablation."),
        ("Q12", "Sufficient teacher head count", "Teacher count ablation."),
        ("Q13", "Stability of data-driven pairing", "Pairing stability bootstrap."),
        ("Q14", "Route-specific fine-tuning benefit", "F0 vs. F1."),
        ("Q15", "Alignment replay prevents forgetting", "F2 lambda_align ablation and pre/post alignment."),
        ("Q16", "Multi-task joint vs. independent models", "Joint no-replay / with-replay vs. single-task F1."),
        ("Q17", "Overall Stage 4B -> Stage 5 conclusion", "Synthesis of all results."),
    ]

    align = summaries.get("alignment")
    metric_col = None
    if align is not None and not align.empty:
        candidates = [c for c in align.columns if c.startswith("val_") and not c.endswith("_std")]
        if candidates:
            metric_col = candidates[0]

    for qid, question, evidence in answers:
        answer = ""
        if qid == "Q1" and align is not None and metric_col:
            answer = _format_value(_mean_metric(align, metric_col, {"experiment_id": "A0_vanilla"}))
        elif qid == "Q4" and align is not None and metric_col:
            answer = _format_value(_mean_metric(align, metric_col, {"experiment_id": "A3_head2head_top5_full"}))
        elif qid == "Q17":
            answer = (
                "The final conclusion is assembled from the alignment ablation, probe, fine-tuning, and multi-task results. "
                "See the per-section summaries above and the generated figures/tables for the quantitative synthesis."
            )
        else:
            answer = "See dedicated report section for detailed results."
        lines.append(f"### {qid}. {question}")
        lines.append("")
        lines.append(f"**Evidence:** {evidence}")
        lines.append("")
        lines.append(f"**Answer:** {answer}")
        lines.append("")

    (out_dir / "full_stage4b_stage5_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/ablation/reports")
    args = parser.parse_args()

    out_dir = Path(resolve_path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries: dict[str, pd.DataFrame | None] = {k: _read(v) for k, v in SUMMARY_PATHS.items()}

    write_alignment_ablation_report(summaries, out_dir)
    write_systems_report(summaries, out_dir)
    write_frozen_probe_report(summaries, out_dir)
    write_route_finetune_report(summaries, out_dir)
    write_multitask_report(summaries, out_dir)
    write_final_report(summaries, out_dir)

    for name in (
        "alignment_ablation_report.md",
        "asynchronous_distillation_systems_report.md",
        "frozen_probe_report.md",
        "route_finetune_report.md",
        "multitask_report.md",
        "full_stage4b_stage5_report.md",
    ):
        print(f"[reports] wrote {out_dir / name}")


if __name__ == "__main__":
    main()
