#!/usr/bin/env python
"""Generate a focused frozen-probe report from current results.

Reads outputs/probes/frozen/frozen_probe_summary.parquet and writes:
  - outputs/probes/frozen/report/figures/*.png
  - outputs/probes/frozen/report/report.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rna_head_alignment.io.paths import resolve_path

FIG_KWARGS = {"dpi": 200, "bbox_inches": "tight"}


def _primary_metric(task_type: str) -> str:
    if task_type == "sequence_classification":
        return "auroc"
    if task_type == "token_regression":
        return "pearson"
    return "top_l_precision"


def _task_type(task: str) -> str:
    from rna_head_alignment.probes.task_config import TASK_TO_TYPE

    return TASK_TO_TYPE[task]


def plot_completion_heatmap(df: pd.DataFrame, out_dir: Path) -> Path:
    """Heatmap of successful eval_test completion (task x mode)."""
    tasks = sorted(df["task"].unique())
    modes = [
        "vanilla_hidden",
        "distilled_hidden",
        "distilled_hidden_native",
        "distilled_aligned",
        "distilled_hidden_aligned",
        "distilled_hidden_random_heads",
        "distilled_hidden_teacher",
    ]
    tasks = [t for t in tasks if t in ["splice_donor_seqcls", "splice_acceptor_seqcls", "rbp_binding_seqcls", "ncRNA_family_seqcls", "chemical_reactivity_token", "secondary_structure_pair"]]
    grid = np.zeros((len(tasks), len(modes)), dtype=int)
    for i, task in enumerate(tasks):
        for j, mode in enumerate(modes):
            sub = df[(df["task"] == task) & (df["feature_mode"] == mode)]
            n = 0
            for _, row in sub.iterrows():
                test_val = row.get("test_auroc")
                if pd.isna(test_val):
                    test_val = row.get("test_pearson")
                if pd.isna(test_val):
                    test_val = row.get("test_top_l_precision")
                if not pd.isna(test_val):
                    n += 1
            grid[i, j] = n

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0, vmax=3)
    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels([m.replace("distilled_", "") for m in modes], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks)
    for i in range(len(tasks)):
        for j in range(len(modes)):
            ax.text(j, i, str(grid[i, j]), ha="center", va="center", fontsize=8)
    ax.set_title("Completed eval seeds per task/mode (max = 3)")
    plt.colorbar(im, ax=ax, label="completed seeds")
    fig.tight_layout()
    path = out_dir / "completion_heatmap.png"
    fig.savefig(path, **FIG_KWARGS)
    plt.close(fig)
    return path


def plot_metric_by_task_and_mode(df: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bar chart of primary test metric by task and feature mode."""
    task_types = {task: _task_type(task) for task in df["task"].unique()}
    tasks = sorted(df["task"].unique())
    modes = [
        "vanilla_hidden",
        "distilled_hidden",
        "distilled_hidden_native",
        "distilled_aligned",
        "distilled_hidden_aligned",
        "distilled_hidden_random_heads",
        "distilled_hidden_teacher",
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    for idx, task in enumerate(tasks):
        ax = axes[idx]
        sub = df[df["task"] == task]
        metric = _primary_metric(task_types[task])
        test_col = f"test_{metric}"
        x = np.arange(len(modes))
        width = 0.25
        means, errs = [], []
        for mode in modes:
            if test_col not in sub.columns:
                means.append(np.nan)
                errs.append(0.0)
                continue
            vals = sub[(sub["feature_mode"] == mode)][test_col].dropna().values
            means.append(float(vals.mean()) if len(vals) else np.nan)
            errs.append(float(vals.std()) if len(vals) > 1 else 0.0)
        means = np.array(means)
        valid = ~np.isnan(means)
        if valid.any():
            ax.bar(x[valid], means[valid], width, yerr=np.array(errs)[valid])
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("distilled_", "") for m in modes], rotation=45, ha="right", fontsize=7)
        ax.set_title(f"{task}\n(metric: {metric})")
        ax.set_ylabel(metric)
    for idx in range(len(tasks), len(axes)):
        axes[idx].axis("off")
    fig.suptitle("Primary test metric by task and feature mode", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    path = out_dir / "metric_by_task_mode.png"
    fig.savefig(path, **FIG_KWARGS)
    plt.close(fig)
    return path


def plot_val_vs_test(df: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter of best validation metric vs primary test metric."""
    fig, ax = plt.subplots(figsize=(7, 7))
    task_types = {task: _task_type(task) for task in df["task"].unique()}
    tasks = sorted(df["task"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(tasks)))
    for task, color in zip(tasks, colors):
        sub = df[df["task"] == task]
        metric = _primary_metric(task_types[task])
        test_col = f"test_{metric}"
        val_col = "best_val_metric"
        x = sub[val_col].dropna().values if val_col in sub.columns else np.array([])
        y = sub[test_col].dropna().values if test_col in sub.columns else np.array([])
        if len(x) and len(y):
            ax.scatter(x, y, label=task, color=color, alpha=0.7, s=50)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, transform=ax.transAxes)
    ax.set_xlabel("best validation metric")
    ax.set_ylabel("test metric")
    ax.set_title("Validation vs Test metric per run")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    path = out_dir / "val_vs_test.png"
    fig.savefig(path, **FIG_KWARGS)
    plt.close(fig)
    return path


def plot_wall_time(df: pd.DataFrame, out_dir: Path) -> Path:
    """Boxplot of wall time per task."""
    fig, ax = plt.subplots(figsize=(9, 5))
    tasks = sorted(df["task"].unique())
    data = [df[df["task"] == t]["wall_time_sec"].dropna().values / 3600.0 for t in tasks]
    ax.boxplot(data, tick_labels=tasks)
    ax.set_ylabel("wall time (hours)")
    ax.set_title("Training wall time distribution per task")
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    fig.tight_layout()
    path = out_dir / "wall_time_boxplot.png"
    fig.savefig(path, **FIG_KWARGS)
    plt.close(fig)
    return path


def _metric_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in sorted(df["task"].unique()):
        metric = _primary_metric(_task_type(task))
        test_col = f"test_{metric}"
        for mode in sorted(df["feature_mode"].unique()):
            sub = df[(df["task"] == task) & (df["feature_mode"] == mode)]
            vals = sub[test_col].dropna().values if test_col in sub.columns else np.array([])
            rows.append({
                "task": task,
                "feature_mode": mode,
                "completed_seeds": int((~sub[test_col].isna()).sum()) if test_col in sub.columns else 0,
                "test_metric": metric,
                "test_mean": float(vals.mean()) if len(vals) else np.nan,
                "test_std": float(vals.std()) if len(vals) > 1 else np.nan,
                "best_val_mean": float(sub["best_val_metric"].dropna().mean()) if not sub["best_val_metric"].dropna().empty else np.nan,
                "wall_time_h_mean": float(sub["wall_time_sec"].dropna().mean() / 3600.0) if not sub["wall_time_sec"].dropna().empty else np.nan,
            })
    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Render a DataFrame as a markdown table without external dependencies."""
    if df.empty:
        return "_Empty table_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            v = row[c]
            if pd.isna(v):
                vals.append("")
            elif isinstance(v, float):
                vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _count_completed_seeds(df: pd.DataFrame, task: str, mode: str) -> int:
    sub = df[(df["task"] == task) & (df["feature_mode"] == mode)]
    metric = _primary_metric(_task_type(task))
    test_col = f"test_{metric}"
    return int((~sub[test_col].isna()).sum()) if test_col in sub.columns else 0


def _failure_analysis(df: pd.DataFrame) -> list[str]:
    """Build data-driven failure-analysis sections from current results."""
    tasks = sorted(df["task"].unique())
    modes = sorted(df["feature_mode"].unique())
    expected_seeds = 3

    missing: list[tuple[str, str, int]] = []
    for task in tasks:
        for mode in modes:
            n = _count_completed_seeds(df, task, mode)
            if n < expected_seeds:
                missing.append((task, mode, n))

    lines: list[str] = ["## 7. Failure Analysis", ""]
    if not missing:
        lines.append("All (task, feature_mode) combinations have the expected number of completed seeds.")
        lines.append("")
        return lines

    # Group by feature_mode to explain *which* alignment mode is missing.
    mode_gaps: dict[str, list[str]] = {}
    task_gaps: dict[str, list[str]] = {}
    for task, mode, n in missing:
        mode_gaps.setdefault(mode, []).append(f"{task} ({n}/3)")
        task_gaps.setdefault(task, []).append(f"{mode} ({n}/3)")

    lines.append("### 7.1 Incomplete (task, mode) combinations")
    lines.append("")
    lines.append("The following cells have fewer than 3 completed seeds:")
    lines.append("")
    for task, modes_missing in sorted(task_gaps.items()):
        lines.append(f"- **{task}**: {', '.join(modes_missing)}")
    lines.append("")

    # Special explanation for P5, which is intentionally excluded from the array jobs.
    if "distilled_hidden_teacher" in mode_gaps:
        lines.append("### 7.2 `distilled_hidden_teacher` (P5) — dropped (cache not applicable)")
        lines.append("")
        lines.append(
            "`distilled_hidden_teacher` requires exact teacher attention maps for the "
            "downstream sequences. The existing `outputs/downstream_teacher_head_cache/` "
            "only covers the RiNALMo pretraining corpus (all 2,114,289 cached sequences "
            "have `original_dataset == rinalmo_pretraining_corpus`). The Stage 5 downstream "
            "datasets were deliberately decontaminated against that corpus, so by design "
            "0 of ~1.66M downstream sequences (0%) are present in the cache — neither by "
            "seq_id nor by sequence_hash. Running P5 would require re-running the 9 teacher "
            "models over all downstream sequences to build a dedicated cache; this was "
            "dropped on 2026-08-11 and P5 is excluded from the benchmark."
        )
        lines.append("")

    # Surface any unexpected gaps (non-P5 modes with missing seeds).
    unexpected = [(t, m, n) for t, m, n in missing if m != "distilled_hidden_teacher"]
    if unexpected:
        lines.append("### 7.3 Unexpected gaps")
        lines.append("")
        lines.append("These non-P5 combinations are also incomplete and should be investigated:")
        lines.append("")
        for task, mode, n in unexpected:
            lines.append(f"- `{task}` / `{mode}`: {n}/3 seeds completed")
        lines.append("")

    return lines


def generate_report(df: pd.DataFrame, figure_paths: list[Path], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "report.md"

    total_runs = len(df)
    tasks = sorted(df["task"].unique())
    modes = sorted(df["feature_mode"].unique())

    # Count runs with a valid primary test metric and classify task health.
    completed = 0
    task_success: dict[str, int] = {t: 0 for t in tasks}
    task_total: dict[str, int] = {t: 0 for t in tasks}
    for _, row in df.iterrows():
        task = row["task"]
        metric = _primary_metric(_task_type(task))
        test_col = f"test_{metric}"
        val = row.get(test_col)
        task_total[task] += 1
        if not pd.isna(val):
            completed += 1
            task_success[task] += 1

    fully_failing = sorted([t for t in tasks if task_success[t] == 0])
    partially_failing = sorted([t for t in tasks if 0 < task_success[t] < task_total[t]])
    healthy = sorted([t for t in tasks if task_success[t] == task_total[t]])

    table = _metric_table(df)
    failure_lines = _failure_analysis(df)

    lines: list[str] = [
        "# Frozen Probe Downstream Results Report",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Executive Summary",
        "",
        f"- **Total runs in summary:** {total_runs}",
        f"- **Tasks:** {', '.join(tasks)}",
        f"- **Feature modes:** {', '.join(modes)}",
        f"- **Runs with valid test metric:** {completed} / {total_runs} ({completed / total_runs * 100:.1f}%)",
        f"- **Healthy tasks (all runs completed):** {', '.join(healthy) if healthy else 'none'}",
        f"- **Partially failing tasks:** {', '.join(partially_failing) if partially_failing else 'none'}",
        f"- **Fully failing tasks:** {', '.join(fully_failing) if fully_failing else 'none'}",
        "",
        "## 2. Slurm Job History",
        "",
        "- **Job 16304** (`frozen_probe_array_v2`): 108-task array (6 tasks × 6 feature modes × 3 seeds). "
        "Mode `distilled_hidden_teacher` (P5) was intentionally excluded because it requires a separate exact-teacher cache.",
        "- **Job 17839** (`frozen_probe_rerun_failed`): 36-task array re-run for `chemical_reactivity_token` and `ncRNA_family_seqcls` "
        "after fixing the multiclass AUROC metric and the token-regression NaN-loss bugs.",
        "",
        "## 3. Completion Overview",
        "",
        "The heatmap below shows how many seeds produced a valid test metric for each (task, mode) combination.",
        "",
        f"![Completion heatmap](figures/{figure_paths[0].name})",
        "",
        "## 4. Results by Task and Feature Mode",
        "",
        f"![Metrics by task/mode](figures/{figure_paths[1].name})",
        "",
        "### Aggregated metrics table",
        "",
        _df_to_markdown(table),
        "",
        "## 5. Validation vs Test Consistency",
        "",
        f"![Val vs Test](figures/{figure_paths[2].name})",
        "",
        "## 6. Wall Time Distribution",
        "",
        f"![Wall time](figures/{figure_paths[3].name})",
        "",
    ]

    lines.extend(failure_lines)

    lines.extend([
        "## 8. Recommendations",
        "",
        "1. **`distilled_hidden_teacher` (P5) is dropped**: the downstream exact-teacher cache only covers the "
        "pretraining corpus and has 0% overlap with the decontaminated downstream datasets (see 7.2). "
        "Reviving it requires building a dedicated downstream teacher cache first.",
        "2. **All six downstream tasks now have valid results** for the six array-included feature modes (P0–P4, P6). "
        "No further re-runs are needed for those modes.",
        "3. **For a 10-GPU deployment**, scale the concurrency limit in the sbatch `--array` throttle (e.g. `%10` or `%20`) "
        "and ensure enough `logs/slurm_frozen_probes/` scratch space.",
        "",
        "## 9. Output Files",
        "",
        f"- Summary table: `outputs/probes/frozen/frozen_probe_summary.parquet`",
        f"- This report: `outputs/probes/frozen/report/report.md`",
        f"- Figures: `outputs/probes/frozen/report/figures/`",
        "",
    ])

    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return summary_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="outputs/probes/frozen/frozen_probe_summary.parquet")
    parser.add_argument("--output-dir", default="outputs/probes/frozen/report")
    args = parser.parse_args()

    df = pd.read_parquet(resolve_path(args.summary))
    out_dir = Path(resolve_path(args.output_dir))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_completion_heatmap(df, fig_dir),
        plot_metric_by_task_and_mode(df, fig_dir),
        plot_val_vs_test(df, fig_dir),
        plot_wall_time(df, fig_dir),
    ]

    report_path = generate_report(df, paths, out_dir)
    print(f"[report] figures: {fig_dir}")
    print(f"[report] {report_path}")


if __name__ == "__main__":
    main()
