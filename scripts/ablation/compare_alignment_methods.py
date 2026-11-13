#!/usr/bin/env python
"""Compare different alignment/feature modes on downstream tasks.

Reads outputs/probes/frozen/frozen_probe_summary.parquet and writes:
  - outputs/probes/frozen/report/figures/alignment_comparison_*.png
  - outputs/probes/frozen/report/alignment_comparison.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rna_head_alignment.io.paths import resolve_path
from rna_head_alignment.probes.task_config import TASK_TO_TYPE

FIG_KWARGS = {"dpi": 200, "bbox_inches": "tight"}

MODE_LABELS: dict[str, str] = {
    "vanilla_hidden": "P0\nvanilla",
    "distilled_hidden": "P1\ndistilled hidden",
    "distilled_hidden_native": "P2\nnative heads",
    "distilled_aligned": "P3\naligned",
    "distilled_hidden_aligned": "P4\nhidden+aligned",
    "distilled_hidden_random_heads": "P6\nrandom heads",
    "distilled_hidden_teacher": "P5\nteacher",
}

MODE_ORDER = [
    "vanilla_hidden",
    "distilled_hidden",
    "distilled_hidden_native",
    "distilled_aligned",
    "distilled_hidden_aligned",
    "distilled_hidden_random_heads",
    "distilled_hidden_teacher",
]


def _primary_metric(task_type: str) -> str:
    if task_type == "sequence_classification":
        return "auroc"
    if task_type == "token_regression":
        return "pearson"
    return "top_l_precision"


def _task_metric(task: str) -> str:
    return _primary_metric(TASK_TO_TYPE[task])


def _test_col(task: str) -> str:
    return f"test_{_task_metric(task)}"


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in df.iterrows():
        task = row["task"]
        mode = row["feature_mode"]
        col = _test_col(task)
        val = row.get(col)
        if col in df.columns and not pd.isna(val):
            rows.append({
                "task": task,
                "feature_mode": mode,
                "metric_name": _task_metric(task),
                "test_score": float(val),
                "best_val": float(row["best_val_metric"]) if "best_val_metric" in df.columns and not pd.isna(row["best_val_metric"]) else np.nan,
            })
    return pd.DataFrame(rows)


def plot_grouped_bars(comp_df: pd.DataFrame, out_dir: Path) -> Path:
    """Grouped bar chart: one group per task, bars = feature modes."""
    tasks = sorted(comp_df["task"].unique())
    modes = [m for m in MODE_ORDER if m in comp_df["feature_mode"].values]
    n_tasks = len(tasks)
    n_modes = len(modes)
    width = 0.11
    x = np.arange(n_tasks)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Mean test score.
    ax = axes[0]
    for offset, mode in enumerate(modes):
        vals = []
        for task in tasks:
            sub = comp_df[(comp_df["task"] == task) & (comp_df["feature_mode"] == mode)]
            vals.append(float(sub["test_score"].mean()) if not sub.empty else np.nan)
        vals = np.array(vals)
        valid = ~np.isnan(vals)
        ax.bar(
            x[valid] + (offset - n_modes / 2 + 0.5) * width,
            vals[valid],
            width,
            label=MODE_LABELS.get(mode, mode),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel("test score")
    ax.set_title("Test score by task and alignment method")
    ax.legend(fontsize=7, loc="best")
    ax.grid(axis="y", alpha=0.3)

    # Number of completed seeds.
    ax = axes[1]
    for offset, mode in enumerate(modes):
        counts = []
        for task in tasks:
            sub = comp_df[(comp_df["task"] == task) & (comp_df["feature_mode"] == mode)]
            counts.append(len(sub))
        counts = np.array(counts)
        ax.bar(
            x + (offset - n_modes / 2 + 0.5) * width,
            counts,
            width,
            label=MODE_LABELS.get(mode, mode),
        )
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, rotation=30, ha="right")
    ax.set_ylabel("completed seeds")
    ax.set_title("Completed seeds by task and alignment method")
    ax.legend(fontsize=7, loc="best")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = out_dir / "alignment_comparison_grouped_bars.png"
    fig.savefig(path, **FIG_KWARGS)
    plt.close(fig)
    return path


def plot_heatmap(comp_df: pd.DataFrame, out_dir: Path) -> Path:
    """Heatmap of mean test score (task x feature mode)."""
    tasks = sorted(comp_df["task"].unique())
    modes = [m for m in MODE_ORDER if m in comp_df["feature_mode"].values]
    grid = np.full((len(tasks), len(modes)), np.nan)
    for i, task in enumerate(tasks):
        for j, mode in enumerate(modes):
            sub = comp_df[(comp_df["task"] == task) & (comp_df["feature_mode"] == mode)]
            if not sub.empty:
                grid[i, j] = float(sub["test_score"].mean())

    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(grid, aspect="auto", cmap="RdYlGn", vmin=0.6, vmax=0.9)
    ax.set_xticks(np.arange(len(modes)))
    ax.set_xticklabels([MODE_LABELS.get(m, m) for m in modes], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(tasks)))
    ax.set_yticklabels(tasks)
    for i in range(len(tasks)):
        for j in range(len(modes)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title("Mean test score: task × alignment method")
    plt.colorbar(im, ax=ax, label="test score")
    fig.tight_layout()
    path = out_dir / "alignment_comparison_heatmap.png"
    fig.savefig(path, **FIG_KWARGS)
    plt.close(fig)
    return path


def plot_ranking(comp_df: pd.DataFrame, out_dir: Path) -> Path:
    """Rank feature modes by average normalized rank per task."""
    tasks = sorted(comp_df["task"].unique())
    modes = sorted(comp_df["feature_mode"].unique())
    ranks: dict[str, list[float]] = {m: [] for m in modes}
    for task in tasks:
        sub = comp_df[comp_df["task"] == task]
        if sub.empty:
            continue
        mean_scores = sub.groupby("feature_mode")["test_score"].mean().sort_values(ascending=False)
        for rank, (mode, _) in enumerate(mean_scores.items(), 1):
            ranks[mode].append(rank)

    avg_ranks = {m: float(np.mean(r)) if r else np.nan for m, r in ranks.items()}
    sorted_modes = sorted(modes, key=lambda m: avg_ranks[m] if not np.isnan(avg_ranks[m]) else 999)

    fig, ax = plt.subplots(figsize=(8, 5))
    y_pos = np.arange(len(sorted_modes))
    vals = [avg_ranks[m] for m in sorted_modes]
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(sorted_modes)))
    ax.barh(y_pos, vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([MODE_LABELS.get(m, m) for m in sorted_modes])
    ax.invert_yaxis()
    ax.set_xlabel("average rank per task (lower is better)")
    ax.set_title("Alignment method ranking across downstream tasks")
    ax.grid(axis="x", alpha=0.3)
    for i, v in enumerate(vals):
        ax.text(v + 0.05, i, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    path = out_dir / "alignment_comparison_ranking.png"
    fig.savefig(path, **FIG_KWARGS)
    plt.close(fig)
    return path


def _format_table(comp_df: pd.DataFrame) -> pd.DataFrame:
    tasks = sorted(comp_df["task"].unique())
    modes = [m for m in MODE_ORDER if m in comp_df["feature_mode"].values]
    rows = []
    for task in tasks:
        row = {"task": task, "metric": _task_metric(task)}
        for mode in modes:
            sub = comp_df[(comp_df["task"] == task) & (comp_df["feature_mode"] == mode)]
            if sub.empty:
                row[mode] = ""
            else:
                mean = float(sub["test_score"].mean())
                std = float(sub["test_score"].std()) if len(sub) > 1 else 0.0
                n = len(sub)
                row[mode] = f"{mean:.4f} ± {std:.4f} (n={n})"
        rows.append(row)
    return pd.DataFrame(rows)


def _df_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Empty table_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        vals = ["" if pd.isna(row[c]) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def generate_report(comp_df: pd.DataFrame, figure_paths: list[Path], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    table = _format_table(comp_df)

    # Ranking.
    tasks = sorted(comp_df["task"].unique())
    modes = sorted(comp_df["feature_mode"].unique())
    rank_rows = []
    for mode in modes:
        avg_ranks = []
        scores = []
        for task in tasks:
            sub = comp_df[(comp_df["task"] == task) & (comp_df["feature_mode"] == mode)]
            if not sub.empty:
                scores.append(float(sub["test_score"].mean()))
        if scores:
            rank_rows.append({
                "feature_mode": mode,
                "mean_test_score": float(np.mean(scores)),
                "std_test_score": float(np.std(scores)),
                "tasks_with_results": len(scores),
            })
    rank_df = pd.DataFrame(rank_rows).sort_values("mean_test_score", ascending=False)

    lines = [
        "# Alignment Method Comparison on Downstream Tasks",
        "",
        f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## 1. Overview",
        "",
        "This report compares the seven frozen-probe feature modes (P0–P6) as different alignment/supervision strategies on the six downstream tasks.",
        "",
        f"- **Tasks**: {', '.join(tasks)}",
        f"- **Feature modes**: {', '.join(modes)}",
        f"- **Successful runs**: {len(comp_df)}",
        "",
        "## 2. Mean Test Score per Task and Alignment Method",
        "",
        f"![Grouped bars](figures/{figure_paths[0].name})",
        "",
        "### Numerical table",
        "",
        _df_to_markdown(table),
        "",
        "## 3. Task × Alignment Heatmap",
        "",
        f"![Heatmap](figures/{figure_paths[1].name})",
        "",
        "## 4. Cross-Task Ranking",
        "",
        "For each task, alignment methods are ranked by mean test score; the plot shows the average rank (lower = better).",
        "",
        f"![Ranking](figures/{figure_paths[2].name})",
        "",
        "### Aggregate ranking table",
        "",
        _df_to_markdown(rank_df),
        "",
        "## 5. Interpretation Notes",
        "",
        "- `vanilla_hidden` (P0) is the non-aligned baseline; all other modes build on top of the Stage 4B head-to-head student.",
        "- `distilled_hidden_random_heads` (P6) is a negative control: it uses mismatched native heads and should underperform the aligned modes if the alignment is meaningful.",
        "- `distilled_hidden_teacher` (P5) is intentionally excluded from the standard array jobs because it requires a separate exact-teacher attention cache. "
        "Consequently it has 0 completed seeds across all tasks and is omitted from the ranking.",
        "- All six downstream tasks now have 3 completed seeds for the six array-included modes (P0–P4, P6) after the rerun of `chemical_reactivity_token` and `ncRNA_family_seqcls`.",
        "",
    ]

    report_path = out_dir / "alignment_comparison.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default="outputs/probes/frozen/frozen_probe_summary.parquet")
    parser.add_argument("--output-dir", default="outputs/probes/frozen/report")
    args = parser.parse_args()

    df = pd.read_parquet(resolve_path(args.summary))
    comp_df = prepare_data(df)

    out_dir = Path(resolve_path(args.output_dir))
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_grouped_bars(comp_df, fig_dir),
        plot_heatmap(comp_df, fig_dir),
        plot_ranking(comp_df, fig_dir),
    ]

    report_path = generate_report(comp_df, paths, out_dir)
    print(f"[alignment-comparison] figures: {fig_dir}")
    print(f"[alignment-comparison] {report_path}")


if __name__ == "__main__":
    main()
