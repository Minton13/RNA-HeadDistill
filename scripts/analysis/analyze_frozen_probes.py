#!/usr/bin/env python3
"""Analyze frozen probe results and generate figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repository root
OUT_DIR = ROOT / "outputs/probes/frozen/analysis"
FIG_DIR = ROOT / "outputs/probes/frozen/figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

TASKS = [
    "splice_donor_seqcls",
    "splice_acceptor_seqcls",
    "rbp_binding_seqcls",
    "ncRNA_family_seqcls",
    "chemical_reactivity_token",
    "secondary_structure_pair",
]

MODES = [
    "vanilla_hidden",
    "distilled_hidden",
    "distilled_hidden_native",
    "distilled_aligned",
    "distilled_hidden_aligned",
    "distilled_hidden_random_heads",
]

SEEDS = [17, 41, 89]

MODE_LABELS = {
    "vanilla_hidden": "P0 vanilla hidden",
    "distilled_hidden": "P1 distilled hidden",
    "distilled_hidden_native": "P2 + native heads",
    "distilled_aligned": "P3 aligned only",
    "distilled_hidden_aligned": "P4 hidden + aligned",
    "distilled_hidden_random_heads": "P6 + random heads",
}

# Primary metric per task type
TASK_PRIMARY_METRIC = {
    "splice_donor_seqcls": "auroc",
    "splice_acceptor_seqcls": "auroc",
    "rbp_binding_seqcls": "auroc",
    "ncRNA_family_seqcls": "auroc",
    "chemical_reactivity_token": "pearson",
    "secondary_structure_pair": "f1",
}


def load_results() -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for mode in MODES:
            for seed in SEEDS:
                p = ROOT / f"outputs/probes/frozen/{task}/{mode}/seed_{seed}/eval_test.json"
                if not p.exists():
                    continue
                data = json.loads(p.read_text())
                row = {
                    "task": task,
                    "mode": mode,
                    "seed": seed,
                    "primary_metric": TASK_PRIMARY_METRIC[task],
                }
                row.update(data)
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    primary = TASK_PRIMARY_METRIC
    rows = []
    for task in TASKS:
        metric = primary[task]
        for mode in MODES:
            sub = df[(df["task"] == task) & (df["mode"] == mode)]
            if sub.empty:
                continue
            vals = sub[metric].dropna()
            rows.append({
                "task": task,
                "mode": mode,
                "metric": metric,
                "mean": vals.mean(),
                "std": vals.std(),
                "n_seeds": len(vals),
                "seeds": vals.tolist(),
            })
    return pd.DataFrame(rows)


def plot_per_task(summary: pd.DataFrame) -> None:
    for task in TASKS:
        metric = TASK_PRIMARY_METRIC[task]
        sub = summary[summary["task"] == task].copy()
        sub["mode_label"] = sub["mode"].map(MODE_LABELS)
        sub = sub.sort_values("mean", ascending=True)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        y_pos = np.arange(len(sub))
        ax.barh(y_pos, sub["mean"], xerr=sub["std"], capsize=4, alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sub["mode_label"])
        ax.set_xlabel(metric.upper())
        ax.set_title(f"{task} - frozen probe test {metric}")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"frozen_probe_{task}.png", dpi=200)
        plt.close(fig)


def plot_heatmap(summary: pd.DataFrame) -> None:
    # Pivot: rows = task, cols = mode
    pivot = summary.pivot(index="task", columns="mode", values="mean")
    pivot = pivot.reindex(index=TASKS, columns=MODES)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(len(MODES)))
    ax.set_xticklabels([MODE_LABELS[m] for m in MODES], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(TASKS)))
    ax.set_yticklabels(TASKS)
    for i in range(len(TASKS)):
        for j in range(len(MODES)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8, color="white")
    ax.set_title("Frozen probe primary metric (test, mean over 3 seeds)")
    fig.colorbar(im, ax=ax, label="metric")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "frozen_probe_heatmap.png", dpi=200)
    plt.close(fig)


def plot_improvement_over_vanilla(summary: pd.DataFrame) -> None:
    rows = []
    for task in TASKS:
        metric = TASK_PRIMARY_METRIC[task]
        vanilla = summary[(summary["task"] == task) & (summary["mode"] == "vanilla_hidden")]["mean"]
        if vanilla.empty:
            continue
        baseline = vanilla.iloc[0]
        for mode in MODES[1:]:
            sub = summary[(summary["task"] == task) & (summary["mode"] == mode)]
            if sub.empty:
                continue
            rows.append({
                "task": task,
                "mode": mode,
                "metric": metric,
                "delta": sub["mean"].iloc[0] - baseline,
                "delta_pct": (sub["mean"].iloc[0] - baseline) / abs(baseline) * 100 if baseline != 0 else np.nan,
            })
    delta_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TASKS))
    width = 0.15
    for i, mode in enumerate(MODES[1:]):
        sub = delta_df[delta_df["mode"] == mode]
        vals = [sub[sub["task"] == t]["delta"].iloc[0] if not sub[sub["task"] == t].empty else 0 for t in TASKS]
        ax.bar(x + i * width, vals, width, label=MODE_LABELS[mode], alpha=0.85)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(TASKS, rotation=20, ha="right")
    ax.set_ylabel("Δ primary metric vs vanilla")
    ax.set_title("Improvement over P0 vanilla hidden")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "frozen_probe_improvement.png", dpi=200)
    plt.close(fig)


def main() -> None:
    df = load_results()
    print(f"Loaded {len(df)} result rows")

    # Save raw results
    df.to_parquet(OUT_DIR / "frozen_probe_all_results.parquet", index=False)
    df.to_csv(OUT_DIR / "frozen_probe_all_results.csv", index=False)

    summary = summarize(df)
    summary.to_parquet(OUT_DIR / "frozen_probe_summary.parquet", index=False)
    summary.to_csv(OUT_DIR / "frozen_probe_summary.csv", index=False)

    # Print summary table
    print("\n=== Frozen probe summary (test, mean ± std over 3 seeds) ===")
    for task in TASKS:
        metric = TASK_PRIMARY_METRIC[task]
        print(f"\n{task} ({metric}):")
        sub = summary[summary["task"] == task].sort_values("mean", ascending=False)
        for _, row in sub.iterrows():
            print(f"  {MODE_LABELS[row['mode']]:30s} {row['mean']:.4f} ± {row['std']:.4f}")

    # Generate plots
    plot_per_task(summary)
    plot_heatmap(summary)
    plot_improvement_over_vanilla(summary)

    # Write markdown report
    report_lines = [
        "# Frozen Probe Analysis Report",
        "",
        f"Total results: {len(df)} (6 tasks × 6 modes × 3 seeds)",
        "",
        "## Summary by task (test set, mean ± std over seeds 17/41/89)",
        "",
    ]
    for task in TASKS:
        metric = TASK_PRIMARY_METRIC[task]
        report_lines.append(f"### {task} ({metric})")
        report_lines.append("")
        report_lines.append("| Mode | Mean | Std |")
        report_lines.append("|------|------|-----|")
        sub = summary[summary["task"] == task].sort_values("mean", ascending=False)
        for _, row in sub.iterrows():
            report_lines.append(f"| {MODE_LABELS[row['mode']]} | {row['mean']:.4f} | {row['std']:.4f} |")
        report_lines.append("")

    # Key comparisons
    report_lines.extend([
        "## Key comparisons",
        "",
        "### P4 (hidden+aligned) vs P1 (distilled hidden only)",
        "",
    ])
    for task in TASKS:
        metric = TASK_PRIMARY_METRIC[task]
        p1 = summary[(summary["task"] == task) & (summary["mode"] == "distilled_hidden")]["mean"]
        p4 = summary[(summary["task"] == task) & (summary["mode"] == "distilled_hidden_aligned")]["mean"]
        if not p1.empty and not p4.empty:
            delta = p4.iloc[0] - p1.iloc[0]
            report_lines.append(f"- {task}: P4={p4.iloc[0]:.4f}, P1={p1.iloc[0]:.4f}, Δ={delta:+.4f}")

    report_lines.extend([
        "",
        "### P4 vs P6 (random heads negative control)",
        "",
    ])
    for task in TASKS:
        metric = TASK_PRIMARY_METRIC[task]
        p4 = summary[(summary["task"] == task) & (summary["mode"] == "distilled_hidden_aligned")]["mean"]
        p6 = summary[(summary["task"] == task) & (summary["mode"] == "distilled_hidden_random_heads")]["mean"]
        if not p4.empty and not p6.empty:
            delta = p4.iloc[0] - p6.iloc[0]
            report_lines.append(f"- {task}: P4={p4.iloc[0]:.4f}, P6={p6.iloc[0]:.4f}, Δ={delta:+.4f}")

    report_path = OUT_DIR / "frozen_probe_analysis_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nReport written to {report_path}")
    print(f"Figures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
