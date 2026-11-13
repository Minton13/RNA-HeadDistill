#!/usr/bin/env python3
"""Comprehensive analysis of frozen probe and multitask results."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repository root
OUT_DIR = ROOT / "outputs/probes/analysis"
FIG_DIR = ROOT / "outputs/probes/figures"
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

TASK_SHORT = {
    "splice_donor_seqcls": "splice donor",
    "splice_acceptor_seqcls": "splice acceptor",
    "rbp_binding_seqcls": "RBP binding",
    "ncRNA_family_seqcls": "ncRNA family",
    "chemical_reactivity_token": "chemical reactivity",
    "secondary_structure_pair": "secondary structure",
}

FROZEN_MODES = [
    "vanilla_hidden",
    "distilled_hidden",
    "distilled_hidden_native",
    "distilled_aligned",
    "distilled_hidden_aligned",
    "distilled_hidden_random_heads",
]

MODE_LABELS = {
    "vanilla_hidden": "P0 vanilla",
    "distilled_hidden": "P1 distilled",
    "distilled_hidden_native": "P2 native",
    "distilled_aligned": "P3 aligned",
    "distilled_hidden_aligned": "P4 hidden+aligned",
    "distilled_hidden_random_heads": "P6 random",
}

TASK_PRIMARY_METRIC = {
    "splice_donor_seqcls": "auroc",
    "splice_acceptor_seqcls": "auroc",
    "rbp_binding_seqcls": "auroc",
    "ncRNA_family_seqcls": "auroc",
    "chemical_reactivity_token": "pearson",
    "secondary_structure_pair": "f1",
}


def load_frozen() -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for mode in FROZEN_MODES:
            for seed in [17, 41, 89]:
                p = ROOT / f"outputs/probes/frozen/{task}/{mode}/seed_{seed}/eval_test.json"
                if not p.exists():
                    continue
                data = json.loads(p.read_text())
                row = {"task": task, "mode": mode, "seed": seed, "experiment": "frozen"}
                row.update(data)
                rows.append(row)
    return pd.DataFrame(rows)


def load_multitask() -> pd.DataFrame:
    rows = []
    for mode in ["joint_no_replay", "joint_with_replay"]:
        for seed in [17, 41, 89]:
            for task in TASKS:
                p = ROOT / f"outputs/probes/multitask/{mode}/seed_{seed}/{task}/eval_test.json"
                if not p.exists():
                    continue
                data = json.loads(p.read_text())
                row = {"task": task, "mode": mode, "seed": seed, "experiment": "multitask"}
                row.update(data)
                rows.append(row)
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    metric_cols = ["auroc", "auprc", "f1", "mcc", "pearson", "spearman", "mae", "rmse"]
    metric_cols = [c for c in metric_cols if c in df.columns]
    rows = []
    for name, group in df.groupby(group_cols):
        row = dict(zip(group_cols, name if isinstance(name, tuple) else [name]))
        for col in metric_cols:
            vals = group[col].dropna()
            row[f"{col}_mean"] = vals.mean() if len(vals) else np.nan
            row[f"{col}_std"] = vals.std() if len(vals) else np.nan
            row[f"{col}_n"] = len(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def plot_frozen_comparison(frozen_df: pd.DataFrame) -> None:
    """Bar plot: P0-P6 for each task, primary metric."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        metric = TASK_PRIMARY_METRIC[task]
        sub = frozen_df[frozen_df["task"] == task]
        means = []
        stds = []
        labels = []
        for mode in FROZEN_MODES:
            vals = sub[sub["mode"] == mode][metric].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std() if len(vals) else np.nan)
            labels.append(MODE_LABELS[mode])

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=3, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric.upper())
        ax.set_title(TASK_SHORT[task])
        ax.grid(axis="y", alpha=0.3)
        ax.axhline(0.5 if metric == "auroc" else 0, color="red", linestyle="--", alpha=0.5, linewidth=0.8)

    fig.suptitle("Frozen probe test performance (mean ± std, 3 seeds)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "frozen_probe_all_tasks.png", dpi=200)
    plt.close(fig)


def plot_multitask_comparison(mt_df: pd.DataFrame) -> None:
    """Bar plot: joint_no_replay vs joint_with_replay for each task."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        metric = TASK_PRIMARY_METRIC[task]
        sub = mt_df[mt_df["task"] == task]

        modes = ["joint_no_replay", "joint_with_replay"]
        means = []
        stds = []
        for mode in modes:
            vals = sub[sub["mode"] == mode][metric].dropna()
            means.append(vals.mean() if len(vals) else np.nan)
            stds.append(vals.std() if len(vals) else np.nan)

        x = np.arange(len(modes))
        ax.bar(x, means, yerr=stds, capsize=4, alpha=0.8, color=["steelblue", "coral"])
        ax.set_xticks(x)
        ax.set_xticklabels(["no replay", "with replay"])
        ax.set_ylabel(metric.upper())
        ax.set_title(TASK_SHORT[task])
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Multitask test performance (mean ± std, 3 seeds)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "multitask_all_tasks.png", dpi=200)
    plt.close(fig)


def plot_frozen_vs_multitask(frozen_df: pd.DataFrame, mt_df: pd.DataFrame) -> None:
    """Compare best frozen probe (P4) vs multitask."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        metric = TASK_PRIMARY_METRIC[task]

        # Frozen P4
        fsub = frozen_df[(frozen_df["task"] == task) & (frozen_df["mode"] == "distilled_hidden_aligned")]
        f_vals = fsub[metric].dropna()

        # Frozen P0
        f0sub = frozen_df[(frozen_df["task"] == task) & (frozen_df["mode"] == "vanilla_hidden")]
        f0_vals = f0sub[metric].dropna()

        # Multitask
        msub = mt_df[(mt_df["task"] == task) & (mt_df["mode"] == "joint_with_replay")]
        m_vals = msub[metric].dropna()

        labels = ["P0 vanilla", "P4 frozen", "multitask"]
        means = [f0_vals.mean() if len(f0_vals) else np.nan,
                 f_vals.mean() if len(f_vals) else np.nan,
                 m_vals.mean() if len(m_vals) else np.nan]
        stds = [f0_vals.std() if len(f0_vals) else np.nan,
                f_vals.std() if len(f_vals) else np.nan,
                m_vals.std() if len(m_vals) else np.nan]

        x = np.arange(len(labels))
        ax.bar(x, means, yerr=stds, capsize=4, alpha=0.8, color=["gray", "steelblue", "coral"])
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15, ha="right")
        ax.set_ylabel(metric.upper())
        ax.set_title(TASK_SHORT[task])
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Frozen P4 vs multitask (test, mean ± std)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "frozen_vs_multitask.png", dpi=200)
    plt.close(fig)


def plot_seed_variance(frozen_df: pd.DataFrame) -> None:
    """Show per-seed variance for each task/mode."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        metric = TASK_PRIMARY_METRIC[task]
        sub = frozen_df[frozen_df["task"] == task]

        for mode in FROZEN_MODES:
            vals = sub[sub["mode"] == mode][metric].dropna()
            if len(vals):
                ax.scatter([MODE_LABELS[mode]] * len(vals), vals, alpha=0.7, s=50)

        ax.set_ylabel(metric.upper())
        ax.set_title(TASK_SHORT[task])
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Per-seed variance (frozen probe test)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "frozen_probe_seed_variance.png", dpi=200)
    plt.close(fig)


def main() -> None:
    frozen_df = load_frozen()
    mt_df = load_multitask()

    print(f"Frozen probe: {len(frozen_df)} rows")
    print(f"Multitask: {len(mt_df)} rows")

    # Save combined data
    frozen_df.to_parquet(OUT_DIR / "frozen_probe_all.parquet", index=False)
    mt_df.to_parquet(OUT_DIR / "multitask_all.parquet", index=False)

    # Summaries
    frozen_summary = summarize(frozen_df, ["task", "mode"])
    mt_summary = summarize(mt_df, ["task", "mode"])
    frozen_summary.to_parquet(OUT_DIR / "frozen_probe_summary.parquet", index=False)
    mt_summary.to_parquet(OUT_DIR / "multitask_summary.parquet", index=False)

    # Plots
    plot_frozen_comparison(frozen_df)
    plot_multitask_comparison(mt_df)
    plot_frozen_vs_multitask(frozen_df, mt_df)
    plot_seed_variance(frozen_df)

    # Report
    report = []
    report.append("# Probe Analysis Report")
    report.append("")
    report.append("## Frozen probe results (test set, mean ± std over 3 seeds)")
    report.append("")
    for task in TASKS:
        metric = TASK_PRIMARY_METRIC[task]
        report.append(f"### {TASK_SHORT[task]} ({metric})")
        report.append("")
        report.append("| Mode | Mean | Std |")
        report.append("|------|------|-----|")
        sub = frozen_summary[frozen_summary["task"] == task].copy()
        sub["mean"] = sub[f"{metric}_mean"]
        sub["std"] = sub[f"{metric}_std"]
        sub = sub.sort_values("mean", ascending=False)
        for _, row in sub.iterrows():
            report.append(f"| {MODE_LABELS[row['mode']]} | {row['mean']:.4f} | {row['std']:.4f} |")
        report.append("")

    report.append("## Multitask results (test set, mean ± std over 3 seeds)")
    report.append("")
    for task in TASKS:
        metric = TASK_PRIMARY_METRIC[task]
        report.append(f"### {TASK_SHORT[task]} ({metric})")
        report.append("")
        report.append("| Mode | Mean | Std |")
        report.append("|------|------|-----|")
        sub = mt_summary[mt_summary["task"] == task].copy()
        sub["mean"] = sub[f"{metric}_mean"]
        sub["std"] = sub[f"{metric}_std"]
        for _, row in sub.iterrows():
            report.append(f"| {row['mode']} | {row['mean']:.4f} | {row['std']:.4f} |")
        report.append("")

    # Key findings
    report.append("## Key findings")
    report.append("")
    report.append("### Frozen probe")
    report.append("")
    for task in TASKS:
        metric = TASK_PRIMARY_METRIC[task]
        p0 = frozen_summary[(frozen_summary["task"] == task) & (frozen_summary["mode"] == "vanilla_hidden")][f"{metric}_mean"]
        p4 = frozen_summary[(frozen_summary["task"] == task) & (frozen_summary["mode"] == "distilled_hidden_aligned")][f"{metric}_mean"]
        if not p0.empty and not p4.empty:
            delta = p4.iloc[0] - p0.iloc[0]
            report.append(f"- **{TASK_SHORT[task]}**: P4 vs P0 Δ = {delta:+.4f}")

    report_path = OUT_DIR / "probe_analysis_report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    print(f"\nReport: {report_path}")
    print(f"Figures: {FIG_DIR}")


if __name__ == "__main__":
    main()
