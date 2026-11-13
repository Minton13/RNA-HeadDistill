#!/usr/bin/env python3
"""Phase 1.1: metric-aligned re-analysis of frozen probe results.

Uses each task's ACTUAL selection metric recorded in training_summary.json:
- sequence classification tasks -> auroc
- chemical_reactivity_token    -> pearson
- secondary_structure_pair     -> top_l_precision
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]  # repository root
OUT_DIR = ROOT / "outputs/probes/analysis_v2"
FIG_DIR = OUT_DIR / "figures"
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

TASK_SHORT = {
    "splice_donor_seqcls": "splice donor",
    "splice_acceptor_seqcls": "splice acceptor",
    "rbp_binding_seqcls": "RBP binding",
    "ncRNA_family_seqcls": "ncRNA family",
    "chemical_reactivity_token": "chemical reactivity",
    "secondary_structure_pair": "secondary structure",
}
MODE_LABELS = {
    "vanilla_hidden": "P0 vanilla",
    "distilled_hidden": "P1 distilled",
    "distilled_hidden_native": "P2 native",
    "distilled_aligned": "P3 aligned",
    "distilled_hidden_aligned": "P4 hidden+aligned",
    "distilled_hidden_random_heads": "P6 random",
}

# Metric-aligned primary metric per task (matches training objective).
PRIMARY = {
    "splice_donor_seqcls": "auroc",
    "splice_acceptor_seqcls": "auroc",
    "rbp_binding_seqcls": "auroc",
    "ncRNA_family_seqcls": "auroc",
    "chemical_reactivity_token": "pearson",
    "secondary_structure_pair": "top_l_precision",
}


def load() -> pd.DataFrame:
    rows = []
    for task in TASKS:
        for mode in MODES:
            for seed in SEEDS:
                d = ROOT / f"outputs/probes/frozen/{task}/{mode}/seed_{seed}"
                et = json.loads((d / "eval_test.json").read_text())
                ev = json.loads((d / "eval_val.json").read_text())
                ts = json.loads((d / "training_summary.json").read_text())
                row = {
                    "task": task,
                    "mode": mode,
                    "seed": seed,
                    "selection_metric": ts["metric_name"],
                    "best_val_metric": ts["best_val_metric"],
                }
                for k, v in et.items():
                    row[f"test_{k}"] = v
                for k, v in ev.items():
                    row[f"val_{k}"] = v
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    df = load()
    df.to_parquet(OUT_DIR / "frozen_probe_all_metric_aligned.parquet", index=False)

    # Consistency check: selection metric uniform per task
    for task in TASKS:
        ms = df[df["task"] == task]["selection_metric"].unique()
        assert len(ms) == 1 and ms[0] == PRIMARY[task], f"{task}: {ms}"
    print("Selection-metric consistency check passed for all 108 cells.")

    # Summary
    rows = []
    for task in TASKS:
        metric = PRIMARY[task]
        for mode in MODES:
            sub = df[(df["task"] == task) & (df["mode"] == mode)]
            vals = sub[f"test_{metric}"].dropna()
            rows.append({
                "task": task, "mode": mode, "metric": metric,
                "mean": vals.mean(), "std": vals.std(), "n": len(vals),
            })
    summary = pd.DataFrame(rows)
    summary.to_parquet(OUT_DIR / "frozen_probe_summary_metric_aligned.parquet", index=False)

    print("\n=== Metric-aligned frozen probe summary (test, mean ± std, 3 seeds) ===")
    for task in TASKS:
        metric = PRIMARY[task]
        print(f"\n{task} ({metric}):")
        sub = summary[summary["task"] == task].sort_values("mean", ascending=False)
        for _, r in sub.iterrows():
            print(f"  {MODE_LABELS[r['mode']]:22s} {r['mean']:.4f} ± {r['std']:.4f}")

    # Figure: 2x3 grid, metric-aligned
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for idx, task in enumerate(TASKS):
        ax = axes[idx]
        metric = PRIMARY[task]
        sub = summary[summary["task"] == task]
        means = [sub[sub["mode"] == m]["mean"].iloc[0] for m in MODES]
        stds = [sub[sub["mode"] == m]["std"].iloc[0] for m in MODES]
        labels = [MODE_LABELS[m] for m in MODES]
        x = np.arange(len(labels))
        colors = ["gray" if m == "vanilla_hidden" else ("crimson" if m == "distilled_hidden_aligned" else "steelblue") for m in MODES]
        ax.bar(x, means, yerr=stds, capsize=3, alpha=0.85, color=colors)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(metric)
        ax.set_title(TASK_SHORT[task])
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Frozen probe test performance — metric aligned with training objective (3 seeds)", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "frozen_probe_metric_aligned.png", dpi=200)
    plt.close(fig)

    # Heatmap of P4-P0 delta
    deltas = []
    for task in TASKS:
        metric = PRIMARY[task]
        p0 = summary[(summary["task"] == task) & (summary["mode"] == "vanilla_hidden")]["mean"].iloc[0]
        p4 = summary[(summary["task"] == task) & (summary["mode"] == "distilled_hidden_aligned")]["mean"].iloc[0]
        p4s = summary[(summary["task"] == task) & (summary["mode"] == "distilled_hidden_aligned")]["std"].iloc[0]
        best = summary[summary["task"] == task]["mean"].max()
        deltas.append({
            "task": task, "metric": metric, "P0": p0, "P4": p4, "P4_std": p4s,
            "delta": p4 - p0, "P4_is_best": abs(p4 - best) < 1e-9,
        })
    delta_df = pd.DataFrame(deltas)
    delta_df.to_csv(OUT_DIR / "p4_vs_p0_delta.csv", index=False)
    print("\n=== P4 vs P0 (metric-aligned) ===")
    print(delta_df.to_string(index=False))

    # Markdown report
    lines = ["# Metric-aligned Frozen Probe Summary (Phase 1.1)", ""]
    lines.append("Primary metric per task = the metric used for model selection during training")
    lines.append("(verified uniform across all 6 modes and 3 seeds per task).")
    lines.append("")
    for task in TASKS:
        metric = PRIMARY[task]
        lines.append(f"## {TASK_SHORT[task]} ({metric})")
        lines.append("")
        lines.append("| Mode | Test mean | Test std |")
        lines.append("|------|-----------|----------|")
        sub = summary[summary["task"] == task].sort_values("mean", ascending=False)
        for _, r in sub.iterrows():
            lines.append(f"| {MODE_LABELS[r['mode']]} | {r['mean']:.4f} | {r['std']:.4f} |")
        lines.append("")
    lines.append("## P4 vs P0 delta")
    lines.append("")
    lines.append("| Task | Metric | P0 | P4 | Δ | P4 best? |")
    lines.append("|------|--------|----|----|---|----------|")
    for _, r in delta_df.iterrows():
        lines.append(f"| {TASK_SHORT[r['task']]} | {r['metric']} | {r['P0']:.4f} | {r['P4']:.4f} | {r['delta']:+.4f} | {'YES' if r['P4_is_best'] else 'no'} |")
    (OUT_DIR / "metric_aligned_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nWrote {OUT_DIR / 'metric_aligned_report.md'}")


if __name__ == "__main__":
    main()
