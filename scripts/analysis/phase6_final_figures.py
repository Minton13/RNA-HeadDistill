"""Phase 6: final publication-style figures for the P4/P4g/A8 story.

Inputs (all under outputs/probes/analysis_v2/):
  bootstrap/point_estimates.csv          P0..P6 per (task, mode, seed), test
  bootstrap/p4_vs_baselines_summary.csv  A3-line paired-bootstrap deltas
  bootstrap_a8/a8_point_estimates.csv    A8-P4g per (task, seed), test
  bootstrap_a8/a8_vs_baselines_summary.csv
  hidden_drift/hidden_drift.csv
  gates/residual_gates.csv

Outputs: outputs/probes/analysis_v2/figures/final/
One plot per file, matplotlib only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("outputs/probes/analysis_v2")
OUT = ROOT / "figures" / "final"
OUT.mkdir(parents=True, exist_ok=True)

METRIC = {
    "splice_donor_seqcls": ("auroc", "AUROC"),
    "splice_acceptor_seqcls": ("auroc", "AUROC"),
    "rbp_binding_seqcls": ("auroc", "AUROC"),
    "ncRNA_family_seqcls": ("auroc", "AUROC"),
    "chemical_reactivity_token": ("pearson", "Pearson"),
    "secondary_structure_pair": ("top_l_precision", "Top-L precision"),
}
TASK_LABEL = {
    "splice_donor_seqcls": "Splice donor\n(L6)",
    "splice_acceptor_seqcls": "Splice acceptor\n(L6)",
    "rbp_binding_seqcls": "RBP binding\n(L6)",
    "ncRNA_family_seqcls": "ncRNA family\n(L6)",
    "chemical_reactivity_token": "Chemical react.\n(L8)",
    "secondary_structure_pair": "Secondary struct.\n(L12)",
}
TASKS = list(METRIC)
# Methods shown in the main figure, in order.
METHODS = ["P0", "P4", "P4g", "A8_P4g"]
METHOD_LABEL = {"P0": "P0 vanilla", "P4": "P4 (H+aligned)", "P4g": "P4g gated", "A8_P4g": "A8-P4g frozen"}


def _load_points() -> pd.DataFrame:
    a3 = pd.read_csv(ROOT / "bootstrap" / "point_estimates.csv")
    a3 = a3.rename(columns={"mode": "method"})
    a8 = pd.read_csv(ROOT / "bootstrap_a8" / "a8_point_estimates.csv")
    a8 = a8.rename(columns={"line": "method"})
    a8 = a8[a8["method"] == "A8_P4g"]
    cols = ["task", "method", "seed"] + list(dict.fromkeys(m for m, _ in METRIC.values()))
    df = pd.concat([a3[[c for c in cols if c in a3.columns]],
                    a8[[c for c in cols if c in a8.columns]]], ignore_index=True)
    return df


def _load_deltas() -> pd.DataFrame:
    a3 = pd.read_csv(ROOT / "bootstrap" / "p4_vs_baselines_summary.csv")
    a3 = a3[a3["comparison"].isin(["P4-P0", "P4g-P0"])].copy()
    a3["method"] = a3["comparison"].str.split("-").str[0]
    a8 = pd.read_csv(ROOT / "bootstrap_a8" / "a8_vs_baselines_summary.csv")
    a8 = a8[a8["comparison"] == "A8_P4g-P0"].copy()
    a8["method"] = "A8_P4g"
    cols = ["task", "method", "metric", "mean", "ci_lo", "ci_hi", "p_pos"]
    return pd.concat([a3[cols], a8[cols]], ignore_index=True)


def _sig_star(p_pos: float) -> str:
    if p_pos >= 0.999 or p_pos <= 0.001:
        return "***"
    if p_pos >= 0.99 or p_pos <= 0.01:
        return "**"
    if p_pos >= 0.975 or p_pos <= 0.025:
        return "*"
    return "ns"


def fig1_main_bars(points: pd.DataFrame, deltas: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharey=False)
    for ax, task in zip(axes.flat, TASKS):
        mcol, mlabel = METRIC[task]
        x = np.arange(len(METHODS))
        for i, meth in enumerate(METHODS):
            sub = points[(points.task == task) & (points.method == meth)][mcol].dropna()
            if len(sub) == 0:
                continue
            mean, std = sub.mean(), sub.std()
            ax.bar(i, mean, yerr=std, capsize=3, color=f"C{METHODS.index(meth)}",
                   alpha=0.85, width=0.7)
            ax.scatter(np.full(len(sub), i), sub, color="black", s=10, zorder=5)
            d = deltas[(deltas.task == task) & (deltas.method == meth)]
            if meth != "P0" and len(d):
                star = _sig_star(float(d.iloc[0]["p_pos"]))
                if star != "ns":
                    star_color = "green" if float(d.iloc[0]["mean"]) > 0 else "red"
                    ax.text(i, mean + std, star, ha="center", va="bottom",
                            fontsize=9, color=star_color)
        p0_mean = points[(points.task == task) & (points.method == "P0")][mcol].mean()
        ax.axhline(p0_mean, color="C0", ls="--", lw=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABEL[m] for m in METHODS], fontsize=8, rotation=20)
        ax.set_title(TASK_LABEL[task], fontsize=10)
        ax.set_ylabel(mlabel, fontsize=9)
        lo = points[points.task == task][mcol].min()
        hi = points[points.task == task][mcol].max()
        pad = max((hi - lo) * 0.25, 1e-3)
        ax.set_ylim(lo - pad, hi + pad * 1.6)
    fig.suptitle("Frozen-probe test performance vs vanilla backbone (mean ± sd, 3 seeds; "
                 "stars: paired bootstrap vs P0)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / "fig1_main_results.png", dpi=200)
    fig.savefig(OUT / "fig1_main_results.pdf")
    plt.close(fig)


def fig2_delta_forest(deltas: pd.DataFrame) -> None:
    order = [(t, m) for t in TASKS for m in ("P4", "P4g", "A8_P4g")]
    fig, ax = plt.subplots(figsize=(8, 7))
    ylabels, y = [], 0
    yticks, yticklabels = [], []
    for task in TASKS:
        for meth in ("P4", "P4g", "A8_P4g"):
            row = deltas[(deltas.task == task) & (deltas.method == meth)]
            if not len(row):
                continue
            r = row.iloc[0]
            color = f"C{METHODS.index(meth)}"
            ax.errorbar(r["mean"], y,
                        xerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                        fmt="o", color=color, capsize=3, ms=5)
            star = _sig_star(float(r["p_pos"]))
            ax.text(r["ci_hi"], y + 0.25, star, fontsize=8, ha="left")
            yticks.append(y)
            yticklabels.append(f"{TASK_LABEL[task].replace(chr(10), ' ')}  {METHOD_LABEL[meth]}")
            y += 1
        y += 0.8  # gap between tasks
    ax.axvline(0, color="gray", ls="--", lw=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=8)
    ax.set_xlabel("Δ primary metric vs P0 (paired bootstrap, 95% CI)")
    ax.set_title("Distilled feature variants minus vanilla backbone")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_delta_forest.png", dpi=200)
    fig.savefig(OUT / "fig2_delta_forest.pdf")
    plt.close(fig)


def fig3_acceptor_repair(points: pd.DataFrame) -> None:
    task = "splice_acceptor_seqcls"
    chain = ["P0", "P1", "P4", "P4g", "A8_P4g"]
    labels = ["P0\nvanilla", "P1\ndistilled hidden", "P4\nH+aligned", "P4g\ngated", "A8-P4g\nfrozen backbone"]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, meth in enumerate(chain):
        sub = points[(points.task == task) & (points.method == meth)]["auroc"].dropna()
        mean, std = sub.mean(), sub.std()
        ax.bar(i, mean, yerr=std, capsize=4, color=f"C{i}", alpha=0.85, width=0.65)
        ax.scatter(np.full(len(sub), i), sub, color="black", s=12, zorder=5)
        ax.text(i, mean + std + 0.004, f"{mean:.3f}", ha="center", fontsize=9)
    ax.axhline(points[(points.task == task) & (points.method == "P0")]["auroc"].mean(),
               color="C0", ls="--", lw=1)
    ax.set_xticks(range(len(chain)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0.53, 0.80)
    ax.set_title("Splice acceptor: distillation damage and repair chain")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_acceptor_repair.png", dpi=200)
    fig.savefig(OUT / "fig3_acceptor_repair.pdf")
    plt.close(fig)


def fig4_hidden_drift() -> None:
    df = pd.read_csv(ROOT / "hidden_drift" / "hidden_drift.csv")
    # Expect columns like route/layer, cosine, cka, rel_mse — be tolerant.
    cols = {c.lower(): c for c in df.columns}
    route_col = next((c for c in df.columns if "route" in c.lower() or "layer" in c.lower()), df.columns[0])
    routes = df[route_col].astype(str).tolist()
    x = np.arange(len(routes))
    width = 0.35
    fig, ax = plt.subplots(figsize=(6.5, 4))
    cos_col = next(c for c in df.columns if "cos" in c.lower())
    cka_col = next((c for c in df.columns if "cka" in c.lower()), None)
    ax.bar(x - width / 2, df[cos_col], width, label="cosine", color="C0", alpha=0.85)
    if cka_col:
        ax.bar(x + width / 2, df[cka_col], width, label="CKA", color="C1", alpha=0.85)
    ax.axhline(1.0, color="gray", ls="--", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(routes)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("similarity to vanilla hidden state")
    ax.set_title("Hidden-state drift after Stage 4B distillation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "fig4_hidden_drift.png", dpi=200)
    plt.close(fig)


def fig5_residual_gates() -> None:
    df = pd.read_csv(ROOT / "gates" / "residual_gates.csv")
    g = df.groupby("task")["residual_gate"].agg(["mean", "std"]).reindex(TASKS)
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(g))
    ax.bar(x, g["mean"], yerr=g["std"], capsize=4, color="C2", alpha=0.85)
    ax.axhline(0, color="gray", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([TASK_LABEL[t] for t in TASKS], fontsize=8)
    ax.set_ylabel("learned residual gate value")
    ax.set_title("P4g residual gates stay near zero (mean ± sd, 3 seeds)")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_residual_gates.png", dpi=200)
    plt.close(fig)


def fig6_length_bins() -> None:
    f = ROOT / "length_bins" / "p4_vs_p0_by_bin.csv"
    if not f.exists():
        return
    df = pd.read_csv(f)
    df = df[df["comparison"].isin(["P4-P0", "P4g-P0"])]
    if not len(df):
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    bins_order = ["L<=128", "129-256", "257-512"]
    plotted = set()
    for j, comp in enumerate(["P4-P0", "P4g-P0"]):
        sub = df[df.comparison == comp]
        for task in TASKS:
            t = sub[sub.task == task]
            if len(t) < 2:  # skip tasks with a single occupied bin
                continue
            t = t.set_index("bin").reindex(bins_order).dropna()
            label = f"{TASK_LABEL[task].splitlines()[0]} {comp.split('-')[0]}"
            ax.plot(range(len(t)), t["mean"], marker="o", ms=5, lw=1.2,
                    color=f"C{TASKS.index(task)}",
                    ls="-" if comp == "P4-P0" else "--", label=label)
            plotted.add(label)
    ax.axhline(0, color="gray", ls=":", lw=1)
    ax.set_xticks(range(len(bins_order)))
    ax.set_xticklabels(bins_order)
    ax.set_xlabel("sequence length bin")
    ax.set_ylabel("Δ vs P0")
    ax.set_title("Length robustness of P4 / P4g gains (tasks with ≥2 bins)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "fig6_length_bins.png", dpi=200)
    plt.close(fig)


def main() -> None:
    points = _load_points()
    deltas = _load_deltas()
    fig1_main_bars(points, deltas)
    fig2_delta_forest(deltas)
    fig3_acceptor_repair(points)
    fig4_hidden_drift()
    fig5_residual_gates()
    fig6_length_bins()
    print(f"[figures] wrote {OUT}")


if __name__ == "__main__":
    main()
