#!/usr/bin/env python
# Provenance: ported from paper/scripts/plot_fig4_downstream.py (RNA_LLM_v2
# monorepo) on 2026-11-13 for the public release.
# Changes vs. original: hard-coded cluster paths replaced by
# --data-root (metrics/bootstrap/*, metrics/downstream/*) and --outdir.
"""Figure 4 — merged downstream-task figure (vector PDF).

  a  frozen-probe test performance vs vanilla backbone, 6 tasks (2x3 grid)
  b  paired-bootstrap delta forest vs P0 for P4 / P4g / A8-P4g / P5
  c  splice-acceptor damage-and-repair chain (P0 -> P1 -> P4 -> P4g -> A8)

Star rule (paired bootstrap, 3 seeds): p_pos >= 0.999 or <= 0.001 -> ***.
Green = significant gain over P0, red = significant loss, gray ns otherwise.

Run: python scripts/reproduction/plot_fig4_downstream.py \
       --data-root /path/to/dataset-rna-head-distill --outdir ./reproduced_figures
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _paper_style import (METHOD_COLORS, TASK_LABEL, TASK_METRIC, TASK_ORDER,  # noqa: E402
                          despine, panel_label, plt)

METHODS_A = ["P0", "P4", "P4g", "A8_P4g"]
MLAB = {"P0": "P0\nvanilla", "P4": "P4\n(H+aligned)", "P4g": "P4g\ngated",
        "A8_P4g": "A8-P4g\nfrozen", "P1": "P1\nhidden only",
        "P5": "P5\nteacher transplant"}


def default_data_root() -> Path:
    env = os.environ.get("RNA_HEAD_DISTILL_DATA")
    return Path(env) if env else Path("./dataset-rna-head-distill")


def resolve(data_root: Path, *candidates: str) -> Path:
    """First existing path under data-root among the candidate relatives."""
    for rel in candidates:
        p = data_root / rel
        if p.exists():
            return p
    raise FileNotFoundError(
        f"none of {[str(data_root / c) for c in candidates]} exists — "
        "is --data-root the downloaded HF dataset root?")


def star(p: float) -> str:
    return "***" if (p >= 0.999 or p <= 0.001) else "ns"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=default_data_root(),
                    help="root of the downloaded HF dataset "
                         "(default: $RNA_HEAD_DISTILL_DATA)")
    ap.add_argument("--outdir", type=Path, default=Path("./reproduced_figures"),
                    help="output directory for fig4_downstream.{pdf,png}")
    args = ap.parse_args()
    OUT = args.outdir
    OUT.mkdir(parents=True, exist_ok=True)

    pe = pd.read_csv(resolve(args.data_root,
                             "metrics/bootstrap/point_estimates.csv"))
    a8 = pd.read_csv(resolve(args.data_root,
                             "metrics/bootstrap/a8_point_estimates.csv",
                             "metrics/bootstrap_a8/a8_point_estimates.csv"))
    s4 = pd.read_csv(resolve(args.data_root,
                             "metrics/bootstrap/p4_vs_baselines_summary.csv"))
    s8 = pd.read_csv(resolve(args.data_root,
                             "metrics/bootstrap/a8_vs_baselines_summary.csv",
                             "metrics/bootstrap_a8/a8_vs_baselines_summary.csv"))
    bs = pd.read_csv(resolve(args.data_root,
                             "metrics/downstream/downstream_primary_bootstrap.csv"))

    def series(task: str, mode: str, metric: str) -> np.ndarray:
        if mode == "A8_P4g":
            sub = a8[(a8.task == task) & (a8.line == "A8_P4g")]
        else:
            sub = pe[(pe.task == task) & (pe["mode"] == mode)]
        return sub[metric].dropna().to_numpy()

    def comp(task: str, name: str):
        """(mean, ci_lo, ci_hi, p_pos) for a bootstrap comparison."""
        if name.startswith("A8"):
            r = s8[(s8.task == task) & (s8.comparison == name)]
            return r[["mean", "ci_lo", "ci_hi", "p_pos"]].iloc[0].to_numpy()
        if name.startswith("P5"):
            r = bs[(bs.task == task) & (bs.mode_tag == "P5") &
                   (bs.baseline_tag == name.split("-")[1])]
            return r[["delta_mean", "ci_lo", "ci_hi", "p_pos"]].iloc[0].to_numpy()
        r = s4[(s4.task == task) & (s4.comparison == name)]
        return r[["mean", "ci_lo", "ci_hi", "p_pos"]].iloc[0].to_numpy()

    fig = plt.figure(figsize=(7.2, 8.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.35], hspace=0.38,
                             left=0.075, right=0.99, top=0.925, bottom=0.055)
    gsa = outer[0].subgridspec(2, 3, hspace=0.55, wspace=0.30)
    gsb = outer[1].subgridspec(1, 2, width_ratios=[1.15, 1.0], wspace=0.34)

    # --------------------------------------------- a: 2x3 main results
    fig.text(0.005, 0.995, "a", fontsize=12, fontweight="bold", va="top")
    fig.text(0.035, 0.995, "Frozen-probe test performance (mean ± sd, 3 seeds)",
             fontsize=9.5, fontweight="bold", va="top")
    for k, task in enumerate(TASK_ORDER):
        ax = fig.add_subplot(gsa[k // 3, k % 3])
        metric, mlab = TASK_METRIC[task]
        base = series(task, "P0", metric).mean()
        ymax = 0.0
        for i, m in enumerate(METHODS_A):
            v = series(task, m, metric)
            mu, sd = v.mean(), v.std(ddof=1)
            ax.bar(i, mu, yerr=sd, capsize=2.5, color=METHOD_COLORS[m],
                   alpha=0.92, edgecolor="black", linewidth=0.5,
                   error_kw=dict(lw=0.8))
            ax.plot(np.full(len(v), i) + np.linspace(-0.13, 0.13, len(v)), v,
                    "o", ms=2.6, mfc="black", mec="black", alpha=0.85, zorder=5)
            ymax = max(ymax, mu + sd)
            if m != "P0":
                d = comp(task, f"{m}-P0")
                st = star(d[3])
                col = "#1a7a1a" if (d[3] >= 0.999 and d[0] > 0) else \
                      "#b00000" if (d[3] <= 0.001 and d[0] < 0) else "#666666"
                ax.text(i, mu + sd, st, ha="center", va="bottom", fontsize=6.4,
                        color=col, fontweight="bold" if st != "ns" else "normal")
        ax.axhline(base, color="#555555", ls="--", lw=0.8, zorder=1)
        ax.set_xticks(range(4), [MLAB[m] for m in METHODS_A], fontsize=5.8)
        ax.set_ylabel(mlab, fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(TASK_LABEL[task], fontsize=7.6, pad=3)
        ax.set_ylim(top=ymax * 1.14 + ax.get_ylim()[0] * 0.0)
        despine(ax)

    # --------------------------------------------- b: delta forest
    axf = fig.add_subplot(gsb[0])
    ylabels = []
    ypos = []
    y = 0.0
    for task in TASK_ORDER:
        for m in ["P4", "P4g", "A8_P4g", "P5"]:
            mean, lo, hi, p = comp(task, f"{m}-P0")
            st = star(p)
            axf.errorbar(mean, y, xerr=[[mean - lo], [hi - mean]],
                         color=METHOD_COLORS[m], marker="o", ms=4.0, capsize=2.5,
                         lw=1.1, zorder=4)
            col = "#1a7a1a" if (p >= 0.999 and mean > 0) else \
                  "#b00000" if (p <= 0.001 and mean < 0) else "#666666"
            axf.text(hi + 0.006, y, st, fontsize=5.6, va="center", color=col,
                     fontweight="bold" if st != "ns" else "normal")
            ylab_m = {"P4": "P4", "P4g": "P4g", "A8_P4g": "A8-P4g", "P5": "P5"}[m]
            ylabels.append(f"{TASK_LABEL[task].splitlines()[0]}  {ylab_m}")
            ypos.append(y)
            y += 1.0
        y += 0.9  # gap between task groups
    axf.axvline(0, color="#555555", ls="--", lw=0.8, zorder=1)
    axf.set_yticks(ypos, ylabels, fontsize=5.2)
    axf.set_xlabel("Δ primary metric vs P0 (paired bootstrap, 95% CI)", fontsize=7)
    axf.tick_params(axis="x", labelsize=6)
    axf.set_xlim(-0.24, 0.28)
    axf.set_ylim(-0.8, y - 0.1)
    despine(axf)
    panel_label(axf, "b", x=-0.62, y=1.02)
    axf.set_title("Distilled variants minus vanilla backbone", fontsize=9.5,
                  fontweight="bold", loc="left", pad=4, x=-0.6)

    # --------------------------------------------- c: acceptor repair chain
    axc = fig.add_subplot(gsb[1])
    task = "splice_acceptor_seqcls"
    metric = "auroc"
    chain = ["P0", "P1", "P4", "P4g", "A8_P4g"]
    for i, m in enumerate(chain):
        v = series(task, m, metric)
        mu, sd = v.mean(), v.std(ddof=1)
        axc.bar(i, mu, yerr=sd, capsize=3, color=METHOD_COLORS[m], alpha=0.92,
                edgecolor="black", linewidth=0.5, error_kw=dict(lw=0.8))
        axc.plot(np.full(len(v), i) + np.linspace(-0.14, 0.14, len(v)), v, "o",
                 ms=3.0, mfc="black", mec="black", alpha=0.85, zorder=5)
        axc.text(i, mu + sd + 0.008, f"{mu:.3f}", ha="center", va="bottom",
                 fontsize=6.2)
    axc.set_xticks(range(len(chain)), [MLAB[m] for m in chain], fontsize=6.2)
    axc.set_ylabel("AUROC", fontsize=7.5)
    axc.tick_params(axis="y", labelsize=6.5)
    axc.margins(y=0.18)
    despine(axc)
    panel_label(axc, "c", x=-0.20, y=1.02)
    axc.set_title("Splice acceptor: distillation damage\nand staged repair",
                  fontsize=9.5, fontweight="bold", loc="left", pad=4)

    fig.savefig(OUT / "fig4_downstream.png")
    fig.savefig(OUT / "fig4_downstream.pdf")
    print("wrote", OUT / "fig4_downstream.pdf")


if __name__ == "__main__":
    main()
