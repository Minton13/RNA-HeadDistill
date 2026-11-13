#!/usr/bin/env python
# Provenance: ported from paper/scripts/plot_fig3_async.py (RNA_LLM_v2
# monorepo) on 2026-11-13 for the public release.
# Changes vs. original: hard-coded cluster paths replaced by
# --data-root (systems/{throughput,memory}_comparison.csv +
# metrics/ablation/alignment_ablation.csv) and --outdir.
"""Figure 3 — merged asynchronous-systems + alignment ablation figure.

  a  throughput + peak VRAM, S0 offline cache vs S1/S2/S3 online teachers
  b  cumulative GPU-hour cost of offline cache vs online extraction
  c  head-to-head alignment ablation (test split): KL and cosine

All disclosure fine print (measured vs extrapolated, n seeds, matched-9
definition) lives in the LaTeX caption, not inside the figure.

Run: python scripts/reproduction/plot_fig3_async.py \
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
from _paper_style import (ABL_COLORS, SYS_OFFLINE, SYS_ONLINE, despine,  # noqa: E402
                          panel_label, plt)

FULL_CORPUS = 2_072_010


def default_data_root() -> Path:
    env = os.environ.get("RNA_HEAD_DISTILL_DATA")
    return Path(env) if env else Path("./dataset-rna-head-distill")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=default_data_root(),
                    help="root of the downloaded HF dataset "
                         "(default: $RNA_HEAD_DISTILL_DATA)")
    ap.add_argument("--outdir", type=Path, default=Path("./reproduced_figures"),
                    help="output directory for fig3_async.{pdf,png}")
    args = ap.parse_args()
    OUT = args.outdir
    OUT.mkdir(parents=True, exist_ok=True)
    SYS = args.data_root / "systems"

    thr = pd.read_csv(SYS / "throughput_comparison.csv").set_index("scenario")["throughput_seq_per_sec"]
    mem = pd.read_csv(SYS / "memory_comparison.csv").set_index("scenario")["peak_vram_gb"]
    t = {k: FULL_CORPUS / thr[k] / 3600.0 for k in thr.index}

    abl = pd.read_csv(args.data_root / "metrics" / "ablation" /
                      "alignment_ablation.csv").set_index("experiment")
    FIG_EXPS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
    EXP_FULL = {e: e + "_" + s for e, s in zip(FIG_EXPS, [
        "single_teacher_top1", "head2head_top3", "head2head_top5_full",
        "consensus_top5", "random_pairing", "linear_adapter", "all_from_L12",
        "adapters_only"])}

    fig = plt.figure(figsize=(7.2, 7.8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.1, 1.0], hspace=0.78,
                          wspace=0.30, left=0.085, right=0.98, top=0.94,
                          bottom=0.05)

    # --------------------------------------------- a: throughput + memory
    scen = ["S0_offline_cached", "S1_online_single_teacher",
            "S2_online_three_teachers", "S3_online_full_estimate"]
    scen_lab = ["S0\noffline", "S1\nonline ×1", "S2\nonline ×3", "S3\nonline ×9 *"]
    ax_thr = fig.add_subplot(gs[0, 0])
    ax_mem = fig.add_subplot(gs[0, 1])
    for ax, vals, ylab in [(ax_thr, thr, "throughput (sequences/s)"),
                           (ax_mem, mem, "peak VRAM (GB)")]:
        for i, s in enumerate(scen):
            c = SYS_OFFLINE if s == "S0_offline_cached" else SYS_ONLINE
            hatch = "//" if s == "S3_online_full_estimate" else None
            ax.bar(i, vals[s], color=c, alpha=0.92 if s == "S0_offline_cached" else 0.78,
                   edgecolor="black", linewidth=0.5, hatch=hatch)
            ax.text(i, vals[s], f"{vals[s]:.2f}" if vals[s] < 10 else f"{vals[s]:.1f}",
                    ha="center", va="bottom", fontsize=6.2)
        ax.set_xticks(range(4), scen_lab, fontsize=6.4)
        ax.set_ylabel(ylab, fontsize=7.5)
        ax.margins(y=0.22)
        despine(ax)
    ax_thr.set_title("throughput (10k-seq microbenchmark)", fontsize=7.5, pad=3)
    ax_mem.set_title("peak GPU memory", fontsize=7.5, pad=3)
    handles = [plt.Rectangle((0, 0), 1, 1, fc=SYS_OFFLINE, ec="black", lw=0.5),
               plt.Rectangle((0, 0), 1, 1, fc=SYS_ONLINE, ec="black", lw=0.5),
               plt.Rectangle((0, 0), 1, 1, fc=SYS_ONLINE, ec="black", lw=0.5,
                             hatch="//")]
    ax_thr.legend(handles, ["offline cache (measured)",
                            "online extraction (measured)", "* S3 extrapolated"],
                  loc="upper right", frameon=False, fontsize=6.0,
                  handlelength=1.4, labelspacing=0.25)

    # --------------------------------------------- b: cumulative cost
    axb = fig.add_subplot(gs[1, :])
    one_time = t["S3_online_full_estimate"]
    N = np.arange(1, 51)
    cost = {
        "offline cache (1× 9-teacher extraction + N× cached reads)":
            one_time + N * t["S0_offline_cached"],
        "online, 1 teacher": N * t["S1_online_single_teacher"],
        "online, 3 teachers": N * t["S2_online_three_teachers"],
        "online, 9 teachers (extrapolated)": N * t["S3_online_full_estimate"],
    }
    styles = [(SYS_OFFLINE, "-", 1.9), (SYS_ONLINE, "-", 1.2),
              (SYS_ONLINE, "-", 1.2), (SYS_ONLINE, "--", 1.2)]
    for (label, y), (c, ls, lw) in zip(cost.items(), styles):
        axb.plot(N, y, color=c, ls=ls, lw=lw, label=label)
    axb.set_yscale("log")
    axb.set_ylim(8, 20000)
    be_offsets = {"online, 1 teacher": (5, 5), "online, 3 teachers": (6, -13),
                  "online, 9 teachers (extrapolated)": (5, -13)}
    for key in ["online, 1 teacher", "online, 3 teachers",
                "online, 9 teachers (extrapolated)"]:
        skey = {"online, 1 teacher": "S1_online_single_teacher",
                "online, 3 teachers": "S2_online_three_teachers",
                "online, 9 teachers (extrapolated)": "S3_online_full_estimate"}[key]
        k = one_time / (t[skey] - t["S0_offline_cached"])
        yk = one_time + k * t["S0_offline_cached"]
        axb.plot(k, yk, marker="o", ms=4.5, mfc="none", mec="black", mew=1.1,
                 zorder=5)
        axb.annotate(f"{k:.1f}", (k, yk), textcoords="offset points",
                     xytext=be_offsets[key], fontsize=6.4, color="black",
                     ha="left", va="top" if be_offsets[key][1] < 0 else "bottom")
    axb.plot([], [], marker="o", ls="none", mfc="none", mec="black", mew=1.1,
             label="break-even N (offline vs online ×1 / ×3 / ×9)")
    axb.set_xlabel("number of student experiments N", fontsize=7.5)
    axb.set_ylabel("cumulative GPU-hours (log)", fontsize=7.5)
    axb.tick_params(labelsize=6.5)
    axb.legend(frameon=False, loc="lower right", fontsize=6.4)
    despine(axb)

    # --------------------------------------------- c: alignment ablation
    ax_kl = fig.add_subplot(gs[2, 0])
    ax_cos = fig.add_subplot(gs[2, 1])
    panels = [(ax_kl, "KL_mean", "KL_std", "KL_matched9_mean", "KL_matched9_std",
               "test row-wise KL ↓"),
              (ax_cos, "Cosine_mean", "Cosine_std", "Cosine_matched9_mean",
               "Cosine_matched9_std", "test cosine similarity ↑")]
    for ax, col, col_std, mcol, mcol_std, ylab in panels:
        x = np.arange(len(FIG_EXPS))
        for i, e in enumerate(FIG_EXPS):
            r = abl.loc[EXP_FULL[e]]
            m, s = r[col], r[col_std]
            color = (ABL_COLORS["main"] if e == "A3" else
                     ABL_COLORS["frozen"] if e == "A8" else ABL_COLORS["ablation"])
            ax.bar(i, m, yerr=s, capsize=2.5, color=color,
                   alpha=0.92 if e == "A3" else 0.78, edgecolor="black",
                   linewidth=0.5, error_kw=dict(lw=0.8))
            ax.text(i, m + (s if np.isfinite(s) else 0), f"{m:.2f}", ha="center",
                    va="bottom", fontsize=5.4)
            mm, ms = r[mcol], r[mcol_std]
            if np.isfinite(mm):
                ax.errorbar([i], [mm], yerr=[[ms], [ms]], marker="D", ms=4.5,
                            color="black", mec="black", mfc=ABL_COLORS["matched"],
                            zorder=5, capsize=2.5, lw=0.8)
        ax.set_xticks(x, FIG_EXPS, fontsize=6.8)
        ax.set_ylabel(ylab, fontsize=7.5)
        ax.tick_params(axis="y", labelsize=6.5)
        ax.margins(y=0.22)
        despine(ax)
    ax_kl.tick_params(axis="x", labelsize=6.8)
    ax_cos.tick_params(axis="x", labelsize=6.8)

    # ---- position-based section headers (after layout is resolved)
    fig.canvas.draw()
    pa = ax_thr.get_position()
    pb = axb.get_position()
    pc = ax_kl.get_position()
    fig.text(0.005, pa.y1 + 0.055, "a", fontsize=12, fontweight="bold", va="top")
    fig.text(0.035, pa.y1 + 0.055, "Offline cache vs online teacher extraction",
             fontsize=9.5, fontweight="bold", va="top")
    fig.text(0.005, pb.y1 + 0.055, "b", fontsize=12, fontweight="bold", va="top")
    fig.text(0.035, pb.y1 + 0.055,
             "Asynchronous amortization over repeated student experiments",
             fontsize=9.5, fontweight="bold", va="top")
    fig.text(0.005, pc.y1 + 0.062, "c", fontsize=12, fontweight="bold", va="top")
    fig.text(0.035, pc.y1 + 0.062,
             "Head-to-head alignment ablation (test split)",
             fontsize=9.5, fontweight="bold", va="top")
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=ABL_COLORS["main"], ec="black", lw=0.5),
        plt.Rectangle((0, 0), 1, 1, fc=ABL_COLORS["ablation"], ec="black", lw=0.5),
        plt.Rectangle((0, 0), 1, 1, fc=ABL_COLORS["frozen"], ec="black", lw=0.5),
        plt.Line2D([0], [0], marker="D", color="none", mfc=ABL_COLORS["matched"],
                   mec="black", ms=4.5),
    ]
    fig.legend(handles, ["A3 full head-to-head (proposed)", "ablations",
                         "A8 adapters-only (frozen backbone)",
                         "matched-9 subset (shared top-3/route heads)"],
               loc="upper center", ncol=4, frameon=False, fontsize=6.0,
               bbox_to_anchor=(0.52, pc.y1 + 0.022), handlelength=1.2,
               columnspacing=0.9)

    fig.savefig(OUT / "fig3_async.png")
    fig.savefig(OUT / "fig3_async.pdf")
    print("wrote", OUT / "fig3_async.pdf")


if __name__ == "__main__":
    main()
