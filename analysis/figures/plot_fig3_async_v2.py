#!/usr/bin/env python
"""Figure 3 (v2, audited rebuild) — verified-scope systems panels + ablation.

Replaces panels a–c of fig3_async with corrected measurement scopes:
  a  forward-only no-gradient throughput (single A100, 10k uniform sample),
     with the production 6-GPU training rate shown as a separate-scope reference
  b  forward-only peak device memory, with the production per-GPU training
     peak shown as a separate-scope reference
  c  measured GPU-hour costs only (student run; teacher cache build lower
     bound). The historical break-even curves are removed: they mixed a
     forward-only student proxy with a modeled cache cost.
  d,e  head-to-head alignment ablation (test split): KL and cosine
     (data unchanged from fig3_async)

Data:
  panels a/b microbenchmark: rna_head_alignment/outputs/systems/asynchronous_distillation/*.csv
  panel a/b production refs + panel c: paper/figures/systems/fig3_v2_measured_costs.csv
    (values transcribed from paper/audit/hardware/hardware_training_facts.md and
     paper/audit/systems/fig3_forensic_audit.md, 2026-11-16)
  panels d/e: paper/tables/alignment_ablation.csv (unchanged)

Run: python paper/scripts/plot_fig3_async_v2.py  (use the project env python)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from _paper_style import (ABL_COLORS, SYS_OFFLINE, SYS_ONLINE, despine,  # noqa: E402
                          panel_label, plt)

ROOT = Path("<REPO_ROOT>")
SYS = ROOT / "rna_head_alignment/outputs/systems/asynchronous_distillation"
OUT = ROOT / "paper/figures/systems"
OUT.mkdir(parents=True, exist_ok=True)

# production-run references (different scope from the microbenchmark bars):
# 6xA100 aggregate throughput and per-GPU peak, from the audited production run
PROD_SEQ_PER_SEC = 18.2           # 6xA100 aggregate, fwd+bwd+AdamW
PROD_MEM_GB = 2.37 * 2**30 / 1e9  # 2.37 GiB/card -> 2.545 GB
PROD_LABEL = "production training\n(6×A100; fwd+bwd+optim.)"
PROD_COLOR = "#555555"

thr = pd.read_csv(SYS / "throughput_comparison.csv").set_index("scenario")["throughput_seq_per_sec"]
mem = pd.read_csv(SYS / "memory_comparison.csv").set_index("scenario")["peak_vram_gb"]

costs = pd.read_csv(OUT / "fig3_v2_measured_costs.csv")
cache = costs[costs["item"].str.startswith("cache_build_")]
student_gpuh = float(costs.loc[costs["item"] == "student_training_run", "gpu_hours"].iloc[0])
fwd_pass_gpuh = float(costs.loc[costs["item"] == "forward_only_corpus_pass", "gpu_hours"].iloc[0])
cache_total = float(cache["gpu_hours"].sum())

abl = pd.read_csv(ROOT / "paper/tables/alignment_ablation.csv").set_index("experiment")
FIG_EXPS = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
EXP_FULL = {e: e + "_" + s for e, s in zip(FIG_EXPS, [
    "single_teacher_top1", "head2head_top3", "head2head_top5_full",
    "consensus_top5", "random_pairing", "linear_adapter", "all_from_L12",
    "adapters_only"])}

fig = plt.figure(figsize=(7.2, 7.8))
gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 1.0], hspace=0.85,
                      wspace=0.30, left=0.085, right=0.98, top=0.965,
                      bottom=0.06)

# ------------------------------------------------- a/b: microbenchmark + production ref
scen = ["S0_offline_cached", "S1_online_single_teacher",
        "S2_online_three_teachers", "S3_online_full_estimate"]
scen_lab = ["S0\noffline", "S1\nonline ×1", "S2\nonline ×3", "S3\nonline ×9 *"]
xpos = [0, 1, 2, 3, 4.8]
ax_thr = fig.add_subplot(gs[0, 0])
ax_mem = fig.add_subplot(gs[0, 1])
for ax, vals, prod_val, prod_fmt, ylab, title in [
        (ax_thr, thr, PROD_SEQ_PER_SEC, ".1f", "sequences/s",
         "forward-only throughput (single A100)"),
        (ax_mem, mem, PROD_MEM_GB, ".2f", "GB (allocated)",
         "forward-only peak device memory (single A100)")]:
    for i, s in enumerate(scen):
        c = SYS_OFFLINE if s == "S0_offline_cached" else SYS_ONLINE
        hatch = "//" if s == "S3_online_full_estimate" else None
        ax.bar(xpos[i], vals[s], color=c, alpha=0.92 if s == "S0_offline_cached" else 0.78,
               edgecolor="black", linewidth=0.5, hatch=hatch)
        ax.text(xpos[i], vals[s], f"{vals[s]:.2f}" if vals[s] < 10 else f"{vals[s]:.1f}",
                ha="center", va="bottom", fontsize=6.2)
    ax.bar(xpos[4], prod_val, color=PROD_COLOR, alpha=0.85, edgecolor="black",
           linewidth=0.5)
    ax.text(xpos[4], prod_val, format(prod_val, prod_fmt), ha="center",
            va="bottom", fontsize=6.2)
    ax.set_xticks(xpos, scen_lab + ["prod.\ntrain †"], fontsize=6.0)
    ax.set_ylabel(ylab, fontsize=7.5)
    ax.set_title(title, fontsize=7.5, pad=3)
    ax.margins(y=0.22)
    despine(ax)
ax_mem.text(xpos[4], PROD_MEM_GB * 1.24, "= 2.37 GiB/card", ha="center",
            va="bottom", fontsize=5.6, color="#333333")
handles = [plt.Rectangle((0, 0), 1, 1, fc=SYS_OFFLINE, ec="black", lw=0.5),
           plt.Rectangle((0, 0), 1, 1, fc=SYS_ONLINE, ec="black", lw=0.5),
           plt.Rectangle((0, 0), 1, 1, fc=SYS_ONLINE, ec="black", lw=0.5,
                         hatch="//"),
           plt.Rectangle((0, 0), 1, 1, fc=PROD_COLOR, ec="black", lw=0.5)]
ax_thr.legend(handles, ["offline cache (measured)",
                        "online extraction (measured)",
                        "* S3 extrapolated",
                        "† production training (different scope)"],
              loc="upper right", frameon=False, fontsize=5.8,
              handlelength=1.4, labelspacing=0.3)
panel_label(ax_thr, "a", x=-0.22, y=1.10)
panel_label(ax_mem, "b", x=-0.18, y=1.10)

# ------------------------------------------------- c: measured GPU-hour costs (no break-even)
axc = fig.add_subplot(gs[1, :])
x0, x1, x2 = 0.0, 1.0, 2.2
axc.bar(x0, fwd_pass_gpuh, width=0.62, color="#bdbdbd", hatch="//",
        edgecolor="black", linewidth=0.5)
axc.text(x0, fwd_pass_gpuh * 1.12, f"{fwd_pass_gpuh:.2f}", ha="center",
         va="bottom", fontsize=6.4)
axc.bar(x1, student_gpuh, width=0.62, color=SYS_OFFLINE, edgecolor="black",
        linewidth=0.5)
axc.text(x1, student_gpuh * 1.12, f"{student_gpuh:.2f}", ha="center",
         va="bottom", fontsize=6.4)
cmap = plt.get_cmap("tab10")
bottom = 0.0
for j, (_, row) in enumerate(cache.iterrows()):
    teacher = row["item"].replace("cache_build_", "")
    axc.bar(x2, row["gpu_hours"], width=0.62, bottom=bottom,
            color=cmap(j % 10), edgecolor="black", linewidth=0.4,
            label=teacher)
    bottom += row["gpu_hours"]
axc.text(x2, cache_total * 1.12, f"≥{cache_total:.0f}", ha="center",
         va="bottom", fontsize=7.2, fontweight="bold")
axc.set_yscale("log")
axc.set_ylim(3, 3000)
axc.set_xlim(-0.7, 3.6)
axc.set_xticks([x0, x1, x2],
               ["forward-only full-corpus pass ‡\n(rescaled microbenchmark;\n"
                "not a training cost)",
                "student training run\n(measured; 6×A100,\n5,000 updates)",
                "teacher cache build\n(measured; 7 of 9\nteachers)"],
               fontsize=6.6)
axc.set_ylabel("GPU-hours (log scale)", fontsize=7.5)
axc.set_title("measured GPU-hour costs — historical break-even panel removed",
              fontsize=7.5, pad=3)
axc.text(0.995, 0.97,
         "‡ 7.51 GPU-h is a single-device forward pass over the corpus,\n"
         "not a student experiment. Cache logs for rinalmo-giga/mega are\n"
         "unrecoverable, so the cache total is a lower bound (est. 300–400).",
         transform=axc.transAxes, ha="right", va="top", fontsize=5.8,
         color="#333333")
axc.legend(frameon=False, loc="upper left", fontsize=6.0, ncol=2,
           title="cache build by teacher", title_fontsize=6.2)
axc.tick_params(labelsize=6.5)
despine(axc)
panel_label(axc, "c", x=-0.045, y=1.06)

# ------------------------------------------------- d/e: alignment ablation (unchanged data)
ax_kl = fig.add_subplot(gs[2, 0])
ax_cos = fig.add_subplot(gs[2, 1])
panels = [(ax_kl, "KL_mean", "KL_std", "KL_matched9_mean", "KL_matched9_std",
           "test row-wise KL ↓", "d"),
          (ax_cos, "Cosine_mean", "Cosine_std", "Cosine_matched9_mean",
           "Cosine_matched9_std", "test cosine similarity ↑", "e")]
for ax, col, col_std, mcol, mcol_std, ylab, letter in panels:
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
    panel_label(ax, letter, x=-0.20, y=1.06)

fig.canvas.draw()
pc = ax_kl.get_position()
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
           bbox_to_anchor=(0.52, pc.y0 - 0.055), handlelength=1.2,
           columnspacing=0.9)

fig.savefig(OUT / "fig3_async_v2.png")
fig.savefig(OUT / "fig3_async_v2.pdf")
print("wrote", OUT / "fig3_async_v2.pdf")
