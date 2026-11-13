#!/usr/bin/env python
# Provenance: ported from paper/scripts/plot_fig2_atlas.py (RNA_LLM_v2
# monorepo) on 2026-11-13 for the public release.
# Changes vs. original: hard-coded cluster paths replaced by
# --data-root (atlas/all_head_atlas.csv + selection/selected_teacher_heads.csv)
# and --outdir.
"""Figure 2 — merged functional-head atlas figure (vector PDF).

  a  per-head probe-score heatmaps for six representative task x model
     combos (2 rows x 3 cols; tall models on row 1, short on row 2 so the
     row heights sum to 45 layers) — square cells (no aspect distortion),
     stats in the title line
  b  cross-task overlap (Jaccard) of top-10 functional heads within a model;
     only cells with J >= 0.10 are annotated (full-matrix numbers were too
     dense), diagonal left blank
  c  percentile of each of the 15 selected teacher heads in its own
     task x model pool, colored by route depth

Run: python scripts/reproduction/plot_fig2_atlas.py \
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
from _paper_style import ROUTE_COLORS, despine, panel_label, plt  # noqa: E402


def default_data_root() -> Path:
    env = os.environ.get("RNA_HEAD_DISTILL_DATA")
    return Path(env) if env else Path("./dataset-rna-head-distill")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=default_data_root(),
                    help="root of the downloaded HF dataset "
                         "(default: $RNA_HEAD_DISTILL_DATA)")
    ap.add_argument("--outdir", type=Path, default=Path("./reproduced_figures"),
                    help="output directory for fig2_atlas.{pdf,png}")
    args = ap.parse_args()
    OUT = args.outdir
    OUT.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ cleaning
    atlas = pd.read_csv(args.data_root / "atlas" / "all_head_atlas.csv",
                        low_memory=False)
    atlas = atlas[~atlas["notes"].fillna("").str.contains("model_load_failed")]
    atlas = atlas[atlas["metric_primary"].notna()]
    per_head = atlas[atlas["head"] >= 0]
    combo_max = per_head.groupby(["task", "model"])["metric_primary"].max()
    deg = set(combo_max[combo_max <= 1e-9].index.tolist())
    valid = per_head[~per_head.apply(lambda r: (r["task"], r["model"]) in deg,
                                     axis=1)].copy()

    sel = pd.read_csv(args.data_root / "selection" /
                      "selected_teacher_heads.csv")

    fig = plt.figure(figsize=(7.2, 7.9))

    # --------------------------------------------------------- a: heatmaps
    # square cells: one cell = CELL inches on both axes
    CELL = 0.045
    FW, FH = 7.2, 7.9
    cw, ch = CELL / FW, CELL / FH          # figure fraction per cell
    # six combos spanning all three route families; rows are balanced by model
    # depth (row 1: 33L + 28L + 12L -> row height 33L; row 2: 12L + 12L + 6L ->
    # 12L) so the two rows sum to 45 layers and the canvas stays <= 7.9 in tall
    picks = [
        ("secondary_structure_pair", "rinalmo_giga", "Secondary structure",
         "RiNALMo-giga", 0, 0),
        ("rbp_binding_seqcls", "carbon_500m", "RBP binding", "CARBON-500M",
         1, 0),
        ("rbp_binding_seqcls", "mrnafm", "RBP binding", "mRNA-FM", 2, 0),
        ("secondary_structure_pair", "rnaernie", "Secondary structure",
         "RNAErnie", 0, 1),
        ("chemical_reactivity_token", "rnafm", "Chemical reactivity",
         "RNA-FM", 1, 1),
        ("splice_site_seqcls", "splicebert", "Splice site", "SpliceBERT", 2, 1),
    ]
    COL_X = [0.75 / FW, 2.95 / FW, 5.15 / FW]   # column left edges (fig fraction)
    ROW_TOP = [0.888, 0.614]                    # heatmap top edges (fig fraction)
    for task, model, lab, mname, cix, rix in picks:
        g = valid[(valid["task"] == task) & (valid["model"] == model)]
        n_layers = int(g["layer"].max()) + 1
        n_heads = int(g["head"].max()) + 1
        mat = np.full((n_layers, n_heads), np.nan)
        for _, r in g.iterrows():
            mat[int(r["layer"]), int(r["head"])] = r["metric_primary"]
        w, h = n_heads * cw, n_layers * ch
        x, top = COL_X[cix], ROW_TOP[rix]
        ax = fig.add_axes([x, top - h, w, h])
        im = ax.imshow(mat, cmap="viridis", interpolation="nearest",
                       aspect="equal")
        med = np.nanmedian(mat)
        t5 = np.sort(mat[~np.isnan(mat)].ravel())[-5:].mean()
        ax.set_title(f"{lab}\n{mname} · {n_layers}L×{n_heads}H\n"
                     f"median {med:.3f} · top-5 {t5:.3f}",
                     fontsize=6.8, pad=4)
        ax.set_xlabel("head", fontsize=7)
        if cix == 0:
            ax.set_ylabel("layer", fontsize=7)
        ax.tick_params(labelsize=6)
        cbax = fig.add_axes([x + w + 0.007, top - h, 0.009, h])
        cb = fig.colorbar(im, cax=cbax)
        cb.ax.tick_params(labelsize=5.5)
        if cix == 2:
            cb.ax.set_ylabel("probe score", fontsize=6)
    fig.text(0.005, 0.975, "a", fontsize=12, fontweight="bold", va="top")
    fig.text(0.032, 0.975, "Probe signal concentrates in a few heads",
             fontsize=9.5, fontweight="bold", va="top")

    # --------------------------------------------------- b: Jaccard overlap
    tasks = sorted(valid["task"].unique())
    K = 10
    jac_sum = pd.DataFrame(0.0, index=tasks, columns=tasks)
    jac_cnt = pd.DataFrame(0, index=tasks, columns=tasks)
    null_jac = []
    for model, gm in valid.groupby("model"):
        tops = {}
        for task, g in gm.groupby("task"):
            top = g.nlargest(K, "metric_primary")
            tops[task] = set(zip(top["layer"], top["head"]))
        n_heads_model = len(gm.groupby(["layer", "head"]).size())
        null_jac.append(K / (2 * n_heads_model - K))
        for i, t1 in enumerate(tasks):
            if t1 not in tops:
                continue
            for t2 in tasks[i:]:
                if t2 not in tops:
                    continue
                a, b = tops[t1], tops[t2]
                j = len(a & b) / len(a | b)
                jac_sum.loc[t1, t2] += j
                jac_sum.loc[t2, t1] += j
                jac_cnt.loc[t1, t2] += 1
                jac_cnt.loc[t2, t1] += 1
    jac = jac_sum / jac_cnt.replace(0, np.nan)
    null_mean = float(np.mean(null_jac))

    short = [t.replace("chemical_reactivity_", "chem_").replace("_regression", "")
             .replace("_seqcls", "").replace("_token", "") for t in tasks]

    axb = fig.add_axes([0.105, 0.118, 0.335, 0.358])
    mask = np.isnan(jac.values)
    im = axb.imshow(np.ma.masked_array(jac.values, mask), cmap="magma", vmin=0,
                    vmax=max(0.3, float(np.nanmax(jac.values))),
                    interpolation="nearest", aspect="equal")
    axb.set_xticks(range(len(tasks)), short, rotation=90, fontsize=5.4)
    axb.set_yticks(range(len(tasks)), short, fontsize=5.4)
    # annotate only clearly-above-chance cells; diagonal carries no information
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            v = jac.values[i, j]
            if mask[i, j] or i == j or v < 0.10:
                continue
            axb.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=5.2,
                     color="w" if v < 0.32 else "black")
    cbax = fig.add_axes([0.455, 0.175, 0.010, 0.288])
    cb = fig.colorbar(im, cax=cbax)
    cb.ax.tick_params(labelsize=5.5)
    cb.ax.set_ylabel("mean Jaccard (top-10 sets)", fontsize=6)
    axb.set_title("Cross-task sharing of top-10 heads within one model\n"
                  f"mean over 26 models (random ≈ {null_mean:.3f}); "
                  "J ≥ 0.10 marked", fontsize=6.8, pad=4)
    panel_label(axb, "b", x=-0.40, y=1.05)
    axb.spines[["left", "bottom"]].set_visible(False)

    # --------------------------------------- c: selected-head percentiles
    axc = fig.add_axes([0.640, 0.118, 0.330, 0.358])
    name_map = {"nucleotide_transformer_v2_500m": "NT-v2-500M",
                "moderngena_base": "ModernGENA-base"}
    labels = []
    for _, r in sel.iterrows():
        m = name_map.get(r["teacher_model"], r["teacher_model"])
        labels.append(f"{m} L{r['teacher_layer']}H{r['teacher_head']}")
    bar_colors = [ROUTE_COLORS[d] for d in sel["route_depth"]]
    axc.barh(range(len(sel)), sel["percentile_in_task_model"], color=bar_colors,
             height=0.72)
    axc.set_yticks(range(len(sel)), labels, fontsize=5.4)
    axc.invert_yaxis()
    axc.set_xlabel("percentile within its task × model pool (100 = best head)",
                   fontsize=6.5)
    axc.set_xlim(0, 108)
    axc.axvline(90, color="grey", ls="--", lw=0.8)
    axc.text(92, 14.6, "top 10%", fontsize=5.6, color="grey", va="bottom",
             ha="left")
    from matplotlib.patches import Patch  # noqa: E402
    route_handles = [Patch(facecolor=c, label=f"route L{d}")
                     for d, c in ROUTE_COLORS.items()]
    axc.legend(handles=route_handles, fontsize=6, loc="upper center",
               bbox_to_anchor=(0.5, -0.16), ncol=3, frameon=False,
               handletextpad=0.5, columnspacing=1.0)
    axc.tick_params(axis="x", labelsize=6)
    axc.set_title("15 selected teacher heads:\nrank percentile in their pools",
                  fontsize=7.2, pad=4)
    panel_label(axc, "c", x=-0.42, y=1.05)
    despine(axc)

    # small pad: everything except the bold "b" letter already sits inside the
    # canvas (panel b's rotated x tick labels get a 1.10 in bottom margin), so
    # bbox="tight" now adds only ~0.2 in of width instead of inflating the height
    fig.savefig(OUT / "fig2_atlas.png", dpi=300, pad_inches=0.02)
    fig.savefig(OUT / "fig2_atlas.pdf", pad_inches=0.02)
    print("wrote", OUT / "fig2_atlas.pdf")


if __name__ == "__main__":
    main()
