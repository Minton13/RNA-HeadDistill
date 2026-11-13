#!/usr/bin/env python
# Provenance: ported from paper/scripts/plot_si_dataflow.py (RNA_LLM_v2
# monorepo) on 2026-11-13 for the public release.
# Changes vs. original: hard-coded cluster paths replaced by
# --data-root (dataflow/*_sankey_{nodes,links}_counts_v2.csv) and --outdir.
"""Figure S1 — data inventory and flow (vector PDF/PNG).

Panel (a): four-column count-weighted Sankey (data source -> species/source
scope -> downstream task -> alignment level), a matplotlib replica of the PPT
Plotly Sankey. Node heights, node order and link widths are read straight
from the v2 layout CSVs (originally exported by
rna_head_alignment/scripts/local/build_pretrain_benchmark_sankey.py, which
needs plotly and is NOT run here); only the vertical spacing is re-stacked
with a minimum per-node pitch so two-line labels never collide.

Panel (b): sequence-count splits — the teacher-attention cache (2,114,289
sequences -> train/val/test) and the six downstream evaluation tasks — as two
compact mini-Sankeys (band width proportional to sqrt(n)).

Run: python scripts/reproduction/plot_si_dataflow.py \
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
from _paper_style import ROUTE_COLORS, panel_label, plt  # noqa: E402

from matplotlib.patches import FancyBboxPatch, PathPatch, Rectangle  # noqa: E402
from matplotlib.path import Path as MplPath  # noqa: E402

NODES_REL = "dataflow/pretrain_benchmark_datasource_level_sankey_nodes_counts_v2.csv"
LINKS_REL = "dataflow/pretrain_benchmark_datasource_level_sankey_links_counts_v2.csv"

INK = "#222222"
GRAY = "#6a6a6a"
TITLE_GRAY = "#555555"
LINK_ALPHA = 0.45

# ---------------- panel (a) geometry (SVG-pixel coordinate space) -----------
COL_X = {0: 40.0, 1: 434.0, 2: 817.0, 3: 1176.0}
NODE_W = 18.0
X_MIN, X_MAX = 20.0, 1580.0
MIN_PITCH = 47.0          # minimum vertical slot per node (px)
COL_GAP = 12.0            # inter-node gap within a column (px)
TOP_H = 88.0              # title / subtitle band
BOT_H = 56.0              # column-title band
COL_COLORS = {0: "#2f5f73", 1: "#6c7a20", 2: "#9a5f1d"}
LEVEL_COLORS = {"L6": ROUTE_COLORS[6], "L8": ROUTE_COLORS[8], "L12": ROUTE_COLORS[12]}
COL_TITLES = ["Data source", "Species / source scope", "Downstream task",
              "Alignment level"]
TITLE_A = "RNA pretraining corpora + benchmark datasets to downstream alignment levels"
SUBTITLE_A = ("Count-weighted links; translucent bands inherit source-column color and "
              "encode amounts (PT = pretraining corpus; BM = benchmark dataset).")

FS_NODE_NAME = 6.9
FS_NODE_SUM = 6.5
FS_TITLE = 10.2
FS_SUBTITLE = 7.0
FS_COL_TITLE = 7.4

# ---------------- panel (b) data (fixed study facts, unchanged) -------------
SPLIT_COLORS = {"train": ROUTE_COLORS[6], "val": ROUTE_COLORS[8],
                "test": ROUTE_COLORS[12]}
CACHE_COLOR = "#8a5ca8"
TASK_COLOR = "#7f9db9"

cache = {"train": 2_072_010, "val": 21_143, "test": 21_136}
cache_total = sum(cache.values())          # 2,114,289
cache_use = {"train": "student distillation training",
             "val": "head pairing + early stopping",
             "test": "alignment eval (row KL / cosine)"}

tasks = {  # name -> (train, val, test)
    "Splice donor": (36_437, 4_670, 4_512),
    "Splice acceptor": (36_475, 4_709, 4_435),
    "RBP binding": (962_667, 120_152, 142_760),
    "ncRNA family": (1_403, 169, 162),
    "Chemical reactivity": (126_909, 26_463, 14_436),
    "Secondary structure": (18_407, 2_273, 2_312),
}
task_splits = {s: sum(v[i] for v in tasks.values())
               for i, s in enumerate(("train", "val", "test"))}
task_use = {"train": "probe/fine-tune training",
            "val": "checkpoint selection",
            "test": "test bootstrap metrics"}

FS_B_NAME = 6.4
FS_B_COUNT = 6.4
FS_B_TITLE = 7.8


def default_data_root() -> Path:
    env = os.environ.get("RNA_HEAD_DISTILL_DATA")
    return Path(env) if env else Path("./dataset-rna-head-distill")


# ============================ panel (a): Sankey ==============================
def load_sankey(data_root: Path):
    nodes = pd.read_csv(data_root / NODES_REL)
    links = pd.read_csv(data_root / LINKS_REL)
    info = {}          # node -> dict(column, height, value, level, pos)
    pos_ctr = {0: 0, 1: 0, 2: 0, 3: 0}
    for _, r in nodes.iterrows():
        c = int(r["column"])
        info[str(r["node"])] = {
            "column": c, "height": float(r["height"]),
            "value": float(r["value"]), "level": str(r["dominant_level"]),
            "pos": pos_ctr[c],
        }
        pos_ctr[c] += 1
    return info, links


def stack_columns(info):
    """Re-stack each column top-down with a minimum per-node pitch."""
    top = {}
    col_block = {}
    for c in range(4):
        members = [n for n, d in info.items() if d["column"] == c]
        members.sort(key=lambda n: info[n]["pos"])
        y = TOP_H
        for n in members:
            top[n] = y
            y += max(info[n]["height"], MIN_PITCH) + COL_GAP
        col_block[c] = y - COL_GAP - TOP_H
    canvas_h = TOP_H + max(col_block.values()) + BOT_H
    return top, canvas_h


def cubic_band(ax, x0, ya0, ya1, x1, yb0, yb1, color, alpha=LINK_ALPHA, zorder=1):
    xm = (x0 + x1) / 2.0
    verts = [(x0, ya0), (xm, ya0), (xm, yb0), (x1, yb0), (x1, yb1), (xm, yb1),
             (xm, ya1), (x0, ya1), (x0, ya0)]
    codes = [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.LINETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4,
             MplPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MplPath(verts, codes), fc=color, ec="none",
                           alpha=alpha, zorder=zorder, lw=0))


def draw_panel_a(ax, data_root: Path):
    info, links = load_sankey(data_root)
    top, canvas_h = stack_columns(info)

    # ---- bands: stack in the same order as the reference layout
    links = links.copy()
    links["ps"] = links["source"].map(lambda s: info[str(s)]["pos"])
    links["pt"] = links["target"].map(lambda s: info[str(s)]["pos"])
    links = links.sort_values(["stage", "ps", "pt", "value"],
                              ascending=[True, True, True, False])
    out_off = {n: 0.0 for n in info}
    in_off = {n: 0.0 for n in info}
    bands = []
    for _, r in links.iterrows():
        s, t = str(r["source"]), str(r["target"])
        w = float(r["stroke_width"])
        stage = int(r["stage"])
        y0 = top[s] + out_off[s]
        y1 = top[t] + in_off[t]
        out_off[s] += w
        in_off[t] += w
        bands.append((w, stage, y0, y1, s, t))
    for w, stage, y0, y1, s, t in sorted(bands, key=lambda b: -b[0]):
        cubic_band(ax, COL_X[stage] + NODE_W, y0, y0 + w,
                   COL_X[stage + 1], y1, y1 + w, COL_COLORS[stage])

    # ---- node bars + two-line labels
    for n, d in info.items():
        c = d["column"]
        color = LEVEL_COLORS[d["level"]] if c == 3 else COL_COLORS[c]
        y, h = top[n], d["height"]
        ax.add_patch(FancyBboxPatch(
            (COL_X[c], y), NODE_W, h,
            boxstyle="round,pad=0,rounding_size=3.2",
            fc=color, ec="none", alpha=0.95, zorder=3, mutation_aspect=1.0))
        yc = y + h / 2.0
        ax.annotate(n, xy=(COL_X[c] + NODE_W, yc), xytext=(8, 3.4),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=FS_NODE_NAME, color=INK, annotation_clip=False)
        ax.annotate(f"sum={int(round(d['value']))}", xy=(COL_X[c] + NODE_W, yc),
                    xytext=(8, -4.2), textcoords="offset points", ha="left",
                    va="center", fontsize=FS_NODE_SUM, color=GRAY,
                    annotation_clip=False)

    # ---- titles
    ax.text(X_MIN + 6, 26, TITLE_A, fontsize=FS_TITLE, fontweight="bold",
            color=INK, ha="left", va="center")
    ax.text(X_MIN + 6, 55, SUBTITLE_A, fontsize=FS_SUBTITLE, color=GRAY,
            ha="left", va="center")
    for c in range(4):
        ax.text(COL_X[c] - 6, canvas_h - 24, COL_TITLES[c],
                fontsize=FS_COL_TITLE, fontweight="bold", color=TITLE_GRAY,
                ha="left", va="center")

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(canvas_h, 0)     # SVG-style y-down
    ax.axis("off")
    return canvas_h


# ======================= panel (b): split mini-Sankeys =======================
def mini_sankey(ax, x_src, x_sp, w_src, w_sp, y_top, y_bot, sources, splits,
                flows, src_colors, gap_src, gap_sp):
    """sources/splits: [(key, total)]; flows: {src_key: {split_key: n}}.
    Returns (source_boxes, split_boxes) as {key: (y_bottom, height)}."""
    H = y_top - y_bot
    s_src = (H - gap_src * (len(sources) - 1)) / sum(np.sqrt(t) for _, t in sources)
    split_stacked = {p: sum(np.sqrt(f.get(p, 0)) for _, f in ((k, flows[k]) for k in flows))
                     for p, _ in splits}
    s_sp = (H - gap_sp * (len(splits) - 1)) / max(sum(split_stacked.values()), 1e-9)
    s = min(s_src, s_sp)

    y = y_top
    src_box = {}
    for k, total in sources:
        h = np.sqrt(total) * s
        src_box[k] = (y - h, h)
        y -= h + gap_src
    y = y_top
    sp_box = {}
    for p, _total in splits:
        h = split_stacked[p] * s
        sp_box[p] = (y - h, h)
        y -= h + gap_sp

    off_s = {k: 0.0 for k, _ in sources}
    off_p = {p: 0.0 for p, _ in splits}
    for k, _t in sources:
        for p, _tp in splits:
            v = flows[k].get(p, 0)
            if v <= 0:
                continue
            h = np.sqrt(v) * s
            y0 = src_box[k][0] + src_box[k][1] - off_s[k] - h
            y1 = sp_box[p][0] + sp_box[p][1] - off_p[p] - h
            cubic_band(ax, x_src + w_src, y0, y0 + h, x_sp, y1, y1 + h,
                       src_colors[k], alpha=LINK_ALPHA, zorder=1)
            off_s[k] += h
            off_p[p] += h
    for k, (yb, h) in src_box.items():
        ax.add_patch(Rectangle((x_src, yb), w_src, h, fc=src_colors[k],
                               ec="none", alpha=0.95, zorder=3))
    for p, (yb, h) in sp_box.items():
        ax.add_patch(Rectangle((x_sp, yb), w_sp, h, fc=SPLIT_COLORS[p],
                               ec="none", alpha=0.95, zorder=3))
    return src_box, sp_box


def split_labels(ax, x_text, sp_box, totals, uses):
    for p, (yb, h) in sp_box.items():
        yc = yb + h / 2.0
        ax.annotate(f"{p} · {totals[p]:,}", xy=(x_text, yc), xytext=(7, 3.4),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=FS_B_NAME, fontweight="bold",
                    color=SPLIT_COLORS[p], annotation_clip=False)
        ax.annotate(uses[p], xy=(x_text, yc), xytext=(7, -4.2),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=FS_B_COUNT, color=GRAY, annotation_clip=False)


def draw_panel_b(ax):
    X0, X1, Y0, Y1 = -1.9, 12.35, -3.0, 101.0
    y_top, y_bot = 84.0, 6.0

    # ---- left: alignment corpus cache
    x_src1, x_sp1, w1 = 0.45, 2.05, 0.34
    src_b, sp_b = mini_sankey(
        ax, x_src1, x_sp1, w1, w1, y_top, y_bot,
        sources=[("cache", cache_total)],
        splits=[(s, cache[s]) for s in ("train", "val", "test")],
        flows={"cache": dict(cache)},
        src_colors={"cache": CACHE_COLOR},
        gap_src=0.0, gap_sp=8.0)
    yb, h = src_b["cache"]
    yc = yb + h / 2
    for dy, txt, fs, col in ((8.5, "teacher-attention cache", FS_B_NAME, INK),
                             (0.0, f"n = {cache_total:,}", FS_B_COUNT, GRAY),
                             (-8.5, "15 heads · fp16 · 8.8 TiB", FS_B_COUNT, GRAY)):
        ax.annotate(txt, xy=(x_src1 - 0.14, yc), xytext=(0, dy),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=fs, color=col, annotation_clip=False)
    split_labels(ax, x_sp1 + w1 + 0.12, sp_b, cache, cache_use)
    ax.text(x_src1, 93, "alignment corpus — teacher-attention cache split",
            fontsize=FS_B_TITLE, fontweight="bold", color=INK, ha="left",
            va="center")

    # ---- right: six downstream evaluation tasks
    x_src2, x_sp2, w2 = 7.45, 9.45, 0.30
    tkeys = list(tasks)
    src_b2, sp_b2 = mini_sankey(
        ax, x_src2, x_sp2, w2, w2, y_top, y_bot,
        sources=[(k, sum(tasks[k])) for k in tkeys],
        splits=[(s, task_splits[s]) for s in ("train", "val", "test")],
        flows={k: {s: tasks[k][j] for j, s in enumerate(("train", "val", "test"))}
               for k in tkeys},
        src_colors={k: TASK_COLOR for k in tkeys},
        gap_src=7.2, gap_sp=8.0)
    for k in tkeys:
        yb, h = src_b2[k]
        yc = yb + h / 2
        ax.annotate(k, xy=(x_src2 - 0.14, yc), xytext=(0, 3.2),
                    textcoords="offset points", ha="right", va="center",
                    fontsize=FS_B_NAME, color=INK, annotation_clip=False)
        ax.annotate(f"{sum(tasks[k]):,}", xy=(x_src2 - 0.14, yc),
                    xytext=(0, -4.4), textcoords="offset points", ha="right",
                    va="center", fontsize=FS_B_COUNT, color=GRAY,
                    annotation_clip=False)
    split_labels(ax, x_sp2 + w2 + 0.12, sp_b2, task_splits, task_use)
    ax.text(x_src2, 93, "six downstream tasks — native splits",
            fontsize=FS_B_TITLE, fontweight="bold", color=INK, ha="left",
            va="center")

    ax.text(-1.55, -1.6, "band width ∝ √n  (counts span four orders of magnitude)",
            fontsize=6.4, ha="left", va="center", style="italic", color=TITLE_GRAY)
    ax.set_xlim(X0, X1)
    ax.set_ylim(Y0, Y1)
    ax.axis("off")


# --------------------------- self-check: text overlaps ----------------------
def check_layout(fig, axes):
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    problems = []
    for tag, ax in axes:
        ab = ax.get_window_extent(r)
        items = []
        for t in ax.texts:
            s = t.get_text().strip()
            if not s or (len(s) == 1 and t.get_fontweight() == "bold"):
                continue            # skip panel letters (intentionally outside)
            items.append((s, t.get_window_extent(r)))
        for s, bb in items:
            if bb.x1 > ab.x1 + 2 or bb.x0 < ab.x0 - 2:
                problems.append(f"[{tag}] text outside axes: {s[:44]!r} "
                                f"(left {ab.x0 - bb.x0:+.1f}px, right {bb.x1 - ab.x1:+.1f}px)")
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                s1, b1 = items[i]
                s2, b2 = items[j]
                ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
                oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
                if ox > 0.5 and oy > 0.5:
                    problems.append(f"[{tag}] overlap {ox:.1f}x{oy:.1f}px: "
                                    f"{s1[:34]!r} x {s2[:34]!r}")
    return problems


# --------------------------------- assemble ---------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=default_data_root(),
                    help="root of the downloaded HF dataset "
                         "(default: $RNA_HEAD_DISTILL_DATA)")
    ap.add_argument("--outdir", type=Path, default=Path("./reproduced_figures"),
                    help="output directory for figS1_dataflow.{pdf,png}")
    args = ap.parse_args()
    OUT = args.outdir
    OUT.mkdir(parents=True, exist_ok=True)

    nodes_df = pd.read_csv(args.data_root / NODES_REL)

    # panel (a) canvas height from the column stacking
    col_block = {}
    for c in range(4):
        rows = nodes_df[nodes_df["column"] == c]
        y = TOP_H
        for _, r in rows.iterrows():
            y += max(float(r["height"]), MIN_PITCH) + COL_GAP
        col_block[c] = y - COL_GAP - TOP_H
    canvas_h = TOP_H + max(col_block.values()) + BOT_H

    FIG_W = 7.2
    ax_a_w_in = FIG_W * 0.994
    px_per_in = (X_MAX - X_MIN) / ax_a_w_in
    ax_a_h_in = canvas_h / px_per_in
    PANELB_H = 1.95
    TOP_M, MID_M, BOT_M = 0.05, 0.24, 0.05
    FIG_H = TOP_M + ax_a_h_in + MID_M + PANELB_H + BOT_M

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax_a = fig.add_axes([0.003, (BOT_M + PANELB_H + MID_M) / FIG_H, 0.994,
                         ax_a_h_in / FIG_H])
    ax_b = fig.add_axes([0.015, BOT_M / FIG_H + 0.004, 0.97,
                         (PANELB_H - 0.02) / FIG_H])

    draw_panel_a(ax_a, args.data_root)
    draw_panel_b(ax_b)
    panel_label(ax_a, "a", x=-0.006, y=1.035)
    panel_label(ax_b, "b", x=-0.014, y=1.10)

    problems = check_layout(fig, [(("a"), ax_a), (("b"), ax_b)])
    for p in problems:
        print("WARN:", p)
    if not problems:
        print("layout check: no text overlaps, all text inside axes")

    fig.savefig(OUT / "figS1_dataflow.png")
    fig.savefig(OUT / "figS1_dataflow.pdf")
    print(f"figure size: {FIG_W:.2f} x {FIG_H:.2f} in; panel (a) canvas "
          f"{canvas_h:.0f}px @ {px_per_in:.1f}px/in")
    print("wrote", OUT / "figS1_dataflow.pdf")


if __name__ == "__main__":
    main()
