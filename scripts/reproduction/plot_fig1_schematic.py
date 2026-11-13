#!/usr/bin/env python
# Provenance: ported from paper/scripts/plot_fig1_schematic.py (RNA_LLM_v2
# monorepo) on 2026-11-13 for the public release.
# Changes vs. original:
#   - hard-coded cluster paths replaced by --data-root / --outdir;
#   - real-data thumbnails are read from the published
#     <data-root>/fig1_real_data/panel_a_probe_atlas.npz and
#     panel_c_attention.npz (heatmap matrices + teacher/student 32x32
#     attention maps + sequence) instead of the 8.8 TiB safetensors cache and
#     the student checkpoint;
#   - graceful fallback to synthetic glyphs is preserved when the npz files
#     are absent.
"""Figure 1 — framework schematic, redrawn in the clean "AI figure" style of
the group-meeting slides (rounded light panels, glyph icons, dashed group
boxes, solid data-flow / dashed supervision arrows).

  a  Supervised discovery of functional attention heads (4-stage flow)
  b  One-time asynchronous teacher cache (teachers retired after one pass)
  c  Routed head-to-head distillation on a 12-layer student with
     depth-routed gradients

Coordinates are in approximate 0.01-inch units per panel so text and glyphs
keep their intended proportions.

Real-data thumbnails (2026-09-16 revision; public-release npz packaging):
  - panel a stage 3 embeds two real per-head probe-score heatmaps (layer x
    head) taken from the head-screening atlas, echoing Fig. 2a;
  - panel c embeds real cached teacher attention maps and the real trained
    student predictions for one cached 32-nt sequence, one rank-1 head per
    route;
  - panel a stage 1 draws a real stem-loop (UUCG tetraloop hairpin) as an
    arc diagram instead of decorative arcs.
Every loader degrades gracefully to the original synthetic glyphs when the
underlying data are unavailable, so the script runs on any machine.

Run: python scripts/reproduction/plot_fig1_schematic.py \
       --data-root /path/to/dataset-rna-head-distill --outdir ./reproduced_figures
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from _paper_style import ROUTE_COLORS, plt  # noqa: E402

from matplotlib.colors import Normalize, PowerNorm  # noqa: E402
from matplotlib.patches import (Arc, Circle, Ellipse, FancyArrowPatch,  # noqa: E402
                                FancyBboxPatch, Rectangle)

NT_COLORS = {"A": "#64B964", "U": "#E6B84C", "G": "#4C9ED9", "C": "#E06565"}
INK = "#222222"
PANEL_EC = "#b8bec8"
LEVEL_COLORS = {"seq": "#4C9ED9", "tok": "#E6B84C", "pair": "#E06565"}
ATTN_CMAP = "magma"                       # student and teacher maps alike
ATTN_NORM = PowerNorm(gamma=0.4, vmin=0.0, vmax=1.0)

# Populated by main(); consumed by the drawing code below.
PROBE_MATS = None
REAL_ATTN = None
OUT = Path("./reproduced_figures")


def default_data_root() -> Path:
    env = os.environ.get("RNA_HEAD_DISTILL_DATA")
    return Path(env) if env else Path("./dataset-rna-head-distill")


# ------------------------------------------------------- real-data loaders

def load_probe_score_matrices(data_root: Path):
    """Panel a stage 3: real per-head probe-score matrices (layer x head).

    Source: <data-root>/fig1_real_data/panel_a_probe_atlas.npz with keys
    mat_k / mat_raw_k (layer x head, NaN filled with lo / NaN preserved),
    label_k, tick_lo_k, tick_hi_k, gamma_k (NaN = plain Normalize).
    Returns a list of dicts with the matrix, a Normalize/PowerNorm and
    end-point tick labels, or None on any failure.
    """
    try:
        z = np.load(data_root / "fig1_real_data" / "panel_a_probe_atlas.npz",
                    allow_pickle=True)
        out = []
        k = 0
        while f"mat_{k}" in z.files:
            mat = z[f"mat_{k}"].astype(np.float64)
            label = str(z[f"label_{k}"])
            tick_lo, tick_hi = str(z[f"tick_lo_{k}"]), str(z[f"tick_hi_{k}"])
            gamma = float(z[f"gamma_{k}"])
            lo, hi = float(tick_lo), float(tick_hi)
            if np.isnan(gamma):
                norm = Normalize(vmin=lo, vmax=hi)
            else:
                norm = PowerNorm(gamma=gamma, vmin=0.0, vmax=hi)
            out.append({"mat": mat, "label": label, "norm": norm,
                        "tick_lo": tick_lo, "tick_hi": tick_hi})
            k += 1
        return out or None
    except Exception:
        return None


def load_real_attention_maps(data_root: Path):
    """Panel c: real teacher attention maps + real student predictions.

    Source: <data-root>/fig1_real_data/panel_c_attention.npz with keys
    teacher_L6/L8/L12, student_L6/L8/L12 ([32, 32] fp32, clip 0..1),
    sequence, head_id_L6/L8/L12.  Returns
    {route: {"teacher": LxL, "student": LxL | None}} or None.
    """
    try:
        z = np.load(data_root / "fig1_real_data" / "panel_c_attention.npz",
                    allow_pickle=True)
        res = {}
        for d in (6, 8, 12):
            teacher = z[f"teacher_L{d}"].astype(np.float32)
            student = (z[f"student_L{d}"].astype(np.float32)
                       if f"student_L{d}" in z.files else None)
            res[d] = {"teacher": np.clip(teacher, 0, 1),
                      "student": np.clip(student, 0, 1) if student is not None
                      else None}
        return res
    except Exception:
        return None


# ---------------------------------------------------------------- helpers


def rbox(ax, x, y, w, h, fc="white", ec=PANEL_EC, lw=0.9, r=3.0, alpha=1.0,
         z=2, ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, alpha=alpha, zorder=z, linestyle=ls)
    ax.add_patch(p)
    return p


def arrow(ax, x0, y0, x1, y1, color=INK, lw=1.3, ls="-", z=6, ms=9):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=ms, color=color, lw=lw, ls=ls,
                                 zorder=z))


def attn_thumb(ax, x, y, s, seed, blocky=False, cmap="magma", z=4, frame=INK):
    rng = np.random.default_rng(seed)
    m = rng.beta(0.4, 0.9, size=(14, 14))
    if blocky:
        m[2:6, 2:6] += 1.3
        m[9:13, 1:5] += 1.1
    m += np.eye(14) * 0.8
    m /= m.max()
    ax.imshow(m, extent=(x, x + s, y, y + s), cmap=cmap, vmin=0, vmax=1.15,
              zorder=z, interpolation="nearest")
    ax.add_patch(Rectangle((x, y), s, s, fill=False, ec=frame, lw=0.6,
                           zorder=z + 1))


def map_thumb(ax, x, y, w, h, mat, cmap, norm, z=4, frame=INK):
    """Real LxL attention/probe map on the schematic axes (row 0 at top)."""
    ax.imshow(mat, extent=(x, x + w, y, y + h), cmap=cmap, norm=norm,
              aspect="auto", zorder=z, interpolation="nearest")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=frame, lw=0.6,
                           zorder=z + 1))


def mini_cbar_v(ax, x, y, w, h, cmap, norm, tick_lo, tick_hi, fs=5.5):
    """Minimal vertical colorbar drawn on the schematic axes."""
    grad = np.linspace(0, 1, 256).reshape(-1, 1)
    ax.imshow(grad, extent=(x, x + w, y, y + h), cmap=cmap, norm=norm,
              aspect="auto", origin="lower", zorder=5, interpolation="nearest")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=INK, lw=0.5, zorder=6))
    ax.text(x + w + 1.6, y + h, tick_hi, fontsize=fs, ha="left", va="top",
            color="#555555", zorder=6)
    ax.text(x + w + 1.6, y, tick_lo, fontsize=fs, ha="left", va="bottom",
            color="#555555", zorder=6)


def mini_cbar_h(ax, x, y, w, h, cmap, norm, tick_lo, tick_hi, fs=5.5):
    """Minimal horizontal colorbar drawn on the schematic axes."""
    grad = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(grad, extent=(x, x + w, y, y + h), cmap=cmap, norm=norm,
              aspect="auto", zorder=5, interpolation="nearest")
    ax.add_patch(Rectangle((x, y), w, h, fill=False, ec=INK, lw=0.5, zorder=6))
    ax.text(x, y - 2.0, tick_lo, fontsize=fs, ha="center", va="top",
            color="#555555", zorder=6)
    ax.text(x + w, y - 2.0, tick_hi, fontsize=fs, ha="center", va="top",
            color="#555555", zorder=6)


def rna_arc_icon(ax, x, y, w, fs=7.0):
    """Real stem-loop as an arc diagram: a UUCG-tetraloop hairpin (5'->3'
    left to right) with 5 nested arcs over the stem and none over the loop."""
    seq = "GGGACUUCGGUCCC"          # 5-bp GC-rich stem + UUCG tetraloop
    pairs = [(0, 13), (1, 12), (2, 11), (3, 10), (4, 9)]
    dx = w / (len(seq) - 1)
    for i, ch in enumerate(seq):
        ax.text(x + i * dx, y, ch, ha="center", va="center", fontsize=fs,
                fontweight="bold", color=NT_COLORS[ch], zorder=4)
    for i, j in pairs:
        ww = (j - i) * dx
        ax.add_patch(Arc((x + (i + j) * dx / 2, y + 2.5), ww, 0.58 * ww,
                         theta1=0, theta2=180, lw=1.2, color="#8a5ca8",
                         zorder=3))


def model_stack_glyph(ax, x, y, w, h, n_layers, z=2):
    """A mini transformer stack: n_layers thin blue boxes."""
    bh = h / n_layers
    for i in range(n_layers):
        shade = 0.30 + 0.5 * i / max(n_layers - 1, 1)
        ax.add_patch(Rectangle((x, y + i * bh), w, bh * 0.80,
                     fc=plt.cm.Blues(shade), ec="white", lw=0.3, zorder=z))


def cache_cylinder(ax, cx, y, w, h, fc="#f2e2c4", ec=INK, z=3):
    ry = w * 0.16
    ax.add_patch(Rectangle((cx - w / 2, y), w, h, fc=fc, ec="none", zorder=z))
    ax.add_patch(Ellipse((cx, y + h), w, 2 * ry, fc="#f9e9c8", ec=ec, lw=0.9,
                         zorder=z + 1))
    ax.add_patch(Arc((cx, y), w, 2 * ry, theta1=180, theta2=360, color=ec,
                     lw=0.9, zorder=z + 1))
    for side in (-1, 1):
        ax.plot([cx + side * w / 2] * 2, [y, y + h], color=ec, lw=0.9,
                zorder=z + 1)


# ================================================================ figure
def build_figure() -> None:
    fig = plt.figure(figsize=(7.2, 7.5))

    # ============================================== panel a: discovery flow
    # axes is 7.20in x 2.55in -> coordinate system 720 x 255 (square units)
    axa = fig.add_axes([0.000, 0.660, 1.000, 0.340])
    axa.set_xlim(0, 720); axa.set_ylim(0, 255); axa.axis("off")
    axa.text(4, 252, "a", fontsize=13, fontweight="bold", va="top")
    axa.text(24, 252, "Supervised discovery of functional attention heads",
             fontsize=10, fontweight="bold", va="top")

    bw_a, gap_a = 160, 25
    boxes = []
    x = 2
    for _ in range(4):
        boxes.append((x, x + bw_a))
        x += bw_a + gap_a
    for x0, x1 in boxes:
        rbox(axa, x0, 12, x1 - x0, 198, fc="#f7f8fa", ec=PANEL_EC, lw=1.0)
    for (x0, x1), (nx0, _) in zip(boxes[:-1], boxes[1:]):
        arrow(axa, x1 + 4, 108, nx0 - 4, 108, lw=1.6, ms=12)

    # ---- a1: RNA input
    x0, x1 = boxes[0]
    cx = (x0 + x1) / 2
    axa.text(cx, 196, "RNA sequences", fontsize=7.6, fontweight="bold",
             ha="center")
    rna_arc_icon(axa, cx - 62, 118, 124, fs=7.2)
    axa.text(cx, 76, "2,114,289 pretraining\nsequences", fontsize=6.2,
             ha="center", va="top", color="#444444")
    axa.text(cx, 44, "length-bucketed batches", fontsize=5.6, ha="center",
             style="italic", color="#777777")

    # ---- a2: 26 frozen foundation models
    x0, x1 = boxes[1]
    cx = (x0 + x1) / 2
    axa.text(cx, 196, "26 frozen foundation models", fontsize=7.6,
             fontweight="bold", ha="center")
    for dx, nl in [(-56, 6), (-30, 9), (-4, 12)]:
        model_stack_glyph(axa, cx + dx, 92, 22, 66, nl)
    attn_thumb(axa, cx + 26, 98, 30, seed=3, blocky=True)
    axa.text(cx, 82, "RiNALMo · RNA-FM · SpliceBERT · NT\n"
             "PlantRNA-FM · UTR-LM · mRNA-FM · …", fontsize=5.9, ha="center",
             va="top", color="#444444")
    axa.text(cx, 44, "attention maps $A^{(l,h)}$ probed head by head",
             fontsize=5.8, ha="center", style="italic", color="#777777")

    # ---- a3: per-head probing on 15 tasks — real probe-score heatmaps
    x0, x1 = boxes[2]
    cx = (x0 + x1) / 2
    axa.text(cx, 196, "Per-head probing · 15 tasks", fontsize=7.6,
             fontweight="bold", ha="center")
    if PROBE_MATS:
        m1, m2 = PROBE_MATS
        # tall 33 layer x 20 head matrix (secondary structure x RiNALMo-giga)
        map_thumb(axa, x0 + 14, 88, 44, 76, m1["mat"], "viridis", m1["norm"])
        mini_cbar_v(axa, x0 + 60.5, 88, 3.2, 76, "viridis", m1["norm"],
                    m1["tick_lo"], m1["tick_hi"])
        axa.text(x0 + 36, 80, m1["label"], fontsize=5.5, ha="center", va="top",
                 color="#333333")
        # wide 6 layer x 16 head matrix (splice site x SpliceBERT)
        map_thumb(axa, x0 + 80, 118, 56, 30, m2["mat"], "viridis", m2["norm"])
        mini_cbar_v(axa, x0 + 139, 118, 3.2, 30, "viridis", m2["norm"],
                    m2["tick_lo"], m2["tick_hi"])
        axa.text(x0 + 108, 110, m2["label"], fontsize=5.5, ha="center",
                 va="top", color="#333333")
    else:  # graceful fallback: synthetic score matrices, generic labels
        for k, (hx, hy, hw, hh, shp) in enumerate([(x0 + 14, 88, 44, 76, (12, 9)),
                                                   (x0 + 80, 118, 56, 30, (6, 14))]):
            rng = np.random.default_rng(11 + k)
            m = rng.beta(0.5, 1.2, size=shp)
            m[-2:, 1:4] += 1.2
            axa.imshow(m / m.max(), extent=(hx, hx + hw, hy, hy + hh),
                       cmap="viridis", aspect="auto", interpolation="nearest",
                       zorder=4)
            axa.add_patch(Rectangle((hx, hy), hw, hh, fill=False, ec=INK, lw=0.6,
                                    zorder=5))
        axa.text(x0 + 36, 80, "task × model atlas\n(schematic)", fontsize=5.5,
                 ha="center", va="top", color="#333333")
    axa.text(cx, 64, "per-head probe score · layer × head", fontsize=5.6,
             ha="center", va="top", style="italic", color="#777777")
    axa.text(cx, 44, "84,066 head × task probe records", fontsize=5.9,
             ha="center", style="italic", color="#777777")

    # ---- a4: rank & select
    x0, x1 = boxes[3]
    cx = (x0 + x1) / 2
    axa.text(cx, 196, "Rank and select", fontsize=7.6, fontweight="bold",
             ha="center")
    for i, d in enumerate([6, 8, 12]):
        yy = 140 - i * 32
        rbox(axa, x0 + 10, yy, 34, 22, fc=ROUTE_COLORS[d], ec=INK, lw=0.7, r=4.0)
        axa.text(x0 + 27, yy + 11, f"L{d}", fontsize=6.4, ha="center",
                 va="center", fontweight="bold", color="white", zorder=6)
        axa.text(x0 + 50, yy + 11, "route · top-5 heads", fontsize=6.2,
                 va="center", color="#333333")
    axa.text(cx, 44, "15 teacher heads · identity preserved\n"
             "(model · layer · head) — no averaging", fontsize=5.9, ha="center",
             va="top", style="italic", color="#777777")

    # ============================================== panel b: async cache
    # axes is 2.13in x 4.50in -> 213 x 450 units
    axb = fig.add_axes([0.004, 0.020, 0.296, 0.600])
    axb.set_xlim(0, 213); axb.set_ylim(0, 450); axb.axis("off")
    axb.text(2, 446, "b", fontsize=13, fontweight="bold", va="top")
    axb.text(22, 446, "One-time asynchronous\n      teacher cache", fontsize=9.5,
             fontweight="bold", va="top")

    teachers = ["SpliceBERT", "PlantRNA-FM", "NT-v2-500M", "RiNALMo", "RNA-FM",
                "UTR-LM", "ModernGENA", "mRNA-FM", "SpliceBERT-MSA"]
    axb.text(106, 396, "9 teacher models (frozen)", fontsize=6.8, ha="center",
             fontweight="bold")
    for i, name in enumerate(teachers):
        r, c = divmod(i, 3)
        xx = 14 + c * 64
        yy = 348 - r * 26
        rbox(axb, xx, yy, 58, 20, fc="#eef3f9", ec="#8fa6c2", lw=0.7, r=3.0)
        axb.text(xx + 29, yy + 10, name, fontsize=4.8, ha="center", va="center",
                 color="#22354a")
    arrow(axb, 106, 266, 106, 236, lw=1.5, ms=11)
    axb.text(113, 251, "one forward pass per sequence\n2.1M sequences · 15 heads",
             fontsize=5.6, ha="left", va="center", color="#444444")

    cache_cylinder(axb, 106, 128, 92, 62)
    axb.text(106, 216, "slim teacher-attention cache", fontsize=6.6, ha="center",
             fontweight="bold")
    axb.text(106, 162, "8.8 TiB · 1,033 shards · fp16", fontsize=6.0, ha="center",
             zorder=6)
    axb.text(106, 144, "15 heads × full $L\\times L$ maps", fontsize=6.0,
             ha="center", zorder=6)

    rbox(axb, 12, 18, 189, 66, fc="#fdf6ec", ec="#d8a23c", lw=1.0, r=4.0, ls="--")
    axb.text(106, 72, "asynchronous", fontsize=6.8, ha="center",
             fontweight="bold", color="#8a5c00")
    axb.text(106, 54, "cache built once — teachers never run again",
             fontsize=5.8, ha="center", color="#5c4600")
    axb.text(106, 32, "every student experiment reads cached maps:\n"
             "0 teacher FLOPs per training step", fontsize=5.8, ha="center",
             va="center", color="#5c4600")

    # ============================================== panel c: routed student
    # axes is 4.95in x 4.50in -> 495 x 450 units
    axc = fig.add_axes([0.308, 0.020, 0.688, 0.600])
    axc.set_xlim(-118, 385); axc.set_ylim(0, 450); axc.axis("off")
    axc.text(-118, 446, "c", fontsize=13, fontweight="bold", va="top")
    axc.text(-96, 446, "Routed head-to-head distillation", fontsize=9.5,
             fontweight="bold", va="top")

    # ---- student stack (layers 1..12 bottom to top)
    bx, bw = 16, 30
    bh, gap = 24.0, 4.5
    y0 = 86
    for i in range(1, 13):
        y = y0 + (i - 1) * (bh + gap)
        axc.add_patch(Rectangle((bx, y), bw, bh,
                      fc=plt.cm.Blues(0.32 + 0.48 * i / 12), ec="white", lw=0.5,
                      zorder=3))
        axc.text(bx + bw / 2, y + bh / 2, str(i), fontsize=5.6, ha="center",
                 va="center", color="white" if i > 5 else INK, zorder=4)
    axc.text(bx + bw / 2, y0 - 12, "RiNALMo-micro student", fontsize=6.4,
             ha="center", va="top", fontweight="bold")
    axc.text(bx + bw / 2, y0 - 26, "12 Transformer layers", fontsize=5.6,
             ha="center", va="top", color="#555555")
    # input arrow into layer 1 (horizontal, from the left, under grad labels)
    yl1 = y0 + bh / 2
    axc.text(bx - 14, yl1, "RNA sequence", fontsize=6.0, style="italic",
             ha="right", va="center")
    arrow(axc, bx - 12, yl1, bx - 1, yl1, lw=1.1, ms=9)

    # ---- gradient-depth bands + labels (horizontal, left of stack)
    bands = [(1, 6, "$\\nabla$ from\nL6+L8+L12", 0.42),
             (7, 8, "$\\nabla$ from\nL8+L12", 0.26),
             (9, 12, "$\\nabla$ from\nL12", 0.14)]
    for lo, hi, lab, al in bands:
        yb = y0 + (lo - 1) * (bh + gap) - 2
        ht = (hi - lo + 1) * (bh + gap) - gap + 4
        axc.add_patch(Rectangle((bx - 7, yb), 4, ht, fc="#2ca02c", alpha=al,
                                ec="none", zorder=2))
        axc.text(bx - 11, yb + ht / 2, lab, fontsize=5.6, ha="right",
                 va="center", color="#1a5c1a")

    # ---- three route rows
    routes = [6, 8, 12]
    for d in routes:
        yl = y0 + (d - 1) * (bh + gap) + bh / 2
        c = ROUTE_COLORS[d]
        axc.add_patch(Circle((bx + bw + 4, yl), 3.4, fc=c, ec=INK, lw=0.7,
                             zorder=6))
        axc.text(bx + bw + 4, yl + 10.5, f"L{d}", fontsize=6.4, color=c,
                 fontweight="bold", ha="center", zorder=6)
        arrow(axc, bx + bw + 8, yl, 74, yl, color=c, lw=1.4, ms=9)
        # native head chip
        rbox(axc, 74, yl - 13, 52, 26, fc="white", ec=c, lw=1.2, r=3.0)
        axc.text(100, yl + 4.5, "native head", fontsize=5.6, ha="center",
                 va="center")
        axc.text(100, yl - 5.5, "$O_{d,s}$ [$L\\times d_h$]", fontsize=5.5,
                 ha="center", va="center", color="#555555")
        arrow(axc, 127, yl, 140, yl, color=c, lw=1.2, ms=8)
        # adapter box
        rbox(axc, 140, yl - 16, 74, 32, fc="#fdfdfd", ec=c, lw=1.5, r=3.5)
        axc.text(177, yl + 6.5, "alignment adapter", fontsize=5.6, ha="center",
                 va="center", fontweight="bold")
        axc.text(177, yl - 6.0, "1 Transformer block\n≈35k params", fontsize=5.5,
                 ha="center", va="center", color="#555555")
        arrow(axc, 215, yl, 226, yl, color=c, lw=1.2, ms=8)
        real = REAL_ATTN.get(d) if REAL_ATTN else None
        # predicted map (real student forward when available)
        if real is not None and real.get("student") is not None:
            map_thumb(axc, 228, yl - 15, 30, 30, real["student"], ATTN_CMAP,
                      ATTN_NORM)
        else:
            attn_thumb(axc, 228, yl - 15, 30, seed=d, blocky=(d == 12),
                       cmap=ATTN_CMAP)
        axc.text(243, yl - 20, "student $\\hat{A}_{d,t}$", fontsize=5.5,
                 ha="center", va="top")
        # loss link to cached teacher map (labeled once, on the top route row)
        axc.plot([260, 272], [yl, yl], color="#a00000", lw=1.0, ls=(0, (2, 2)),
                 zorder=5)
        if d == 12:
            axc.text(266, yl + 18, "per-head\nloss", fontsize=5.5, ha="center",
                     va="bottom", color="#a00000")
        if real is not None and real.get("teacher") is not None:
            map_thumb(axc, 274, yl - 15, 30, 30, real["teacher"], ATTN_CMAP,
                      ATTN_NORM)
        else:
            attn_thumb(axc, 274, yl - 15, 30, seed=d + 40, blocky=(d == 12),
                       cmap=ATTN_CMAP)
        axc.text(289, yl - 20, "teacher $A_{d,t}$", fontsize=5.5, ha="center",
                 va="top")

    # shared attention colorbar under the bottom (L6) route-row thumbnails
    if REAL_ATTN is not None:
        yl6 = y0 + 5 * (bh + gap) + bh / 2
        mini_cbar_h(axc, 232, yl6 - 48, 68, 5.5, ATTN_CMAP, ATTN_NORM, "0", "1")
        axc.text(266, yl6 - 39.5, "attention weight", fontsize=5.6, ha="center",
                 va="bottom", color="#555555")

    # dashed feed from cache (panel b) into teacher thumbs
    fx = 318
    y_lo = y0 + 5 * (bh + gap) + bh / 2
    y_hi = y0 + 11 * (bh + gap) + bh / 2
    axc.plot([fx, fx], [y_lo, y_hi], color="#8a5c00", lw=1.1, ls=(0, (4, 3)),
             zorder=1)
    for d in routes:
        yl = y0 + (d - 1) * (bh + gap) + bh / 2
        arrow(axc, fx, yl, 304, yl, color="#8a5c00", lw=1.0, ls=(0, (4, 3)), ms=7)
    axc.text(fx + 6, (y_lo + y_hi) / 2, "cached maps\nfrom (b)", fontsize=5.6,
             color="#8a5c00", ha="left", va="center")

    # ---- loss box at bottom (includes the task-depth mapping line)
    rbox(axc, 72, 4, 296, 58, fc="#f7f7f7", ec="#999999", lw=0.9, r=3.5)
    axc.text(82, 52, "one predicted student head per preserved teacher head",
             fontsize=6.0, fontweight="bold", va="center")
    axc.text(82, 38, "$\\mathcal{L}_{d,t} = \\overline{\\mathrm{KL}}"
             "\\left(A_{d,t}\\,\\|\\,\\hat{A}_{d,t}\\right) + \\mathrm{MSE}"
             " + 0.1\\,(1 - \\cos)$     ·     "
             "$\\mathcal{L} = \\bar{\\mathcal{L}}_6 + \\bar{\\mathcal{L}}_8"
             " + \\bar{\\mathcal{L}}_{12}$",
             fontsize=6.4, va="center", color="#a00000")
    axc.text(82, 24, "task routing — L6: splice · RBP · ncRNA;  "
             "L8: chemical reactivity;  L12: secondary structure",
             fontsize=5.6, va="center", color="#333333")
    axc.text(82, 11, "full-$L^2$ matching on $[0,L_i]^2$ · no consensus, "
             "no head averaging", fontsize=5.6, va="center", style="italic",
             color="#555555")

    fig.savefig(OUT / "fig1_schematic.png", dpi=300)
    fig.savefig(OUT / "fig1_schematic.pdf")
    print("wrote", OUT / "fig1_schematic.pdf",
          "| probe atlas:", "real" if PROBE_MATS else "fallback",
          "| attention:", "real" if REAL_ATTN else "fallback",
          "| student:", "real" if (REAL_ATTN and REAL_ATTN[12]["student"] is not None)
          else "fallback")


def main() -> None:
    global PROBE_MATS, REAL_ATTN, OUT
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=default_data_root(),
                    help="root of the downloaded HF dataset "
                         "(default: $RNA_HEAD_DISTILL_DATA)")
    ap.add_argument("--outdir", type=Path, default=Path("./reproduced_figures"),
                    help="output directory for fig1_schematic.{pdf,png}")
    args = ap.parse_args()
    OUT = args.outdir
    OUT.mkdir(parents=True, exist_ok=True)
    PROBE_MATS = load_probe_score_matrices(args.data_root)
    REAL_ATTN = load_real_attention_maps(args.data_root)
    build_figure()


if __name__ == "__main__":
    main()
