# Provenance: ported from paper/scripts/_paper_style.py (RNA_LLM_v2 monorepo)
# on 2026-11-13 for the public release. Unmodified except for this header.
"""Shared publication style for all paper figures (v2, merged panels).

Import this module first in every plotting script; it sets rcParams once and
provides the shared color maps and the panel-letter helper.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "axes.linewidth": 0.8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ---- feature modes (frozen probes); identical across every figure
METHOD_COLORS = {
    "P0": "#4C72B0",      # vanilla — neutral blue
    "P1": "#DD8452",      # distilled hidden (damaged) — orange
    "P2": "#55A868",
    "P4": "#55A868",      # main method (hidden+aligned) — green
    "P4g": "#C44E52",     # gated repair — red
    "A8_P4g": "#8172B3",  # frozen-backbone repair — purple
    "P5": "#937860",      # teacher transplant control — brown
    "P6": "#999999",      # random-heads negative control — gray
}
METHOD_LABELS = {
    "P0": "P0 vanilla", "P1": "P1 distilled hidden", "P4": "P4 hidden+aligned",
    "P4g": "P4g gated", "A8_P4g": "A8-P4g frozen", "P5": "P5 teacher transplant",
    "P6": "P6 random heads",
}

# ---- route banks
ROUTE_COLORS = {6: "#4C72B0", 8: "#DD8452", 12: "#55A868"}

# ---- downstream tasks
TASK_ORDER = [
    "splice_donor_seqcls", "splice_acceptor_seqcls", "rbp_binding_seqcls",
    "ncRNA_family_seqcls", "chemical_reactivity_token", "secondary_structure_pair",
]
TASK_LABEL = {
    "splice_donor_seqcls": "Splice donor\n(L6)",
    "splice_acceptor_seqcls": "Splice acceptor\n(L6)",
    "rbp_binding_seqcls": "RBP binding\n(L6)",
    "ncRNA_family_seqcls": "ncRNA family\n(L6)",
    "chemical_reactivity_token": "Chemical react.\n(L8)",
    "secondary_structure_pair": "Secondary struct.\n(L12)",
}
TASK_LABEL_FLAT = {k: v.replace("\n", " ") for k, v in TASK_LABEL.items()}
TASK_METRIC = {
    "splice_donor_seqcls": ("auroc", "AUROC"),
    "splice_acceptor_seqcls": ("auroc", "AUROC"),
    "rbp_binding_seqcls": ("auroc", "AUROC"),
    "ncRNA_family_seqcls": ("auroc", "AUROC"),
    "chemical_reactivity_token": ("pearson", "Pearson r"),
    "secondary_structure_pair": ("top_l_precision", "Top-L precision"),
}

# ---- ablation
ABL_COLORS = {"main": "#d62728", "ablation": "#1f77b4", "frozen": "#8c8c8c",
              "matched": "#ffd166"}

# ---- systems
SYS_OFFLINE = "#d62728"
SYS_ONLINE = "#1f77b4"


def panel_label(ax, letter: str, x: float = -0.16, y: float = 1.08) -> None:
    """Bold lowercase panel letter at the top-left corner of an axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=12, fontweight="bold",
            va="top", ha="left", clip_on=False)


def despine(ax) -> None:
    ax.spines[["top", "right"]].set_visible(False)
