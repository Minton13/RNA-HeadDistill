"""Shared style and I/O for standalone, code-generated figure panels."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from figstyle import (ROOT, PAPER, INK, GRAY, CARD_EC, ROUTE, TINT, MID,
                      YELLOW_FC, YELLOW_EC, TAN_FC, TAN_EC, PANEL_FC,
                      METHOD, card, header, clean)

PANELS = ROOT / 'panels'
TRANSPARENT_PANELS = ROOT / 'panels_transparent'
DATA_OUT = PANELS / 'recovered_data'
PANELS.mkdir(parents=True, exist_ok=True)
TRANSPARENT_PANELS.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'Arial', 'font.size': 9.0,
    'axes.titlesize': 10.0, 'axes.labelsize': 15.0,
    'axes.linewidth': 0.7, 'axes.facecolor': 'white', 'figure.facecolor': 'white',
    'xtick.labelsize': 9.0, 'ytick.labelsize': 9.0,
    'xtick.major.width': 0.6, 'ytick.major.width': 0.6,
    'xtick.major.size': 3.0, 'ytick.major.size': 3.0,
    'legend.fontsize': 8.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'savefig.dpi': 400, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': False, 'savefig.facecolor': 'white',
})


def panel_figure(letter, title, subtitle='', figsize=(6.4, 5.2), route=6,
                 card_fc=None, card_ec=None):
    """Create a clean panel with one concise title and no explanatory note."""
    fig = plt.figure(figsize=figsize, facecolor='white')
    wide = figsize[0] >= 9.0
    fig.text(.030, .965, letter, fontsize=11.0 if wide else 10.5,
             fontweight='bold', va='top', ha='left', color=INK)
    # Offset scales with the actual canvas, keeping a1/a2 labels separated.
    title_x = .030 + (7.0 * len(letter) + 8.0) / 72.0 / figsize[0]
    fig.text(title_x, .965, title, fontsize=10.5 if wide else 9.5,
             fontweight='bold', va='top', ha='left', color=INK)
    return fig


def save_panel(fig, stem):
    """Save vector PDF, white-background PNG and transparent-background PNG."""
    # Enforce one physical size for every axis and colorbar title, including
    # labels created indirectly by seaborn colorbars.
    for axis in fig.axes:
        axis.xaxis.label.set_size(15.0)
        axis.yaxis.label.set_size(15.0)
    fig.savefig(PANELS / f'{stem}.pdf', facecolor='white')
    fig.savefig(PANELS / f'{stem}.png', facecolor='white')
    fig.savefig(TRANSPARENT_PANELS / f'{stem}.png', transparent=True,
                facecolor='none', edgecolor='none')
    plt.close(fig)
    print(PANELS / f'{stem}.pdf')


def digitize_matrix(path, n_rows, n_cols, cmap_name, vmin, vmax):
    """Recover a matrix by cell averaging and nearest-colormap inversion.

    Used only when the original numeric CSV is absent. The input is the raster
    embedded in the supplied manuscript PDF, not a manually estimated matrix.
    """
    image = Image.open(path).convert('RGB').resize(
        (n_cols, n_rows), Image.Resampling.BOX)
    rgb = np.asarray(image, dtype=float)[..., :3]
    colors = plt.get_cmap(cmap_name)(np.linspace(0, 1, 256))[:, :3] * 255
    delta = rgb[..., None, :] - colors[None, None, :, :]
    indices = np.argmin(np.sum(delta * delta, axis=-1), axis=-1)
    values = vmin + indices / 255.0 * (vmax - vmin)
    return values
