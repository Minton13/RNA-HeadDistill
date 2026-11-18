from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT.parent / 'paper'
OUT = ROOT / 'figures'
OUT.mkdir(parents=True, exist_ok=True)

INK = '#1A1A1A'
GRAY = '#666666'
CARD_EC = '#9AA2B0'
ROUTE = {6: '#4C9ED9', 8: '#58B368', 12: '#8E6FBF'}
TINT = {6: '#E3EEF8', 8: '#E6F3E8', 12: '#EFE9F6'}
MID = {6: '#BCD7EE', 8: '#C2E2C8', 12: '#D9CCEA'}
YELLOW_FC, YELLOW_EC = '#FDF6DC', '#CDB15A'
TAN_FC, TAN_EC = '#F7F0E1', '#D8C9A3'
PANEL_FC = '#F7F9FB'
METHOD = {
    'P0': '#9AA2B0', 'P1': '#D65F5F', 'P4': '#4C9ED9',
    'P4g': '#58B368', 'A8_P4g': '#8E6FBF', 'P5': '#B79A67',
    'P6': '#777777',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.2,
    'axes.titlesize': 7.8, 'axes.labelsize': 7.0,
    'axes.linewidth': 0.75, 'xtick.labelsize': 6.3, 'ytick.labelsize': 6.3,
    'legend.fontsize': 6.2, 'figure.dpi': 120, 'savefig.dpi': 320,
    'savefig.bbox': None, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.spines.top': False, 'axes.spines.right': False,
    'text.color': INK, 'axes.labelcolor': INK, 'xtick.color': INK,
    'ytick.color': INK,
})


def card(fig, rect, fc=PANEL_FC, ec=CARD_EC, lw=0.9, z=-5):
    p = Rectangle((rect[0], rect[1]), rect[2], rect[3], transform=fig.transFigure,
                  facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z)
    fig.add_artist(p)
    return p


def header(fig, letter, title, y, x=0.022, size=10.2):
    fig.text(x, y, letter, fontsize=13.5, fontweight='bold', va='top', ha='left')
    fig.text(x + 0.024, y, title, fontsize=size, fontweight='bold', va='top', ha='left')


def clean(ax, grid=False):
    """CNS-style axes: only left/bottom spines and outward ticks; no grid."""
    ax.set_facecolor('white')
    ax.grid(False)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color('#222222')
    ax.spines[['left', 'bottom']].set_linewidth(0.7)
    ax.tick_params(direction='out', length=3.0, width=0.6, pad=2.5,
                   colors='#222222')


def save(fig, stem):
    fig.savefig(OUT / f'{stem}.pdf', facecolor='white')
    fig.savefig(OUT / f'{stem}.png', facecolor='white')
    plt.close(fig)
    print(OUT / f'{stem}.pdf')
