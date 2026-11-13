#!/usr/bin/env python
# Provenance: ported from paper/scripts/plot_si_pilot.py (RNA_LLM_v2
# monorepo) on 2026-11-13 for the public release.
# Changes vs. original: hard-coded cluster paths replaced by
# --data-root (pilot/pilot_mcc_heatmap_*.csv + pilot/pilot_decoder_curves.csv)
# and --outdir. The decoder curves are read from the published aggregated
# pilot_decoder_curves.csv (the raw per-run training logs of the legacy
# pilot project are NOT part of the data package — see REPRODUCTION.md).
"""SI pilot-study figures (Fig. S2 / Fig. S3).

Produces:
  figS_pilot_atlas.{pdf,png}
      Per-head mean MCC heatmaps (layer x head) for RiNALMo micro/mega/giga
      on bpRNA-90 secondary structure, scheme global_t=0.50, from the
      aggregated tables (pilot/pilot_mcc_heatmap_*.csv). Top-3 heads boxed.
  figS_pilot_decoders.{pdf,png}
      Validation-MCC training curves for the top-3-head U-Net vs transformer
      decoders (mega, giga), from the aggregated pilot/pilot_decoder_curves.csv
      (columns: epoch, val_mcc, val_f1, model, decoder; each block is the
      final/longest run attempt).

Layer/head indices are 1-based (legacy pilot-project convention).

Run: python scripts/reproduction/plot_si_pilot.py \
       --data-root /path/to/dataset-rna-head-distill --outdir ./reproduced_figures
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).parent))
import _paper_style as ps  # noqa: F401,E402  (sets rcParams)

MODELS = ["micro", "mega", "giga"]
TOP3 = {
    "mega": [(29, 11), (30, 4), (30, 13)],
    "giga": [(32, 2), (32, 14), (33, 13)],
}  # 1-based, from the pilot ranking; micro top-3 computed from the table


def default_data_root() -> Path:
    env = os.environ.get("RNA_HEAD_DISTILL_DATA")
    return Path(env) if env else Path("./dataset-rna-head-distill")


def plot_atlas(pilot_dir: Path, out: Path) -> None:
    grids = {}
    vmax = 0.0
    for m in MODELS:
        df = pd.read_csv(pilot_dir / f"pilot_mcc_heatmap_{m}.csv")
        piv = df.pivot(index="layer", columns="head", values="mcc_mean")
        grids[m] = piv
        vmax = max(vmax, float(np.nanmax(piv.values)))
    # micro top-3 from the aggregated table
    micro_top3 = [tuple(x) for x in pd.read_csv(pilot_dir / "pilot_mcc_heatmap_micro.csv")
                  .nlargest(3, "mcc_mean")[["layer", "head"]].values]
    TOP3["micro"] = micro_top3
    print("micro top-3 (1-based):", micro_top3)

    fig, axes = ps.plt.subplots(1, 3, figsize=(7.0, 2.9),
                                constrained_layout=True)
    im = None
    for ax, m, letter in zip(axes, MODELS, "abc"):
        piv = grids[m]
        im = ax.imshow(piv.values, cmap="magma", origin="upper",
                       aspect="equal", vmin=0.0, vmax=vmax,
                       extent=None, interpolation="nearest")
        n_layers, n_heads = piv.shape
        x_step = 2
        x_idx = np.arange(0, n_heads, x_step)
        ax.set_xticks(x_idx + 0.5)
        ax.set_xticklabels(piv.columns[x_idx], fontsize=6)
        y_step = 5 if n_layers > 20 else 2
        y_idx = np.arange(0, n_layers, y_step)
        if n_layers - 1 - y_idx[-1] >= 2:
            y_idx = np.append(y_idx, n_layers - 1)
        ax.set_yticks(y_idx + 0.5)
        ax.set_yticklabels(piv.index[y_idx], fontsize=6)
        ax.set_xlabel("head (1-based)")
        ax.set_ylabel("layer (1-based)")
        ax.set_title(f"RiNALMo-{m}", pad=3)
        ax.tick_params(length=0)
        for sp in ax.spines.values():
            sp.set_visible(False)
        for (layer, head) in TOP3[m]:
            ax.add_patch(Rectangle((head - 1, layer - 1), 1, 1,
                                   fill=False, edgecolor="#00e5ff",
                                   lw=1.4))
        ps.panel_label(ax, letter, x=-0.10, y=1.06)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015, aspect=30)
    cbar.set_label("mean per-sample MCC (global $t$=0.50)")
    cbar.ax.tick_params(labelsize=7)
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figS_pilot_atlas.{ext}")
    ps.plt.close(fig)
    print("wrote", out / "figS_pilot_atlas.pdf")


def plot_decoders(pilot_dir: Path, out: Path) -> None:
    curves = {}
    all_curves = pd.read_csv(pilot_dir / "pilot_decoder_curves.csv")
    for m in ["mega", "giga"]:
        for dec in ["unet", "transformer"]:
            d = all_curves[(all_curves.model == m)
                           & (all_curves.decoder == dec)].sort_values("epoch")
            curves[(m, dec)] = d
            print(f"{m}/{dec}: epochs={len(d)} best val_mcc={d.val_mcc.max():.4f}")

    fig, axes = ps.plt.subplots(1, 2, figsize=(7.0, 2.5), sharey=True,
                                constrained_layout=True)
    colors = {"unet": "#C44E52", "transformer": "#4C72B0"}
    labels = {"unet": "U-Net decoder", "transformer": "transformer decoder"}
    for ax, m, letter in zip(axes, ["mega", "giga"], "ab"):
        for dec in ["unet", "transformer"]:
            d = curves[(m, dec)]
            ax.plot(d.epoch, d.val_mcc, color=colors[dec], lw=1.2,
                    label=labels[dec])
            b = d.loc[d.val_mcc.idxmax()]
            ax.plot([b.epoch], [b.val_mcc], "o", ms=3.5, color=colors[dec])
            ax.annotate(f"{b.val_mcc:.3f}", (b.epoch, b.val_mcc),
                        textcoords="offset points", xytext=(4, -8),
                        fontsize=6.5, color=colors[dec])
        ax.set_xlabel("epoch")
        ax.set_title(f"RiNALMo-{m} top-3 heads", pad=3)
        ps.panel_label(ax, letter, x=-0.12, y=1.06)
        ps.despine(ax)
    axes[0].set_ylabel("validation MCC")
    axes[1].legend(frameon=False, loc="lower right")
    for ext in ("pdf", "png"):
        fig.savefig(out / f"figS_pilot_decoders.{ext}")
    ps.plt.close(fig)
    print("wrote", out / "figS_pilot_decoders.pdf")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=default_data_root(),
                    help="root of the downloaded HF dataset "
                         "(default: $RNA_HEAD_DISTILL_DATA)")
    ap.add_argument("--outdir", type=Path, default=Path("./reproduced_figures"),
                    help="output directory for figS_pilot_*.{pdf,png}")
    args = ap.parse_args()
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    pilot_dir = args.data_root / "pilot"

    if all((pilot_dir / f"pilot_mcc_heatmap_{m}.csv").exists() for m in MODELS):
        plot_atlas(pilot_dir, out)
    else:
        print(f"SKIP figS_pilot_atlas: pilot_mcc_heatmap_*.csv missing under "
              f"{pilot_dir}")
    if (pilot_dir / "pilot_decoder_curves.csv").exists():
        plot_decoders(pilot_dir, out)
    else:
        print(f"SKIP figS_pilot_decoders: pilot_decoder_curves.csv missing "
              f"under {pilot_dir}")


if __name__ == "__main__":
    main()
