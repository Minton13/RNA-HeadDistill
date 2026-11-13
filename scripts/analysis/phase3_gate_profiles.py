#!/usr/bin/env python3
"""Phase 3.2: learned gate-weight profiles over aligned/native heads.

For every frozen-probe cell with learned softmax gates, extract the gate
vector from ``best_checkpoint.pt``, convert to softmax weights, and map each
entry to its teacher-head identity (via the slim-cache safetensors metadata,
which defines the adapter/head ordering per route).  Also records the P4g
residual gate value.

Outputs under ``outputs/probes/analysis_v2/gates/``:

* ``gate_weights_per_cell.csv``  one row per (task, mode, seed, head)
* ``gate_weights_summary.csv``   mean +/- std across seeds per (task, mode, head)
* ``residual_gates.csv``         P4g residual gate values
* ``gate_heatmap_p4.png``        tasks x heads heatmap for P4
* ``gate_heatmap_p4g.png``       same for the gated variant (if present)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT / "scripts" / "ablation") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "ablation"))

from run_frozen_probe import _load_head_ids  # noqa: E402
from rna_head_alignment.probes.task_config import TASK_TO_ROUTE_DEPTH  # noqa: E402

FROZEN_ROOT = Path("outputs/probes/frozen")
OUT_DIR = Path("outputs/probes/analysis_v2/gates")
SLIM_CACHE_ROOT = Path("outputs/teacher_attention_cache_slim/stage4b_head2head/full")

GATED_MODES = (
    "distilled_hidden_native",
    "distilled_aligned",
    "distilled_hidden_aligned",
    "distilled_hidden_aligned_gated",
    "distilled_hidden_random_heads",
)
SEEDS = (17, 41, 89)


def _first_shard_idx(root: Path) -> int:
    buckets_dir = root / "buckets"
    candidates = sorted(buckets_dir.glob("shard_*.safetensors"))
    if not candidates:
        candidates = sorted(root.glob("shard_*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No slim buckets under {root}")
    return int(candidates[0].stem.split("_")[1])


def _heatmap(df: pd.DataFrame, mode: str, out_path: Path) -> None:
    sub = df[df["mode"] == mode]
    if sub.empty:
        return
    piv = sub.pivot_table(index="task", columns="head_id", values="weight_mean", aggfunc="first")
    fig, ax = plt.subplots(figsize=(max(4, 0.9 * piv.shape[1] + 2), 0.7 * piv.shape[0] + 2))
    im = ax.imshow(piv.to_numpy(), aspect="auto", cmap="viridis")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels(piv.index, fontsize=9)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.to_numpy()[i, j]
            if v == v:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="softmax gate weight (mean over seeds)")
    ax.set_title(f"Learned task gates — {mode}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    head_ids_by_route = _load_head_ids(SLIM_CACHE_ROOT, _first_shard_idx(SLIM_CACHE_ROOT))

    rows: list[dict] = []
    gate_rows: list[dict] = []
    for task, route in TASK_TO_ROUTE_DEPTH.items():
        head_ids = head_ids_by_route.get(route, [])
        for mode in GATED_MODES:
            for seed in SEEDS:
                ckpt_path = FROZEN_ROOT / task / mode / f"seed_{seed}" / "best_checkpoint.pt"
                if not ckpt_path.exists():
                    continue
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False, mmap=True)
                sd = ckpt["feature_extractor_state_dict"]
                gates = sd.get("gates")
                if gates is None:
                    continue
                weights = torch.softmax(gates.float(), dim=0).numpy()
                for i, w in enumerate(weights):
                    rows.append({
                        "task": task,
                        "route_depth": route,
                        "mode": mode,
                        "seed": seed,
                        "head_index": i,
                        "head_id": head_ids[i] if i < len(head_ids) else f"idx_{i}",
                        "gate_logit": float(gates[i]),
                        "weight": float(w),
                    })
                rg = sd.get("residual_gate")
                if rg is not None:
                    gate_rows.append({
                        "task": task, "mode": mode, "seed": seed,
                        "residual_gate": float(rg.flatten()[0]),
                    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "gate_weights_per_cell.csv", index=False)
    if df.empty:
        print("[gates] no gated cells found")
        return

    summary = (
        df.groupby(["task", "route_depth", "mode", "head_index", "head_id"])["weight"]
        .agg(weight_mean="mean", weight_std="std", n="count")
        .reset_index()
    )
    summary.to_csv(OUT_DIR / "gate_weights_summary.csv", index=False)
    pd.DataFrame(gate_rows).to_csv(OUT_DIR / "residual_gates.csv", index=False)

    _heatmap(summary, "distilled_hidden_aligned", OUT_DIR / "gate_heatmap_p4.png")
    _heatmap(summary, "distilled_hidden_aligned_gated", OUT_DIR / "gate_heatmap_p4g.png")

    print(f"[gates] wrote {OUT_DIR} ({len(df)} rows)")


if __name__ == "__main__":
    main()
