#!/usr/bin/env python3
"""Phase 3.1: downstream metric by sequence-length bin.

Joins the per-sequence prediction dumps (Phase 1.2) with sequence lengths from
the downstream task parquets and reports the primary task metric per length
bin for each feature mode, plus a paired-bootstrap P4 - P0 delta per bin.

Outputs under ``outputs/probes/analysis_v2/length_bins/``:

* ``metrics_by_length_bin.csv``  per (task, mode, seed, bin)
* ``p4_vs_p0_by_bin.csv``        paired bootstrap delta per (task, bin)
* ``length_bins_{task}.png``     one figure per task
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "analysis"))

from phase1_bootstrap import LOADERS, MODES, TASKS  # noqa: E402
from rna_head_alignment.probes.task_config import downstream_data_dir  # noqa: E402

OUT_DIR = Path("outputs/probes/analysis_v2/length_bins")

BINS = [(0, 128, "L<=128"), (128, 256, "129-256"), (256, 512, "257-512"), (512, 10**9, "513+")]
PLOT_MODES = ("P0", "P1", "P4", "P4g")
N_BOOT = 300
SEEDS = (17, 41, 89)


def _bin_index(lengths: np.ndarray, lo: int, hi: int) -> np.ndarray:
    return np.where((lengths > lo) & (lengths <= hi))[0]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)

    metric_rows: list[dict] = []
    delta_rows: list[dict] = []

    for task, spec in TASKS.items():
        parquet = Path(downstream_data_dir(task)) / "test.parquet"
        if not parquet.exists():
            print(f"[lenbin] skip {task}: {parquet} missing")
            continue
        len_map = pd.read_parquet(parquet, columns=["seq_id", "length"]).set_index("seq_id")["length"]

        loader = LOADERS[spec["type"]]
        dumps: dict[tuple[str, int], dict] = {}
        for mode_key, mode_name in MODES.items():
            for seed in SEEDS:
                d = loader(task, mode_key, seed, "test")
                if d is not None:
                    dumps[(mode_key, seed)] = d

        for (mode_key, seed), d in sorted(dumps.items()):
            seq_ids = d["seq_ids"]
            lengths = len_map.reindex(seq_ids).to_numpy()
            if np.isnan(lengths.astype(float)).any():
                # Keep only sequences with known length.
                valid = ~pd.isna(lengths)
            else:
                valid = np.ones(len(seq_ids), dtype=bool)
            lengths = np.asarray(lengths, dtype=float)
            for lo, hi, name in BINS:
                idx = _bin_index(lengths, lo, hi)
                idx = idx[valid[idx]]
                if len(idx) < 5:
                    continue
                metric_rows.append({
                    "task": task, "mode": mode_key, "seed": seed, "bin": name,
                    "n": len(idx), "metric": spec["primary"],
                    "value": d["primary_fn"](idx),
                })

        # Paired bootstrap P4-family vs P0 per bin (same resample for A and B).
        for mode_a in ("P4", "P4g"):
            for lo, hi, name in BINS:
                per_seed = []
                for seed in SEEDS:
                    da = dumps.get((mode_a, seed))
                    db = dumps.get(("P0", seed))
                    if da is None or db is None:
                        continue
                    pos_b = {sid: i for i, sid in enumerate(db["seq_ids"])}
                    keep_a = np.array([i for i, s in enumerate(da["seq_ids"]) if s in pos_b])
                    map_b = np.array([pos_b[da["seq_ids"][i]] for i in keep_a])
                    lengths = np.asarray(len_map.reindex(da["seq_ids"][keep_a]), dtype=float)
                    bidx = _bin_index(lengths, lo, hi)
                    if len(bidx) < 5:
                        continue
                    n = len(bidx)
                    deltas = np.empty(N_BOOT)
                    for r in range(N_BOOT):
                        idx = rng.integers(0, n, n)
                        deltas[r] = da["primary_fn"](keep_a[bidx][idx]) - db["primary_fn"](map_b[bidx][idx])
                    per_seed.append(deltas)
                if per_seed:
                    pooled = np.nanmean(np.stack(per_seed), axis=0)
                    delta_rows.append({
                        "task": task, "comparison": f"{mode_a}-P0", "bin": name,
                        "metric": spec["primary"], "n_seeds": len(per_seed),
                        "mean": float(np.nanmean(pooled)),
                        "ci_lo": float(np.nanpercentile(pooled, 2.5)),
                        "ci_hi": float(np.nanpercentile(pooled, 97.5)),
                        "p_pos": float((pooled > 0).mean()),
                    })

    md = pd.DataFrame(metric_rows)
    dd = pd.DataFrame(delta_rows)
    md.to_csv(OUT_DIR / "metrics_by_length_bin.csv", index=False)
    dd.to_csv(OUT_DIR / "p4_vs_p0_by_bin.csv", index=False)

    # One figure per task: primary metric vs length bin, mean+/-std over seeds.
    for task, spec in TASKS.items():
        sub = md[md["task"] == task]
        if sub.empty:
            continue
        bin_names = [b[2] for b in BINS]
        fig, ax = plt.subplots(figsize=(6, 4))
        for mode in PLOT_MODES:
            ms = sub[sub["mode"] == mode]
            if ms.empty:
                continue
            means, stds, xs = [], [], []
            for x, bn in enumerate(bin_names):
                vals = ms[ms["bin"] == bn]["value"]
                if len(vals):
                    xs.append(x)
                    means.append(vals.mean())
                    stds.append(vals.std() if len(vals) > 1 else 0.0)
            ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=mode)
        ax.set_xticks(range(len(bin_names)))
        ax.set_xticklabels(bin_names)
        ax.set_ylabel(spec["primary"])
        ax.set_title(f"{task}: {spec['primary']} by length bin (test, mean±sd over seeds)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"length_bins_{task}.png", dpi=200)
        plt.close(fig)

    print(f"[lenbin] wrote {OUT_DIR} ({len(md)} metric rows, {len(dd)} delta rows)")


if __name__ == "__main__":
    main()
