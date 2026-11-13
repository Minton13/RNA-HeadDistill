#!/usr/bin/env python3
"""Phase 1.2: paired bootstrap statistics for frozen-probe predictions.

Reads per-sequence prediction dumps produced by ``phase1_dump_predictions.py``
(under ``outputs/probes/predictions/``) and computes paired bootstrap
confidence intervals for metric differences between feature modes, on the
test split.  The primary metric per task is the training selection metric:

* seqcls tasks        -> auroc
* chemical_reactivity -> pearson
* secondary_structure -> top_l_precision (+ F1 at val-calibrated threshold)

Outputs under ``outputs/probes/analysis_v2/bootstrap/``:

* ``point_estimates.csv``        test metric per (task, mode, seed)
* ``p4_vs_baselines_per_seed.csv``
* ``p4_vs_baselines_summary.csv`` cross-seed pooled deltas + 95% CI
* ``bootstrap_p4_vs_p0_forest.png``
* ``bootstrap_p4_vs_p0_forest_per_seed.png``
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

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rna_head_alignment.probes.metrics import (  # noqa: E402
    _top_k_overlap_per_sample,
    sequence_classification_metrics,
)

PRED_ROOT = Path("outputs/probes/predictions")
OUT_DIR = Path("outputs/probes/analysis_v2/bootstrap")

TASKS: dict[str, dict] = {
    "splice_donor_seqcls": {"type": "seqcls", "primary": "auroc"},
    "splice_acceptor_seqcls": {"type": "seqcls", "primary": "auroc"},
    "rbp_binding_seqcls": {"type": "seqcls", "primary": "auroc"},
    "ncRNA_family_seqcls": {"type": "seqcls", "primary": "auroc"},
    "chemical_reactivity_token": {"type": "token", "primary": "pearson"},
    "secondary_structure_pair": {"type": "pairwise", "primary": "top_l_precision"},
}

MODES: dict[str, str] = {
    "P0": "vanilla_hidden",
    "P1": "distilled_hidden",
    "P2": "distilled_hidden_native",
    "P3": "distilled_aligned",
    "P4": "distilled_hidden_aligned",
    "P4g": "distilled_hidden_aligned_gated",
    "P5": "distilled_hidden_teacher",
    "P6": "distilled_hidden_random_heads",
}

COMPARISONS = [
    ("P4", "P0"), ("P4", "P6"), ("P4", "P2"), ("P4", "P1"), ("P1", "P0"),
    ("P4g", "P0"), ("P4g", "P4"),
]

SEEDS = (17, 41, 89)
N_BOOT = 1000
RNG_SEED = 2024
F1_THRESHOLD_GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


# --------------------------------------------------------------------- loaders
def _load_seqcls(task: str, mode: str, seed: int, split: str):
    path = PRED_ROOT / task / MODES[mode] / f"seed_{seed}" / f"pred_{split}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    labels = df["label"].to_numpy()
    logits = np.stack(df["logits"].to_numpy())
    if logits.ndim > 1 and logits.shape[1] == 1:
        logits = logits[:, 0]
    n_cls = logits.shape[1] if logits.ndim > 1 else 2
    return {
        "seq_ids": df["seq_id"].to_numpy(),
        "primary_fn": lambda idx: sequence_classification_metrics(
            labels[idx], logits[idx], n_cls
        )["auroc"],
        "secondary": {
            "auprc": lambda idx: sequence_classification_metrics(
                labels[idx], logits[idx], n_cls
            )["auprc"],
        },
    }


def _load_token(task: str, mode: str, seed: int, split: str):
    import torch

    path = PRED_ROOT / task / MODES[mode] / f"seed_{seed}" / f"pred_{split}.pt"
    if not path.exists():
        return None
    seqs = torch.load(path, map_location="cpu", weights_only=False)
    # Per-sequence sufficient statistics for Pearson over concatenated points.
    stats = []
    for s in seqs:
        y = np.asarray(s["labels"], dtype=np.float64)
        p = np.asarray(s["preds"], dtype=np.float64)
        stats.append((len(y), y.sum(), p.sum(), (y * y).sum(), (p * p).sum(), (y * p).sum()))
    stats = np.array(stats)  # [N, 6]

    def pearson(idx):
        n, sy, sp, syy, spp, syp = stats[idx].sum(axis=0)
        cov = syp - sy * sp / n
        vy = syy - sy * sy / n
        vp = spp - sp * sp / n
        if vy <= 0 or vp <= 0:
            return float("nan")
        return float(cov / np.sqrt(vy * vp))

    def mae(idx):
        # per-seq abs error sums; recompute lazily from stored seqs
        vals = np.array([np.abs(s["labels"] - s["preds"]).sum() for s in seqs])
        ns = stats[:, 0]
        return float(vals[idx].sum() / ns[idx].sum())

    return {
        "seq_ids": np.array([s["seq_id"] for s in seqs]),
        "primary_fn": pearson,
        "secondary": {"mae": mae},
    }


def _pair_counts(probs, labels, mask, thr):
    p = probs.numpy()
    y = labels.numpy()
    m = mask.numpy()
    pred = (p >= thr) & m
    pos = (y > 0.5) & m
    tp = float((pred & pos).sum())
    fp = float((pred & ~pos).sum())
    fn = float((~pred & pos).sum())
    return tp, fp, fn


def _load_pairwise(task: str, mode: str, seed: int, split: str):
    import torch

    path = PRED_ROOT / task / MODES[mode] / f"seed_{seed}" / f"pred_{split}.pt"
    if not path.exists():
        return None
    seqs = torch.load(path, map_location="cpu", weights_only=False)
    top_l = np.full(len(seqs), np.nan)
    counts = {t: np.zeros((len(seqs), 3)) for t in F1_THRESHOLD_GRID}
    for i, s in enumerate(seqs):
        L = int(s["length"])
        probs = s["probs"].float()
        if L > 1:
            top_l[i] = _top_k_overlap_per_sample(probs, s["labels"], L, k=L)
        for t in F1_THRESHOLD_GRID:
            counts[t][i] = _pair_counts(probs, s["labels"], s["mask"], t)

    def top_l_precision(idx):
        return float(np.nanmean(top_l[idx]))

    def f1_at(t):
        def fn(idx):
            tp, fp, fn_ = counts[t][idx].sum(axis=0)
            denom = 2 * tp + fp + fn_
            return float(2 * tp / denom) if denom > 0 else float("nan")

        return fn

    return {
        "seq_ids": np.array([s["seq_id"] for s in seqs]),
        "primary_fn": top_l_precision,
        "f1_by_threshold": {t: f1_at(t) for t in F1_THRESHOLD_GRID},
        "secondary": {},  # F1 threshold calibrated on val; handled in driver
        "_seqs_len": [int(s["length"]) for s in seqs],
    }


LOADERS = {"seqcls": _load_seqcls, "token": _load_token, "pairwise": _load_pairwise}


# -------------------------------------------------------------------- driver
def _paired_boot(fn_a, fn_b, n, rng, n_boot=N_BOOT):
    deltas = np.full(n_boot, np.nan)
    for r in range(n_boot):
        idx = rng.integers(0, n, size=n)
        deltas[r] = fn_a(idx) - fn_b(idx)
    return deltas


def _summarize(deltas: np.ndarray) -> dict[str, float]:
    d = deltas[~np.isnan(deltas)]
    if len(d) == 0:
        return {"mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_pos": float("nan")}
    return {
        "mean": float(d.mean()),
        "ci_lo": float(np.percentile(d, 2.5)),
        "ci_hi": float(np.percentile(d, 97.5)),
        "p_pos": float((d > 0).mean()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if "--plots-only" in sys.argv:
        ps = pd.read_csv(OUT_DIR / "p4_vs_baselines_per_seed.csv")
        sm = pd.read_csv(OUT_DIR / "p4_vs_baselines_summary.csv")
    else:
        _compute()

        ps = pd.read_csv(OUT_DIR / "p4_vs_baselines_per_seed.csv")
        sm = pd.read_csv(OUT_DIR / "p4_vs_baselines_summary.csv")

    _plot(ps, sm)

def _compute() -> None:
    rng = np.random.default_rng(RNG_SEED)

    point_rows: list[dict] = []
    per_seed_rows: list[dict] = []
    summary_rows: list[dict] = []
    missing: list[str] = []

    for task, spec in TASKS.items():
        loader = LOADERS[spec["type"]]

        # Val-calibrated F1 thresholds for secondary_structure.
        f1_thr: dict[tuple[str, int], float] = {}
        if spec["type"] == "pairwise":
            for mode in MODES:
                for seed in SEEDS:
                    val = loader(task, mode, seed, "val")
                    if val is None:
                        continue
                    n_val = len(val["seq_ids"])
                    best_t, best_f1 = None, -1.0
                    all_idx = np.arange(n_val)
                    for t, fn in val["f1_by_threshold"].items():
                        f1 = fn(all_idx)
                        if f1 == f1 and f1 > best_f1:
                            best_f1, best_t = f1, t
                    if best_t is not None:
                        f1_thr[(mode, seed)] = best_t

        # Point estimates on full test.
        test_data: dict[tuple[str, int], dict] = {}
        for mode in MODES:
            for seed in SEEDS:
                d = loader(task, mode, seed, "test")
                if d is None:
                    missing.append(f"{task}/{mode}/seed_{seed}")
                    continue
                test_data[(mode, seed)] = d
                n = len(d["seq_ids"])
                idx = np.arange(n)
                row = {
                    "task": task,
                    "mode": mode,
                    "seed": seed,
                    "n_test": n,
                    spec["primary"]: d["primary_fn"](idx),
                }
                for name, fn in d["secondary"].items():
                    row[name] = fn(idx)
                if spec["type"] == "pairwise" and (mode, seed) in f1_thr:
                    row["f1_val_threshold"] = d["f1_by_threshold"][f1_thr[(mode, seed)]](idx)
                    row["f1_threshold"] = f1_thr[(mode, seed)]
                point_rows.append(row)

        # Paired bootstrap comparisons.
        for mode_a, mode_b in COMPARISONS:
            per_seed_deltas: list[np.ndarray] = []
            for seed in SEEDS:
                da = test_data.get((mode_a, seed))
                db = test_data.get((mode_b, seed))
                if da is None or db is None:
                    continue
                # Align by seq_id (inner join, preserve A order).
                pos_b = {sid: i for i, sid in enumerate(db["seq_ids"])}
                keep_a = [i for i, sid in enumerate(da["seq_ids"]) if sid in pos_b]
                map_b = np.array([pos_b[da["seq_ids"][i]] for i in keep_a])
                keep_a = np.array(keep_a)
                n = len(keep_a)

                def fn_a(idx, _da=da, _keep=keep_a):
                    return _da["primary_fn"](_keep[idx])

                def fn_b(idx, _db=db, _map=map_b):
                    return _db["primary_fn"](_map[idx])

                deltas = _paired_boot(fn_a, fn_b, n, rng)
                per_seed_deltas.append(deltas)
                s = _summarize(deltas)
                per_seed_rows.append({
                    "task": task, "comparison": f"{mode_a}-{mode_b}", "seed": seed,
                    "metric": spec["primary"], **s,
                })

                # Secondary metric: F1 at val-calibrated threshold (pairwise only).
                if spec["type"] == "pairwise" and (mode_a, seed) in f1_thr and (mode_b, seed) in f1_thr:
                    ta, tb = f1_thr[(mode_a, seed)], f1_thr[(mode_b, seed)]

                    def fa(idx, _da=da, _keep=keep_a, _t=ta):
                        return _da["f1_by_threshold"][_t](_keep[idx])

                    def fb(idx, _db=db, _map=map_b, _t=tb):
                        return _db["f1_by_threshold"][_t](_map[idx])

                    d2 = _paired_boot(fa, fb, n, rng)
                    s2 = _summarize(d2)
                    per_seed_rows.append({
                        "task": task, "comparison": f"{mode_a}-{mode_b}", "seed": seed,
                        "metric": "f1_val_threshold", **s2,
                    })

            if per_seed_deltas:
                pooled = np.nanmean(np.stack(per_seed_deltas), axis=0)
                s = _summarize(pooled)
                s["seed_mean_std"] = float(np.nanstd([d.mean() for d in per_seed_deltas]))
                summary_rows.append({
                    "task": task, "comparison": f"{mode_a}-{mode_b}",
                    "metric": spec["primary"], "n_seeds": len(per_seed_deltas), **s,
                })

    pt = pd.DataFrame(point_rows)
    ps = pd.DataFrame(per_seed_rows)
    sm = pd.DataFrame(summary_rows)
    pt.to_csv(OUT_DIR / "point_estimates.csv", index=False)
    ps.to_csv(OUT_DIR / "p4_vs_baselines_per_seed.csv", index=False)
    sm.to_csv(OUT_DIR / "p4_vs_baselines_summary.csv", index=False)

    if missing:
        (OUT_DIR / "missing_dumps.json").write_text(json.dumps(missing, indent=2))
        print(f"[bootstrap] WARNING missing dumps: {len(missing)}")

def _plot(ps: pd.DataFrame, sm: pd.DataFrame) -> None:
    # ------------------------------------------------------------- forest plot
    sub = sm[(sm["comparison"] == "P4-P0")].set_index("task")
    if not sub.empty:
        order = [t for t in TASKS if t in sub.index]
        fig, ax = plt.subplots(figsize=(7, 0.7 * len(order) + 1.5))
        ypos = np.arange(len(order))[::-1]
        for y, t in zip(ypos, order):
            r = sub.loc[t]
            ax.errorbar(
                r["mean"], y,
                xerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                fmt="o", capsize=4,
            )
            ax.text(r["ci_hi"], y + 0.25, f"Δ={r['mean']:+.3f} (P(>0)={r['p_pos']:.2f})",
                    fontsize=8, va="bottom")
        ax.axvline(0, linestyle="--", linewidth=1)
        ax.set_yticks(ypos)
        ax.set_yticklabels([f"{t}\n({TASKS[t]['primary']})" for t in order], fontsize=9)
        ax.set_xlabel("P4 − P0 (paired bootstrap mean Δ, 95% CI)")
        ax.set_title("Frozen probe: distilled hidden+aligned (P4) vs vanilla (P0), test")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "bootstrap_p4_vs_p0_forest.png", dpi=200)
        plt.close(fig)

    # Per-seed forest for P4-P0 (primary metric only; secondary rows share seeds).
    sub2 = ps[(ps["comparison"] == "P4-P0")]
    sub2 = sub2[sub2.apply(lambda r: r["metric"] == TASKS[r["task"]]["primary"], axis=1)]
    if not sub2.empty:
        order = [t for t in TASKS if t in set(sub2["task"])]
        fig, axes = plt.subplots(
            len(order), 1, figsize=(7, 1.4 * len(order) + 1.0), sharex=False
        )
        if len(order) == 1:
            axes = [axes]
        for ax, t in zip(axes, order):
            rows = sub2[sub2["task"] == t].set_index("seed").reindex(SEEDS).dropna(how="all")
            ypos = np.arange(len(rows))[::-1]
            for y, (seed, r) in zip(ypos, rows.iterrows()):
                ax.errorbar(
                    r["mean"], y,
                    xerr=[[r["mean"] - r["ci_lo"]], [r["ci_hi"] - r["mean"]]],
                    fmt="o", capsize=4,
                )
            ax.axvline(0, linestyle="--", linewidth=1)
            ax.set_yticks(ypos)
            ax.set_yticklabels([f"seed {s}" for s in rows.index], fontsize=8)
            ax.set_title(f"{t} — {TASKS[t]['primary']} Δ (P4−P0)", fontsize=9)
        axes[-1].set_xlabel("paired bootstrap mean Δ, 95% CI")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "bootstrap_p4_vs_p0_forest_per_seed.png", dpi=200)
        plt.close(fig)

    print(f"[bootstrap] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
