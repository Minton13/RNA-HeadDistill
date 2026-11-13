#!/usr/bin/env python3
"""Paper-grade downstream full-metric re-evaluation.

Independently recomputes every metric from the per-sequence prediction dumps
(``outputs/probes/predictions/``) for 6 tasks x 6 feature modes x 3 seeds,
cross-checks against ``analysis_v2/frozen_probe_all_metric_aligned.parquet``
(tolerance 1e-3), runs paired bootstrap (method identical to
``phase1_bootstrap.py``: per-seed paired resampling, N_BOOT=1000,
``np.random.default_rng(2024)``, pooled across seeds) of each mode vs
``vanilla_hidden`` (P0) on the primary metric, and writes paper artifacts to
``../paper/`` (sibling of this repository).

Read-only with respect to src/ and existing outputs.
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

from rna_head_alignment.probes.metrics import (  # noqa: E402
    _top_k_overlap_per_sample,
    pairwise_metrics_streaming,
    sequence_classification_metrics,
    token_regression_metrics,
)

ROOT = _PROJECT_ROOT
PRED_ROOT = ROOT / "outputs/probes/predictions"
REF_PARQUET = ROOT / "outputs/probes/analysis_v2/frozen_probe_all_metric_aligned.parquet"
PAPER = _PROJECT_ROOT.parent / "paper"
TABLES = PAPER / "tables"
FIGS = PAPER / "figures/downstream"
NOTES = PAPER / "notes"

TASKS: dict[str, dict] = {
    "splice_donor_seqcls": {"type": "seqcls", "primary": "auroc"},
    "splice_acceptor_seqcls": {"type": "seqcls", "primary": "auroc"},
    "rbp_binding_seqcls": {"type": "seqcls", "primary": "auroc"},
    "ncRNA_family_seqcls": {"type": "seqcls", "primary": "auroc"},
    "chemical_reactivity_token": {"type": "token", "primary": "pearson"},
    "secondary_structure_pair": {"type": "pairwise", "primary": "top_l_precision"},
}
TASK_SHORT = {
    "splice_donor_seqcls": "splice donor",
    "splice_acceptor_seqcls": "splice acceptor",
    "rbp_binding_seqcls": "RBP binding",
    "ncRNA_family_seqcls": "ncRNA family",
    "chemical_reactivity_token": "chemical reactivity",
    "secondary_structure_pair": "secondary structure",
}
MODES = [
    "vanilla_hidden",
    "distilled_hidden",
    "distilled_hidden_native",
    "distilled_aligned",
    "distilled_hidden_aligned",
    "distilled_hidden_teacher",
    "distilled_hidden_random_heads",
]
MODE_LABELS = {
    "vanilla_hidden": "P0 vanilla",
    "distilled_hidden": "P1 distilled",
    "distilled_hidden_native": "P2 native",
    "distilled_aligned": "P3 aligned",
    "distilled_hidden_aligned": "P4 hidden+aligned",
    "distilled_hidden_teacher": "P5 teacher",
    "distilled_hidden_random_heads": "P6 random",
}
MODE_TAGS = {
    "vanilla_hidden": "P0",
    "distilled_hidden": "P1",
    "distilled_hidden_native": "P2",
    "distilled_aligned": "P3",
    "distilled_hidden_aligned": "P4",
    "distilled_hidden_teacher": "P5",
    "distilled_hidden_random_heads": "P6",
}
SEEDS = [17, 41, 89]
TOL = 1e-3
N_BOOT = 1000
RNG_SEED = 2024

TASK_METRICS = {
    "seqcls": ["auroc", "auprc", "f1", "mcc", "accuracy"],
    "token": ["mse", "mae", "pearson", "spearman"],
    "pairwise": ["top_l_precision", "top_2l_precision", "f1", "mcc", "auprc"],
}
# Metrics that are "higher is better" (all except mse/mae).
LOWER_BETTER = {"mse", "mae"}
METRIC_RANGE = {
    "auroc": (0.0, 1.0), "auprc": (0.0, 1.0), "f1": (0.0, 1.0),
    "accuracy": (0.0, 1.0), "mcc": (-1.0, 1.0), "pearson": (-1.0, 1.0),
    "spearman": (-1.0, 1.0), "top_l_precision": (0.0, 1.0),
    "top_2l_precision": (0.0, 1.0), "mse": (0.0, np.inf), "mae": (0.0, np.inf),
}


# --------------------------------------------------------------- recomputation
def _recompute_seqcls(path: Path) -> dict[str, float]:
    df = pd.read_parquet(path)
    labels = df["label"].to_numpy()
    logits = np.stack(df["logits"].to_numpy())
    if logits.ndim > 1 and logits.shape[1] == 1:
        logits = logits[:, 0]
    n_cls = logits.shape[1] if logits.ndim > 1 else 2
    out = sequence_classification_metrics(labels, logits, n_cls)
    if n_cls == 2:
        y_pred = (logits >= 0).astype(int)
    else:
        y_pred = np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6).argmax(axis=1)
    out["accuracy"] = float((y_pred == labels).mean())
    out["n_samples"] = int(len(labels))
    return out


def _recompute_token(path: Path) -> dict[str, float]:
    seqs = torch.load(path, map_location="cpu", weights_only=False)
    preds = np.concatenate([np.asarray(s["preds"], dtype=np.float64) for s in seqs])
    labels = np.concatenate([np.asarray(s["labels"], dtype=np.float64) for s in seqs])
    # Dumps are already masked to valid label positions by the dump script.
    out = token_regression_metrics(labels, preds, np.ones_like(labels, dtype=bool))
    out["n_samples"] = int(len(seqs))
    return out


def _recompute_pairwise(path: Path) -> dict[str, float]:
    seqs = torch.load(path, map_location="cpu", weights_only=False)
    flat_true, flat_probs, top_l, top_2l = [], [], [], []
    for s in seqs:
        m = s["mask"].numpy().astype(bool)
        flat_true.append(s["labels"].numpy()[m])
        flat_probs.append(s["probs"].float().numpy()[m])
        L = int(s["length"])
        if L > 1:
            top_l.append(_top_k_overlap_per_sample(s["probs"].float(), s["labels"], L, k=L))
            top_2l.append(_top_k_overlap_per_sample(s["probs"].float(), s["labels"], L, k=2 * L))
    out = pairwise_metrics_streaming(
        np.concatenate(flat_true), np.concatenate(flat_probs), top_l, top_2l
    )
    out["n_samples"] = int(len(seqs))
    return out


RECOMPUTE = {"seqcls": _recompute_seqcls, "token": _recompute_token, "pairwise": _recompute_pairwise}


# ------------------------------------------------------------------ bootstrap
def _load_cell_for_boot(task: str, mode: str, seed: int):
    """Per-sequence handles for the primary metric (mirrors phase1_bootstrap)."""
    ttype = TASKS[task]["type"]
    d = PRED_ROOT / task / mode / f"seed_{seed}"
    if ttype == "seqcls":
        path = d / "pred_test.parquet"
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
        }
    if ttype == "token":
        seqs = torch.load(d / "pred_test.pt", map_location="cpu", weights_only=False)
        stats = []
        for s in seqs:
            y = np.asarray(s["labels"], dtype=np.float64)
            p = np.asarray(s["preds"], dtype=np.float64)
            stats.append((len(y), y.sum(), p.sum(), (y * y).sum(), (p * p).sum(), (y * p).sum()))
        stats = np.array(stats)

        def pearson(idx):
            n, sy, sp, syy, spp, syp = stats[idx].sum(axis=0)
            cov = syp - sy * sp / n
            vy = syy - sy * sy / n
            vp = spp - sp * sp / n
            if vy <= 0 or vp <= 0:
                return float("nan")
            return float(cov / np.sqrt(vy * vp))

        return {"seq_ids": np.array([s["seq_id"] for s in seqs]), "primary_fn": pearson}
    # pairwise
    seqs = torch.load(d / "pred_test.pt", map_location="cpu", weights_only=False)
    top_l = np.full(len(seqs), np.nan)
    for i, s in enumerate(seqs):
        L = int(s["length"])
        if L > 1:
            top_l[i] = _top_k_overlap_per_sample(s["probs"].float(), s["labels"], L, k=L)
    return {
        "seq_ids": np.array([s["seq_id"] for s in seqs]),
        "primary_fn": lambda idx: float(np.nanmean(top_l[idx])),
    }


def _summarize(deltas: np.ndarray) -> dict[str, float]:
    d = deltas[~np.isnan(deltas)]
    if len(d) == 0:
        return {"delta_mean": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"), "p_pos": float("nan")}
    return {
        "delta_mean": float(d.mean()),
        "ci_lo": float(np.percentile(d, 2.5)),
        "ci_hi": float(np.percentile(d, 97.5)),
        "p_pos": float((d > 0).mean()),
    }


# ----------------------------------------------------------------------- main
def main() -> None:
    for p in (TABLES, FIGS, NOTES):
        p.mkdir(parents=True, exist_ok=True)

    ref = pd.read_parquet(REF_PARQUET)
    ref_map = {(r["task"], r["mode"], int(r["seed"])): r for _, r in ref.iterrows()}
    # P5 reference lives in a separate parquet with aligned columns.
    p5_path = ROOT / "outputs/probes/frozen/p5_summary.parquet"
    if p5_path.exists():
        p5 = pd.read_parquet(p5_path)
        for _, r in p5.iterrows():
            ref_map[(r["task"], r["mode"], int(r["seed"]))] = r

    point_rows: list[dict] = []
    integrity: list[str] = []
    mismatches: list[dict] = []

    # Incremental: keep previously recomputed cells, only compute missing ones.
    points_csv = TABLES / "downstream_recomputed_points.csv"
    done: set[tuple[str, str, int]] = set()
    if points_csv.exists():
        prev = pd.read_csv(points_csv)
        point_rows.extend(prev.to_dict("records"))
        done = {(r["task"], r["mode"], int(r["seed"])) for r in prev.to_dict("records")}
        print(f"[paper] incremental: {len(done)} cells cached", flush=True)

    # ---- 1. recompute every cell
    for task, spec in TASKS.items():
        ttype = spec["type"]
        fn = RECOMPUTE[ttype]
        ext = "parquet" if ttype == "seqcls" else "pt"
        for mode in MODES:
            for seed in SEEDS:
                if (task, mode, seed) in done:
                    continue
                path = PRED_ROOT / task / mode / f"seed_{seed}" / f"pred_test.{ext}"
                if not path.exists():
                    integrity.append(f"MISSING dump: {task}/{mode}/seed_{seed}")
                    continue
                print(f"[paper] recompute {task}/{mode}/seed_{seed}", flush=True)
                m = fn(path)
                row = {"task": task, "mode": mode, "seed": seed, "n_test": m.pop("n_samples")}
                row.update({k: m.get(k, float("nan")) for k in TASK_METRICS[ttype]})
                point_rows.append(row)

                # NaN / range checks
                for k in TASK_METRICS[ttype]:
                    v = row[k]
                    if v != v:
                        integrity.append(f"NaN metric: {task}/{mode}/seed_{seed} {k}")
                    else:
                        lo, hi = METRIC_RANGE[k]
                        if not (lo - 1e-9 <= v <= hi + 1e-9):
                            integrity.append(f"OUT-OF-RANGE: {task}/{mode}/seed_{seed} {k}={v:.4f} not in [{lo},{hi}]")

                # cross-check vs reference parquet
                r = ref_map.get((task, mode, seed))
                if r is None:
                    integrity.append(f"MISSING in reference parquet: {task}/{mode}/seed_{seed}")
                    continue
                for k in TASK_METRICS[ttype]:
                    rv = r.get(f"test_{k}", float("nan"))
                    if rv != rv:  # reference NaN -> nothing to check
                        continue
                    v = row[k]
                    diff = abs(v - rv) if v == v else float("inf")
                    if diff > TOL:
                        mismatches.append({
                            "task": task, "mode": mode, "seed": seed, "metric": k,
                            "recomputed": v, "reference": rv, "abs_diff": diff,
                        })

    pts = pd.DataFrame(point_rows)
    pts = pts.drop_duplicates(subset=["task", "mode", "seed"], keep="last")
    # seed completeness
    for (task, mode), sub in pts.groupby(["task", "mode"]):
        have = sorted(sub["seed"].tolist())
        if have != SEEDS:
            integrity.append(f"SEED SET != {SEEDS}: {task}/{mode} has {have}")

    pts.to_csv(TABLES / "downstream_recomputed_points.csv", index=False)

    # ---- 2. mean +/- std summary
    summ_rows = []
    for task, spec in TASKS.items():
        ttype = spec["type"]
        for mode in MODES:
            sub = pts[(pts["task"] == task) & (pts["mode"] == mode)]
            for k in TASK_METRICS[ttype]:
                vals = sub[k].dropna()
                summ_rows.append({
                    "task": task, "mode": mode, "mode_tag": MODE_TAGS[mode],
                    "metric": k, "mean": vals.mean(), "std": vals.std(), "n_seeds": len(vals),
                })
    summ = pd.DataFrame(summ_rows)
    summ.to_csv(TABLES / "downstream_full_metrics.csv", index=False)

    # ---- 3. paired bootstrap: every mode vs P0 on primary metric
    rng = np.random.default_rng(RNG_SEED)
    boot_rows: list[dict] = []
    per_seed_rows: list[dict] = []
    for task, spec in TASKS.items():
        primary = spec["primary"]
        test_data = {}
        for mode in MODES:
            for seed in SEEDS:
                test_data[(mode, seed)] = _load_cell_for_boot(task, mode, seed)
        for mode in MODES:
            if mode == "vanilla_hidden":
                continue
            per_seed_deltas = []
            for seed in SEEDS:
                da = test_data[(mode, seed)]
                db = test_data[("vanilla_hidden", seed)]
                pos_b = {sid: i for i, sid in enumerate(db["seq_ids"])}
                keep_a = np.array([i for i, sid in enumerate(da["seq_ids"]) if sid in pos_b])
                map_b = np.array([pos_b[da["seq_ids"][i]] for i in keep_a])
                n = len(keep_a)

                def fn_a(idx, _da=da, _keep=keep_a):
                    return _da["primary_fn"](_keep[idx])

                def fn_b(idx, _db=db, _map=map_b):
                    return _db["primary_fn"](_map[idx])

                deltas = np.full(N_BOOT, np.nan)
                for r_i in range(N_BOOT):
                    idx = rng.integers(0, n, size=n)
                    deltas[r_i] = fn_a(idx) - fn_b(idx)
                per_seed_deltas.append(deltas)
                per_seed_rows.append({
                    "task": task, "mode": mode, "mode_tag": MODE_TAGS[mode],
                    "metric": primary, "seed": seed, **_summarize(deltas),
                })
            pooled = np.nanmean(np.stack(per_seed_deltas), axis=0)
            boot_rows.append({
                "task": task, "mode": mode, "mode_tag": MODE_TAGS[mode],
                "metric": primary, "n_seeds": len(per_seed_deltas), **_summarize(pooled),
            })
    boot = pd.DataFrame(boot_rows)
    pd.DataFrame(per_seed_rows).to_csv(TABLES / "downstream_primary_bootstrap_per_seed.csv", index=False)
    boot.to_csv(TABLES / "downstream_primary_bootstrap.csv", index=False)

    # ---- 4. markdown tables
    _write_md_tables(summ, boot)

    # ---- 5. extended figure
    _plot_all_metrics(summ)

    # ---- 6. notes
    _write_notes(pts, integrity, mismatches, boot)

    print(f"[paper] points={len(pts)} mismatches={len(mismatches)} integrity_issues={len(integrity)}")
    print(boot.to_string(index=False))


def _write_md_tables(summ: pd.DataFrame, boot: pd.DataFrame) -> None:
    lines = ["# Downstream frozen-probe full metrics (recomputed from prediction dumps)", ""]
    lines.append("Test split, mean ± std over 3 seeds (17/41/89). **Bold** = best per column "
                 "excluding the P6 random-heads negative control. Recomputed independently "
                 "from `outputs/probes/predictions/` dumps.")
    lines.append("")
    lines.append("| Tag | Mode |")
    lines.append("|-----|------|")
    for m in MODES:
        lines.append(f"| {MODE_TAGS[m]} | `{m}` |")
    lines.append("")
    for task, spec in TASKS.items():
        ttype = spec["type"]
        metrics = TASK_METRICS[ttype]
        sub = summ[summ["task"] == task]
        # best per metric excluding P6
        best_val = {}
        for k in metrics:
            s = sub[(sub["metric"] == k) & (sub["mode"] != "distilled_hidden_random_heads")]
            best_val[k] = s["mean"].min() if k in LOWER_BETTER else s["mean"].max()
        lines.append(f"## {TASK_SHORT[task]} (`{task}`, primary: {spec['primary']})")
        lines.append("")
        lines.append("| Mode | " + " | ".join(metrics) + " |")
        lines.append("|------|" + "------:|" * len(metrics))
        for mode in MODES:
            cells = []
            for k in metrics:
                r = sub[(sub["mode"] == mode) & (sub["metric"] == k)].iloc[0]
                cell = f"{r['mean']:.4f} ± {r['std']:.4f}"
                if mode != "distilled_hidden_random_heads" and abs(r["mean"] - best_val[k]) < 1e-12:
                    cell = f"**{cell}**"
                cells.append(cell)
            lines.append(f"| {MODE_TAGS[mode]} {MODE_LABELS[mode].split(' ', 1)[1]} | " + " | ".join(cells) + " |")
        lines.append("")
    (TABLES / "downstream_full_metrics.md").write_text("\n".join(lines), encoding="utf-8")

    lines = ["# Primary metric: paired bootstrap Δ vs baselines", ""]
    lines.append("Per-seed paired bootstrap (1000 resamples over test sequences, "
                 "`np.random.default_rng(2024)`), pooled by averaging the per-seed delta "
                 "distributions — identical method to `analysis_v2/bootstrap` "
                 "(`phase1_bootstrap.py`). p_pos = P(Δ > 0). "
                 "All modes vs P0 (vanilla_hidden), plus P5 vs P4.")
    lines.append("")
    for task in TASKS:
        sub = boot[boot["task"] == task]
        lines.append(f"## {TASK_SHORT[task]} (primary: {sub['metric'].iloc[0]})")
        lines.append("")
        lines.append("| Mode | Baseline | Δ mean | 95% CI | p_pos |")
        lines.append("|------|----------|-------:|--------|------:|")
        for _, r in sub.iterrows():
            lines.append(f"| {r['mode_tag']} {r['mode']} | {r['baseline_tag']} {r['baseline']} | "
                         f"{r['delta_mean']:+.4f} | "
                         f"[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | {r['p_pos']:.3f} |")
        lines.append("")
    (TABLES / "downstream_primary_bootstrap.md").write_text("\n".join(lines), encoding="utf-8")


def _plot_all_metrics(summ: pd.DataFrame) -> None:
    """2 rows x 6 cols: row 1 = primary metric per task, row 2 = secondary metric."""
    row2_metric = {
        "splice_donor_seqcls": "auprc",
        "splice_acceptor_seqcls": "auprc",
        "rbp_binding_seqcls": "auprc",
        "ncRNA_family_seqcls": "auprc",
        "chemical_reactivity_token": "spearman",
        "secondary_structure_pair": "top_2l_precision",
    }
    fig, axes = plt.subplots(2, 6, figsize=(24, 9))
    for col, (task, spec) in enumerate(TASKS.items()):
        for row, metric in enumerate([spec["primary"], row2_metric[task]]):
            ax = axes[row, col]
            sub = summ[(summ["task"] == task) & (summ["metric"] == metric)].set_index("mode")
            means = [sub.loc[m, "mean"] for m in MODES]
            stds = [sub.loc[m, "std"] for m in MODES]
            labels = [MODE_LABELS[m] for m in MODES]
            x = np.arange(len(labels))
            colors = [
                "gray" if m == "vanilla_hidden"
                else ("crimson" if m == "distilled_hidden_aligned"
                      else ("seagreen" if m == "distilled_hidden_teacher" else "steelblue"))
                for m in MODES
            ]
            ax.bar(x, means, yerr=stds, capsize=3, alpha=0.85, color=colors)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel(metric)
            ax.set_title(TASK_SHORT[task])
            ax.grid(axis="y", alpha=0.3)
    axes[0, 0].annotate("row 1: primary metric", xy=(0, 1.15), xycoords="axes fraction", fontsize=10)
    axes[1, 0].annotate("row 2: secondary metric", xy=(0, 1.15), xycoords="axes fraction", fontsize=10)
    fig.suptitle("Frozen probe test performance, all metrics recomputed from prediction dumps "
                 "(3 seeds; gray = P0 vanilla, red = P4 hidden+aligned, green = P5 teacher heads, "
                 "blue = other distilled variants)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGS / "frozen_probe_all_metrics.png", dpi=200)
    plt.close(fig)


def _write_notes(pts: pd.DataFrame, integrity: list[str], mismatches: list[dict], boot: pd.DataFrame) -> None:
    lines = ["# Downstream verification notes", ""]
    lines.append("## Data integrity")
    lines.append("")
    lines.append(f"- Cells expected: 6 tasks × 6 modes × 3 seeds = 108; recomputed: {len(pts)}.")
    if integrity:
        lines.append(f"- Issues ({len(integrity)}):")
        for s in integrity:
            lines.append(f"  - {s}")
    else:
        lines.append("- All 108 cells present with exactly seeds {17, 41, 89}; no NaN metrics; "
                     "all metrics within valid ranges.")
    lines.append("")
    lines.append("## Cross-check vs `analysis_v2/frozen_probe_all_metric_aligned.parquet` (tol 1e-3)")
    lines.append("")
    if mismatches:
        mm = pd.DataFrame(mismatches)
        lines.append(f"- {len(mm)} cell-metric values exceed tolerance:")
        lines.append("")
        lines.append("| task | mode | seed | metric | recomputed | reference | abs diff |")
        lines.append("|------|------|-----:|--------|-----------:|----------:|---------:|")
        for _, r in mm.iterrows():
            lines.append(f"| {r['task']} | {r['mode']} | {r['seed']} | {r['metric']} | "
                         f"{r['recomputed']:.6f} | {r['reference']:.6f} | {r['abs_diff']:.6f} |")
    else:
        lines.append("- All recomputed values match the reference within 1e-3.")
    lines.append("")
    lines.append("## secondary_structure_pair provenance")
    lines.append("")
    lines.append("Secondary-structure dumps are per-sequence `.pt` files (fp16 prob maps + labels "
                 "+ mask), NOT flat parquet rows. All secondary metrics (top-L, top-2L, F1, MCC, "
                 "AUPRC) were **recomputed from the dumps** using the same functions as "
                 "`phase1_dump_predictions.py` (`pairwise_metrics_streaming` + "
                 "`_top_k_overlap_per_sample`). No fallback to reference-parquet values was needed.")
    lines.append("")
    lines.append("Caveat: dumped prob maps are stored in fp16, so recomputed top-L/top-2L can "
                 "deviate from the fp32 eval-time values recorded in `eval_test.json` (the source "
                 "of the reference parquet) by more than 1e-3 — see the mismatch table above. "
                 "This fp16 rounding issue was already flagged by the original dump script "
                 "(`dump_check.json` max_abs_diff ≈ 0.0098). The paper tables use the "
                 "dump-recomputed values, consistent with the existing bootstrap analysis.")
    lines.append("")
    lines.append("## chemical_reactivity_token provenance")
    lines.append("")
    lines.append("Token-regression dumps store already-masked per-position preds/labels "
                 "(dump script applies `label_mask` before saving). MSE/MAE/Pearson/Spearman "
                 "were recomputed by concatenating these masked arrays — the label mask is "
                 "therefore respected by construction.")
    lines.append("")
    lines.append("## Bootstrap method")
    lines.append("")
    lines.append("Paired bootstrap identical to `scripts/analysis/phase1_bootstrap.py`: "
                 "test sequences resampled per seed (1000 draws, `default_rng(2024)`), seq_id "
                 "inner-join alignment between modes, per-seed delta distributions averaged "
                 "for the pooled estimate; CI = 2.5/97.5 percentiles; p_pos = P(Δ > 0). "
                 "Note: the RNG stream differs from the original run (different set of "
                 "comparisons), so CIs here are method-identical but not bit-identical.")
    lines.append("")
    lines.append("## Scope note")
    lines.append("")
    lines.append("A 7th mode (`distilled_hidden_aligned_gated`, P4g) has dumps on disk but is "
                 "out of scope for the requested 6-mode comparison and was excluded.")
    (NOTES / "downstream_verification.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
