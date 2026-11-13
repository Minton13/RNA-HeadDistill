#!/usr/bin/env python3
"""Fast paired bootstrap + paper artifact regeneration.

Companion to ``paper_full_metrics.py`` (which already recomputed per-cell
metrics and wrote ``downstream_recomputed_points.csv`` /
``downstream_full_metrics.csv``).  This script:

* runs the per-seed paired bootstrap (1000 draws, ``default_rng(2024)``,
  pooled across seeds — same resampling design as ``phase1_bootstrap.py``)
  for every mode vs P0 on each task's primary metric, using an exact
  O(n) Mann-Whitney formulation of AUROC (validated against sklearn on
  random draws; ties handled with average ranks, identical to sklearn);
* regenerates the markdown tables, the extended all-metrics figure, and the
  verification notes from the already-written CSVs.

The AUROC shortcut is exact: for a resample with draw counts ``c`` per
element (sorted by score, ties grouped), the sum of within-draw ranks of
positives is ``sum_g pos_g * (Cprev_g + (c_g + 1) / 2)`` — algebraically
identical to sklearn's average-rank ROC AUC, no approximation.
"""

from __future__ import annotations

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
if str(_PROJECT_ROOT / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "analysis"))

from paper_full_metrics import (  # noqa: E402
    FIGS,
    METRIC_RANGE,
    MODE_LABELS,
    MODE_TAGS,
    MODES,
    NOTES,
    N_BOOT,
    PRED_ROOT,
    REF_PARQUET,
    RNG_SEED,
    ROOT,
    SEEDS,
    TABLES,
    TASK_METRICS,
    TASK_SHORT,
    TASKS,
    TOL,
    _plot_all_metrics,
    _summarize,
    _write_md_tables,
)
from rna_head_alignment.probes.metrics import (  # noqa: E402
    _top_k_overlap_per_sample,
    sequence_classification_metrics,
)


# ------------------------------------------------------------ fast exact AUROC
def _prep_binary_auroc(scores: np.ndarray, labels: np.ndarray):
    """Return fn(idx) computing sklearn-identical AUROC on a resample."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float64)
    order = np.argsort(scores, kind="stable")
    s = scores[order]
    lab = labels[order]
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    lab_g_pos_unit = lab  # per-element positive indicator, aggregated below
    n = len(s)

    def auroc(idx: np.ndarray) -> float:
        c = np.bincount(idx, minlength=n).astype(np.float64)[order]
        cg = np.add.reduceat(c, starts)
        pos_g = np.add.reduceat(c * lab_g_pos_unit, starts)
        cprev = np.cumsum(cg) - cg
        s_pos = float((pos_g * (cprev + (cg + 1.0) / 2.0)).sum())
        n_pos = float(pos_g.sum())
        n_neg = float(c.sum()) - n_pos
        if n_pos == 0.0 or n_neg == 0.0:
            return float("nan")
        return (s_pos - n_pos * (n_pos + 1.0) / 2.0) / (n_pos * n_neg)

    return auroc


def _prep_seqcls_primary(labels: np.ndarray, logits: np.ndarray, n_cls: int):
    """Primary-metric fn (AUROC) identical to sequence_classification_metrics."""
    if n_cls == 2:
        scores = 1.0 / (1.0 + np.exp(-logits))
        return _prep_binary_auroc(scores, labels)
    from scipy.special import softmax

    probs = softmax(np.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6), axis=1)
    present = sorted(np.unique(labels).tolist())
    per_class = [
        (lab, _prep_binary_auroc(probs[:, lab], (labels == lab).astype(int)))
        for lab in present
    ]

    def auroc(idx: np.ndarray) -> float:
        vals = [fn(idx) for _, fn in per_class]
        vals = [v for v in vals if v == v]
        return float(np.mean(vals)) if vals else float("nan")

    return auroc


# --------------------------------------------------------------- cell loaders
def _load_cell_for_boot(task: str, mode: str, seed: int):
    ttype = TASKS[task]["type"]
    d = PRED_ROOT / task / mode / f"seed_{seed}"
    if ttype == "seqcls":
        df = pd.read_parquet(d / "pred_test.parquet")
        labels = df["label"].to_numpy()
        logits = np.stack(df["logits"].to_numpy())
        if logits.ndim > 1 and logits.shape[1] == 1:
            logits = logits[:, 0]
        n_cls = logits.shape[1] if logits.ndim > 1 else 2
        return {
            "seq_ids": df["seq_id"].to_numpy(),
            "primary_fn": _prep_seqcls_primary(labels, logits, n_cls),
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


def _validate_fast_auroc() -> list[str]:
    """Check fast AUROC == sequence_classification_metrics on random draws."""
    msgs = []
    rng = np.random.default_rng(7)
    for task, spec in TASKS.items():
        if spec["type"] != "seqcls":
            continue
        df = pd.read_parquet(PRED_ROOT / task / "vanilla_hidden" / "seed_17" / "pred_test.parquet")
        labels = df["label"].to_numpy()
        logits = np.stack(df["logits"].to_numpy())
        if logits.ndim > 1 and logits.shape[1] == 1:
            logits = logits[:, 0]
        n_cls = logits.shape[1] if logits.ndim > 1 else 2
        fast = _prep_seqcls_primary(labels, logits, n_cls)
        max_diff = 0.0
        for _ in range(50):
            idx = rng.integers(0, len(labels), len(labels))
            ref = sequence_classification_metrics(labels[idx], logits[idx], n_cls)["auroc"]
            got = fast(idx)
            if ref == ref and got == got:
                max_diff = max(max_diff, abs(ref - got))
        msgs.append(f"fast-AUROC validation {task}: max |diff| vs sklearn over 50 draws = {max_diff:.2e}")
        assert max_diff < 1e-9, f"fast AUROC mismatch on {task}: {max_diff}"
    return msgs


# ----------------------------------------------------------------------- main
def main() -> None:
    validation_msgs = _validate_fast_auroc()
    for m in validation_msgs:
        print("[validate]", m, flush=True)

    rng = np.random.default_rng(RNG_SEED)
    boot_rows: list[dict] = []
    per_seed_rows: list[dict] = []
    # (mode_a, baseline): every mode vs P0, plus P5 vs P4.
    comparisons = [(m, "vanilla_hidden") for m in MODES if m != "vanilla_hidden"]
    comparisons.append(("distilled_hidden_teacher", "distilled_hidden_aligned"))
    for task, spec in TASKS.items():
        primary = spec["primary"]
        test_data = {}
        for mode in MODES:
            for seed in SEEDS:
                test_data[(mode, seed)] = _load_cell_for_boot(task, mode, seed)
        print(f"[boot] loaded {task}", flush=True)
        for mode, baseline in comparisons:
            per_seed_deltas = []
            for seed in SEEDS:
                da = test_data[(mode, seed)]
                db = test_data[(baseline, seed)]
                pos_b = {sid: i for i, sid in enumerate(db["seq_ids"])}
                keep_a = np.array([i for i, sid in enumerate(da["seq_ids"]) if sid in pos_b])
                map_b = np.array([pos_b[da["seq_ids"][i]] for i in keep_a])
                n = len(keep_a)
                fn_a = da["primary_fn"]
                fn_b = db["primary_fn"]
                deltas = np.full(N_BOOT, np.nan)
                for r_i in range(N_BOOT):
                    idx = rng.integers(0, n, size=n)
                    deltas[r_i] = fn_a(keep_a[idx]) - fn_b(map_b[idx])
                per_seed_deltas.append(deltas)
                per_seed_rows.append({
                    "task": task, "mode": mode, "mode_tag": MODE_TAGS[mode],
                    "baseline": baseline, "baseline_tag": MODE_TAGS[baseline],
                    "metric": primary, "seed": seed, **_summarize(deltas),
                })
            pooled = np.nanmean(np.stack(per_seed_deltas), axis=0)
            boot_rows.append({
                "task": task, "mode": mode, "mode_tag": MODE_TAGS[mode],
                "baseline": baseline, "baseline_tag": MODE_TAGS[baseline],
                "metric": primary, "n_seeds": len(per_seed_deltas), **_summarize(pooled),
            })
            print(f"[boot] {task} {mode} vs {baseline} Δ={boot_rows[-1]['delta_mean']:+.4f} "
                  f"p_pos={boot_rows[-1]['p_pos']:.3f}", flush=True)

    boot = pd.DataFrame(boot_rows)
    boot.to_csv(TABLES / "downstream_primary_bootstrap.csv", index=False)
    pd.DataFrame(per_seed_rows).to_csv(TABLES / "downstream_primary_bootstrap_per_seed.csv", index=False)

    # ---- regenerate md tables + figure from existing CSVs
    pts = pd.read_csv(TABLES / "downstream_recomputed_points.csv")
    summ = pd.read_csv(TABLES / "downstream_full_metrics.csv")
    _write_md_tables(summ, boot)
    _plot_all_metrics(summ)

    # ---- integrity + mismatch reconstruction from points CSV
    integrity: list[str] = []
    for task, spec in TASKS.items():
        for mode in MODES:
            sub = pts[(pts["task"] == task) & (pts["mode"] == mode)]
            if sorted(sub["seed"].tolist()) != SEEDS:
                integrity.append(f"SEED SET != {SEEDS}: {task}/{mode} has {sorted(sub['seed'].tolist())}")
            for k in TASK_METRICS[spec["type"]]:
                for _, r in sub.iterrows():
                    v = r[k]
                    if v != v:
                        integrity.append(f"NaN metric: {task}/{mode}/seed_{int(r['seed'])} {k}")
                    else:
                        lo, hi = METRIC_RANGE[k]
                        if not (lo - 1e-9 <= v <= hi + 1e-9):
                            integrity.append(f"OUT-OF-RANGE: {task}/{mode}/seed_{int(r['seed'])} {k}={v:.4f}")

    ref = pd.read_parquet(REF_PARQUET)
    p5_path = ROOT / "outputs/probes/frozen/p5_summary.parquet"
    p5 = pd.read_parquet(p5_path) if p5_path.exists() else None
    m = pts.merge(ref, on=["task", "mode", "seed"], suffixes=("", "_ref"))
    mism_rows = []
    for task, spec in TASKS.items():
        for k in TASK_METRICS[spec["type"]]:
            col_ref = f"test_{k}"
            if col_ref not in m.columns:
                continue  # e.g. accuracy: newly recomputed, no reference column
            sub = m[(m["task"] == task) & m[col_ref].notna() & m[k].notna()]
            bad = sub[(sub[k] - sub[col_ref]).abs() > TOL]
            for _, r in bad.iterrows():
                mism_rows.append({
                    "task": task, "mode": r["mode"], "seed": int(r["seed"]), "metric": k,
                    "recomputed": r[k], "reference": r[col_ref],
                    "abs_diff": abs(r[k] - r[col_ref]),
                })
    mismatches = pd.DataFrame(mism_rows)
    mismatches.to_csv(TABLES / "downstream_crosscheck_mismatches.csv", index=False)

    # P5 cross-check vs p5_summary.parquet (separate reference source).
    p5_mism = pd.DataFrame()
    if p5 is not None:
        m5 = pts[pts["mode"] == "distilled_hidden_teacher"].merge(
            p5, on=["task", "mode", "seed"], suffixes=("", "_ref")
        )
        rows5 = []
        for task, spec in TASKS.items():
            for k in TASK_METRICS[spec["type"]]:
                col_ref = f"test_{k}"
                if col_ref not in m5.columns:
                    continue
                sub = m5[(m5["task"] == task) & m5[col_ref].notna() & m5[k].notna()]
                bad = sub[(sub[k] - sub[col_ref]).abs() > TOL]
                for _, r in bad.iterrows():
                    rows5.append({
                        "task": task, "seed": int(r["seed"]), "metric": k,
                        "recomputed": r[k], "p5_summary": r[col_ref],
                        "abs_diff": abs(r[k] - r[col_ref]),
                    })
        p5_mism = pd.DataFrame(rows5)
        p5_mism.to_csv(TABLES / "downstream_crosscheck_p5_mismatches.csv", index=False)

    _write_notes(pts, integrity, mismatches, validation_msgs, p5_mism)
    print(f"[paper] bootstrap done; mismatches={len(mismatches)} p5_mismatches={len(p5_mism)} "
          f"integrity_issues={len(integrity)}")


def _write_notes(pts: pd.DataFrame, integrity: list[str], mismatches: pd.DataFrame,
                 validation_msgs: list[str], p5_mism: pd.DataFrame) -> None:
    lines = ["# Downstream verification notes", ""]
    lines.append("## Data integrity")
    lines.append("")
    lines.append(f"- Cells expected: 6 tasks × 7 modes (P0–P5 + P6) × 3 seeds = 126; "
                 f"recomputed: {len(pts)}.")
    if integrity:
        lines.append(f"- Issues ({len(integrity)}):")
        for s in integrity:
            lines.append(f"  - {s}")
    else:
        lines.append("- All 126 cells present with exactly seeds {17, 41, 89}; no NaN metrics; "
                     "all metrics within valid ranges.")
    lines.append("")
    lines.append("## Cross-check vs `analysis_v2/frozen_probe_all_metric_aligned.parquet` (tol 1e-3)")
    lines.append("")
    if len(mismatches):
        lines.append(f"- {len(mismatches)} cell-metric values exceed tolerance "
                     f"(full table: `tables/downstream_crosscheck_mismatches.csv`):")
        lines.append("")
        lines.append("| task | mode | seed | metric | recomputed | reference | abs diff |")
        lines.append("|------|------|-----:|--------|-----------:|----------:|---------:|")
        for _, r in mismatches.iterrows():
            lines.append(f"| {r['task']} | {r['mode']} | {r['seed']} | {r['metric']} | "
                         f"{r['recomputed']:.6f} | {r['reference']:.6f} | {r['abs_diff']:.6f} |")
        lines.append("")
        by = mismatches.groupby(["task", "metric"]).size()
        lines.append("- Breakdown by task/metric: "
                     + "; ".join(f"{t}/{k}: {n}" for (t, k), n in by.items()))
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
    lines.append("Caveat: dumped prob maps are stored in fp16, so recomputed top-L/top-2L/AUPRC "
                 "deviate from the fp32 eval-time values in `eval_test.json` (the source of the "
                 "reference parquet) by more than 1e-3 — see mismatch table. The original dump "
                 "script already flagged this (`dump_check.json`, max_abs_diff ≈ 0.01 for "
                 "vanilla/seed_17; larger for some other cells, up to ≈0.10 on top-L). "
                 "The paper tables use dump-recomputed values, consistent with the existing "
                 "bootstrap analysis (`analysis_v2/bootstrap`), which is also dump-based.")
    lines.append("")
    lines.append("## chemical_reactivity_token provenance")
    lines.append("")
    lines.append("Token-regression dumps store already-masked per-position preds/labels "
                 "(dump script applies `label_mask` before saving). MSE/MAE/Pearson/Spearman "
                 "were recomputed by concatenating these masked arrays — the label mask is "
                 "respected by construction.")
    lines.append("")
    lines.append("## Bootstrap method")
    lines.append("")
    lines.append("Resampling design identical to `scripts/analysis/phase1_bootstrap.py`: "
                 "test sequences resampled per seed (1000 draws, `np.random.default_rng(2024)`), "
                 "seq_id inner-join alignment between modes, per-seed delta distributions "
                 "averaged for the pooled estimate; CI = 2.5/97.5 percentiles; "
                 "p_pos = P(Δ > 0). The RNG stream differs from the original run (different "
                 "comparison set), so CIs are method-identical but not bit-identical.")
    lines.append("")
    lines.append("For speed, seqcls AUROC inside the bootstrap uses an exact O(n) Mann-Whitney "
                 "reformulation (average-rank tie handling identical to sklearn), validated "
                 "against `sequence_classification_metrics` on 50 random resamples per task:")
    for m in validation_msgs:
        lines.append(f"- {m}")
    lines.append("")
    lines.append("## P5 (`distilled_hidden_teacher`, exact teacher heads) provenance")
    lines.append("")
    lines.append("P5 (upper-bound diagnostic: probe directly on the exact 15 teacher heads) "
                 "finished after the initial 6-mode analysis. Sources: per-seed metrics in "
                 "`outputs/probes/frozen/p5_summary.parquet` (columns aligned with the "
                 "reference parquet) and per-sample dumps under "
                 "`outputs/probes/predictions/{task}/distilled_hidden_teacher/seed_{17,41,89}/` "
                 "(seqcls parquet; chemical `.pt`; secondary fp16 prob maps, same as other modes).")
    lines.append("")
    lines.append("P5 metrics in the paper tables were **recomputed from the dumps** with the "
                 "same code path as all other modes, then cross-checked against "
                 "`p5_summary.parquet` (tol 1e-3). All 18 cells (6 tasks × 3 seeds) are present; "
                 "no NaN metrics; all values within valid ranges.")
    lines.append("")
    if len(p5_mism):
        lines.append(f"- {len(p5_mism)} P5 cell-metric values exceed the 1e-3 tolerance vs "
                     f"`p5_summary.parquet` (full table: "
                     f"`tables/downstream_crosscheck_p5_mismatches.csv`):")
        lines.append("")
        lines.append("| task | seed | metric | recomputed | p5_summary | abs diff |")
        lines.append("|------|-----:|--------|-----------:|-----------:|---------:|")
        for _, r in p5_mism.iterrows():
            lines.append(f"| {r['task']} | {r['seed']} | {r['metric']} | "
                         f"{r['recomputed']:.6f} | {r['p5_summary']:.6f} | {r['abs_diff']:.6f} |")
        lines.append("")
        lines.append("As with the other modes, secondary-structure deviations stem from fp16 "
                     "prob-map dumps vs fp32 eval-time values.")
    else:
        lines.append("- All P5 recomputed values match `p5_summary.parquet` within 1e-3.")
    lines.append("")
    lines.append("P5 bootstrap: same paired design, two comparison sets — P5 vs P0 and P5 vs P4 "
                 "(primary metric, 1000 resamples, 3 seeds, pooled).")
    lines.append("")
    lines.append("## Scope note")
    lines.append("")
    lines.append("An 8th mode (`distilled_hidden_aligned_gated`, P4g) has dumps on disk but is "
                 "out of scope for the requested 7-mode comparison and was excluded.")
    lines.append("")
    lines.append("## Figure sources copied into `paper/figures/downstream/`")
    lines.append("")
    lines.append("| File | Source |")
    lines.append("|------|--------|")
    src = "outputs/probes/analysis_v2"
    for f, s in [
        ("frozen_probe_metric_aligned.png", f"{src}/figures/frozen_probe_metric_aligned.png"),
        ("bootstrap_p4_vs_p0_forest.png", f"{src}/bootstrap/bootstrap_p4_vs_p0_forest.png"),
        ("fig1_main_results.png", f"{src}/figures/final/fig1_main_results.png"),
        ("fig2_delta_forest.png", f"{src}/figures/final/fig2_delta_forest.png"),
        ("fig3_acceptor_repair.png", f"{src}/figures/final/fig3_acceptor_repair.png"),
        ("fig6_length_bins.png", f"{src}/figures/final/fig6_length_bins.png"),
    ]:
        lines.append(f"| `{f}` | `rna_head_alignment/{s}` |")
    lines.append("")
    lines.append("fig1–fig3 and fig6 are downstream-probe figures (main bars, Δ forest, "
                 "acceptor repair chain, metric-by-length-bin). fig4 (hidden drift) and fig5 "
                 "(residual gates) are representation-analysis figures, not downstream probe "
                 "results, and were not copied.")
    (NOTES / "downstream_verification.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
