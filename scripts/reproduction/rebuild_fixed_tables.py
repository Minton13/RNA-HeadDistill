#!/usr/bin/env python
# Provenance: ported from paper/scripts/rebuild_fixed_tables.py (RNA_LLM_v2
# monorepo) on 2026-11-13 for the public release.
# Changes vs. original: hard-coded cluster paths replaced by --data-root /
# --outdir; each table is skipped with a clear message when its raw per-seed
# artifacts are not staged.
"""Rebuild the ablation / fine-tuning paper tables from RAW per-seed artifacts.

NOTE (public release): the raw per-seed evaluation artifacts
(per-seed summary.parquet / eval JSON trees) are NOT part of the published
HF dataset, which ships the final aggregated CSVs under metrics/ablation/.
This script is kept for provenance and re-runs only when the raw artifacts
are staged under:

  <data-root>/metrics/ablation/raw/alignment/                    (A8 dir)
  <data-root>/metrics/ablation/raw/alignment_fullbb/             (A1-A7 dirs)
  <data-root>/metrics/ablation/raw/finetune/<task>/<mode>/seed_*/lambda_*/
  <data-root>/metrics/ablation/raw/multitask/<mode>/seed_*/
  <data-root>/metrics/ablation/raw/frozen_probe_summary.parquet

Tables rebuilt (to <outdir>/tables/):
  - alignment_ablation.csv/.md   (A1-A7 full-backbone re-run; A8 frozen by design)
  - alignment_ablation_delta.csv/.md
  - route_finetune.csv/.md
  - alignment_replay.csv/.md
  - multitask.csv/.md

For the published (canonical) tables, use reproduce_tables.py instead.

Run: python scripts/reproduction/rebuild_fixed_tables.py \
       --data-root /path/to/dataset-rna-head-distill --outdir ./reproduced_figures
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# Matched-subset comparability set: the 9 teacher heads (top-3 per route) shared by
# A2/A3/A5/A6 target sets, as (route_depth, teacher_model, teacher_layer_head).
MATCHED9 = {
    (6, "nucleotide_transformer_v2_500m", "L4:H3"),
    (6, "plantrna_fm", "L8:H18"),
    (6, "splicebert", "L5:H15"),
    (8, "plantrna_fm", "L10:H7"),
    (8, "plantrna_fm", "L9:H7"),
    (8, "utrlm_mrl", "L2:H10"),
    (12, "plantrna_fm", "L8:H2"),
    (12, "rinalmo_giga", "L31:H1"),
    (12, "rinalmo_giga", "L31:H13"),
}

SEEDS = [17, 41, 89]
LAMBDAS = ["0.0", "0.01", "0.1", "0.5"]
PRIMARY = {
    "chemical_reactivity_token": "pearson",
    "ncRNA_family_seqcls": "auroc",
    "rbp_binding_seqcls": "auroc",
    "secondary_structure_pair": "top_l_precision",
    "splice_acceptor_seqcls": "auroc",
    "splice_donor_seqcls": "auroc",
}
TASKS = list(PRIMARY)

# Populated by main() from --data-root / --outdir.
ABL = Path("UNUSED")
FULLBB = Path("UNUSED")
FT = Path("UNUSED")
MT = Path("UNUSED")
FROZEN = Path("UNUSED")
TAB = Path("UNUSED")


def default_data_root() -> Path:
    env = os.environ.get("RNA_HEAD_DISTILL_DATA")
    return Path(env) if env else Path("./dataset-rna-head-distill")


def mean_std(vals):
    # Sample std (ddof=1) for n>1, 0.0 for n=1 — the convention used across
    # all paper tables so they agree.
    v = [float(x) for x in vals if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not v:
        return (np.nan, np.nan, 0)
    return (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))


def agg_student(summary_path: Path) -> dict[str, float]:
    """Validated recipe: mean of `mean` over baseline=='student' rows per metric."""
    s = pd.read_parquet(summary_path)
    st = s[s.baseline == "student"]
    return {m: float(g["mean"].mean()) for m, g in st.groupby("metric")}


def agg_student_detailed(summary_path: Path) -> tuple[dict[str, float], dict[int, dict[str, float]], dict[str, float] | None]:
    """Per-seed aggregation with route split and matched-subset (MATCHED9) values.

    Returns (own_target_macro, per_route_macro, matched9_macro_or_None).
    Per-head values are count-weighted over length bins, then macro-averaged.
    """
    s = pd.read_parquet(summary_path)
    st = s[s.baseline == "student"].copy()
    st["wm"] = st["mean"] * st["count"]
    per_head = (
        st.groupby(["route_depth", "teacher_model", "teacher_layer_head", "metric"])
        .apply(lambda g: g["wm"].sum() / g["count"].sum(), include_groups=False)
        .unstack("metric")
    )
    own = {m: float(per_head[m].mean()) for m in per_head.columns}
    per_route = {
        int(r): {m: float(g[m].mean()) for m in per_head.columns}
        for r, g in per_head.groupby(level="route_depth")
    }
    matched_idx = [i for i in per_head.index if tuple(i) in MATCHED9]
    matched = None
    if len(matched_idx) == len(MATCHED9):
        sub = per_head.loc[matched_idx]
        matched = {m: float(sub[m].mean()) for m in per_head.columns}
    return own, per_route, matched


def read_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


# ---------------------------------------------------------------- md helpers
def _pm(m, s, n, prec=3):
    """mean ± std (n=k) cell, matching alignment_ablation.md conventions."""
    if n == 0 or (isinstance(m, float) and np.isnan(m)):
        return "n/a"
    return f"{m:.{prec}f} ± {s:.{prec}f} (n={n})"


def _lam(lam: str) -> str:
    return lam.rstrip("0").rstrip(".") if "." in lam else lam


# English translations for the Chinese disclosure notes stored in the CSV `notes`
# column. Lookup is per clause (notes may join several clauses with ";").
NOTE_EN = {
    "F1 三 seed 比特级相同(std=0 为结构性)":
        "the three F1 seeds are bit-identical; std = 0 is structural, not evidence of stability",
    "F1 三 seed 比特级相同(结构性)":
        "the three F1 seeds are bit-identical (structural)",
    "F1/F2 下游指标来自修复后重估 (softmax 修复 + best-checkpoint 重选)":
        "F1/F2 downstream metrics come from the post-fix re-evaluation "
        "(softmax fix + best-checkpoint reselection)",
    "F1/F2 为后补运行 (2026-09-09 fix)":
        "F1/F2 are backfilled runs (2026-09-09 fix)",
    "lambda=0 三 seed 比特级相同(std=0 为结构性)":
        "lambda=0 rows are bit-identical across seeds (std = 0 is structural)",
    "test 指标来自修复后重估":
        "test metrics come from the post-fix re-evaluation",
    "joint AUROC 取自修复后 eval(_fixed)":
        "joint AUROC is taken from the repaired evals (eval_*_fixed.json)",
    "宏观 F1≈0.04/MCC≈0 为单类退化":
        "macro F1 ~ 0.04 / MCC ~ 0 is single-class degeneracy",
    "独立 F1 为修复后重估":
        "independent F1 is from the post-fix re-evaluation",
    "joint 训练使该任务 AUROC 降至随机水平(≈0.49)":
        "joint training collapses this task's AUROC to chance level (~0.49)",
    "joint 模型 F1=0.7228 为全阳退化(阈值 0)":
        "joint model F1 = 0.7228 is all-positive degeneracy (threshold 0)",
    "joint 与 F1 的 F1/MCC=0 均为阈值退化,AUROC 有效":
        "F1/MCC = 0 for both joint and independent F1 are threshold degeneracy; AUROC is valid",
}


def _note_footnotes(df: pd.DataFrame) -> list[str]:
    """Turn the CSV `notes` column into bilingual footnote bullets (original kept)."""
    seen, bullets = set(), []
    for _, r in df.iterrows():
        note = str(r.get("notes", "") or "").strip()
        if not note:
            continue
        for clause in [c.strip() for c in note.split(";") if c.strip()]:
            if clause in seen:
                continue
            seen.add(clause)
            en = NOTE_EN.get(clause)
            bullets.append(f"- {clause}" + (f" — {en}." if en else ""))
    return bullets


# ---------------------------------------------------------------- ablation
def ablation_row_fullbb(exp_id: str, out_root: Path, seeds, status: str) -> dict:
    """Aggregate one fullbb experiment from per-seed eval summary parquets."""
    mets = {"KL": "rowwise_kl", "MSE": "mse", "Cosine": "cosine",
            "Pearson": "pearson", "TopL": "top_l_overlap"}
    collected: dict[str, list[float]] = {k: [] for k in mets}
    per_route: dict[int, dict[str, list[float]]] = {}
    matched: dict[str, list[float]] = {k: [] for k in mets}
    n_matched = 0
    wall, vram = [], []
    for s in seeds:
        summ = out_root / f"{exp_id}_seed{s}" / "test" / "summary.parquet"
        if summ.exists():
            own, routes, m9 = agg_student_detailed(summ)
            for short, m in mets.items():
                collected[short].append(own.get(m, np.nan))
            for r, vals in routes.items():
                slot = per_route.setdefault(r, {k: [] for k in mets})
                for short, m in mets.items():
                    slot[short].append(vals.get(m, np.nan))
            if m9 is not None:
                n_matched += 1
                for short, m in mets.items():
                    matched[short].append(m9.get(m, np.nan))
        tp = out_root / f"seed_{s}" / "throughput_memory.csv"
        if tp.exists():
            t = pd.read_csv(tp)
            if "wall_time_sec" in t.columns and len(t):
                wall.append(float(t["wall_time_sec"].iloc[-1]) / 3600.0)
            if "peak_vram_gb" in t.columns and len(t):
                vram.append(float(t["peak_vram_gb"].max()))
    row = {"experiment": exp_id, "status": status,
           "n_seeds_metrics": mean_std(collected["Cosine"])[2]}
    for short in mets:
        m, sd, _ = mean_std(collected[short])
        row[f"{short}_mean"], row[f"{short}_std"] = m, sd
    for r in (6, 8, 12):
        for short in ("KL", "Cosine"):
            vals = per_route.get(r, {}).get(short, [])
            m, sd, _ = mean_std(vals)
            row[f"{short}_L{r}_mean"], row[f"{short}_L{r}_std"] = m, sd
    for short in ("KL", "Cosine"):
        m, sd, _ = mean_std(matched[short])
        row[f"{short}_matched9_mean"], row[f"{short}_matched9_std"] = m, sd
    row["n_seeds_matched9"] = n_matched
    row["wall_h_mean"], row["wall_h_std"] = mean_std(wall)[:2]
    row["vram_mean"], row["vram_std"] = mean_std(vram)[:2]
    return row


def rebuild_alignment_ablation() -> None:
    """Full-backbone (corrected) core ablation table.

    A1-A7 come from raw/alignment_fullbb (activation_checkpointing=false,
    step-0 backbone-gradient guard). A8 stays on the original raw/alignment
    directory (frozen backbone by design). A0 is the untrained-student manifest.
    """
    if not FULLBB.exists() or not ABL.exists():
        print(f"[skip] alignment_ablation: raw per-seed artifacts not staged "
              f"(need {FULLBB} and {ABL}); use the canonical CSV shipped under "
              f"metrics/ablation/ instead")
        return
    status_note = {
        "A1_single_teacher_top1": "full-backbone training x3 (from vanilla init)",
        "A2_head2head_top3": "full-backbone training x3 (from vanilla init)",
        "A3_head2head_top5_full": "full-backbone training x3 (from vanilla init; main-method config)",
        "A4_consensus_top5": "full-backbone training x3 (from vanilla init; consensus targets have no head identity)",
        "A5_random_pairing": "full-backbone training x3 (from vanilla init)",
        "A6_linear_adapter": "full-backbone training x3 (from vanilla init)",
        "A7_all_from_L12": "full-backbone training x3 (from vanilla init; all heads read from L12)",
    }
    order = ["A1_single_teacher_top1", "A2_head2head_top3", "A3_head2head_top5_full",
             "A4_consensus_top5", "A5_random_pairing", "A6_linear_adapter", "A7_all_from_L12"]
    rows = [{"experiment": "A0_vanilla",
             "status": "vanilla manifest x3 (no alignment training, no metrics)",
             "n_seeds_metrics": 0}]
    for exp in order:
        rows.append(ablation_row_fullbb(exp, FULLBB / exp, SEEDS, status_note[exp]))
    # A8: frozen by design; reuse the original directory and layout.
    a8 = ablation_row_fullbb(
        "A8_adapters_only", ABL / "A8_adapters_only", SEEDS,
        "adapters-only x3 (backbone frozen BY DESIGN; original run, not affected by the bug)")
    rows.append(a8)

    df = pd.DataFrame(rows)
    df.to_csv(TAB / "alignment_ablation.csv", index=False)

    def pm(m, s, n, prec=3, sci=False):
        if n == 0 or (isinstance(m, float) and np.isnan(m)):
            return "n/a"
        return f"{m:.{prec}e} ± {s:.{prec}e} (n={n})" if sci else f"{m:.{prec}f} ± {s:.{prec}f} (n={n})"

    lines = ["# Table. Core alignment ablation (A0-A8), test split — FULL-BACKBONE re-run (2026-09-14)",
             "",
             "| Experiment | Status | KL ↓ (own targets) | MSE ↓ | Cosine ↑ (own targets) | Pearson ↑ | Top-L ↑ | KL ↓ matched-9 | Cosine ↑ matched-9 | Wall (h) | VRAM (GB) |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        n = int(r["n_seeds_metrics"])
        _nm = r.get("n_seeds_matched9", 0)
        nm = int(_nm) if _nm == _nm else 0  # NaN-safe (A0 row has no matched9 key)
        lines.append(
            f"| {r['experiment']} | {r['status']} | "
            f"{pm(r['KL_mean'], r['KL_std'], n)} | {pm(r['MSE_mean'], r['MSE_std'], n, sci=True)} | "
            f"{pm(r['Cosine_mean'], r['Cosine_std'], n)} | {pm(r['Pearson_mean'], r['Pearson_std'], n)} | "
            f"{pm(r['TopL_mean'], r['TopL_std'], n)} | "
            f"{pm(r.get('KL_matched9_mean', np.nan), r.get('KL_matched9_std', np.nan), nm)} | "
            f"{pm(r.get('Cosine_matched9_mean', np.nan), r.get('Cosine_matched9_std', np.nan), nm)} | "
            f"{pm(r.get('wall_h_mean', np.nan), r.get('wall_h_std', np.nan), n, prec=2)} | "
            f"{pm(r.get('vram_mean', np.nan), r.get('vram_std', np.nan), n)} |"
        )
    lines.append("")
    lines.append("Methodological notes:")
    lines.append("- FULL-BACKBONE re-run: an earlier version of this table (kept as `alignment_ablation.*.bak_frozen`) "
                 "unknowingly trained A1-A7 with a frozen backbone — hook-captured activations detach under "
                 "activation checkpointing, so the backbone received zero gradients. The corrected runs use "
                 "activation_checkpointing=false plus a fail-loud step-0 backbone-gradient guard.")
    lines.append("- Own-target macro metrics average over each experiment's own target heads, so absolute values "
                 "are NOT comparable across experiments with different target scopes (A1: 3 heads; A2: 9; "
                 "A3/A5/A6: 15; A7: 15 read from L12; A4: route-consensus maps).")
    lines.append("- Matched-9 columns restrict every experiment to the same 9 teacher heads (top-3 per route: "
                 "L6: nucleotide_transformer_v2_500m/L4:H3, plantrna_fm/L8:H18, splicebert/L5:H15; "
                 "L8: plantrna_fm/L10:H7, plantrna_fm/L9:H7, utrlm_mrl/L2:H10; "
                 "L12: plantrna_fm/L8:H2, rinalmo_giga/L31:H1, rinalmo_giga/L31:H13). "
                 "Defined only for A2/A3/A5/A6 (A1 covers 3 of the 9; A4/A7 lack matching per-head rows).")
    lines.append("- A8 is the intended frozen-backbone comparator (adapters only) and was not affected by the bug.")
    lines.append("- The earlier warm-start A3 rows (seed17 reused stage4b; seeds 41/89 warm-started) are retained "
                 "in the frozen-domain backup for provenance only and are superseded by the A3 row above.")
    (TAB / "alignment_ablation.md").write_text("\n".join(lines) + "\n")
    print("[rebuild] alignment_ablation.csv/.md (fullbb)")
    print(df[["experiment", "KL_mean", "Cosine_mean", "KL_matched9_mean", "Cosine_matched9_mean",
              "KL_L6_mean", "KL_L8_mean", "KL_L12_mean"]].to_string())

    # ------------------------------------------------ delta table (fullbb, matched-9 where defined)
    higher_better = {"KL": False, "Cosine": True}
    comparability = {
        "A1_single_teacher_top1": "partial: target scope differs (3 heads vs 15); own-target macro only",
        "A2_head2head_top3": "matched-9 subset (top-3/route); same training budget",
        "A4_consensus_top5": "NO for absolute values (route-consensus targets); listed for reference",
        "A5_random_pairing": "matched-9 subset; same 15 targets overall",
        "A6_linear_adapter": "matched-9 subset; same 15 targets overall",
        "A7_all_from_L12": "same 15 targets but all read from L12; own-target macro only",
        "A8_adapters_only": "same 15 targets; frozen backbone by design",
    }
    a3 = df[df.experiment == "A3_head2head_top5_full"].iloc[0]
    drows = []
    for comp, note in comparability.items():
        c = df[df.experiment == comp].iloc[0]
        for basis, suffix in (("own-target", ""), ("matched-9", "_matched9")):
            for short in ("KL", "Cosine"):
                am, bm = a3.get(f"{short}{suffix}_mean"), c.get(f"{short}{suffix}_mean")
                if np.isnan(am) or np.isnan(bm):
                    continue
                d = am - bm
                win = (d > 0) if higher_better[short] else (d < 0)
                drows.append({"comparison": f"A3 vs {comp}", "basis": basis, "metric": short,
                              "higher_is_better": higher_better[short],
                              "A3_mean": am, "comparator_mean": bm,
                              "delta_A3minusComp": d, "A3_wins": bool(win),
                              "comparability": note})
    ddf = pd.DataFrame(drows)
    ddf.to_csv(TAB / "alignment_ablation_delta.csv", index=False)
    dlines = ["# Table. Δ comparisons against the full method (A3), FULL-BACKBONE re-run (2026-09-14)",
              "",
              "Δ = A3 − comparator (seed means, n=3). For ↑ metrics Δ>0 favours A3; for ↓ metrics Δ<0 favours A3.",
              "\"matched-9\" restricts both sides to the same 9 teacher heads (see alignment_ablation.md).",
              "Seed-level spread only (n=3); no per-sample bootstrap at this level.",
              "",
              "| Comparison | Basis | Metric | A3 | Comparator | Δ | A3 wins | Comparability |",
              "|---|---|---|---|---|---|---|---|"]
    for _, r in ddf.iterrows():
        dlines.append(
            f"| {r['comparison']} | {r['basis']} | {r['metric']} {'↑' if r['higher_is_better'] else '↓'} | "
            f"{r['A3_mean']:.4f} | {r['comparator_mean']:.4f} | {r['delta_A3minusComp']:+.4f} | "
            f"{'yes' if r['A3_wins'] else 'no'} | {r['comparability'].split(':')[0]} |")
    dlines.append("")
    dlines.append("Headline: on the matched-9 subset A2 (top-3/route) beats A3 (top-5/route), and A3 ≈ A5 "
                  "(random pairing) — see alignment_ablation.md notes and the manuscript ablation paragraph.")
    (TAB / "alignment_ablation_delta.md").write_text("\n".join(dlines) + "\n")
    print("[rebuild] alignment_ablation_delta.csv/.md (fullbb)")


# ---------------------------------------------------------------- finetune tables
def ft_cell(task: str, mode: str, seed: int, lam: str) -> Path:
    return FT / task / mode / f"seed_{seed}" / f"lambda_{lam}"


def ft_test_metric(task: str, mode: str, seed: int, lam: str) -> float:
    """Test primary metric for one cell; prefers fixed evals when present."""
    d = ft_cell(task, mode, seed, lam)
    metric = PRIMARY[task]
    for name in ("eval_test_fixed.json", "eval_test.json"):
        e = read_json(d / name)
        if e and not np.isnan(float(e.get(metric, np.nan))):
            return float(e[metric])
    return np.nan


def ft_align(task: str, seed: int, lam: str) -> tuple[dict, dict]:
    s = read_json(ft_cell(task, "F2", seed, lam) / "training_summary.json")
    return s.get("pre_alignment", {}) or {}, s.get("post_alignment", {}) or {}


def macro(align: dict, metric: str) -> float:
    vals = [align.get(f"L{d}_{metric}", np.nan) for d in (6, 8, 12)]
    vals = [v for v in vals if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan


def rebuild_route_finetune() -> None:
    if not FT.exists() or not FROZEN.exists():
        print(f"[skip] route_finetune: raw artifacts not staged (need {FT} and "
              f"{FROZEN}); use the canonical CSV under metrics/ablation/ instead")
        return
    frozen = pd.read_parquet(FROZEN)
    rows = []
    for task in TASKS:
        pm_ = PRIMARY[task]
        f0 = frozen[(frozen.task == task) & (frozen.feature_mode == "distilled_hidden")]
        f0_m, f0_s, f0_n = mean_std(f0[f"test_{pm_}"])
        f1_m, f1_s, f1_n = mean_std([ft_test_metric(task, "F1", s, "0.0") for s in SEEDS])
        f2_m, f2_s, f2_n = mean_std([ft_test_metric(task, "F2", s, "0.1") for s in SEEDS])
        cos_pre, cos_post = [], []
        for s in SEEDS:
            pre, post = ft_align(task, s, "0.1")
            cos_pre.append(macro(pre, "cosine"))
            cos_post.append(macro(post, "cosine"))
        notes = {
            "ncRNA_family_seqcls": "F1/F2 下游指标来自修复后重估 (softmax 修复 + best-checkpoint 重选)",
            "secondary_structure_pair": "F1/F2 为后补运行 (2026-09-09 fix);F1 三 seed 比特级相同(结构性)",
        }.get(task, "")
        if task in ("chemical_reactivity_token", "rbp_binding_seqcls"):
            notes = "F1 三 seed 比特级相同(std=0 为结构性)"
        rows.append({
            "task": task, "primary_metric": pm_,
            "F0_mean": f0_m, "F0_std": f0_s, "F0_n": f0_n,
            "F1_mean": f1_m, "F1_std": f1_s, "F1_n": f1_n,
            "F2_replay_mean": f2_m, "F2_replay_std": f2_s, "F2_replay_n": f2_n,
            "F2_align_cos_pre": round(float(np.nanmean(cos_pre)), 4) if cos_pre else np.nan,
            "F2_align_cos_post": round(float(np.nanmean(cos_post)), 4) if cos_post else np.nan,
            "notes": notes,
        })
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "route_finetune.csv", index=False)
    write_route_finetune_md(df)
    print("[rebuild] route_finetune.csv/.md")
    print(df.to_string())


def rebuild_alignment_replay() -> None:
    if not FT.exists():
        print(f"[skip] alignment_replay: raw artifacts not staged (need {FT}); "
              f"use the canonical CSV under metrics/ablation/ instead")
        return
    rows = []
    for task in TASKS:
        pm_ = PRIMARY[task]
        for lam in LAMBDAS:
            t_m, t_s, t_n = mean_std([ft_test_metric(task, "F2", s, lam) for s in SEEDS])
            cos_pre, cos_post, kl_pre, kl_post = [], [], [], []
            for s in SEEDS:
                pre, post = ft_align(task, s, lam)
                cos_pre.append(macro(pre, "cosine")); cos_post.append(macro(post, "cosine"))
                kl_pre.append(macro(pre, "rowwise_kl")); kl_post.append(macro(post, "rowwise_kl"))
            note = ""
            if lam == "0.0":
                note = "lambda=0 三 seed 比特级相同(std=0 为结构性)"
            if task == "ncRNA_family_seqcls":
                note = (note + "; " if note else "") + "test 指标来自修复后重估"
            rows.append({
                "task": task, "primary_metric": pm_, "lambda_align": lam,
                "test_mean": t_m, "test_std": t_s, "n_seeds": t_n,
                "align_cos_pre": round(float(np.nanmean(cos_pre)), 4),
                "align_cos_post": round(float(np.nanmean(cos_post)), 4),
                "delta_cos": round(float(np.nanmean(cos_post) - np.nanmean(cos_pre)), 4),
                "align_kl_pre": round(float(np.nanmean(kl_pre)), 4),
                "align_kl_post": round(float(np.nanmean(kl_post)), 4),
                "delta_kl": round(float(np.nanmean(kl_post) - np.nanmean(kl_pre)), 4),
                "notes": note,
            })
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "alignment_replay.csv", index=False)
    write_alignment_replay_md(df)
    print("[rebuild] alignment_replay.csv/.md")


def rebuild_multitask() -> None:
    if not MT.exists() or not FT.exists():
        print(f"[skip] multitask: raw artifacts not staged (need {MT} and {FT}); "
              f"use the canonical CSV under metrics/ablation/ instead")
        return
    rows = []
    for task in TASKS:
        pm_ = PRIMARY[task]
        f1_m, f1_s, f1_n = mean_std([ft_test_metric(task, "F1", s, "0.0") for s in SEEDS])
        row = {"task": task, "primary_metric": pm_,
               "independent_F1_mean": f1_m, "independent_F1_std": f1_s, "independent_F1_n": f1_n}
        for mode in ("joint_no_replay", "joint_with_replay"):
            vals, hours = [], []
            for s in SEEDS:
                d = MT / mode / f"seed_{s}" / task
                e = read_json(d / "eval_test_fixed.json") or read_json(d / "eval_test.json")
                v = float(e.get(pm_, np.nan)) if e else np.nan
                vals.append(v)
                ts = read_json(MT / mode / f"seed_{s}" / "training_summary.json")
                if ts.get("wall_time_sec"):
                    hours.append(float(ts["wall_time_sec"]) / 3600.0)
            m, sd, n = mean_std(vals)
            row[f"{mode}_mean"], row[f"{mode}_std"], row[f"{mode}_n"] = m, sd, n
            row[f"{mode}_gpu_hours_total"] = round(sum(hours), 3) if hours else np.nan
            # forgetting: macro cosine / rowwise_kl deltas from post-hoc measurement
            fcos, fkl = [], []
            for s in SEEDS:
                f = read_json(MT / mode / f"seed_{s}" / "alignment_forgetting.json")
                if f and "macro" in f:
                    fcos.append(f["macro"]["cosine"]["delta"])
                    fkl.append(f["macro"]["rowwise_kl"]["delta"])
            fm, fs_, fn = mean_std(fcos)
            km, ks, kn = mean_std(fkl)
            row[f"{mode}_forget_cos_delta_mean"] = round(fm, 4) if fn else np.nan
            row[f"{mode}_forget_cos_delta_std"] = round(fs_, 4) if fn else np.nan
            row[f"{mode}_forget_kl_delta_mean"] = round(km, 4) if kn else np.nan
            row[f"{mode}_forget_kl_delta_std"] = round(ks, 4) if kn else np.nan
        notes = {
            "ncRNA_family_seqcls": "joint AUROC 取自修复后 eval(_fixed);宏观 F1≈0.04/MCC≈0 为单类退化;独立 F1 为修复后重估",
            "rbp_binding_seqcls": "joint 训练使该任务 AUROC 降至随机水平(≈0.49)",
            "splice_acceptor_seqcls": "joint 模型 F1=0.7228 为全阳退化(阈值 0)",
            "splice_donor_seqcls": "joint 与 F1 的 F1/MCC=0 均为阈值退化,AUROC 有效",
        }.get(task, "")
        row["notes"] = notes
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(TAB / "multitask.csv", index=False)
    write_multitask_md(df)
    print("[rebuild] multitask.csv/.md")
    print(df.to_string())


# ---------------------------------------------------------------- md writers
def write_route_finetune_md(df: pd.DataFrame) -> None:
    lines = [
        "# Table. Route-specific fine-tuning vs frozen probe (F0/F1/F2), test split, mean ± std over seeds 17/41/89",
        "",
        "F0: linear/MLP probe on the frozen Stage-4B head-to-head student (`distilled_hidden` features).",
        "F1: task fine-tuning (backbone up to the task route + route adapters + probe unfrozen), no alignment replay.",
        "F2: same as F1 with alignment replay at λ_align = 0.1 (replay λ sweep in `alignment_replay.md`).",
        "Alignment retention: mean teacher-head cosine similarity on the replay sample before→after "
        "fine-tuning (avg over routes L6/L8/L12; only recorded for F2 runs).",
        "",
        "| Task | Primary metric ↑ | F0 frozen probe | F1 fine-tune (no replay) | F2 fine-tune + replay (λ=0.1) | Alignment cos pre→post (F2, λ=0.1) |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['task']} | {r['primary_metric']} | {_pm(r['F0_mean'], r['F0_std'], r['F0_n'])} | "
            f"{_pm(r['F1_mean'], r['F1_std'], r['F1_n'])} | "
            f"{_pm(r['F2_replay_mean'], r['F2_replay_std'], r['F2_replay_n'])} | "
            f"{r['F2_align_cos_pre']:.4f} → {r['F2_align_cos_post']:.4f} |"
        )
    lines.append("")
    lines.append("Notes (disclosures from the CSV `notes` column; Chinese original kept, with translation):")
    lines.extend(_note_footnotes(df))
    lines.append("")
    lines.append(
        "All fine-tuned rows (F1/F2) score below the F0 frozen probe on every task; replay at λ=0.1 "
        "does not rescue downstream performance, though it improves alignment retention "
        "(cos pre→post increases on five of six tasks; rbp_binding_seqcls is essentially flat at −0.002)."
    )
    (TAB / "route_finetune.md").write_text("\n".join(lines) + "\n")


def write_alignment_replay_md(df: pd.DataFrame) -> None:
    lines = [
        "# Table. Alignment-replay ablation over λ_align (F2 runs), test split, mean ± std over seeds 17/41/89",
        "",
        "Forgetting is measured on a 1,024-sequence replay sample as the change in teacher-head alignment "
        "from before to after fine-tuning, averaged over routes L6/L8/L12. "
        "Δcos = post − pre cosine (positive = alignment improved); ΔKL = post − pre row-wise KL "
        "(negative = alignment improved). λ = 0 is the no-replay control (replay batches sampled but zero-weighted).",
        "",
        "| Task | Primary metric ↑ | λ_align | Downstream (test) | Δcos (post−pre) | ΔKL (post−pre) | Post cosine |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['task']} | {r['primary_metric']} | {_lam(str(r['lambda_align']))} | "
            f"{_pm(r['test_mean'], r['test_std'], r['n_seeds'])} | "
            f"{r['delta_cos']:+.4f} | {r['delta_kl']:+.4f} | {r['align_cos_post']:.4f} |"
        )
    lines.append("")
    lines.append("Notes (disclosures from the CSV `notes` column; Chinese original kept, with translation):")
    lines.extend(_note_footnotes(df))
    lines.append("")
    lines.append(
        "Without replay (λ=0) fine-tuning degrades alignment on the binary sequence-classification tasks "
        "(Δcos < 0, ΔKL > 0); with replay, alignment improves monotonically in λ while the downstream "
        "cost stays small but grows with λ. All fine-tuned downstream numbers remain below the F0 "
        "frozen-probe baseline (see route_finetune.md)."
    )
    (TAB / "alignment_replay.md").write_text("\n".join(lines) + "\n")


def write_multitask_md(df: pd.DataFrame) -> None:
    lines = [
        "# Table. Multi-task joint model vs independent single-task fine-tunes, test split, mean ± std over seeds 17/41/89",
        "",
        "Independent = per-task F1 route fine-tune (no replay). Joint = one student fine-tuned on all six "
        "tasks jointly, without / with alignment replay (λ_align = 0.1).",
        "",
        "| Task | Primary metric ↑ | Independent F1 | Joint (no replay) | Joint (+ replay) |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        lines.append(
            f"| {r['task']} | {r['primary_metric']} | "
            f"{_pm(r['independent_F1_mean'], r['independent_F1_std'], r['independent_F1_n'])} | "
            f"{_pm(r['joint_no_replay_mean'], r['joint_no_replay_std'], r['joint_no_replay_n'])} | "
            f"{_pm(r['joint_with_replay_mean'], r['joint_with_replay_std'], r['joint_with_replay_n'])} |"
        )
    r0 = df.iloc[0]
    lines.append("")
    lines.append(
        f"Compute: joint no-replay {r0['joint_no_replay_gpu_hours_total']:.2f} GPU-h total over 3 seeds; "
        f"joint with-replay {r0['joint_with_replay_gpu_hours_total']:.2f} GPU-h total. "
        "One joint run replaces six task runs but trains one shared model."
    )
    lines.append("")
    lines.append(
        "Alignment forgetting (post-hoc measurement, macro over routes L6/L8/L12; identical across task "
        "rows because it is a property of the joint run): no-replay "
        f"Δcos {r0['joint_no_replay_forget_cos_delta_mean']:+.4f} ± {r0['joint_no_replay_forget_cos_delta_std']:.4f}, "
        f"ΔKL {r0['joint_no_replay_forget_kl_delta_mean']:+.4f} ± {r0['joint_no_replay_forget_kl_delta_std']:.4f}; "
        f"with-replay Δcos {r0['joint_with_replay_forget_cos_delta_mean']:+.4f} ± {r0['joint_with_replay_forget_cos_delta_std']:+.4f}, "
        f"ΔKL {r0['joint_with_replay_forget_kl_delta_mean']:+.4f} ± {r0['joint_with_replay_forget_kl_delta_std']:+.4f}. "
        "Replay markedly increases alignment retention during joint training "
        "(Δcos +0.036 vs +0.005 without replay; ΔKL −0.199 vs −0.030)."
    )
    lines.append("")
    lines.append("Notes (disclosures from the CSV `notes` column; Chinese original kept, with translation):")
    lines.extend(_note_footnotes(df))
    lines.append("")
    lines.append(
        "Joint training is worse than independent F1 on every task except ncRNA_family_seqcls, and "
        "independent F1 is itself far below the F0 frozen probe (see route_finetune.md)."
    )
    (TAB / "multitask.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    global ABL, FULLBB, FT, MT, FROZEN, TAB
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=default_data_root(),
                    help="root of the downloaded HF dataset "
                         "(default: $RNA_HEAD_DISTILL_DATA)")
    ap.add_argument("--outdir", type=Path, default=Path("./reproduced_figures"),
                    help="output root; tables go to <outdir>/tables/")
    args = ap.parse_args()
    raw = args.data_root / "metrics" / "ablation" / "raw"
    ABL = raw / "alignment"
    FULLBB = raw / "alignment_fullbb"
    FT = raw / "finetune"
    MT = raw / "multitask"
    FROZEN = raw / "frozen_probe_summary.parquet"
    TAB = args.outdir / "tables"
    TAB.mkdir(parents=True, exist_ok=True)
    rebuild_alignment_ablation()
    rebuild_route_finetune()
    rebuild_alignment_replay()
    rebuild_multitask()


if __name__ == "__main__":
    main()
