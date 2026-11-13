#!/usr/bin/env python
# Provenance: ported from paper/scripts/rebuild_pairing_stability.py
# (RNA_LLM_v2 monorepo) on 2026-11-13 for the public release.
# Changes vs. original: hard-coded cluster paths replaced by --data-root /
# --outdir; skips with a clear message when the raw bootstrap parquets are
# not staged (the published HF dataset ships the final pairing_stability.csv
# under metrics/ablation/).
"""Rebuild pairing_stability.{csv,md} from the completed bootstrap products.

Inputs (under <data-root>/metrics/ablation/):
  - pairing_stability/pairing_stability_summary.parquet  (75 rows: 5 bootstraps x 15 teacher heads)
  - pairing_stability/pairing_stability_per_head.parquet (15 rows: per-head mode/match stats)
  - pairing_stability/bootstrap{b}_seed17/pair_search_metadata.json  (n_shards, method metadata)
  - supplementary_indices/pairing_stability_bootstrap{b}_nshards16_seed17.parquet (row counts)

Definitions used in the table:
  - pairing frequency (per teacher head): mode frequency of the head's student-head
    assignment across the 5 bootstraps (from pairing_stability_per_head.parquet).
  - pairing frequency (per bootstrap): fraction of the 15 teacher heads whose assignment
    in that bootstrap equals the head's mode assignment. Averaged over bootstraps this is
    exactly the mean mode frequency.
  - top-1 agreement: fraction of assignments equal to the reference pairing
    (reference_student_head; computed earlier on 4 slim shards — protocol differs).

Run: python scripts/reproduction/rebuild_pairing_stability.py \
       --data-root /path/to/dataset-rna-head-distill --outdir ./reproduced_figures
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

N_HEADS = 15


def default_data_root() -> Path:
    env = os.environ.get("RNA_HEAD_DISTILL_DATA")
    return Path(env) if env else Path("./dataset-rna-head-distill")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=default_data_root(),
                    help="root of the downloaded HF dataset "
                         "(default: $RNA_HEAD_DISTILL_DATA)")
    ap.add_argument("--outdir", type=Path, default=Path("./reproduced_figures"),
                    help="output root; tables go to <outdir>/tables/")
    args = ap.parse_args()
    PS = args.data_root / "metrics" / "ablation" / "pairing_stability"
    IDX = args.data_root / "metrics" / "ablation" / "supplementary_indices"
    TAB = args.outdir / "tables"

    if not (PS / "pairing_stability_summary.parquet").exists() \
            or not (PS / "pairing_stability_per_head.parquet").exists():
        print(f"[skip] pairing_stability: raw bootstrap parquets not staged "
              f"under {PS}; use the canonical pairing_stability.csv shipped "
              f"under metrics/ablation/ (reproduce_tables.py copies it)")
        return
    TAB.mkdir(parents=True, exist_ok=True)

    summary = pd.read_parquet(PS / "pairing_stability_summary.parquet")
    per_head = pd.read_parquet(PS / "pairing_stability_per_head.parquet")
    assert len(per_head) == N_HEADS, f"expected {N_HEADS} teacher heads, got {len(per_head)}"

    # ---- per-bootstrap stats ------------------------------------------------
    boot_rows = []
    for b, g in summary.groupby("bootstrap", sort=True):
        meta = json.loads((PS / f"bootstrap{b}_seed17/pair_search_metadata.json").read_text())
        idx_pq = IDX / f"pairing_stability_bootstrap{b}_nshards16_seed17.parquet"
        index_rows = len(pd.read_parquet(idx_pq)) if idx_pq.exists() else np.nan
        m = g.merge(per_head[["teacher_key", "mode_student_head"]], on="teacher_key")
        assert len(m) == N_HEADS
        pairing_freq = float((m["student_head"] == m["mode_student_head"]).mean())
        top1 = float((m["student_head"] == m["reference_student_head"]).mean())
        boot_rows.append({
            "section": "bootstrap",
            "bootstrap": f"bootstrap{b}",
            "seed": int(meta["seed"]),
            "n_shards": int(meta["n_shards"]),
            "index_rows": int(index_rows) if not np.isnan(index_rows) else "",
            "teacher_key": "",
            "route_depth": "",
            "mode_student_head": "",
            "reference_student_head": "",
            "n_distinct_student_heads": "",
            "pairing_frequency": pairing_freq,
            "top1_agreement_vs_reference": top1,
            "frac_always_same_pair": "",
            "mean_matching_score": float(g["matching_score"].mean()),
            "notes": "",
        })

    # ---- aggregate ----------------------------------------------------------
    mean_mode_freq = float(per_head["mode_frequency"].mean())
    mean_match_ref = float(per_head["matches_reference_rate"].mean())
    frac_always = float((per_head["mode_frequency"] >= 1.0).mean())
    pf_std = float(np.std([r["pairing_frequency"] for r in boot_rows], ddof=1))
    t1_std = float(np.std([r["top1_agreement_vs_reference"] for r in boot_rows], ddof=1))
    # cross-check the identity: mean of per-bootstrap pairing frequency == mean mode frequency
    assert abs(np.mean([r["pairing_frequency"] for r in boot_rows]) - mean_mode_freq) < 1e-9
    assert abs(np.mean([r["top1_agreement_vs_reference"] for r in boot_rows]) - mean_match_ref) < 1e-9
    agg_row = {
        "section": "aggregate",
        "bootstrap": "aggregate (5 bootstraps x 15 heads)",
        "seed": 17,
        "n_shards": 16,
        "index_rows": "",
        "teacher_key": "",
        "route_depth": "",
        "mode_student_head": "",
        "reference_student_head": "",
        "n_distinct_student_heads": "",
        "pairing_frequency": mean_mode_freq,
        "top1_agreement_vs_reference": mean_match_ref,
        "frac_always_same_pair": frac_always,
        "mean_matching_score": float(summary["matching_score"].mean()),
        "notes": f"pairing_frequency std across bootstraps={pf_std:.4f}; "
                 f"top1_agreement std={t1_std:.4f}; n_bootstraps=5; n_teacher_heads=15",
    }

    # ---- per-head -----------------------------------------------------------
    head_rows = []
    for _, r in per_head.iterrows():
        head_rows.append({
            "section": "per_head",
            "bootstrap": "",
            "seed": 17,
            "n_shards": "",
            "index_rows": "",
            "teacher_key": r["teacher_key"],
            "route_depth": int(r["route_depth"]),
            "mode_student_head": int(r["mode_student_head"]),
            "reference_student_head": int(r["reference_student_head"]),
            "n_distinct_student_heads": int(r["n_distinct_student_heads"]),
            "pairing_frequency": float(r["mode_frequency"]),
            "top1_agreement_vs_reference": float(r["matches_reference_rate"]),
            "frac_always_same_pair": "",
            "mean_matching_score": float(r["mean_matching_score"]),
            "notes": "",
        })

    df = pd.DataFrame(boot_rows + [agg_row] + head_rows)
    df.to_csv(TAB / "pairing_stability.csv", index=False)

    # ---- markdown -----------------------------------------------------------
    lines = [
        "# Table. Pairing stability across bootstrap subsets (5 bootstraps x 16 shards, seed 17)",
        "",
        "| Bootstrap | Seed | Shards | Index rows | Pairing frequency (vs mode) | Top-1 agreement (vs reference) | Mean match score |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in boot_rows:
        lines.append(
            f"| {r['bootstrap']} | {r['seed']} | {r['n_shards']} | {r['index_rows']:,} | "
            f"{r['pairing_frequency']:.3f} | {r['top1_agreement_vs_reference']:.3f} | "
            f"{r['mean_matching_score']:.3f} |"
        )
    lines.append(
        f"| **Aggregate** | 17 | 16 | — | **{mean_mode_freq:.3f} ± {pf_std:.3f}** | "
        f"**{mean_match_ref:.3f} ± {t1_std:.3f}** | {agg_row['mean_matching_score']:.3f} |"
    )
    lines += [
        "",
        "Per-teacher-head stability (15 supervised teacher heads, routes L6/L8/L12):",
        "",
        "| Teacher head | Route | Mode student head | Mode freq | #distinct | Ref head | Match ref rate | Mean match score |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in per_head.iterrows():
        lines.append(
            f"| {r['teacher_key']} | L{int(r['route_depth'])} | H{int(r['mode_student_head'])} | "
            f"{r['mode_frequency']:.2f} | {int(r['n_distinct_student_heads'])} | "
            f"H{int(r['reference_student_head'])} | {r['matches_reference_rate']:.2f} | "
            f"{r['mean_matching_score']:.3f} |"
        )
    lines += [
        "",
        "Each bootstrap re-runs the pairing search on a different 16-shard subset of the alignment corpus "
        "(teacher attention from the full-cache buckets; vanilla RiNALMo-micro student, 20 candidate heads "
        "scored with a combined MSE/cosine/Pearson/Spearman metric; bipartite assignment). "
        "Pairing frequency = fraction of bootstraps in which a head takes its mode assignment "
        "(per-bootstrap column: fraction of heads matching their mode in that bootstrap). "
        "Top-1 agreement = fraction of assignments equal to the reference pairing. "
        f"Aggregate: mean pairing frequency {mean_mode_freq:.3f} (6/15 heads identical in all 5 bootstraps), "
        f"mean top-1 agreement {mean_match_ref:.3f}. "
        "Caveat: the reference pairing was computed on 4 slim shards under a different protocol, so the "
        "top-1 agreement is a cross-protocol comparison; the within-protocol bootstrap pairing frequency "
        "is the primary stability statistic.",
    ]
    (TAB / "pairing_stability.md").write_text("\n".join(lines) + "\n")
    print("[rebuild] pairing_stability.csv/.md")
    print(f"  mean pairing frequency = {mean_mode_freq:.4f} (expected ~0.7467)")
    print(f"  mean top-1 agreement   = {mean_match_ref:.4f} (expected ~0.5333)")
    print(f"  frac heads always same = {frac_always:.4f} (expected 0.4000)")
    print(f"  per-bootstrap pairing freq: {[r['pairing_frequency'] for r in boot_rows]}")
    print(f"  per-bootstrap top1:         {[r['top1_agreement_vs_reference'] for r in boot_rows]}")


if __name__ == "__main__":
    main()
