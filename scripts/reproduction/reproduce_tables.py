#!/usr/bin/env python
# Provenance: new script written on 2026-11-13 for the public release (no
# monorepo counterpart; it consolidates the table-rebuild logic of
# paper/scripts/rebuild_fixed_tables.py, rebuild_pairing_stability.py and
# paper/notes/_analysis_head_function.py against the published release data
# layout).
"""Rebuild every paper table CSV into <outdir>/tables/.

Two strategies per table (recorded in <outdir>/tables/tables_manifest.json):

  rebuilt    recomputed from derived data shipped in the release dataset
             (e.g. head_functionality_by_task.csv from atlas/all_head_atlas.csv,
             alignment_ablation_delta.csv from metrics/ablation/
             alignment_ablation.csv);
  canonical  copied verbatim from the canonical CSV shipped in the release dataset
             (tables whose rebuild requires raw per-seed artifacts that are
             not part of the data package; see rebuild_fixed_tables.py /
             rebuild_pairing_stability.py for the full provenance pipeline).

Paper anchors:
  Table 1   selected_teacher_heads.csv          (canonical, selection/)
  Table 2   downstream_paper_metrics.csv        (canonical, figures_source/)
  Support   downstream_full_metrics.csv         (canonical, metrics/downstream/)
  Table S1  pilot_grid_summary.csv              (canonical, pilot/)
  Table S2  pilot_rna3db_val_curve.csv          (canonical, pilot/)
  Table S3  head_functionality_stats.csv        (canonical, figures_source/)
  Support   head_functionality_by_task.csv     (rebuilt from atlas/)
  Table S4  route_finetune.csv + multitask.csv  (canonical, metrics/ablation/)
  Table S5  alignment_ablation.csv              (canonical, metrics/ablation/)

Run: python scripts/reproduction/reproduce_tables.py \
       --data-root /path/to/dataset-rna-head-distill --outdir ./reproduced_figures
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def default_data_root() -> Path:
    env = os.environ.get("RNA_HEAD_DISTILL_DATA")
    return Path(env) if env else Path("./dataset-rna-head-distill")


# ----------------------------------------------------------- rebuild recipes

def rebuild_head_functionality_stats(data_root: Path, out: Path) -> Path:
    """Supporting per-task/model head-score concentration from the full atlas.

    This per-task summary has a different schema and scope from the published
    Table S3, whose canonical Source Data are copied separately.

    Mirrors the aggregation in the original analysis (cleaning: drop
    model_load_failed rows, drop null metrics, drop degenerate task x model
    combos whose best head scores <= 1e-9; then per-combo summary stats).
    """
    atlas = pd.read_csv(data_root / "atlas" / "all_head_atlas.csv",
                        low_memory=False)
    atlas = atlas[~atlas["notes"].fillna("").str.contains("model_load_failed")]
    atlas = atlas[atlas["metric_primary"].notna()]
    per_head = atlas[atlas["head"] >= 0].copy()
    combo_max = per_head.groupby(["task", "model"])["metric_primary"].max()
    deg = set(combo_max[combo_max <= 1e-9].index.tolist())
    valid = per_head[~per_head.apply(lambda r: (r["task"], r["model"]) in deg,
                                     axis=1)].copy()

    rows = []
    for (task, model), g in valid.groupby(["task", "model"]):
        s = g["metric_primary"]
        med = float(s.median())
        t5 = float(s.nlargest(min(5, len(s))).mean())
        rows.append({
            "task": task,
            "model": model,
            "model_type": g["model_type"].iloc[0],
            "n_heads": int(len(g)),
            "mean": round(float(s.mean()), 4),
            "std": round(float(s.std()), 4),
            "min": round(float(s.min()), 4),
            "median": round(med, 4),
            "max": round(float(s.max()), 4),
            "top5_mean": round(t5, 4),
            "top5_over_median": round(t5 / med, 2) if med > 1e-6 else np.nan,
            "top5_minus_median": round(t5 - med, 4),
        })
    stats = pd.DataFrame(rows).sort_values(["task", "model"])
    dst = out / "head_functionality_by_task.csv"
    stats.to_csv(dst, index=False)
    return dst


def rebuild_alignment_ablation_delta(data_root: Path, out: Path) -> Path:
    """Delta table — recomputed from the canonical alignment_ablation.csv."""
    src = data_root / "metrics" / "ablation" / "alignment_ablation.csv"
    df = pd.read_csv(src)
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
                am = a3.get(f"{short}{suffix}_mean")
                bm = c.get(f"{short}{suffix}_mean")
                if np.isnan(am) or np.isnan(bm):
                    continue
                d = am - bm
                win = (d > 0) if higher_better[short] else (d < 0)
                drows.append({"comparison": f"A3 vs {comp}", "basis": basis,
                              "metric": short,
                              "higher_is_better": higher_better[short],
                              "A3_mean": am, "comparator_mean": bm,
                              "delta_A3minusComp": d, "A3_wins": bool(win),
                              "comparability": note})
    dst = out / "alignment_ablation_delta.csv"
    pd.DataFrame(drows).to_csv(dst, index=False)
    return dst


# ------------------------------------------------------------- table registry
# (output name, paper anchor, strategy, rebuild fn | canonical candidates)
CANONICAL = [
    ("selected_teacher_heads.csv", "Table 1",
     ["selection/selected_teacher_heads.csv"]),
    ("downstream_paper_metrics.csv", "Table 2",
     ["figures_source/Table2.csv"]),
    ("downstream_full_metrics.csv", "Table 2 (full metric support)",
     ["metrics/downstream/downstream_full_metrics.csv"]),
    ("downstream_primary_bootstrap.csv", "Table 2 (bootstrap support)",
     ["metrics/downstream/downstream_primary_bootstrap.csv"]),
    ("downstream_primary_bootstrap_per_seed.csv", "Table 2 (per-seed support)",
     ["metrics/downstream/downstream_primary_bootstrap_per_seed.csv"]),
    ("pilot_grid_summary.csv", "Table S1",
     ["pilot/pilot_grid_summary.csv"]),
    ("pilot_grid_experiments.csv", "Table S1 (per-experiment support)",
     ["pilot/pilot_grid_experiments.csv"]),
    ("pilot_rna3db_val_curve.csv", "Table S2",
     ["pilot/pilot_rna3db_val_curve.csv"]),
    ("head_functionality_stats.csv", "Table S3",
     ["figures_source/TableS3.csv"]),
    ("route_finetune.csv", "Table S4a",
     ["metrics/ablation/route_finetune.csv"]),
    ("multitask.csv", "Table S4b",
     ["metrics/ablation/multitask.csv"]),
    ("alignment_ablation.csv", "Table S5",
     ["metrics/ablation/alignment_ablation.csv"]),
    ("alignment_replay.csv", "SI (replay lambda sweep)",
     ["metrics/ablation/alignment_replay.csv"]),
    ("pairing_stability.csv", "SI (pairing stability)",
     ["metrics/ablation/pairing_stability.csv"]),
    ("loss_ablation.csv", "SI (loss ablation)",
     ["metrics/ablation/loss_ablation.csv"]),
    ("data_scaling.csv", "SI (data scaling)",
     ["metrics/ablation/data_scaling.csv"]),
    ("teacher_count_ablation.csv", "SI (teacher-count ablation)",
     ["metrics/ablation/teacher_count_ablation.csv"]),
    ("pilot_decoder_curves.csv", "Fig. S3 source",
     ["pilot/pilot_decoder_curves.csv"]),
    ("pilot_mcc_heatmap_micro.csv", "Fig. S2 source",
     ["pilot/pilot_mcc_heatmap_micro.csv"]),
    ("pilot_mcc_heatmap_mega.csv", "Fig. S2 source",
     ["pilot/pilot_mcc_heatmap_mega.csv"]),
    ("pilot_mcc_heatmap_giga.csv", "Fig. S2 source",
     ["pilot/pilot_mcc_heatmap_giga.csv"]),
]

REBUILT = [
    ("head_functionality_by_task.csv", "SI (per-task head-score support)", rebuild_head_functionality_stats,
     ["atlas/all_head_atlas.csv"]),
    ("alignment_ablation_delta.csv", "Table S5 (delta support)",
     rebuild_alignment_ablation_delta,
     ["metrics/ablation/alignment_ablation.csv"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", type=Path, default=default_data_root(),
                    help="root of the downloaded release dataset "
                         "(default: $RNA_HEAD_DISTILL_DATA)")
    ap.add_argument("--outdir", type=Path, default=Path("./reproduced_figures"),
                    help="output root; tables go to <outdir>/tables/")
    ap.add_argument("--allow-missing", action="store_true",
                    help="warn instead of failing when an input is absent")
    args = ap.parse_args()
    out = args.outdir / "tables"
    out.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    missing: list[str] = []

    for name, anchor, fn, inputs in REBUILT:
        if not all((args.data_root / rel).exists() for rel in inputs):
            print(f"[missing] {name}: inputs not found "
                  f"({', '.join(inputs)} under {args.data_root})")
            manifest[name] = {"paper_anchor": anchor, "source": "rebuilt",
                              "inputs": inputs, "status": "missing"}
            missing.append(name)
            continue
        dst = fn(args.data_root, out)
        print(f"[rebuilt ] {name}  ({anchor})")
        manifest[name] = {"paper_anchor": anchor, "source": "rebuilt",
                          "inputs": inputs, "status": "ok",
                          "output": str(dst.relative_to(args.outdir))}

    for name, anchor, candidates in CANONICAL:
        src = next((args.data_root / rel for rel in candidates
                    if (args.data_root / rel).exists()), None)
        if src is None:
            print(f"[missing] {name}: none of {candidates} found under "
                  f"{args.data_root}")
            manifest[name] = {"paper_anchor": anchor, "source": "canonical",
                              "inputs": candidates, "status": "missing"}
            missing.append(name)
            continue
        shutil.copy2(src, out / name)
        rel = str(src.relative_to(args.data_root))
        print(f"[canonical] {name}  ({anchor})  <- {rel}")
        manifest[name] = {"paper_anchor": anchor, "source": "canonical",
                          "inputs": [rel], "status": "ok",
                          "output": f"tables/{name}"}

    with open(out / "tables_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)
    print(f"wrote {out / 'tables_manifest.json'} "
          f"({sum(1 for m in manifest.values() if m['status'] == 'ok')} ok, "
          f"{len(missing)} missing)")

    if missing and not args.allow_missing:
        sys.exit(f"ERROR: {len(missing)} table(s) could not be produced: "
                 f"{missing}. Pass --allow-missing to downgrade to a warning.")


if __name__ == "__main__":
    main()
