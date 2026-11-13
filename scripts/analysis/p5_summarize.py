#!/usr/bin/env python
"""Summarize P5 (distilled_hidden_teacher) frozen-probe cells.

Writes ``outputs/probes/frozen/p5_summary.parquet`` with the same column
layout as ``outputs/probes/analysis_v2/frozen_probe_all_metric_aligned.parquet``
and prints a P5 vs P4 (distilled_hidden_aligned) vs P0 (vanilla_hidden)
primary-metric comparison table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "outputs" / "probes" / "frozen"
ALIGNED = ROOT / "outputs" / "probes" / "analysis_v2" / "frozen_probe_all_metric_aligned.parquet"
OUT = FROZEN / "p5_summary.parquet"

TASKS = [
    "splice_donor_seqcls",
    "splice_acceptor_seqcls",
    "rbp_binding_seqcls",
    "ncRNA_family_seqcls",
    "chemical_reactivity_token",
    "secondary_structure_pair",
]
SEEDS = (17, 41, 89)
MODE = "distilled_hidden_teacher"

METRIC_COLS = [
    "auroc", "auprc", "f1", "mcc",                      # sequence classification
    "mse", "mae", "pearson", "spearman",                # token regression
    "top_l_precision", "top_2l_precision",              # pairwise
]
PRIMARY_METRIC = {
    "splice_donor_seqcls": "test_auroc",
    "splice_acceptor_seqcls": "test_auroc",
    "rbp_binding_seqcls": "test_auroc",
    "ncRNA_family_seqcls": "test_auroc",
    "chemical_reactivity_token": "test_pearson",
    "secondary_structure_pair": "test_top_l_precision",
}


def collect_cells() -> tuple[pd.DataFrame, list[dict]]:
    rows: list[dict] = []
    problems: list[dict] = []
    for task in TASKS:
        for seed in SEEDS:
            cell = FROZEN / task / MODE / f"seed_{seed}"
            row: dict = {"task": task, "mode": MODE, "seed": seed}
            problem: dict | None = None
            test_path = cell / "eval_test.json"
            val_path = cell / "eval_val.json"
            summ_path = cell / "training_summary.json"
            if not test_path.exists():
                problem = {"task": task, "seed": seed, "issue": "missing eval_test.json"}
            else:
                test = json.loads(test_path.read_text())
                val = json.loads(val_path.read_text()) if val_path.exists() else {}
                summ = json.loads(summ_path.read_text()) if summ_path.exists() else {}
                row["selection_metric"] = summ.get("metric_name")
                row["best_val_metric"] = summ.get("best_val_metric")
                for m in METRIC_COLS:
                    row[f"test_{m}"] = test.get(m, np.nan)
                    row[f"val_{m}"] = val.get(m, np.nan)
                # sanity: NaN or out-of-range primary metric
                pm = PRIMARY_METRIC[task]
                v = row.get(pm)
                bad = v is None or (isinstance(v, float) and (np.isnan(v) or v < 0.0 or v > 1.0))
                if bad:
                    problem = {"task": task, "seed": seed, "issue": f"bad {pm}={v}"}
            if problem:
                problems.append(problem)
            rows.append(row)
    return pd.DataFrame(rows), problems


def main() -> int:
    df, problems = collect_cells()
    df.to_parquet(OUT, index=False)
    print(f"[p5-summary] wrote {OUT} ({len(df)} rows)")
    if problems:
        print("[p5-summary] PROBLEMS:")
        for p in problems:
            print("  ", p)
    else:
        print("[p5-summary] all 18 cells have valid test metrics")

    # Comparison table P5 vs P4 vs P0 on the primary metric.
    aligned = pd.read_parquet(ALIGNED)
    base = aligned[aligned["mode"].isin(["vanilla_hidden", "distilled_hidden_aligned"])]
    combined = pd.concat([base, df], ignore_index=True)
    print("\nP5 vs P4 vs P0 (primary test metric, mean ± std over seeds 17/41/89):")
    header = f"{'task':28s} {'metric':20s} {'P0 vanilla':>12s} {'P4 aligned':>12s} {'P5 teacher':>12s}"
    print(header)
    print("-" * len(header))
    for task in TASKS:
        pm = PRIMARY_METRIC[task]
        cells = {}
        for mode, label in (("vanilla_hidden", "P0"), ("distilled_hidden_aligned", "P4"), (MODE, "P5")):
            vals = combined[(combined["task"] == task) & (combined["mode"] == mode)][pm].dropna()
            cells[label] = f"{vals.mean():.4f}±{vals.std():.4f}" if len(vals) else "MISSING"
        print(f"{task:28s} {pm:20s} {cells['P0']:>12s} {cells['P4']:>12s} {cells['P5']:>12s}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
