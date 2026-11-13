"""Small-student (from-scratch 9.5M) bootstrap: S1 vs S0 distillation verdict.

Compares the three small-student probe lines on the test split:

* S0_hidden   = pure-MLM 40k-step student, hidden-only probe   (small-model P0)
* S1_hidden   = MLM + head-distilled student, hidden-only      (small-model P1)
* S1_aligned  = MLM + head-distilled student, hidden + aligned (small-model P4)

Prediction dumps live under ``outputs/probes/predictions_small/{cell}/`` and
are produced by ``scripts/server/small_pred_dump_array.sbatch``.

Outputs under ``outputs/probes/frozen_small/``:

* ``small_student_summary.parquet``   - per-cell test metrics (all seeds)
* ``bootstrap_small/small_vs_summary.csv`` - paired bootstrap comparisons
* ``SMALL_STUDENT_REPORT.md``         - preregistered verdict
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "analysis"))

import phase1_bootstrap as pb  # noqa: E402

OUT_DIR = Path("outputs/probes/frozen_small/bootstrap_small")
REPORT = Path("outputs/probes/frozen_small/SMALL_STUDENT_REPORT.md")
SUMMARY_PARQUET = Path("outputs/probes/frozen_small/small_student_summary.parquet")

# pb.LOADERS expect a MODES key (P1 -> distilled_hidden, P4 -> distilled_hidden_aligned);
# the per-line PRED_ROOT keeps the three cells apart.
LINES = {
    "S0_hidden": ("P1", "outputs/probes/predictions_small/s0_hidden"),
    "S1_hidden": ("P1", "outputs/probes/predictions_small/s1_hidden"),
    "S1_aligned": ("P4", "outputs/probes/predictions_small/s1_aligned"),
}

COMPARISONS = [
    ("S1_hidden", "S0_hidden"),
    ("S1_aligned", "S0_hidden"),
    ("S1_aligned", "S1_hidden"),
]


def _load(task: str, line: str, seed: int, split: str):
    mode, root = LINES[line]
    pb.PRED_ROOT = Path(root)
    return pb.LOADERS[pb.TASKS[task]["type"]](task, mode, seed, split)


def _collect_eval_summary() -> pd.DataFrame:
    """Aggregate eval_test.json / eval_val.json from every trained cell."""
    rows: list[dict] = []
    for f in sorted(Path("outputs/probes/frozen_small").glob("*/*/*/seed_*/eval_*.json")):
        parts = f.parts
        cell, task, mode, seed = parts[3], parts[4], parts[5], parts[6]
        split = f.stem.replace("eval_", "")
        d = json.loads(f.read_text())
        rows.append({
            "cell": cell, "task": task, "mode": mode, "seed": seed, "split": split,
            **{k: v for k, v in d.items() if isinstance(v, (int, float))},
        })
    return pd.DataFrame(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = _collect_eval_summary()
    summary.to_parquet(SUMMARY_PARQUET, index=False)
    print(f"[small] summary parquet: {SUMMARY_PARQUET} ({len(summary)} rows)")

    rng = np.random.default_rng(pb.RNG_SEED)
    point_rows: list[dict] = []
    per_seed_rows: list[dict] = []
    summary_rows: list[dict] = []
    missing: list[str] = []

    for task, spec in pb.TASKS.items():
        test_data: dict[tuple[str, int], dict] = {}
        for line in LINES:
            for seed in pb.SEEDS:
                d = _load(task, line, seed, "test")
                if d is None:
                    missing.append(f"{task}/{line}/seed_{seed}")
                    continue
                test_data[(line, seed)] = d
                idx = np.arange(len(d["seq_ids"]))
                point_rows.append({
                    "task": task, "line": line, "seed": seed,
                    "n_test": len(idx), spec["primary"]: d["primary_fn"](idx),
                })

        for line_a, line_b in COMPARISONS:
            per_seed_deltas: list[np.ndarray] = []
            for seed in pb.SEEDS:
                da = test_data.get((line_a, seed))
                db = test_data.get((line_b, seed))
                if da is None or db is None:
                    continue
                pos_b = {sid: i for i, sid in enumerate(db["seq_ids"])}
                keep_a = np.array([i for i, sid in enumerate(da["seq_ids"]) if sid in pos_b])
                map_b = np.array([pos_b[da["seq_ids"][i]] for i in keep_a])
                n = len(keep_a)

                def fn_a(idx, _da=da, _keep=keep_a):
                    return _da["primary_fn"](_keep[idx])

                def fn_b(idx, _db=db, _map=map_b):
                    return _db["primary_fn"](_map[idx])

                deltas = pb._paired_boot(fn_a, fn_b, n, rng)
                per_seed_deltas.append(deltas)
                per_seed_rows.append({
                    "task": task, "comparison": f"{line_a}-{line_b}", "seed": seed,
                    "metric": spec["primary"], **pb._summarize(deltas),
                })

            if per_seed_deltas:
                pooled = np.nanmean(np.stack(per_seed_deltas), axis=0)
                s = pb._summarize(pooled)
                s["seed_mean_std"] = float(np.nanstd([d.mean() for d in per_seed_deltas]))
                summary_rows.append({
                    "task": task, "comparison": f"{line_a}-{line_b}",
                    "metric": spec["primary"], "n_seeds": len(per_seed_deltas), **s,
                })

    pd.DataFrame(point_rows).to_csv(OUT_DIR / "small_point_estimates.csv", index=False)
    pd.DataFrame(per_seed_rows).to_csv(OUT_DIR / "small_vs_per_seed.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "small_vs_summary.csv", index=False)
    (OUT_DIR / "missing_dumps.json").write_text(json.dumps(missing, indent=2))

    # ------------------------------------------------------- preregistered verdict
    lines = ["# Small-student (from-scratch 9.5M) distillation verdict", ""]
    lines.append("Prediction dumps: `outputs/probes/predictions_small/`; "
                 "bootstrap: paired, 3 seeds (17/41/89), RNG seed 2024.")
    lines.append("")
    lines.append("| task | metric | S1_hidden−S0 | p_pos | S1_aligned−S0 | p_pos | S1_aligned−S1_hidden | p_pos |")
    lines.append("|---|---|---|---|---|---|---|---|")
    key = {(r["task"], r["comparison"]): r for r in summary_rows}
    for task, spec in pb.TASKS.items():
        row = [task, spec["primary"]]
        for comp in ("S1_hidden-S0_hidden", "S1_aligned-S0_hidden", "S1_aligned-S1_hidden"):
            r = key.get((task, comp))
            if r is None:
                row += ["NA", "NA"]
            else:
                row += [f"{r['mean']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]", f"{r['p_pos']:.3f}"]
        lines.append("| " + " | ".join(row) + " |")

    # Preregistered criteria:
    #   1. secondary top-L precision S1_aligned - S0_hidden >= +0.10 with p_pos >= 0.975
    #   2. >= 4/6 tasks positive (S1_aligned vs S0_hidden) and none significantly negative
    sec = key.get(("secondary_structure_pair", "S1_aligned-S0_hidden"))
    crit1 = bool(sec is not None and sec["mean"] >= 0.10 and sec["p_pos"] >= 0.975)
    pos = neg_sig = n_valid = 0
    for task, _spec in pb.TASKS.items():
        r = key.get((task, "S1_aligned-S0_hidden"))
        if r is None:
            continue
        n_valid += 1
        if r["mean"] > 0:
            pos += 1
        if r["p_pos"] <= 0.025:
            neg_sig += 1
    crit2 = bool(n_valid == 6 and pos >= 4 and neg_sig == 0)

    lines += [
        "",
        "## Preregistered criteria",
        "",
        f"1. secondary top-L precision S1_aligned−S0 ≥ +0.10 (p_pos ≥ 0.975): "
        f"**{'PASS' if crit1 else 'FAIL'}**"
        + (f" (mean={sec['mean']:+.4f}, p_pos={sec['p_pos']:.3f})" if sec is not None else " (no data)"),
        f"2. ≥4/6 tasks positive and none significantly negative: "
        f"**{'PASS' if crit2 else 'FAIL'}** (positive={pos}/{n_valid}, significantly negative={neg_sig})",
        "",
        f"Missing dumps: {len(missing)}",
    ]
    REPORT.write_text("\n".join(lines) + "\n")
    print(f"[small] wrote {OUT_DIR} and {REPORT} (missing: {len(missing)})")
    for row in summary_rows:
        print(f"  {row['task']:32s} {row['comparison']:28s} "
              f"{row['metric']:16s} mean={row['mean']:+.4f} "
              f"[{row['ci_lo']:+.4f},{row['ci_hi']:+.4f}] p_pos={row['p_pos']:.3f}")


if __name__ == "__main__":
    main()
