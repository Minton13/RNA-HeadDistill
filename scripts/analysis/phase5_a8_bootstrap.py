"""Phase 5: paired bootstrap for the A8 (frozen-backbone) P4g probe line.

Compares A8-P4g (dumps under ``outputs/probes/predictions_a8/``) against
P0 / P4 / P4g (dumps under ``outputs/probes/predictions/``) on the test
split, reusing the loaders and statistics from ``phase1_bootstrap.py``.

Output: ``outputs/probes/analysis_v2/bootstrap_a8/``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "analysis"))

import phase1_bootstrap as pb  # noqa: E402

OUT_DIR = Path("outputs/probes/analysis_v2/bootstrap_a8")

# label -> (mode_key, predictions_root)
LINES = {
    "A8_P4g": ("P4g", "outputs/probes/predictions_a8"),
    "P0": ("P0", "outputs/probes/predictions"),
    "P4": ("P4", "outputs/probes/predictions"),
    "P4g": ("P4g", "outputs/probes/predictions"),
}

COMPARISONS = [("A8_P4g", "P0"), ("A8_P4g", "P4"), ("A8_P4g", "P4g")]


def _load(task: str, line: str, seed: int, split: str):
    mode, root = LINES[line]
    pb.PRED_ROOT = Path(root)
    return pb.LOADERS[pb.TASKS[task]["type"]](task, mode, seed, split)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
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

    pd.DataFrame(point_rows).to_csv(OUT_DIR / "a8_point_estimates.csv", index=False)
    pd.DataFrame(per_seed_rows).to_csv(OUT_DIR / "a8_vs_baselines_per_seed.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(OUT_DIR / "a8_vs_baselines_summary.csv", index=False)
    import json
    (OUT_DIR / "missing_dumps.json").write_text(json.dumps(missing, indent=2))
    print(f"[a8-bootstrap] wrote {OUT_DIR} (missing: {len(missing)})")
    for row in summary_rows:
        print(f"  {row['task']:32s} {row['comparison']:14s} "
              f"{row['metric']:16s} mean={row['mean']:+.4f} "
              f"[{row['ci_lo']:+.4f},{row['ci_hi']:+.4f}] p_pos={row['p_pos']:.3f}")


if __name__ == "__main__":
    main()
