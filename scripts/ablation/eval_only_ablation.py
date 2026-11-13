#!/usr/bin/env python
"""Eval-only driver for existing ablation run directories.

Loads ``config_resolved.yaml`` from a run's seed directory, picks the best
checkpoint (``best_val.pt`` with ``latest.pt`` fallback), and runs
:func:`evaluate_ablation_student` without any training.  Used to backfill
missing evaluations (A2, data_scaling 500k seed 17/89, loss-ablation test
splits, teacher_count top5 test).

Example:
    PYTHONPATH=src python scripts/ablation/eval_only_ablation.py \
        --run-dir outputs/ablation/alignment/A2_head2head_top3/seed_17 \
        --splits val,test
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rna_head_alignment.ablation.ablation_runner import evaluate_ablation_student
from rna_head_alignment.io.paths import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, help="Seed run dir containing config_resolved.yaml and checkpoints/")
    parser.add_argument("--splits", default="val,test", help="Comma-separated splits to evaluate")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path (default: checkpoints/best_val.pt)")
    parser.add_argument("--overwrite", action="store_true", help="Re-evaluate splits whose summary.parquet already exists")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    cfg_path = run_dir / "config_resolved.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(cfg_path)
    config = load_yaml(cfg_path)

    checkpoint = Path(args.checkpoint) if args.checkpoint else run_dir / "checkpoints" / "best_val.pt"
    if not checkpoint.exists():
        checkpoint = run_dir / "checkpoints" / "latest.pt"
    if not checkpoint.exists():
        raise FileNotFoundError(f"no checkpoint under {run_dir}/checkpoints")

    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    # Skip splits that already have a summary unless --overwrite.
    out_root = Path(config["output_root"]) / config.get("run_name", run_dir.name)
    todo: list[str] = []
    for split in splits:
        if not args.overwrite and (out_root / split / "summary.parquet").exists():
            print(f"[eval-only] {split} already evaluated at {out_root/split}; skipping")
        else:
            todo.append(split)
    if not todo:
        print("[eval-only] nothing to do")
        return

    print(f"[eval-only] run={config.get('run_name')} checkpoint={checkpoint} splits={todo}")
    report = evaluate_ablation_student(
        config=config,
        checkpoint_path=checkpoint,
        output_dir=config["output_root"],
        splits=todo,
    )
    report["checkpoint"] = str(checkpoint)
    report["eval_only"] = True

    # Merge with any pre-existing eval report.
    rep_path = run_dir / "eval_report.json"
    if rep_path.exists():
        try:
            old = json.loads(rep_path.read_text(encoding="utf-8"))
            old.setdefault("splits", {}).update(report.get("splits", {}))
            for k in ("checkpoint", "eval_only"):
                old[k] = report[k]
            report = old
        except Exception:
            pass
    rep_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(f"[eval-only] wrote {rep_path}")
    print(json.dumps({k: v.get("n_rows") for k, v in report["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
