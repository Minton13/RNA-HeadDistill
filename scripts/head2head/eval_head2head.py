#!/usr/bin/env python
"""CLI entry point for Stage 4B head-to-head evaluation.

Example:

    PYTHONPATH=src python scripts/head2head/eval_head2head.py \
        --config configs/head2head/eval_head2head.yaml \
        --checkpoint runs/head2head/rinalmo_head2head_stage4b/checkpoints/best_val.pt \
        --run-name rinalmo_head2head_stage4b
"""

from __future__ import annotations

import argparse
import json

from rna_head_alignment.head2head.eval import evaluate_head2head_student


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="eval_head2head.yaml")
    parser.add_argument("--checkpoint", required=True, help="path to trained model checkpoint")
    parser.add_argument("--run-name", default=None, help="run identifier")
    parser.add_argument("--output-dir", default=None, help="override output root")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--splits", default=None, help="comma list of splits, e.g. val,test")
    parser.add_argument("--max-shards", type=int, default=None, help="limit shards per split")
    args = parser.parse_args()

    splits = [s.strip() for s in args.splits.split(",")] if args.splits else None
    report = evaluate_head2head_student(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        run_name=args.run_name,
        output_dir=args.output_dir,
        device=args.device,
        splits=splits,
        max_shards=args.max_shards,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
