#!/usr/bin/env python
"""MLM pretraining CLI for the from-scratch small student (S0 arm).

Single GPU smoke:

    PYTHONPATH=src python scripts/head2head/train_small_mlm.py \
        --config configs/head2head/small_student/mlm_s0_smoke.yaml

Multi-GPU via torchrun:

    torchrun --standalone --nproc_per_node=6 \
        scripts/head2head/train_small_mlm.py \
        --config configs/head2head/small_student/mlm_s0a_6gpu.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from rna_head_alignment.head2head.train_mlm import launch_mlm_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="training YAML config")
    parser.add_argument("--run-name", default=None, help="override config run_name")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from latest checkpoint (sets training.resume=true)",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.resume:
        config.setdefault("training", {})["resume"] = True

    run_name = config.get("run_name", "small_mlm_run")
    run_dir = Path(config.get("output_dir", "runs/small_student")) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "config.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    (run_dir / "metadata.json").write_text(
        json.dumps({"run_name": run_name, "run_dir": str(run_dir)}, indent=2),
        encoding="utf-8",
    )

    launch_mlm_training(config, run_dir)


if __name__ == "__main__":
    main()
