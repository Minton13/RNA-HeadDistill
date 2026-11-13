#!/usr/bin/env python
"""Stage 4B head-to-head alignment training CLI.

Single-GPU smoke:

    PYTHONPATH=src python scripts/head2head/train_head2head.py \
        --config configs/head2head/train_head2head_smoke.yaml

Multi-GPU via torchrun:

    torchrun --nproc_per_node=6 scripts/head2head/train_head2head.py \
        --config configs/head2head/train_head2head_6gpu.yaml \
        --run-name rinalmo_head2head_stage4b

Resume from latest checkpoint:

    torchrun --nproc_per_node=6 scripts/head2head/train_head2head.py \
        --config configs/head2head/train_head2head_6gpu.yaml \
        --resume
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from rna_head_alignment.head2head.train_ddp import launch_training


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _comma_ints(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="training YAML config")
    parser.add_argument("--run-name", default=None, help="override config run_name")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="resume from latest checkpoint (sets training.resume=true)",
    )
    parser.add_argument(
        "--shards",
        default=None,
        type=_comma_ints,
        help="override training shard list, comma-separated integers",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="spawn this many processes locally (default: env WORLD_SIZE or 1)",
    )
    args = parser.parse_args()

    config = _load_yaml(args.config)
    if args.run_name is not None:
        config["run_name"] = args.run_name
    if args.resume:
        config.setdefault("training", {})["resume"] = True
    if args.shards is not None:
        config["shards"] = args.shards

    run_name = config.get("run_name", "head2head_run")
    output_dir = Path(config.get("output_dir", "runs/head2head"))
    run_dir = output_dir

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)
    (run_dir / "metadata.json").write_text(
        json.dumps({"run_name": run_name, "run_dir": str(run_dir)}, indent=2),
        encoding="utf-8",
    )

    launch_training(config, run_dir, world_size=args.world_size)


if __name__ == "__main__":
    main()
