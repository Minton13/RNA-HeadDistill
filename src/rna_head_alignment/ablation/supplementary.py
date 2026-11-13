"""Supplementary ablation drivers for Stage 4B/5.

Implements data-scaling, loss-combination, teacher-count, and pairing-stability
ablations on top of the core :mod:`rna_head_alignment.ablation.ablation_runner`.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml

from rna_head_alignment.ablation.ablation_runner import (
    evaluate_ablation_student,
    get_experiment_config,
    load_registry,
    train_ablation,
)
from rna_head_alignment.io.paths import load_yaml, project_root, resolve_path, save_yaml


LOSS_VARIANTS = {
    "kl_only": {"kl": 1.0, "mse": 0.0, "cos": 0.0},
    "mse_only": {"kl": 0.0, "mse": 1.0, "cos": 0.0},
    "kl_mse": {"kl": 1.0, "mse": 1.0, "cos": 0.0},
    "kl_mse_cos": {"kl": 1.0, "mse": 1.0, "cos": 0.1},
}


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _base_experiment_id(ablation_type: str, registry: dict[str, Any]) -> str:
    """Return the registered base experiment (usually A3) for a supplementary ablation."""
    key = {
        "data_scaling": "sizes",
        "loss_ablation": "variants",
        "teacher_count_ablation": "counts",
        "pairing_stability": "n_bootstrap",
    }.get(ablation_type)
    if key is None:
        raise ValueError(f"unknown ablation type {ablation_type!r}")
    for exp_id, exp in registry.get("experiments", {}).items():
        if exp.get("category") == "supplementary" and key in exp:
            return exp.get("base_experiment", exp_id)
    raise ValueError(f"no registered supplementary ablation for {ablation_type}")


def _make_data_scaling_index(
    size: int,
    seed: int,
    train_index_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Create a deterministic train split index containing ``size`` sequences."""
    train_df = pd.read_parquet(resolve_path(train_index_path))
    size = min(int(size), len(train_df))
    rng = np.random.default_rng(seed)
    sampled = train_df.sample(n=size, random_state=rng)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"data_scaling_size{size}_seed{seed}.parquet"
    sampled.to_parquet(out_path, index=False)
    return out_path


def _make_pairing_stability_index(
    bootstrap_idx: int,
    n_shards: int,
    seed: int,
    train_index_path: str | Path,
    output_dir: str | Path,
) -> Path:
    """Create a bootstrap train split from ``n_shards`` random source shards."""
    train_df = pd.read_parquet(resolve_path(train_index_path))
    shard_ids = train_df["original_shard_id"].apply(
        lambda x: str(x[0]) if isinstance(x, (list, np.ndarray)) and len(x) else str(x)
    )
    unique_shards = sorted(shard_ids.unique())
    rng = np.random.default_rng(seed + bootstrap_idx)
    selected = sorted(rng.choice(unique_shards, size=min(n_shards, len(unique_shards)), replace=False))
    selected_set = set(selected)
    sampled = train_df[shard_ids.isin(selected_set)].copy()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"pairing_stability_bootstrap{bootstrap_idx}_nshards{n_shards}_seed{seed}.parquet"
    sampled.to_parquet(out_path, index=False)
    return out_path


def _resolve_output_root(ablation_type: str, variant: str, seed: int) -> Path:
    root = Path("outputs/ablation/alignment")
    if ablation_type == "data_scaling":
        return root / "data_scaling" / f"size{variant}_seed{seed}"
    if ablation_type == "loss_ablation":
        return root / "loss_ablation" / f"{variant}_seed{seed}"
    if ablation_type == "teacher_count_ablation":
        return root / "teacher_count_ablation" / f"top{variant}_seed{seed}"
    if ablation_type == "pairing_stability":
        return root / "pairing_stability" / f"bootstrap{variant}_seed{seed}"
    raise ValueError(ablation_type)


def build_supplementary_config(
    registry_path: str | Path,
    ablation_type: str,
    variant: str,
    seed: int,
    base_config_path: str | Path | None = None,
    index_root: str | Path = "outputs/ablation/alignment/supplementary_indices",
) -> dict[str, Any]:
    """Resolve a full training config for one supplementary experiment."""
    registry = load_registry(registry_path)
    base_exp_id = _base_experiment_id(ablation_type, registry)
    config = get_experiment_config(registry, base_exp_id, seed, base_config_path)

    # Alias expected by train/eval helpers.
    if "full_slim_cache_root" not in config and "slim_cache_root" in config:
        config["full_slim_cache_root"] = config["slim_cache_root"]

    output_root = _resolve_output_root(ablation_type, variant, seed)
    config["output_root"] = str(output_root)
    config["run_dir"] = str(output_root / f"seed_{seed}")
    config["experiment_id"] = f"{base_exp_id}_{ablation_type}_{variant}"
    config["run_name"] = f"{base_exp_id}_{ablation_type}_{variant}_seed{seed}"

    if ablation_type == "data_scaling":
        size = int(variant)
        train_index_path = config.get("train_split_index") or config.get("train_index")
        index_path = _make_data_scaling_index(size, seed, train_index_path, index_root)
        config["train_split_index"] = str(index_path)
        config["dataset_fraction"] = size / max(len(pd.read_parquet(resolve_path(train_index_path))), 1)

    elif ablation_type == "loss_ablation":
        if variant not in LOSS_VARIANTS:
            raise ValueError(f"unknown loss variant {variant!r}; choose from {list(LOSS_VARIANTS)}")
        lambdas = LOSS_VARIANTS[variant]
        config["training"]["loss_lambdas"] = dict(lambdas)

    elif ablation_type == "teacher_count_ablation":
        count = int(variant)
        if count == 5:
            config["teacher_head_subset"] = "all"
        else:
            config["teacher_head_subset"] = f"top{count}_per_route"

    elif ablation_type == "pairing_stability":
        bootstrap_idx = int(variant)
        n_shards = 16
        train_index_path = config.get("train_split_index") or config.get("train_index")
        index_path = _make_pairing_stability_index(
            bootstrap_idx, n_shards, seed, train_index_path, index_root
        )
        config["train_split_index"] = str(index_path)
        config["dataset_fraction"] = None  # opaque shard-limited index

    return config


def run_supplementary_experiment(
    registry_path: str | Path,
    ablation_type: str,
    variant: str,
    seed: int,
    base_config_path: str | Path | None = None,
    skip_eval: bool = False,
) -> dict[str, Any]:
    """Run one supplementary ablation experiment."""
    config = build_supplementary_config(registry_path, ablation_type, variant, seed, base_config_path)
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config_resolved.yaml")

    # Reproducibility.
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    train_summary = train_ablation(config, run_dir)
    train_summary["ablation_type"] = ablation_type
    train_summary["variant"] = variant
    train_summary["seed"] = seed

    if not skip_eval:
        checkpoint_path = run_dir / "checkpoints" / "best_val.pt"
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / "checkpoints" / "latest.pt"
        if checkpoint_path.exists():
            eval_report = evaluate_ablation_student(
                config=config,
                checkpoint_path=checkpoint_path,
                output_dir=config["output_root"],
                splits=["val", "test"],
            )
            train_summary["eval"] = eval_report
            (run_dir / "eval_report.json").write_text(
                json.dumps(eval_report, indent=2, default=str), encoding="utf-8"
            )
        else:
            train_summary["eval"] = {"error": "no checkpoint found"}

    return train_summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run a single supplementary ablation.")
    parser.add_argument("--registry", default="configs/ablation/experiment_registry.yaml")
    parser.add_argument("--base-config", default=None)
    parser.add_argument(
        "--ablation-type",
        required=True,
        choices=["data_scaling", "loss_ablation", "teacher_count_ablation", "pairing_stability"],
    )
    parser.add_argument("--variant", required=True, help="size, loss variant, top-N count, or bootstrap index")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args(argv)

    summary = run_supplementary_experiment(
        registry_path=args.registry,
        ablation_type=args.ablation_type,
        variant=args.variant,
        seed=args.seed,
        base_config_path=args.base_config,
        skip_eval=args.skip_eval,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
