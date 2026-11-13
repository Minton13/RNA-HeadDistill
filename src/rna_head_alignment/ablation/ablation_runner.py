"""Core alignment ablation runner (A0-A8).

Implements single-GPU training and evaluation for every experiment in
``configs/ablation/experiment_registry.yaml``.  It builds the student model
according to the experiment's teacher-head subset, aggregation, pairing, adapter,
route and backbone-trainability policies, trains with bf16 + AdamW + cosine
scheduler, and evaluates with the existing Stage-4B evaluation helpers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import socket
import subprocess
import time
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch import nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover
    SummaryWriter = None  # type: ignore[misc,assignment]

from rna_head_alignment.head2head.eval import (
    _evaluate_batch,
    _head_key_to_identity,
    _summarize,
)
from rna_head_alignment.head2head.full_sequence_dataset import (
    ROUTE_KEYS,
    FullSlimBucketDataset,
    LengthBucketedSampler,
    full_variable_length_collate,
)
from rna_head_alignment.head2head.gradient_routing import (
    compute_route_losses,
    gradient_diagnostics,
    total_route_loss,
)
from rna_head_alignment.head2head.head_manifest import (
    ROUTE_DEPTHS,
    make_head_key,
)
from rna_head_alignment.head2head.losses import route_alignment_loss
from rna_head_alignment.head2head.model import RiNALMoHeadToHeadStudent
from rna_head_alignment.head2head.student_backbone import RiNALMoStudentBackbone
from rna_head_alignment.io.paths import load_yaml, project_root, resolve_path, save_yaml


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Update ``base`` recursively with ``override``."""
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_registry(registry_path: str | Path) -> dict[str, Any]:
    """Load the experiment registry YAML."""
    cfg = load_yaml(registry_path)
    if "experiments" not in cfg:
        raise ValueError(f"registry {registry_path} missing 'experiments'")
    return cfg


def get_experiment_config(
    registry: dict[str, Any],
    experiment_id: str,
    seed: int,
    base_config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve a full training/eval config for one (experiment, seed)."""
    experiments = registry.get("experiments", {})
    if experiment_id not in experiments:
        raise ValueError(f"unknown experiment {experiment_id!r}")
    exp = experiments[experiment_id]

    base = {
        "project_root": registry.get("project_root", "."),
        "run_name": f"{experiment_id}_seed{seed}",
        "experiment_id": experiment_id,
        "seed": seed,
    }
    # Copy base_data paths.
    base_data = registry.get("base_data", {})
    base.update(base_data)
    # Normalize index key names used by the trainer/eval.
    if "train_index" in base and "train_split_index" not in base:
        base["train_split_index"] = base["train_index"]
    if "val_index" in base and "val_split_index" not in base:
        base["val_split_index"] = base["val_index"]
    if "test_index" in base and "test_split_index" not in base:
        base["test_split_index"] = base["test_index"]
    if "slim_cache_root" in base and "full_slim_cache_root" not in base:
        base["full_slim_cache_root"] = base["slim_cache_root"]
    base["vanilla_checkpoint"] = registry.get("vanilla_checkpoint", "")

    # Base training hyperparameters.
    base_training = registry.get("base_training", {})
    base["student"] = {"model": base_training.get("student", "rinalmo_micro")}
    base["student"]["model_dir"] = "assets/models/rinalmo-micro"

    model_cfg: dict[str, Any] = {
        "adapter_dim": 96,
        "adapter_ffn_mult": 2,
        "activation_checkpointing": base_training.get("activation_checkpointing", False),
    }
    base["model"] = model_cfg

    optimizer_cfg: dict[str, Any] = {
        "lr": base_training.get("adapter_lr", 1.0e-4),
        "weight_decay": base_training.get("adapter_weight_decay", 1.0e-4),
        "backbone_lr": base_training.get("backbone_lr", 2.0e-5),
        "adapter_lr": base_training.get("adapter_lr", 3.0e-4),
        "backbone_weight_decay": base_training.get("backbone_weight_decay", 0.01),
        "adapter_weight_decay": base_training.get("adapter_weight_decay", 1.0e-4),
        "fused": True,
    }
    base["optimizer"] = optimizer_cfg

    scheduler_cfg: dict[str, Any] = {
        "warmup_steps": base_training.get("warmup_steps", 1000),
        "min_lr_ratio": base_training.get("min_lr_ratio", 0.05),
    }
    base["scheduler"] = scheduler_cfg

    loss_cfg = base_training.get("loss", {})
    base["training"] = {
        "max_steps": int(exp.get("max_steps", base_training.get("max_steps", 20000))),
        "eval_every": base_training.get("eval_every", 1000),
        "save_every": base_training.get("save_every", 2000),
        "log_every": base_training.get("log_every", 20),
        "grad_accum": base_training.get("dynamic_batch", {}).get(
            "gradient_accumulation_steps", 1
        ),
        "grad_clip": base_training.get("grad_clip", 1.0),
        "precision": base_training.get("dtype", "bf16"),
        "resume": False,
        "route_weights": loss_cfg.get("route_weights", {"L6": 1.0, "L8": 1.0, "L12": 1.0}),
        "loss_lambdas": {
            "kl": loss_cfg.get("kl_weight", 1.0),
            "mse": loss_cfg.get("mse_weight", 1.0),
            "cos": loss_cfg.get("cosine_weight", 0.1),
        },
        "normalize_within_route_by_head_count": loss_cfg.get(
            "normalize_within_route_by_head_count", True
        ),
        "baselines": [],
        "early_stop_metric": base_training.get("early_stop_metric", "val_loss/total"),
        "early_stop_mode": base_training.get("early_stop_mode", "min"),
        "early_stop_patience_evals": base_training.get("early_stop_patience_evals", 6),
        "keep_last_n_checkpoints": base_training.get("keep_last_n_checkpoints", 3),
        "keep_best_checkpoint": base_training.get("keep_best_checkpoint", True),
        "oom_budget_decay": 0.8,
        "oom_min_budget": 4096,
    }

    base["runtime"] = {
        "seed": seed,
        "max_len": 1024,
        "num_workers": 0,
        "lru_cache_size": 1,
    }
    base["dynamic_batch"] = base_training.get("dynamic_batch", {
        "max_sequences_per_batch": 4,
        "max_pair_tokens": 524288,
        "bucket_size_samples": 1000,
    })

    # Experiment-specific overrides.
    exp_cfg: dict[str, Any] = {
        "experiment_id": experiment_id,
        "scientific_question": exp.get("scientific_question", ""),
        "category": exp.get("category", ""),
        "dataset_fraction": float(exp.get("dataset_fraction", 1.0)),
        "initialization_checkpoint": exp.get("initialization_checkpoint", ""),
        "output_root": exp.get("output_root", f"outputs/ablation/alignment/{experiment_id}"),
        "teacher_head_subset": exp.get("teacher_head_subset", []),
        "teacher_aggregation_policy": exp.get("teacher_aggregation_policy", "none"),
        "student_pairing_policy": exp.get("student_pairing_policy", "none"),
        "adapter_type": exp.get("adapter_type", "transformer"),
        "route_policy": exp.get("route_policy", "route_specific"),
        "backbone_trainability": exp.get("backbone_trainability", "full"),
        "gpu_count": int(exp.get("gpu_count", 1)),
    }
    _deep_update(base, exp_cfg)

    # Backbone trainability -> model.train_backbone.
    base["model"]["train_backbone"] = base["backbone_trainability"] == "full"
    base["model"]["adapter_type"] = base["adapter_type"]

    if base_config_path is not None:
        base_override = load_yaml(base_config_path)
        _deep_update(base, base_override)

    base["run_dir"] = str(Path(base["output_root"]) / f"seed_{seed}")
    return base


# ---------------------------------------------------------------------------
# Teacher-head selection and pairing
# ---------------------------------------------------------------------------


def _load_target_head_ids(target_manifest_path: str | Path) -> dict[int, list[str]]:
    """Return route depth -> ordered list of head_keys from the target manifest."""
    df = pd.read_parquet(resolve_path(target_manifest_path))
    df = df[df["keep"].fillna(True)].copy()
    # Sort by route and rank (with NaN ranks last).
    df["_rank"] = pd.to_numeric(df["rank"], errors="coerce")
    df = df.sort_values(["route_depth", "_rank"]).reset_index(drop=True)
    out: dict[int, list[str]] = {depth: [] for depth in ROUTE_DEPTHS}
    for _, row in df.iterrows():
        out[int(row["route_depth"])].append(str(row["head_key"]))
    return out


def _select_teacher_heads(
    all_head_ids: dict[int, list[str]],
    subset_policy: str | list[str],
) -> dict[int, list[str]]:
    """Apply ``top1_per_route`` / ``top3_per_route`` / ``all`` subsetting."""
    if isinstance(subset_policy, list):
        # Explicit list of head keys.
        selected_set = set(subset_policy)
        return {
            depth: [h for h in heads if h in selected_set]
            for depth, heads in all_head_ids.items()
        }

    if subset_policy == "top1_per_route":
        n = 1
    elif subset_policy == "top3_per_route":
        n = 3
    elif subset_policy in ("all", "top5_per_route"):
        n = None
    else:
        raise ValueError(f"unknown teacher_head_subset {subset_policy!r}")

    return {
        depth: (heads[:n] if n is not None else heads)
        for depth, heads in all_head_ids.items()
    }


def _build_data_driven_specs(
    pair_manifest_path: str | Path,
    selected_head_ids: dict[int, list[str]],
) -> tuple[dict[int, list[tuple[str, int]]], dict[int, list[int]]]:
    """Build adapter specs and native student-head lists from pair manifest."""
    df = pd.read_parquet(resolve_path(pair_manifest_path))
    df = df[df["keep"].fillna(True)].copy()

    def _row_head_key(row: pd.Series) -> str:
        route_key = f"L{int(row['route_depth'])}"
        return make_head_key(
            str(row["teacher_model"]),
            int(row["teacher_layer"]),
            int(row["teacher_head"]),
            route_key,
            str(row["task"]),
        )

    df["head_key"] = df.apply(_row_head_key, axis=1)
    row_by_key = {row["head_key"]: row for _, row in df.iterrows()}

    adapter_specs: dict[int, list[tuple[str, int]]] = {}
    student_heads: dict[int, list[int]] = {}
    for depth in ROUTE_DEPTHS:
        specs: list[tuple[str, int]] = []
        heads: list[int] = []
        for head_key in selected_head_ids.get(depth, []):
            row = row_by_key.get(head_key)
            if row is None:
                raise KeyError(
                    f"head_pair_manifest missing head_key {head_key!r}"
                )
            specs.append((str(row["adapter_key"]), int(row["student_head"])))
            heads.append(int(row["student_head"]))
        adapter_specs[depth] = specs
        student_heads[depth] = heads
    return adapter_specs, student_heads


def _build_random_specs(
    selected_head_ids: dict[int, list[str]],
    num_heads_per_route: int,
    seed: int,
) -> tuple[dict[int, list[tuple[str, int]]], dict[int, list[int]]]:
    """Randomly assign a native student head to each selected teacher head."""
    rng = np.random.default_rng(seed)
    adapter_specs: dict[int, list[tuple[str, int]]] = {}
    student_heads: dict[int, list[int]] = {}
    for depth in ROUTE_DEPTHS:
        heads = selected_head_ids.get(depth, [])
        chosen = [int(rng.integers(0, num_heads_per_route)) for _ in heads]
        specs = [(f"L{depth}_H{i}", chosen[i]) for i in range(len(heads))]
        adapter_specs[depth] = specs
        student_heads[depth] = chosen
    return adapter_specs, student_heads


def _build_consensus_specs(
    selected_head_ids: dict[int, list[str]],
    num_heads_per_route: int,
) -> tuple[dict[int, list[tuple[str, int]]], dict[int, list[int]]]:
    """One adapter per route, paired with the first native head."""
    adapter_specs: dict[int, list[tuple[str, int]]] = {}
    student_heads: dict[int, list[int]] = {}
    for depth in ROUTE_DEPTHS:
        if selected_head_ids.get(depth):
            adapter_specs[depth] = [(f"L{depth}_consensus", 0)]
            student_heads[depth] = [0]
        else:
            adapter_specs[depth] = []
            student_heads[depth] = []
    return adapter_specs, student_heads


def _build_ablation_specs(
    config: dict[str, Any],
    num_heads: int,
) -> tuple[dict[int, list[tuple[str, int]]], dict[int, list[int]], dict[int, list[str]]]:
    """Return (adapter_specs, student_heads, selected_head_ids) for an experiment."""
    target_manifest_path = config.get("target_head_manifest")
    pair_manifest_path = config.get("head_pair_manifest")
    subset_policy = config.get("teacher_head_subset", "all")
    pairing_policy = config.get("student_pairing_policy", "none")
    aggregation_policy = config.get("teacher_aggregation_policy", "identity")

    all_head_ids = _load_target_head_ids(target_manifest_path)
    selected_head_ids = _select_teacher_heads(all_head_ids, subset_policy)

    if aggregation_policy == "route_consensus":
        adapter_specs, student_heads = _build_consensus_specs(selected_head_ids, num_heads)
        # For consensus the head IDs are synthetic.
        eval_head_ids = {
            depth: [f"L{depth}_consensus"] if selected_head_ids.get(depth) else []
            for depth in ROUTE_DEPTHS
        }
    elif pairing_policy == "data_driven":
        adapter_specs, student_heads = _build_data_driven_specs(
            pair_manifest_path, selected_head_ids
        )
        eval_head_ids = selected_head_ids
    elif pairing_policy == "random":
        adapter_specs, student_heads = _build_random_specs(
            selected_head_ids, num_heads, int(config["seed"])
        )
        eval_head_ids = selected_head_ids
    elif pairing_policy == "none":
        adapter_specs = {depth: [] for depth in ROUTE_DEPTHS}
        student_heads = {depth: [] for depth in ROUTE_DEPTHS}
        eval_head_ids = selected_head_ids
    else:
        raise ValueError(f"unknown student_pairing_policy {pairing_policy!r}")

    return adapter_specs, student_heads, eval_head_ids


def _rebase_specs_to_L12(
    adapter_specs: dict[int, list[tuple[str, int]]],
    student_heads: dict[int, list[int]],
    eval_head_ids: dict[int, list[str]],
    num_heads: int,
) -> tuple[dict[int, list[tuple[str, int]]], dict[int, list[int]], dict[int, list[str]]]:
    """Collapse all routes into a single L12 route (A7)."""
    all_specs: list[tuple[str, int]] = []
    all_eval_heads: list[str] = []
    for depth in ROUTE_DEPTHS:
        all_specs.extend(adapter_specs.get(depth, []))
        all_eval_heads.extend(eval_head_ids.get(depth, []))

    # Cycle through native heads so the backbone can provide K == T outputs.
    cycled_student_heads = [
        i % num_heads for i in range(len(all_specs))
    ]
    new_specs = [
        (f"L12_H{i}", cycled_student_heads[i]) for i in range(len(all_specs))
    ]
    new_student_heads = {12: cycled_student_heads}
    new_adapter_specs = {12: new_specs}
    new_eval_head_ids = {12: all_eval_heads}
    return new_adapter_specs, new_student_heads, new_eval_head_ids


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def build_ablation_model(
    config: dict[str, Any],
    device: torch.device,
) -> RiNALMoHeadToHeadStudent:
    """Build a student model for the resolved experiment config."""
    model_cfg = config.get("model", {})
    adapter_dim = int(model_cfg.get("adapter_dim", 96))
    ffn_mult = int(model_cfg.get("adapter_ffn_mult", 2))
    adapter_type = str(model_cfg.get("adapter_type", "transformer"))
    train_backbone = bool(model_cfg.get("train_backbone", True))
    act_ckpt = bool(model_cfg.get("activation_checkpointing", False))
    route_policy = config.get("route_policy", "route_specific")

    # Default routes for route_specific; L12 only for all_from_L12.
    routes: tuple[int, ...] = ROUTE_DEPTHS if route_policy == "route_specific" else (12,)

    # Probe a temporary backbone to learn num_heads.
    tmp_backbone = RiNALMoStudentBackbone(
        model_dir=resolve_path(config["student"]["model_dir"]),
        routes=routes,
    )
    num_heads = tmp_backbone.num_heads
    del tmp_backbone

    adapter_specs, student_heads, eval_head_ids = _build_ablation_specs(config, num_heads)
    if route_policy == "all_from_L12":
        adapter_specs, student_heads, eval_head_ids = _rebase_specs_to_L12(
            adapter_specs, student_heads, eval_head_ids, num_heads
        )

    backbone = RiNALMoStudentBackbone(
        model_dir=resolve_path(config["student"]["model_dir"]),
        routes=routes,
        student_heads=student_heads,
    )
    if act_ckpt:
        try:
            backbone.model.gradient_checkpointing_enable()
        except AttributeError:
            pass
    for p in backbone.parameters():
        p.requires_grad_(train_backbone)

    student = RiNALMoHeadToHeadStudent(
        backbone=backbone,
        adapter_specs=adapter_specs,
        adapter_dim=adapter_dim,
        ffn_multiplier=ffn_mult,
        adapter_type=adapter_type,
    )
    # Stash experiment metadata on the model for eval/serialization.
    student.teacher_aggregation_policy = config.get("teacher_aggregation_policy", "identity")
    student.route_policy = route_policy
    student.eval_head_ids = eval_head_ids

    # Load initialization checkpoint if requested.
    init_ckpt = config.get("initialization_checkpoint", "")
    if init_ckpt:
        init_path = resolve_path(init_ckpt)
        if init_path.exists():
            ckpt = torch.load(init_path, map_location="cpu", weights_only=False)
            state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
            missing, unexpected = student.load_state_dict(state, strict=False)
            if missing:
                print(f"[ablation] init checkpoint missing {len(missing)} keys: {missing[:5]}")

    return student.to(device)


# ---------------------------------------------------------------------------
# Target transforms (consensus / all-from-L12)
# ---------------------------------------------------------------------------


def _subset_batch_to_selected_heads(
    batch: dict[str, Any],
    selected_head_ids: dict[int, list[str]],
    route_head_keys: dict[int, list[str]] | None,
) -> dict[str, Any]:
    """Slice teacher targets/availability to the experiment-selected heads.

    The slim-cache buckets always contain all 5 heads per route in a fixed
    order.  Ablations such as A1/A2 only train on a subset, so the batch must
    be narrowed to match the model's adapter count.
    """
    if route_head_keys is None:
        return batch
    for depth, heads in route_head_keys.items():
        route_key = f"L{depth}"
        selected = selected_head_ids.get(depth, [])
        if not selected or len(selected) == len(heads):
            continue
        index = {h: i for i, h in enumerate(heads)}
        indices = [index[h] for h in selected if h in index]
        if not indices:
            continue
        idx = torch.tensor(indices, dtype=torch.long, device=batch[f"teacher_{route_key}"].device)
        batch[f"teacher_{route_key}"] = batch[f"teacher_{route_key}"].index_select(1, idx)
        if f"head_availability_{route_key}" in batch:
            batch[f"head_availability_{route_key}"] = batch[f"head_availability_{route_key}"].index_select(1, idx)
    return batch


def transform_teacher_targets(batch: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    """Apply consensus and/or all-from-L12 transformations to a collated batch."""
    aggregation = config.get("teacher_aggregation_policy", "identity")
    route_policy = config.get("route_policy", "route_specific")

    if route_policy == "all_from_L12":
        teachers = []
        availabilities = []
        for route_key in ROUTE_KEYS:
            tkey = f"teacher_{route_key}"
            akey = f"head_availability_{route_key}"
            if tkey in batch:
                teachers.append(batch[tkey])
                availabilities.append(batch.get(akey, torch.ones(batch[tkey].shape[:2], dtype=torch.bool, device=batch[tkey].device)))
        if teachers:
            batch["teacher_L12"] = torch.cat(teachers, dim=1)
            batch["head_availability_L12"] = torch.cat(availabilities, dim=1)
        for route_key in ROUTE_KEYS:
            if route_key != "L12":
                batch.pop(f"teacher_{route_key}", None)
                batch.pop(f"head_availability_{route_key}", None)

    if aggregation == "route_consensus":
        route_keys_present = [rk for rk in ROUTE_KEYS if f"teacher_{rk}" in batch]
        for route_key in route_keys_present:
            teacher = batch[f"teacher_{route_key}"]
            avail_key = f"head_availability_{route_key}"
            if avail_key in batch:
                avail = batch[avail_key].float()
                while avail.dim() < teacher.dim():
                    avail = avail.unsqueeze(-1)
                masked = teacher * avail
                consensus = masked.sum(dim=1, keepdim=True) / avail.sum(dim=1, keepdim=True).clamp_min(1.0)
            else:
                consensus = teacher.mean(dim=1, keepdim=True)
            batch[f"teacher_{route_key}"] = consensus
            batch[avail_key] = torch.ones(
                teacher.shape[0], 1, dtype=torch.bool, device=teacher.device
            )
    return batch


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _lambdas_from_config(loss_lambdas: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(loss_lambdas.get("kl", 1.0)),
        float(loss_lambdas.get("mse", 1.0)),
        float(loss_lambdas.get("cos", 0.1)),
    )


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | None,
    step: int,
    metrics: dict[str, Any],
    path: Path,
    config: dict[str, Any] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }
    if config is not None:
        payload["config"] = config
    torch.save(payload, path)


def _cleanup_old_checkpoints(checkpoint_dir: Path, keep_last_n: int) -> None:
    step_paths = sorted(
        checkpoint_dir.glob("step_*.pt"),
        key=lambda p: int(p.stem.split("_")[1]),
    )
    for old in step_paths[:-keep_last_n]:
        old.unlink(missing_ok=True)


def _log_train_step(
    writer: Any | None,
    step: int,
    metrics: dict[str, float],
    lr: float,
) -> None:
    if writer is None:
        return
    for key, val in metrics.items():
        writer.add_scalar(f"train/{key}", val, step)
    writer.add_scalar("train/lr", lr, step)


def _log_val_step(writer: Any | None, step: int, metrics: dict[str, float]) -> None:
    if writer is None:
        return
    for key, val in metrics.items():
        writer.add_scalar(f"val/{key}", val, step)


def _run_val_pass(
    model: RiNALMoHeadToHeadStudent,
    val_loader: DataLoader,
    device: torch.device,
    loss_fn: Any,
    route_weights: dict[str, float],
    use_amp: bool,
    amp_dtype: torch.dtype,
    normalize_within_route: bool,
    config: dict[str, Any],
    route_head_keys: dict[int, list[str]] | None = None,
) -> dict[str, float]:
    model.eval()
    route_totals: dict[str, float] = {rk: 0.0 for rk in ROUTE_KEYS}
    route_counts: dict[str, int] = {rk: 0 for rk in ROUTE_KEYS}
    with torch.no_grad():
        for batch in val_loader:
            batch = _to_device(batch, device)
            batch = _subset_batch_to_selected_heads(
                batch,
                getattr(model, "eval_head_ids", {}),
                route_head_keys,
            )
            batch = transform_teacher_targets(batch, config)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred = model(batch["input_ids"], batch["attention_mask"])
                route_losses = compute_route_losses(
                    pred,
                    batch,
                    loss_fn=loss_fn,
                    normalize_within_route=normalize_within_route,
                )
            for rk, (loss, _per_head) in route_losses.items():
                route_totals[rk] += float(loss.detach().mean().item())
                route_counts[rk] += 1
    metrics: dict[str, float] = {}
    weighted = 0.0
    for rk in ROUTE_KEYS:
        if route_counts[rk]:
            val = route_totals[rk] / route_counts[rk]
            metrics[f"loss/{rk}"] = val
            weighted += route_weights.get(rk, 1.0) * val
    metrics["loss/total"] = weighted
    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train_ablation(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """Single-GPU training entry point."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config_resolved.yaml")

    seed = int(config["seed"])
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    full_cache_root = resolve_path(config["full_slim_cache_root"])
    sequence_manifest_path = resolve_path(config["sequence_manifest"])
    train_index = resolve_path(config["train_split_index"])
    val_index = resolve_path(config["val_split_index"])

    tokenizer = RiNALMoStudentBackbone(
        model_dir=resolve_path(config["student"]["model_dir"])
    ).tokenizer()

    max_len = int(config["runtime"]["max_len"])

    train_dataset = FullSlimBucketDataset(
        full_cache_root,
        train_index,
        sequence_manifest_path,
        tokenizer,
        max_len=max_len,
        cache_size=int(config["runtime"].get("lru_cache_size", 1)),
    )
    val_dataset = FullSlimBucketDataset(
        full_cache_root,
        val_index,
        sequence_manifest_path,
        tokenizer,
        max_len=max_len,
        cache_size=1,
    )

    dynamic_cfg = config["dynamic_batch"]
    max_pair_tokens = int(dynamic_cfg.get("max_pair_tokens", 524288))
    max_sequences = int(dynamic_cfg.get("max_sequences_per_batch", 4))
    bucket_size_samples = dynamic_cfg.get("bucket_size_samples")

    train_sampler = LengthBucketedSampler(
        train_dataset.lengths(),
        max_pair_tokens=max_pair_tokens,
        max_sequences_per_batch=max_sequences,
        bucket_size_samples=bucket_size_samples,
        rank=0,
        world_size=1,
        seed=seed,
        shuffle=True,
        drop_last=False,
    )
    val_sampler = LengthBucketedSampler(
        val_dataset.lengths(),
        max_pair_tokens=max_pair_tokens,
        max_sequences_per_batch=max_sequences,
        bucket_size_samples=bucket_size_samples,
        rank=0,
        world_size=1,
        seed=seed,
        shuffle=False,
        drop_last=False,
    )

    num_workers = int(config["runtime"].get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=full_variable_length_collate,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=full_variable_length_collate,
        num_workers=0,
        pin_memory=False,
    )

    model = build_ablation_model(config, device)
    model.train()
    print(f"[ablation] params: {model.count_parameters()}", flush=True)

    # Optimizer with separate backbone / adapter groups.
    opt_cfg = config["optimizer"]
    backbone_lr = float(opt_cfg.get("backbone_lr", opt_cfg["lr"]))
    adapter_lr = float(opt_cfg.get("adapter_lr", opt_cfg["lr"]))
    backbone_wd = float(opt_cfg.get("backbone_weight_decay", opt_cfg["weight_decay"]))
    adapter_wd = float(opt_cfg.get("adapter_weight_decay", opt_cfg["weight_decay"]))
    fused = bool(opt_cfg.get("fused", True)) and device.type == "cuda"

    param_groups = [
        {
            "params": [p for p in model.backbone.parameters() if p.requires_grad],
            "lr": backbone_lr,
            "weight_decay": backbone_wd,
            "name": "backbone",
        },
        {
            "params": [p for p in model.adapters.parameters() if p.requires_grad],
            "lr": adapter_lr,
            "weight_decay": adapter_wd,
            "name": "adapters",
        },
    ]
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    try:
        optimizer = torch.optim.AdamW(param_groups, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"], fused=fused)
    except TypeError:
        optimizer = torch.optim.AdamW(param_groups, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])

    sched_cfg = config["scheduler"]
    max_steps = int(config["training"]["max_steps"])
    warmup_steps = int(sched_cfg.get("warmup_steps", 0))
    min_lr_ratio = float(sched_cfg.get("min_lr_ratio", 0.01))

    def lr_lambda(step: int) -> float:
        step = min(step, max_steps)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine * (1.0 - min_lr_ratio) + min_lr_ratio

    scheduler = LambdaLR(optimizer, lr_lambda)

    precision = str(config["training"].get("precision", "fp32"))
    use_amp = precision in ("bf16", "fp16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else torch.float32)

    route_weights = dict(config["training"].get("route_weights", {"L6": 1.0, "L8": 1.0, "L12": 1.0}))
    lambdas = _lambdas_from_config(config["training"].get("loss_lambdas", {"kl": 1.0, "mse": 1.0, "cos": 0.1}))
    loss_fn = partial(route_alignment_loss, lambdas=lambdas)
    normalize_within_route = bool(config["training"].get("normalize_within_route_by_head_count", True))

    grad_accum = int(config["training"].get("grad_accum", 1))
    grad_clip = float(config["training"].get("grad_clip", 1.0))
    log_every = int(config["training"].get("log_every", 20))
    save_every = int(config["training"].get("save_every", 2000))
    eval_every = int(config["training"].get("eval_every", 1000))
    grad_diag_every = int(config["training"].get("grad_diagnostics_every", 100))
    early_stop_patience = int(config["training"].get("early_stop_patience_evals", 6))
    keep_last_n = int(config["training"].get("keep_last_n_checkpoints", 3))
    oom_decay = float(config["training"].get("oom_budget_decay", 0.8))
    oom_min_budget = int(config["training"].get("oom_min_budget", 4096))

    writer = SummaryWriter(log_dir=run_dir / "tensorboard") if SummaryWriter else None
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    throughput_rows: list[dict[str, Any]] = []
    best_val_loss = float("inf")
    evals_without_improvement = 0
    step = 0
    epoch = 0
    train_iter = iter(train_loader)

    start_time = time.perf_counter()
    data_wait_start = time.perf_counter()

    current_max_pair_tokens = max_pair_tokens
    current_max_sequences = max_sequences

    while step < max_steps:
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        try:
            batch_fetch_start = time.perf_counter()
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)
        data_wait_time = time.perf_counter() - data_wait_start
        batch = _to_device(batch, device)
        batch = _subset_batch_to_selected_heads(
            batch,
            getattr(model, "eval_head_ids", {}),
            getattr(train_dataset, "route_head_keys", None),
        )
        batch = transform_teacher_targets(batch, config)

        step_start = time.perf_counter()

        oom_happened = False
        try:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                pred = model(batch["input_ids"], batch["attention_mask"])
                route_losses_with_per_head = compute_route_losses(
                    pred,
                    batch,
                    loss_fn=loss_fn,
                    normalize_within_route=normalize_within_route,
                )
                route_losses = {k: v[0] for k, v in route_losses_with_per_head.items()}
                per_head = {k: v[1] for k, v in route_losses_with_per_head.items()}
                total_loss_tensor = total_route_loss(route_losses, route_weights) / grad_accum
            total_loss_tensor.backward()
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and device.type == "cuda":
                oom_happened = True
                torch.cuda.empty_cache()
                optimizer.zero_grad(set_to_none=True)
                current_max_pair_tokens = max(oom_min_budget, int(current_max_pair_tokens * oom_decay))
                current_max_sequences = max(1, int(current_max_sequences * oom_decay))
                train_sampler = LengthBucketedSampler(
                    train_dataset.lengths(),
                    max_pair_tokens=current_max_pair_tokens,
                    max_sequences_per_batch=current_max_sequences,
                    bucket_size_samples=bucket_size_samples,
                    rank=0,
                    world_size=1,
                    seed=seed,
                    shuffle=True,
                    drop_last=False,
                )
                train_sampler.set_epoch(epoch)
                train_loader = DataLoader(
                    train_dataset,
                    batch_sampler=train_sampler,
                    collate_fn=full_variable_length_collate,
                    num_workers=num_workers,
                    pin_memory=device.type == "cuda",
                )
                train_iter = iter(train_loader)
                print(f"[ablation] OOM at step {step}; reduced budget to "
                      f"{current_max_pair_tokens}/{current_max_sequences}", flush=True)
            else:
                raise

        if oom_happened:
            data_wait_start = time.perf_counter()
            continue

        # Fail-loud guard (2026-09-11 incident): with activation checkpointing
        # enabled, hook-captured activations are detached from the autograd
        # graph, so backbone params get requires_grad=True but never receive
        # gradients and silently stay frozen.  Detect this on the first step.
        if step == 0 and bool(config.get("model", {}).get("train_backbone", True)):
            bb_grad = any(
                p.grad is not None and bool((p.grad != 0).any())
                for p in model.backbone.parameters()
            )
            if not bb_grad:
                raise RuntimeError(
                    "train_backbone=True but no backbone gradients after the first "
                    "backward pass (check activation_checkpointing / detached "
                    "activation captures). Refusing to train a silently-frozen "
                    "backbone."
                )

        total_loss = float(total_loss_tensor.detach().item()) * grad_accum
        if not math.isfinite(total_loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {total_loss}")

        step_time = time.perf_counter() - step_start
        wall_time = time.perf_counter() - start_time
        peak_vram_gb = (
            torch.cuda.max_memory_allocated(device) / 1e9
            if device.type == "cuda"
            else 0.0
        )

        train_metrics: dict[str, float] = {"loss/total": total_loss}
        for rk, loss in route_losses.items():
            train_metrics[f"loss/{rk}"] = float(loss.detach().mean().item())
        for rk, heads in per_head.items():
            for comps in heads:
                for key, val in comps.items():
                    tag = f"component/{rk}/{key}"
                    train_metrics[tag] = train_metrics.get(tag, 0.0) + float(val.detach().mean().item())
            for key in ["kl", "mse", "cos", "total"]:
                tag = f"component/{rk}/{key}"
                if tag in train_metrics:
                    train_metrics[tag] /= len(heads)

        grad_norms: dict[str, float] = {}
        diagnostics: dict[str, float] = {}
        if (step + 1) % grad_diag_every == 0 or step == 0:
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += float(p.grad.norm() ** 2)
            grad_norms["total"] = total_norm**0.5
            diagnostics = gradient_diagnostics(model)
            if diagnostics.get("nan_inf_grad_params", 0.0) > 0:
                raise RuntimeError(f"NaN/Inf gradients at step {step}")
        train_metrics.update({f"grad/{k}": v for k, v in grad_norms.items()})

        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        seqs = int(batch["lengths"].sum().item())
        pair_tokens = int((batch["lengths"].float() ** 2).sum().item())
        throughput_rows.append({
            "step": step,
            "batch_size": len(batch["lengths"]),
            "total_seqs": seqs,
            "total_pair_tokens": pair_tokens,
            "step_time_sec": step_time,
            "data_wait_time_sec": data_wait_time,
            "wall_time_sec": wall_time,
            "throughput_seq_per_sec": seqs / step_time if step_time > 0 else 0.0,
            "peak_vram_gb": peak_vram_gb,
        })

        if step % log_every == 0 or step == 0:
            _log_train_step(writer, step, train_metrics, current_lr)
            route_str = " ".join(
                f"{rk}={route_losses[rk].detach().mean().item():.4f}"
                for rk in ROUTE_KEYS if rk in route_losses
            )
            print(
                f"[step {step}] loss={total_loss:.4f} {route_str} "
                f"lr={current_lr:.2e} step={step_time:.2f}s data_wait={data_wait_time:.2f}s "
                f"vram={peak_vram_gb:.2f}GB",
                flush=True,
            )

        train_metrics.update({
            "lr": current_lr,
            "step_time_sec": step_time,
            "data_wait_time_sec": data_wait_time,
            "wall_time_sec": wall_time,
            "peak_vram_gb": peak_vram_gb,
        })
        train_rows.append({"step": step, **train_metrics})

        if (step + 1) % save_every == 0:
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                step + 1,
                train_metrics,
                checkpoint_dir / f"step_{step + 1:07d}.pt",
                config=config,
            )
            _save_checkpoint(
                model,
                optimizer,
                scheduler,
                step + 1,
                train_metrics,
                checkpoint_dir / "latest.pt",
                config=config,
            )
            if keep_last_n > 0:
                _cleanup_old_checkpoints(checkpoint_dir, keep_last_n)

        if (step + 1) % eval_every == 0 and len(val_loader) > 0:
            val_metrics = _run_val_pass(
                model,
                val_loader,
                device,
                loss_fn,
                route_weights,
                use_amp,
                amp_dtype,
                normalize_within_route,
                config,
                route_head_keys=getattr(val_dataset, "route_head_keys", None),
            )
            _log_val_step(writer, step + 1, val_metrics)
            val_loss = val_metrics.get("loss/total", float("inf"))
            print(f"[step {step + 1}] val loss={val_loss:.4f}", flush=True)
            val_metrics["wall_time_sec"] = time.perf_counter() - start_time
            val_rows.append({"step": step + 1, **val_metrics})
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                evals_without_improvement = 0
                if config["training"].get("keep_best_checkpoint", True):
                    _save_checkpoint(
                        model,
                        optimizer,
                        scheduler,
                        step + 1,
                        val_metrics,
                        checkpoint_dir / "best_val.pt",
                        config=config,
                    )
            else:
                evals_without_improvement += 1
            if evals_without_improvement >= early_stop_patience:
                print(f"[ablation] early stopping at step {step + 1}", flush=True)
                break

        step += 1
        data_wait_start = time.perf_counter()

    # Final checkpoint.
    final_metrics = train_rows[-1] if train_rows else {}
    _save_checkpoint(
        model,
        optimizer,
        scheduler,
        step,
        final_metrics,
        checkpoint_dir / "latest.pt",
        config=config,
    )
    if train_rows:
        pd.DataFrame(train_rows).to_csv(run_dir / "train_metrics.csv", index=False)
    if val_rows:
        pd.DataFrame(val_rows).to_csv(run_dir / "val_metrics.csv", index=False)
    if throughput_rows:
        pd.DataFrame(throughput_rows).to_csv(run_dir / "throughput_memory.csv", index=False)
    if writer:
        writer.close()

    return {
        "run_dir": str(run_dir),
        "best_val_loss": best_val_loss,
        "final_step": step,
        "total_wall_time_sec": time.perf_counter() - start_time,
        "peak_vram_gb": torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0,
    }


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def _load_model_from_checkpoint(
    checkpoint_path: Path,
    config: dict[str, Any],
    device: torch.device,
) -> RiNALMoHeadToHeadStudent:
    """Load a trained ablation model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_config = ckpt.get("config") if isinstance(ckpt, dict) else None
    effective_config = dict(config)
    if checkpoint_config is not None:
        _deep_update(effective_config, checkpoint_config)

    model = build_ablation_model(effective_config, device)
    state = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[ablation-eval] missing keys ({len(missing)}): {missing[:5]}")
    model.eval()
    return model


def evaluate_ablation_student(
    config: dict[str, Any],
    checkpoint_path: str | Path,
    output_dir: str | Path,
    splits: list[str] | None = None,
    device: str | torch.device = "auto",
) -> dict[str, Any]:
    """Evaluate an ablation checkpoint using existing Stage-4B eval helpers."""
    device = torch.device(
        get_device(device) if str(device).lower() == "auto" else device
    )
    cfg = config
    run_name = cfg.get("run_name", "ablation_run")
    out_root = Path(output_dir) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    precision = str(cfg.get("training", {}).get("precision", "bf16"))
    if precision == "bf16":
        amp_dtype = torch.bfloat16
    elif precision == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    use_amp = precision in ("bf16", "fp16") and device.type == "cuda"

    model = _load_model_from_checkpoint(Path(checkpoint_path), cfg, device)
    tokenizer = model.backbone.tokenizer()

    full_cache_root = resolve_path(cfg["full_slim_cache_root"])
    sequence_manifest_path = resolve_path(cfg["sequence_manifest"])

    split_paths: dict[str, Path] = {}
    for split in ["val", "test"]:
        key = f"{split}_split_index"
        if key in cfg:
            split_paths[split] = resolve_path(cfg[key])
    if not split_paths:
        for split in ["val", "test"]:
            p = full_cache_root / "index" / f"{split}.parquet"
            if p.exists():
                split_paths[split] = p
    if splits is not None:
        split_paths = {k: v for k, v in split_paths.items() if k in splits}
    if not split_paths:
        raise ValueError("no val/test split indices configured or found")

    metric_names = list(cfg.get("metrics", [
        "mse", "mae", "rowwise_kl", "cosine", "pearson", "spearman",
        "top_l_overlap", "top_2l_overlap",
    ]))
    baselines = list(cfg.get("baselines", ["native_head", "linear_projection", "random_student_head"]))
    length_bins = list(cfg.get("length_bins", [0, 64, 128, 256, 384, 512]))
    max_len = int(cfg.get("runtime", {}).get("max_len", 1024))

    dynamic_cfg = cfg.get("dynamic_batch", {})
    max_pair_tokens = int(dynamic_cfg.get("max_pair_tokens", 524288))
    max_sequences = int(dynamic_cfg.get("max_sequences_per_batch", 8))
    bucket_size_samples = dynamic_cfg.get("bucket_size_samples")

    baseline_projections = _build_eval_linear_projections(model)
    head_ids_by_route = getattr(model, "eval_head_ids", {d: [] for d in ROUTE_DEPTHS})

    report: dict[str, Any] = {
        "run_name": run_name,
        "output_root": str(out_root),
        "splits": {},
    }

    for split_name, split_path in split_paths.items():
        dataset = FullSlimBucketDataset(
            full_cache_root,
            split_path,
            sequence_manifest_path,
            tokenizer,
            max_len=max_len,
            cache_size=1,
        )
        sampler = LengthBucketedSampler(
            dataset.lengths(),
            max_pair_tokens=max_pair_tokens,
            max_sequences_per_batch=max_sequences,
            bucket_size_samples=bucket_size_samples,
            rank=0,
            world_size=1,
            seed=42,
            shuffle=False,
            drop_last=False,
        )
        loader = DataLoader(
            dataset,
            batch_sampler=sampler,
            collate_fn=full_variable_length_collate,
            num_workers=0,
        )

        rows: list[dict[str, Any]] = []
        for batch in loader:
            batch = _to_device(batch, device)
            batch = _subset_batch_to_selected_heads(
                batch,
                getattr(model, "eval_head_ids", {}),
                getattr(dataset, "route_head_keys", None),
            )
            batch = transform_teacher_targets(batch, cfg)
            rows.extend(
                _evaluate_batch(
                    model,
                    batch,
                    head_ids_by_route,
                    baseline_projections,
                    metric_names,
                    baselines,
                    split_name,
                    length_bins,
                    device,
                    use_amp,
                    amp_dtype,
                )
            )

        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        per_head_path = split_dir / "per_head_metrics.parquet"
        df.to_parquet(per_head_path, index=False)

        group_by = cfg.get("group_by", ["task", "route_depth", "teacher_model", "teacher_layer_head", "length_bin"])
        summary = _summarize(df, group_by, length_bins)
        summary_path = split_dir / "summary.parquet"
        summary.to_parquet(summary_path, index=False)

        report["splits"][split_name] = {
            "per_head_metrics": str(per_head_path),
            "summary": str(summary_path),
            "n_rows": len(df),
            "n_batches": len(loader),
        }

    return report


def _build_eval_linear_projections(model: RiNALMoHeadToHeadStudent) -> dict[int, nn.ModuleList]:
    """Random per-head linear projections for the linear-projection baseline."""
    projs: dict[int, nn.ModuleList] = {}
    device = next(model.parameters()).device
    for depth, specs in model.adapter_specs.items():
        route_projs = nn.ModuleList(
            nn.Linear(model.backbone.head_dim, model.backbone.head_dim, bias=False)
            for _ in specs
        )
        for proj in route_projs:
            nn.init.xavier_uniform_(proj.weight)
            proj.requires_grad_(False)
        projs[depth] = route_projs.to(device)
    return projs


# ---------------------------------------------------------------------------
# Top-level experiment driver
# ---------------------------------------------------------------------------


def _save_run_metadata(run_dir: Path, config: dict[str, Any]) -> None:
    """Save resolved config, Slurm job ID, git commit and code hash."""
    run_dir.mkdir(parents=True, exist_ok=True)
    save_yaml(config, run_dir / "config_resolved.yaml")

    git_commit = "unknown"
    git_dirty = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project_root(), text=True
        ).strip()
        subprocess.check_output(["git", "diff", "--quiet"], cwd=project_root())
        git_dirty = "clean"
    except subprocess.CalledProcessError:
        git_dirty = "dirty"
    except FileNotFoundError:
        git_commit = "unknown"
        git_dirty = "unknown"

    metadata = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "unknown"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "host": socket.gethostname(),
        "run_dir": str(run_dir),
    }
    (run_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


def _reuse_existing_run(
    existing_run_dir: Path,
    target_run_dir: Path,
) -> dict[str, Any]:
    """Reuse an existing run by symlinking its artifacts into a fresh run dir."""
    target_run_dir.parent.mkdir(parents=True, exist_ok=True)
    if target_run_dir.exists():
        if target_run_dir.is_symlink():
            target_run_dir.unlink()
        else:
            shutil.rmtree(target_run_dir)
    target_run_dir.mkdir(parents=True, exist_ok=True)

    # Symlink the checkpoint directory so the authoritative weights are not duplicated.
    ckpt_src = existing_run_dir / "checkpoints"
    ckpt_dst = target_run_dir / "checkpoints"
    if ckpt_src.exists():
        try:
            os.symlink(ckpt_src.resolve(), ckpt_dst, target_is_directory=True)
            method = "symlink"
        except OSError:
            shutil.copytree(ckpt_src, ckpt_dst)
            method = "copy"
    else:
        method = "no_checkpoints"

    # Copy lightweight metadata files (never modify the source run).
    for name in ["config_resolved.yaml", "config.yaml", "metadata.json", "train_metrics.csv", "val_metrics.csv", "throughput_memory.csv"]:
        src = existing_run_dir / name
        if src.exists():
            shutil.copy2(src, target_run_dir / name)

    (target_run_dir / "REUSED_FROM").write_text(
        f"{existing_run_dir.resolve()}\nmethod={method}\n", encoding="utf-8"
    )
    return {
        "run_dir": str(target_run_dir),
        "reused_from": str(existing_run_dir),
        "method": method,
        "status": "reused",
    }


def _write_vanilla_manifest(config: dict[str, Any], run_dir: Path) -> dict[str, Any]:
    """A0: write a manifest for downstream probe evaluation."""
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "experiment_id": config["experiment_id"],
        "seed": config["seed"],
        "run_dir": str(run_dir),
        "student_model_dir": config["student"]["model_dir"],
        "vanilla_checkpoint": config.get("vanilla_checkpoint", ""),
        "status": "vanilla_manifest",
        "downstream_probe_ready": True,
    }
    (run_dir / "vanilla_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    save_yaml(config, run_dir / "config_resolved.yaml")
    return {"run_dir": str(run_dir), "status": "vanilla_manifest"}


def run_ablation_experiment(
    registry_path: str | Path,
    experiment_id: str,
    seed: int,
    base_config_path: str | Path | None = None,
    run_dir_override: str | Path | None = None,
    skip_eval: bool = False,
    eval_splits: list[str] | None = None,
) -> dict[str, Any]:
    """Run one ablation experiment and return a status dict."""
    registry = load_registry(registry_path)
    config = get_experiment_config(registry, experiment_id, seed, base_config_path)
    if run_dir_override is not None:
        config["run_dir"] = str(Path(run_dir_override))
    run_dir = Path(config["run_dir"])
    run_dir.mkdir(parents=True, exist_ok=True)
    _save_run_metadata(run_dir, config)

    experiment = registry["experiments"][experiment_id]

    # A0: vanilla baseline manifest only.
    if experiment.get("max_steps", 0) == 0 or experiment.get("gpu_count", 1) == 0:
        _save_run_metadata(run_dir, config)
        return _write_vanilla_manifest(config, run_dir)

    # A3 seed 17: reuse the existing full run if configured.
    reuse = experiment.get("reuse_existing_run")
    if reuse and int(reuse.get("seed")) == seed:
        existing = resolve_path(reuse["run_dir"])
        if existing.exists():
            train_summary = _reuse_existing_run(existing, run_dir)
            _save_run_metadata(run_dir, config)
            if not skip_eval:
                checkpoint_path = run_dir / "checkpoints" / "best_val.pt"
                if not checkpoint_path.exists():
                    checkpoint_path = run_dir / "checkpoints" / "latest.pt"
                eval_report = evaluate_ablation_student(
                    config=config,
                    checkpoint_path=checkpoint_path,
                    output_dir=config["output_root"],
                    splits=eval_splits,
                )
                train_summary["eval"] = eval_report
                (run_dir / "eval_report.json").write_text(
                    json.dumps(eval_report, indent=2, default=str), encoding="utf-8"
                )
            return train_summary

    # Train.
    train_summary = train_ablation(config, run_dir)

    # Evaluate.
    if not skip_eval:
        checkpoint_path = run_dir / "checkpoints" / "best_val.pt"
        if not checkpoint_path.exists():
            checkpoint_path = run_dir / "checkpoints" / "latest.pt"
        eval_report = evaluate_ablation_student(
            config=config,
            checkpoint_path=checkpoint_path,
            output_dir=config["output_root"],
            splits=eval_splits,
        )
        train_summary["eval"] = eval_report
        (run_dir / "eval_report.json").write_text(
            json.dumps(eval_report, indent=2, default=str), encoding="utf-8"
        )

    return train_summary


# Local helper kept for parity with io.paths.get_device (avoid circular import).
def get_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/ablation/experiment_registry.yaml")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--eval-splits", default=None, help="comma list e.g. val,test")
    args = parser.parse_args(argv)

    eval_splits = [s.strip() for s in args.eval_splits.split(",")] if args.eval_splits else None
    summary = run_ablation_experiment(
        registry_path=args.registry,
        experiment_id=args.experiment_id,
        seed=args.seed,
        base_config_path=args.base_config,
        run_dir_override=args.run_dir,
        skip_eval=args.skip_eval,
        eval_splits=eval_splits,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
