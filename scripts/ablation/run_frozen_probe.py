#!/usr/bin/env python
"""CLI for a single frozen downstream probe experiment.

Example:
    PYTHONPATH=src python scripts/ablation/run_frozen_probe.py \
        --task splice_donor_seqcls \
        --feature_mode distilled_hidden_aligned \
        --seed 17

For the exact-teacher upper-bound mode, the downstream teacher cache must
already exist (built by ``build_downstream_teacher_cache.py``):

    PYTHONPATH=src python scripts/ablation/run_frozen_probe.py \
        --task secondary_structure_pair \
        --feature_mode distilled_hidden_teacher \
        --seed 17
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

# Allow running without package installation.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rna_head_alignment.head2head.eval import (
    _load_checkpoint as _load_h2h_checkpoint,
    _load_model_for_eval,
)
from rna_head_alignment.head2head.student_backbone import RiNALMoStudentBackbone
from rna_head_alignment.io.paths import get_device, load_yaml, resolve_path
from rna_head_alignment.probes.datasets import get_downstream_dataloaders
from rna_head_alignment.probes.feature_extractor import ProbeFeatureExtractor
from rna_head_alignment.probes.pair_probe import PairProbe
from rna_head_alignment.probes.sequence_probe import SequenceProbe
from rna_head_alignment.probes.task_config import (
    TASK_TO_ROUTE_DEPTH,
    TASK_TO_TYPE,
    default_teacher_cache_dir,
    downstream_data_dir,
    validate_feature_mode,
    validate_task,
)
from rna_head_alignment.probes.token_probe import TokenProbe
from rna_head_alignment.probes.train import ProbeTrainer
from rna_head_alignment.probes.eval import evaluate_probe, write_eval_outputs


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_checkpoint(
    checkpoint_path: str | None,
    registry_path: str | Path = "configs/ablation/experiment_registry.yaml",
) -> Path:
    if checkpoint_path:
        return resolve_path(checkpoint_path)
    registry = load_yaml(registry_path)
    ckpt = registry.get("full_checkpoint_best")
    if not ckpt:
        raise ValueError(
            "No checkpoint provided and experiment_registry.yaml does not set "
            "full_checkpoint_best"
        )
    return resolve_path(ckpt)


def _load_head_ids(slim_cache_root: str | Path, shard_idx: int) -> dict[int, list[str]]:
    """Load route head IDs from the bucketed slim-cache safetensors metadata."""
    from safetensors import safe_open

    buckets_dir = Path(slim_cache_root) / "buckets"
    candidates = sorted(buckets_dir.glob(f"shard_{shard_idx:05d}_*.safetensors"))
    if not candidates:
        candidates = sorted(Path(slim_cache_root).glob(f"shard_{shard_idx:05d}_*.safetensors"))
    if not candidates:
        raise FileNotFoundError(f"No bucket for shard {shard_idx} under {slim_cache_root}")
    with safe_open(str(candidates[0]), framework="pt") as f:
        meta = f.metadata() or {}
    return {
        depth: json.loads(meta.get(f"route_heads_L{depth}", "[]"))
        for depth in (6, 8, 12)
    }


def _load_stage4b_student(
    checkpoint_path: str | Path,
    device: torch.device,
) -> Any:
    """Load the full Stage 4B head-to-head student, preserving adapter pairing."""
    ckpt = _load_h2h_checkpoint(checkpoint_path, device)
    cfg = ckpt.get("config") or {}

    slim_cache_root = cfg.get("slim_cache_root", "outputs/teacher_attention_cache_slim/stage4b_head2head/full")
    splits = cfg.get("splits", {})
    first_shard = None
    for key in ("train", "val", "test"):
        shards = splits.get(key, [])
        if shards:
            first_shard = int(shards[0])
            break
    if first_shard is None:
        # Fall back to the first bucket on disk.
        buckets_dir = Path(slim_cache_root) / "buckets"
        candidates = sorted(buckets_dir.glob("shard_*.safetensors"))
        if not candidates:
            candidates = sorted(Path(slim_cache_root).glob("shard_*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No slim buckets under {slim_cache_root}")
        first_shard = int(candidates[0].stem.split("_")[1])

    head_ids_by_route = _load_head_ids(slim_cache_root, first_shard)
    model, _ckpt = _load_model_for_eval(
        checkpoint_path, cfg, device, teacher_head_ids=head_ids_by_route
    )
    return model


def _load_vanilla_backbone(device: torch.device) -> RiNALMoStudentBackbone:
    """Load the vanilla RiNALMo-micro backbone (no alignment training)."""
    backbone = RiNALMoStudentBackbone(model_dir="assets/models/rinalmo-micro")
    backbone.to(device)
    return backbone


def _build_probe(
    task_type: str,
    input_dim: int,
    num_classes: int | None = None,
) -> torch.nn.Module:
    if task_type == "sequence_classification":
        if num_classes is None:
            raise ValueError("num_classes required")
        return SequenceProbe(input_dim=input_dim, num_classes=num_classes)
    if task_type == "token_regression":
        return TokenProbe(input_dim=input_dim)
    if task_type == "pairwise":
        return PairProbe(input_dim=input_dim, symmetric=True)
    raise ValueError(task_type)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="Downstream task name")
    parser.add_argument(
        "--feature-mode",
        required=True,
        help="One of the P0-P6 frozen probe feature modes",
    )
    parser.add_argument("--seed", type=int, default=17, help="Random seed")
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Override Stage 4B checkpoint (defaults to registry full_checkpoint_best)",
    )
    parser.add_argument(
        "--teacher-cache-dir",
        default=None,
        help="Override exact-teacher cache directory for P5",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML config with runtime/training overrides",
    )
    parser.add_argument(
        "--output-root",
        default="outputs/probes/frozen",
        help="Root directory for probe outputs",
    )
    parser.add_argument("--device", default="auto", help="cpu/cuda/auto")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Use AMP fp16 on CUDA",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip val/test evaluation after training",
    )
    parser.add_argument(
        "--eval-splits",
        type=str,
        default="val,test",
        help="Comma-separated splits to evaluate after training (default: val,test).",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Max epochs per (lr, wd) grid point. Defaults are task-dependent.",
    )
    parser.add_argument(
        "--max-training-hours",
        type=float,
        default=None,
        help="Hard wall-time budget for the entire fit (all grid points).",
    )
    parser.add_argument(
        "--lrs",
        type=str,
        default=None,
        help="Comma-separated learning rates for the optimizer grid (default: built-in).",
    )
    parser.add_argument(
        "--wds",
        type=str,
        default=None,
        help="Comma-separated weight decays for the optimizer grid (default: built-in).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="Early-stopping patience (default: 10).",
    )
    args = parser.parse_args()

    task = args.task
    feature_mode = args.feature_mode
    validate_task(task)
    validate_feature_mode(feature_mode)

    _set_seed(args.seed)
    device = torch.device(get_device(args.device))

    cfg: dict[str, Any] = load_yaml(args.config) if args.config else {}
    runtime_cfg = cfg.get("runtime", {})
    training_cfg = cfg.get("training", {})

    max_len = int(runtime_cfg.get("max_len", args.max_len))
    batch_size = int(runtime_cfg.get("batch_size", args.batch_size))
    num_workers = int(runtime_cfg.get("num_workers", args.num_workers))
    mixed_precision = bool(training_cfg.get("mixed_precision", args.mixed_precision))

    output_dir = (
        Path(args.output_root)
        / task
        / feature_mode
        / f"seed_{args.seed}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ model
    print(f"[frozen-probe] loading model for mode={feature_mode}", flush=True)
    if feature_mode == "vanilla_hidden":
        model = _load_vanilla_backbone(device)
    else:
        checkpoint_path = _resolve_checkpoint(args.checkpoint)
        model = _load_stage4b_student(checkpoint_path, device)
    # Re-seed after model loading so DataLoader shuffling and probe init are
    # deterministic per args.seed regardless of any RNG side effects from
    # checkpoint / transformers loading.
    _set_seed(args.seed)
    print(f"[frozen-probe] model loaded", flush=True)

    # Freeze the entire pretrained model.
    for p in model.parameters():
        p.requires_grad_(False)

    route_depth = TASK_TO_ROUTE_DEPTH[task]
    teacher_cache_dir = (
        args.teacher_cache_dir
        or cfg.get("teacher_cache_dir")
        or default_teacher_cache_dir(task)
    )
    feature_extractor = ProbeFeatureExtractor(
        model=model,
        feature_mode=feature_mode,
        task=task,
        route_depth=route_depth,
        teacher_cache_dir=teacher_cache_dir,
    )
    feature_extractor.to(device)

    # Lazily create gates/projection before the optimizer.  For the exact-teacher
    # mode we cannot run a dummy lookup against the cache, so we explicitly
    # initialize for the canonical 5 heads per route.
    if feature_mode == "distilled_hidden_teacher":
        feature_extractor._ensure_components(5)
    else:
        dummy_ids = torch.zeros((1, 10), dtype=torch.long, device=device)
        dummy_mask = torch.ones((1, 10), dtype=torch.long, device=device)
        with torch.no_grad():
            _ = feature_extractor(dummy_ids, dummy_mask, seq_ids=["dummy"])

    task_type = TASK_TO_TYPE[task]
    num_classes: int | None = None

    # --------------------------------------------------------------- data
    data_dir = cfg.get("data_dir") or downstream_data_dir(task)
    print(f"[frozen-probe] loading downstream data from {data_dir}", flush=True)
    tokenizer = model.backbone.tokenizer() if hasattr(model, "backbone") else model.tokenizer()
    train_loader, val_loader, test_loader, label_map = get_downstream_dataloaders(
        task=task,
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        num_workers=num_workers,
    )
    print(f"[frozen-probe] data loaded: train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}", flush=True)
    if label_map is not None:
        num_classes = len(label_map)
        label_map_serializable = {int(k) if hasattr(k, "item") else k: int(v) for k, v in label_map.items()}
        with open(output_dir / "label_map.json", "w", encoding="utf-8") as fh:
            json.dump(label_map_serializable, fh, indent=2, default=str)

    # --------------------------------------------------------------- probe
    probe = _build_probe(task_type, feature_extractor.output_dim(), num_classes)
    probe.to(device)
    print(f"[frozen-probe] probe built: {sum(p.numel() for p in probe.parameters())} params", flush=True)

    # --------------------------------------------------------------- train
    # Task-aware epoch budget: huge datasets cannot finish 50 epochs in 24h.
    train_size = len(train_loader.dataset)
    if args.max_epochs is not None:
        max_epochs = args.max_epochs
    elif train_size > 500_000:
        max_epochs = 5
    elif train_size > 100_000:
        max_epochs = 10
    else:
        max_epochs = 20
    max_epochs = int(training_cfg.get("max_epochs", max_epochs))
    max_training_hours = float(training_cfg.get("max_training_hours", args.max_training_hours or 0.0))

    # Optional optimizer-grid override (--lrs/--wds form the cartesian product).
    optimizer_grid = None
    if args.lrs or args.wds:
        from rna_head_alignment.probes.train import OPTIMIZER_GRID

        default_lrs = sorted({lr for lr, _ in OPTIMIZER_GRID})
        default_wds = sorted({wd for _, wd in OPTIMIZER_GRID})
        lrs = [float(x) for x in args.lrs.split(",")] if args.lrs else default_lrs
        wds = [float(x) for x in args.wds.split(",")] if args.wds else default_wds
        optimizer_grid = [(lr, wd) for lr in lrs for wd in wds]

    print(f"[frozen-probe] starting training: max_epochs={max_epochs} max_training_hours={max_training_hours}", flush=True)
    trainer = ProbeTrainer(
        feature_extractor=feature_extractor,
        probe=probe,
        task=task,
        device=device,
        output_dir=output_dir,
        max_epochs=max_epochs,
        patience=int(training_cfg.get("patience", args.patience or 10)),
        mixed_precision=mixed_precision,
        max_training_hours=max_training_hours,
        optimizer_grid=optimizer_grid,
    )
    try:
        summary = trainer.fit(train_loader, val_loader, num_classes=num_classes)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            print(f"[frozen-probe] OOM during training, early stopping: {exc}", flush=True)
            torch.cuda.empty_cache()
            summary = {
                "task": task,
                "metric_name": "oom",
                "best_val_metric": float("nan"),
                "best_cfg": None,
                "trainable_params": sum(p.numel() for p in probe.parameters()),
                "wall_time_sec": 0.0,
                "oom": True,
            }
        else:
            raise
    print(json.dumps(summary, indent=2, default=str))

    # --------------------------------------------------------------- eval
    if not args.skip_eval:
        eval_splits = [s.strip() for s in args.eval_splits.split(",") if s.strip()]
        split_to_loader = {"train": train_loader, "val": val_loader, "test": test_loader}
        for split in eval_splits:
            loader = split_to_loader.get(split)
            if loader is None:
                print(f"[frozen-probe] unknown eval split {split}, skipping", flush=True)
                continue
            try:
                result = evaluate_probe(
                    probe,
                    feature_extractor,
                    loader,
                    task,
                    device,
                    num_classes=num_classes,
                    mixed_precision=mixed_precision,
                )
                write_eval_outputs(output_dir, split, result)
                print(f"[{split}] {result['metrics']}")
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    print(f"[frozen-probe] OOM during {split} eval, skipping: {exc}", flush=True)
                    torch.cuda.empty_cache()
                else:
                    raise

    # --------------------------------------------------------------- manifest
    manifest = {
        "task": task,
        "feature_mode": feature_mode,
        "seed": args.seed,
        "task_type": task_type,
        "route_depth": route_depth,
        "checkpoint": str(_resolve_checkpoint(args.checkpoint)) if feature_mode != "vanilla_hidden" else None,
        "optimizer_grid": optimizer_grid,
        "output_dir": str(output_dir),
        "summary": summary,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
