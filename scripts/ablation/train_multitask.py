#!/usr/bin/env python
"""CLI for training the multi-task joint Stage 5 model.

Examples
--------
Without replay:
    PYTHONPATH=src python scripts/ablation/train_multitask.py \
        --mode joint_no_replay --seed 17

With alignment replay:
    PYTHONPATH=src python scripts/ablation/train_multitask.py \
        --mode joint_with_replay --seed 17 --lambda-align 0.1
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
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rna_head_alignment.head2head.eval import (
    _load_checkpoint as _load_h2h_checkpoint,
    _load_model_for_eval,
)
from rna_head_alignment.io.paths import get_device, load_yaml, resolve_path
from rna_head_alignment.probes.datasets import get_downstream_dataloaders
from rna_head_alignment.probes.finetune import build_replay_subset_index
from rna_head_alignment.probes.multitask import MultiTaskHead2HeadStudent, MultiTaskTrainer
from rna_head_alignment.probes.pair_probe import PairProbe
from rna_head_alignment.probes.sequence_probe import SequenceProbe
from rna_head_alignment.probes.task_config import (
    TASK_TO_ROUTE_DEPTH,
    TASK_TO_TYPE,
    downstream_data_dir,
)
from rna_head_alignment.probes.token_probe import TokenProbe


TASKS = [
    "splice_donor_seqcls",
    "splice_acceptor_seqcls",
    "rbp_binding_seqcls",
    "ncRNA_family_seqcls",
    "chemical_reactivity_token",
    "secondary_structure_pair",
]


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
        raise ValueError("No checkpoint provided and registry missing full_checkpoint_best")
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


def _load_stage4b_student(checkpoint_path: str | Path, device: torch.device) -> Any:
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
        buckets_dir = Path(slim_cache_root) / "buckets"
        candidates = sorted(buckets_dir.glob("shard_*.safetensors"))
        if not candidates:
            candidates = sorted(Path(slim_cache_root).glob("shard_*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No slim buckets under {slim_cache_root}")
        first_shard = int(candidates[0].stem.split("_")[1])
    head_ids_by_route = _load_head_ids(slim_cache_root, first_shard)
    model, _ = _load_model_for_eval(checkpoint_path, cfg, device, teacher_head_ids=head_ids_by_route)
    return model


def _build_probe(task_type: str, input_dim: int, num_classes: int | None = None) -> torch.nn.Module:
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
    parser.add_argument("--mode", required=True, choices=["joint_no_replay", "joint_with_replay"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-root", default="outputs/probes/multitask")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--save-every", type=int, default=1_000)
    parser.add_argument("--backbone-lr", type=float, default=2.0e-6)
    parser.add_argument("--adapter-lr", type=float, default=3.0e-5)
    parser.add_argument("--probe-lr", type=float, default=1.0e-4)
    parser.add_argument("--lambda-align", type=float, default=0.1)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--size-proportional", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--debug-steps", type=int, default=10)
    parser.add_argument("--eval-subset-size", type=int, default=None, help="Evaluate on a fixed-size subset of val during training for speed")
    args = parser.parse_args()

    _set_seed(args.seed)
    device = torch.device(get_device(args.device))

    cfg: dict[str, Any] = load_yaml(args.config) if args.config else {}
    runtime_cfg = cfg.get("runtime", {})
    training_cfg = cfg.get("training", {})

    max_len = int(runtime_cfg.get("max_len", args.max_len))
    batch_size = int(runtime_cfg.get("batch_size", args.batch_size))
    num_workers = int(runtime_cfg.get("num_workers", args.num_workers))
    mixed_precision = bool(training_cfg.get("mixed_precision", args.mixed_precision))

    output_dir = Path(args.output_root) / args.mode / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    student = _load_stage4b_student(checkpoint_path, device)
    # Re-seed after loading the pretrained checkpoint.  Some model-loading paths
    # (e.g. transformers / checkpoint unpickling) can advance or reset the CPU
    # RNG state; we want DataLoader shuffling and probe initialization to be
    # deterministic per args.seed.
    _set_seed(args.seed)
    tokenizer = student.backbone.tokenizer()

    task_loaders: dict[str, Any] = {}
    val_loaders: dict[str, Any] = {}
    test_loaders: dict[str, Any] = {}
    probes: dict[str, torch.nn.Module] = {}
    task_num_classes: dict[str, int | None] = {}
    label_maps: dict[str, dict | None] = {}

    first_task = TASKS[0]
    for task in TASKS:
        data_dir = cfg.get("data_dir") or downstream_data_dir(task)
        train_loader, val_loader, test_loader, label_map = get_downstream_dataloaders(
            task=task,
            data_dir=data_dir,
            tokenizer=tokenizer,
            batch_size=batch_size,
            max_len=max_len,
            num_workers=num_workers,
        )
        task_loaders[task] = train_loader
        val_loaders[task] = val_loader
        test_loaders[task] = test_loader
        label_maps[task] = label_map
        num_classes = len(label_map) if label_map is not None else None
        task_num_classes[task] = num_classes
        probes[task] = _build_probe(TASK_TO_TYPE[task], student.backbone.hidden_size, num_classes)
        probes[task].to(device)

    replay_dataset = None
    replay_loader = None
    if args.mode == "joint_with_replay":
        registry = load_yaml("configs/ablation/experiment_registry.yaml")
        slim_cache_root = cfg.get("slim_cache_root") or registry.get("base_data", {}).get("slim_cache_root", "outputs/teacher_attention_cache_slim/stage4b_head2head/full")
        index_path = output_dir / "replay_subset_index.parquet"
        if index_path.exists():
            index_df = pd.read_parquet(index_path)
        else:
            index_df = build_replay_subset_index(
                slim_cache_root=resolve_path(slim_cache_root),
                n_sequences=args.replay_size,
                seed=args.seed,
                output_path=index_path,
            )
        from rna_head_alignment.probes.finetune import ReplaySubsetDataset, replay_collate

        replay_dataset = ReplaySubsetDataset(
            index_df=index_df,
            slim_cache_root=resolve_path(slim_cache_root),
            tokenizer=tokenizer,
            max_len=max_len,
        )
        replay_loader = torch.utils.data.DataLoader(
            replay_dataset,
            batch_size=batch_size,
            collate_fn=replay_collate,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
        )

    multi_task_model = MultiTaskHead2HeadStudent(student=student, probes=probes)
    multi_task_model.to(device)

    trainer = MultiTaskTrainer(
        multi_task_model=multi_task_model,
        task_loaders=task_loaders,
        task_num_classes=task_num_classes,
        device=device,
        output_dir=output_dir,
        replay_loader=replay_loader,
        lambda_align=args.lambda_align,
        backbone_lr=args.backbone_lr,
        adapter_lr=args.adapter_lr,
        probe_lr=args.probe_lr,
        max_steps=int(training_cfg.get("max_steps", args.max_steps)),
        eval_every=int(training_cfg.get("eval_every", args.eval_every)),
        save_every=int(training_cfg.get("save_every", args.save_every)),
        grad_clip=float(training_cfg.get("grad_clip", 1.0)),
        mixed_precision=mixed_precision,
        size_proportional=args.size_proportional,
        warmup_steps=int(training_cfg.get("warmup_steps", args.warmup_steps)),
        debug_steps=int(training_cfg.get("debug_steps", args.debug_steps)),
        eval_subset_size=int(training_cfg.get("eval_subset_size", args.eval_subset_size or 0)) or None,
    )
    summary = trainer.fit(val_loaders)
    print(json.dumps(summary, indent=2, default=str))

    eval_subset_size = int(training_cfg.get("eval_subset_size", args.eval_subset_size or 0)) or None
    if not args.skip_eval:
        from rna_head_alignment.probes.eval import evaluate_probe, write_eval_outputs
        from rna_head_alignment.probes.feature_extractor import ProbeFeatureExtractor

        extractor = ProbeFeatureExtractor(
            model=multi_task_model.student,
            feature_mode="distilled_hidden",
            task=TASKS[0],
            route_depth=TASK_TO_ROUTE_DEPTH[TASKS[0]],
        )
        extractor.to(device)
        for task in TASKS:
            extractor.task = task
            extractor.route_depth = TASK_TO_ROUTE_DEPTH[task]
            extractor.route_key = f"L{TASK_TO_ROUTE_DEPTH[task]}"
            for split, loader in [("train", task_loaders[task]), ("val", val_loaders[task]), ("test", test_loaders[task])]:
                # Use a subset for final evaluation to keep wall time reasonable.
                eval_loader = loader
                if eval_subset_size is not None and split != "val" and hasattr(loader, "dataset"):
                    n = min(eval_subset_size, len(loader.dataset))
                    subset = torch.utils.data.Subset(loader.dataset, range(n))
                    eval_loader = torch.utils.data.DataLoader(
                        subset,
                        batch_size=loader.batch_size or 16,
                        collate_fn=loader.collate_fn,
                        shuffle=False,
                        num_workers=0,
                        pin_memory=False,
                    )
                result = evaluate_probe(
                    multi_task_model.probes[task],
                    extractor,
                    eval_loader,
                    task,
                    device,
                    num_classes=task_num_classes[task],
                    mixed_precision=mixed_precision,
                )
                task_output = output_dir / task
                task_output.mkdir(parents=True, exist_ok=True)
                write_eval_outputs(task_output, split, result)
                print(f"[{task}/{split}] {result['metrics']}")

    manifest = {
        "mode": args.mode,
        "seed": args.seed,
        "lambda_align": args.lambda_align,
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "summary": summary,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
