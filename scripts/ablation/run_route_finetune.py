#!/usr/bin/env python
"""CLI for route-specific fine-tuning of a Stage 4B student on a downstream task.

Examples
--------
F1 (no replay):
    PYTHONPATH=src python scripts/ablation/run_route_finetune.py \
        --task splice_donor_seqcls --feature-mode F1 --seed 17

F2 (with alignment replay):
    PYTHONPATH=src python scripts/ablation/run_route_finetune.py \
        --task chemical_reactivity_token --feature-mode F2 --seed 17 \
        --lambda-align 0.1
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
from rna_head_alignment.probes.finetune import (
    RouteSpecificFineTuner,
    build_replay_subset_index,
)
from rna_head_alignment.probes.pair_probe import PairProbe
from rna_head_alignment.probes.sequence_probe import SequenceProbe
from rna_head_alignment.probes.task_config import (
    TASK_TO_ROUTE_DEPTH,
    TASK_TO_TYPE,
    downstream_data_dir,
    validate_task,
)
from rna_head_alignment.probes.token_probe import TokenProbe


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


def _load_or_build_replay_index(
    slim_cache_root: str | Path,
    output_dir: Path,
    n_sequences: int,
    seed: int,
) -> pd.DataFrame:
    import pandas as pd

    index_path = output_dir / "replay_subset_index.parquet"
    if index_path.exists():
        return pd.read_parquet(index_path)
    df = build_replay_subset_index(
        slim_cache_root=slim_cache_root,
        n_sequences=n_sequences,
        seed=seed,
        output_path=index_path,
    )
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--feature-mode", required=True, choices=["F0", "F1", "F2"])
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--output-root", default="outputs/probes/finetune")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=2_000)
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--backbone-lr", type=float, default=2.0e-6)
    parser.add_argument("--adapter-lr", type=float, default=3.0e-5)
    parser.add_argument("--probe-lr", type=float, default=1.0e-4)
    parser.add_argument("--lambda-align", type=float, default=0.1)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()

    task = args.task
    validate_task(task)
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
        / args.feature_mode
        / f"seed_{args.seed}"
        / f"lambda_{args.lambda_align}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = _resolve_checkpoint(args.checkpoint)
    print(f"[route-finetune] loading student from {checkpoint_path}", flush=True)
    student = _load_stage4b_student(checkpoint_path, device)
    route_depth = TASK_TO_ROUTE_DEPTH[task]
    tokenizer = student.backbone.tokenizer()

    data_dir = cfg.get("data_dir") or downstream_data_dir(task)
    print(f"[route-finetune] loading downstream data from {data_dir}", flush=True)
    train_loader, val_loader, test_loader, label_map = get_downstream_dataloaders(
        task=task,
        data_dir=data_dir,
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        num_workers=num_workers,
    )
    print(f"[route-finetune] data loaded: train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}", flush=True)
    num_classes = len(label_map) if label_map is not None else None

    replay_dataset = None
    if args.feature_mode == "F2":
        registry = load_yaml("configs/ablation/experiment_registry.yaml")
        slim_cache_root = cfg.get("slim_cache_root") or registry.get("base_data", {}).get("slim_cache_root", "outputs/teacher_attention_cache_slim/stage4b_head2head/full")
        index_df = _load_or_build_replay_index(
            slim_cache_root=resolve_path(slim_cache_root),
            output_dir=output_dir,
            n_sequences=args.replay_size,
            seed=args.seed,
        )
        from rna_head_alignment.probes.finetune import ReplaySubsetDataset

        replay_dataset = ReplaySubsetDataset(
            index_df=index_df,
            slim_cache_root=resolve_path(slim_cache_root),
            tokenizer=tokenizer,
            max_len=max_len,
        )

    task_type = TASK_TO_TYPE[task]
    probe = _build_probe(task_type, student.backbone.hidden_size, num_classes)
    probe.to(device)

    finetuner = RouteSpecificFineTuner(
        student=student,
        task=task,
        route_depth=route_depth,
        feature_mode=args.feature_mode,
        device=device,
        output_dir=output_dir,
        replay_dataset=replay_dataset,
        lambda_align=args.lambda_align,
        backbone_lr=args.backbone_lr,
        adapter_lr=args.adapter_lr,
        probe_lr=args.probe_lr,
        max_steps=int(training_cfg.get("max_steps", args.max_steps)),
        eval_every=int(training_cfg.get("eval_every", args.eval_every)),
        save_every=int(training_cfg.get("save_every", args.save_every)),
        grad_clip=float(training_cfg.get("grad_clip", 1.0)),
        mixed_precision=mixed_precision,
    )
    summary = finetuner.fit(probe, train_loader, val_loader, num_classes=num_classes)
    print(json.dumps(summary, indent=2, default=str))

    if not args.skip_eval:
        from rna_head_alignment.probes.eval import evaluate_probe, write_eval_outputs
        from rna_head_alignment.probes.feature_extractor import ProbeFeatureExtractor

        extractor = ProbeFeatureExtractor(
            model=student,
            feature_mode="distilled_hidden",
            task=task,
            route_depth=route_depth,
        )
        extractor.to(device)
        for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
            result = evaluate_probe(
                probe,
                extractor,
                loader,
                task,
                device,
                num_classes=num_classes,
                mixed_precision=mixed_precision,
            )
            write_eval_outputs(output_dir, split, result)
            print(f"[{split}] {result['metrics']}")

    manifest = {
        "task": task,
        "feature_mode": args.feature_mode,
        "seed": args.seed,
        "lambda_align": args.lambda_align,
        "route_depth": route_depth,
        "checkpoint": str(checkpoint_path),
        "output_dir": str(output_dir),
        "summary": summary,
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, default=str)


if __name__ == "__main__":
    main()
