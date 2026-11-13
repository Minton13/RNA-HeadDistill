"""Stage 4B evaluation logic for the bucketed full slim cache.

This module is the library counterpart to ``scripts/head2head/eval_head2head_full.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from rna_head_alignment.head2head.eval import (
    _build_linear_projections,
    _evaluate_batch,
    _load_model_for_eval,
    _summarize,
)
from rna_head_alignment.head2head.full_sequence_dataset import (
    FullSlimBucketDataset,
    LengthBucketedSampler,
    full_variable_length_collate,
)
from rna_head_alignment.head2head.head_manifest import ROUTE_DEPTHS
from rna_head_alignment.io.paths import get_device, load_yaml, resolve_path


def load_route_head_keys(full_cache_root: str | Path) -> dict[int, list[str]]:
    """Load route head-key lists from the first bucket file's metadata."""
    from safetensors import safe_open

    bucket_manifest_path = Path(full_cache_root) / "manifests" / "slim_cache_manifest.parquet"
    bucket_df = pd.read_parquet(bucket_manifest_path).dropna(subset=["file_path"])
    if bucket_df.empty:
        raise FileNotFoundError(f"no bucket files in {bucket_manifest_path}")
    first_path = Path(str(bucket_df["file_path"].iloc[0]))
    if not first_path.is_absolute():
        candidate_cwd = Path.cwd() / first_path
        candidate_root = Path(full_cache_root) / first_path
        first_path = candidate_cwd if candidate_cwd.exists() else candidate_root
    with safe_open(str(first_path), framework="pt") as f:
        metadata = f.metadata() or {}
    return {
        depth: json.loads(metadata.get(f"route_heads_L{depth}", "[]"))
        for depth in ROUTE_DEPTHS
    }


def evaluate_head2head_student_full(
    config_path: str | Path,
    checkpoint_path: str | Path,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    device: str | torch.device = "auto",
    splits: list[str] | None = None,
    max_buckets: int | None = None,
) -> dict[str, Any]:
    """Evaluate a trained head-to-head student on the full bucketed slim cache."""
    cfg = load_yaml(config_path)
    run_name = run_name or cfg.get("run_name")
    if not run_name:
        raise ValueError("run_name must be provided or set in config")

    if output_dir is not None:
        out_root = Path(output_dir) / run_name
    else:
        out_root = (
            resolve_path(cfg.get("output_dir", "outputs/alignment/stage4b_head2head/eval"))
            / run_name
        )
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(get_device(device) if str(device).lower() == "auto" else device)
    precision = cfg.get("precision", "bf16")
    if precision == "bf16":
        amp_dtype = torch.bfloat16
    elif precision == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    use_amp = precision in ("bf16", "fp16") and device.type == "cuda"

    full_cache_root = resolve_path(cfg["full_slim_cache_root"])
    sequence_manifest_path = cfg["sequence_manifest"]
    route_head_keys = load_route_head_keys(full_cache_root)

    model, _ckpt = _load_model_for_eval(
        checkpoint_path, cfg, device, route_head_keys=route_head_keys
    )
    tokenizer = model.backbone.tokenizer()

    splits_cfg = cfg.get("splits", {})
    split_paths: dict[str, Path] = {}
    for split in ["val", "test"]:
        key = f"{split}_index"
        if key in splits_cfg:
            split_paths[split] = Path(splits_cfg[key])
    if not split_paths:
        index_root = Path(full_cache_root) / "index"
        for split in ["val", "test"]:
            p = index_root / f"{split}.parquet"
            if p.exists():
                split_paths[split] = p

    if splits is not None:
        split_paths = {k: v for k, v in split_paths.items() if k in splits}
    if not split_paths:
        raise ValueError("no val/test split indices configured or found")

    baseline_projections = _build_linear_projections(model)
    metric_names = list(cfg.get("metrics", []))
    baselines = list(
        cfg.get("baselines", ["native_head", "linear_projection", "random_student_head"])
    )
    length_bins = list(cfg.get("length_bins", [0, 64, 128, 256, 384, 512]))
    max_len = int(cfg.get("max_len", 1024))
    batch_size = int(cfg.get("batch_size", 8))

    report: dict[str, Any] = {
        "run_name": run_name,
        "output_root": str(out_root),
        "splits": {},
    }

    for split_name, split_path in split_paths.items():
        print(f"[h2h-eval-full] building dataset for split={split_name} from {split_path}", flush=True)
        dataset = FullSlimBucketDataset(
            full_cache_root,
            split_path,
            sequence_manifest_path,
            tokenizer,
            max_len=max_len,
            cache_size=1,
        )
        print(f"[h2h-eval-full] split={split_name} dataset ready: n={len(dataset)}", flush=True)

        # Use the same length/pair-token aware batching as training to avoid
        # OOM on long sequences while keeping throughput high on short ones.
        dynamic_cfg = cfg.get("dynamic_batch", {})
        max_pair_tokens = int(dynamic_cfg.get("max_pair_tokens", 524288))
        max_sequences = int(dynamic_cfg.get("max_sequences_per_batch", batch_size))
        sampler = LengthBucketedSampler(
            dataset.lengths(),
            max_pair_tokens=max_pair_tokens,
            max_sequences_per_batch=max_sequences,
            bucket_size_samples=dynamic_cfg.get("bucket_size_samples"),
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
        batch_idx = 0
        n_batches = len(loader)
        for batch in loader:
            rows.extend(
                _evaluate_batch(
                    model,
                    batch,
                    route_head_keys,
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
            batch_idx += 1
            if batch_idx % 10 == 0 or batch_idx == n_batches:
                print(
                    f"[h2h-eval-full] split={split_name} batch={batch_idx}/{n_batches} "
                    f"rows={len(rows)} mem_allocated={torch.cuda.memory_allocated(device) / 1e9:.2f}GB",
                    flush=True,
                )

        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        per_head_path = split_dir / "per_head_metrics.parquet"
        df.to_parquet(per_head_path, index=False)

        summary = _summarize(df, cfg.get("group_by"), length_bins)
        summary_path = split_dir / "summary.parquet"
        summary.to_parquet(summary_path, index=False)

        report["splits"][split_name] = {
            "per_head_metrics": str(per_head_path),
            "summary": str(summary_path),
            "n_rows": len(df),
            "n_batches": len(loader),
        }

    return report
