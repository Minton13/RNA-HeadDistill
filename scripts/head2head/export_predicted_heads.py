#!/usr/bin/env python
"""Export predicted teacher-head attention maps per slim shard.

Loads a trained :class:`RiNALMoHeadToHeadStudent` and writes one ``.pt`` file
per evaluated shard containing the model's predicted maps together with seq_ids,
lengths, and the teacher-head identity lists.

Example:

    PYTHONPATH=src python scripts/head2head/export_predicted_heads.py \
        --config configs/head2head/eval_head2head.yaml \
        --checkpoint runs/head2head/rinalmo_head2head_stage4b/checkpoints/best_val.pt \
        --run-name rinalmo_head2head_stage4b \
        --split val
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from rna_head_alignment.head2head.eval import _load_head_ids, _load_model_for_eval
from rna_head_alignment.head2head.head_manifest import ROUTE_DEPTHS
from rna_head_alignment.head2head.sequence_dataset import (
    SlimShardDataset,
    variable_length_collate,
)
from rna_head_alignment.io.paths import load_yaml, resolve_path


def _resolve_shard_indices(cfg: dict[str, Any], split: str) -> list[int]:
    splits_cfg = cfg.get("splits", {})
    key = f"{split}_shards"
    shards = list(splits_cfg.get(key) or [])
    if not shards and split == "val" and cfg.get("val_shards"):
        shards = list(cfg["val_shards"])
    if not shards and split == "test" and cfg.get("test_shards"):
        shards = list(cfg["test_shards"])
    if not shards:
        raise ValueError(f"No shard indices configured for split {split!r}")
    return shards


def export_predicted_heads(
    config_path: str | Path,
    checkpoint_path: str | Path,
    run_name: str | None = None,
    split: str = "val",
    shards: list[int] | None = None,
    out_root: str | Path | None = None,
    device: str = "auto",
    batch_size: int | None = None,
    max_len: int | None = None,
) -> dict[str, Any]:
    cfg = load_yaml(config_path)
    run_name = run_name or cfg.get("run_name")
    if not run_name:
        raise ValueError("run_name must be provided or set in config")

    if out_root is not None:
        out_dir = Path(out_root)
    else:
        out_dir = (
            resolve_path(
                cfg.get(
                    "predicted_heads_output_dir",
                    "outputs/alignment/stage4b_head2head/predicted_teacher_heads",
                )
            )
            / run_name
            / split
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(device)
    precision = cfg.get("precision", "bf16")
    if precision == "bf16":
        amp_dtype = torch.bfloat16
    elif precision == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    use_amp = precision in ("bf16", "fp16") and device.type == "cuda"

    model, _ckpt = _load_model_for_eval(checkpoint_path, cfg, device)
    tokenizer = model.backbone.tokenizer()

    shard_indices = shards if shards is not None else _resolve_shard_indices(cfg, split)
    if not shard_indices:
        raise ValueError(f"No shards to export for split {split!r}")
    head_ids_by_route = _load_head_ids(cfg["slim_cache_root"], shard_indices[0])

    batch_size = int(batch_size or cfg.get("batch_size", 8))
    max_len = int(max_len or cfg.get("max_len", 512))
    written: list[str] = []

    for shard_idx in shard_indices:
        shard_path = resolve_path(cfg["slim_cache_root"]) / f"shard_{shard_idx:05d}.pt"
        dataset = SlimShardDataset(
            [str(shard_path)], tokenizer, max_len=max_len, cache_size=1
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=variable_length_collate,
            shuffle=False,
            num_workers=0,
        )

        preds: dict[int, list[torch.Tensor]] = {d: [] for d in ROUTE_DEPTHS}
        teachers: dict[int, list[torch.Tensor]] = {d: [] for d in ROUTE_DEPTHS}
        seq_ids: list[str] = []
        lengths: list[torch.Tensor] = []

        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
                out = model(input_ids, attention_mask)
            for depth in ROUTE_DEPTHS:
                preds[depth].append(out[f"pred_L{depth}"].cpu())
                teachers[depth].append(batch[f"teacher_L{depth}"].cpu())
            seq_ids.extend(batch["seq_ids"])
            lengths.append(batch["lengths"].cpu())

        payload: dict[str, Any] = {
            "seq_ids": seq_ids,
            "shard_id": f"{shard_idx:05d}",
            "lengths": torch.cat(lengths),
        }
        for depth in ROUTE_DEPTHS:
            payload[f"pred_L{depth}"] = torch.cat(preds[depth], dim=0)
            payload[f"teacher_L{depth}"] = torch.cat(teachers[depth], dim=0)
            payload[f"head_ids_L{depth}"] = head_ids_by_route[depth]

        out_path = out_dir / f"shard_{shard_idx:05d}.pt"
        torch.save(payload, out_path)
        written.append(str(out_path))

    return {"run_name": run_name, "split": split, "output_dir": str(out_dir), "files": written}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="eval_head2head.yaml")
    parser.add_argument("--checkpoint", required=True, help="path to trained model checkpoint")
    parser.add_argument("--run-name", default=None, help="run identifier")
    parser.add_argument("--shards", default=None, help="comma list of shard indices, overrides config split")
    parser.add_argument("--split", default="val", choices=["val", "test"], help="split to export")
    parser.add_argument("--out-root", default=None, help="override output root")
    parser.add_argument("--device", default="auto", help="cpu, cuda, or auto")
    parser.add_argument("--batch-size", type=int, default=None, help="override batch size")
    parser.add_argument("--max-len", type=int, default=None, help="override max length")
    args = parser.parse_args()

    shard_indices = [int(s) for s in args.shards.split(",") if s.strip()] if args.shards else None
    report = export_predicted_heads(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        run_name=args.run_name,
        split=args.split,
        shards=shard_indices,
        out_root=args.out_root,
        device=args.device,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
