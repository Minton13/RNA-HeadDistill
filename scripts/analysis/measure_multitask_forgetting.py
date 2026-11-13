#!/usr/bin/env python3
"""Measure alignment forgetting for multitask joint runs (publication fix).

The multitask joint training never recorded pre/post alignment metrics
(``pre_alignment``/``post_alignment`` empty in training_summary.json).  This
script measures them post hoc from existing checkpoints — no retraining:

  * pre  = Stage-4B student before joint training (registry full_checkpoint_best)
  * post = student part of ``outputs/probes/multitask/{mode}/seed_{s}/best_checkpoint.pt``

Alignment metrics (per route L6/L8/L12: rowwise KL, cosine, MSE) are computed
against cached teacher attention on a replay subset rebuilt with the *current*
``build_replay_subset_index`` (the deployed F2-era index parquets lack the
``sequence_manifest_index`` column; rebuilding yields a correct seq->bucket-row
mapping).  Per seed the same subset is used for pre and post.

Outputs:
  outputs/probes/multitask/{mode}/seed_{s}/alignment_forgetting.json
  outputs/probes/multitask/alignment_forgetting_summary.csv  (aggregate)

Usage:
    PYTHONPATH=src python scripts/analysis/measure_multitask_forgetting.py \
        [--modes joint_no_replay joint_with_replay] [--seeds 17 41 89] \
        [--max-seqs 1024]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/ablation"))

from train_multitask import _load_stage4b_student, _resolve_checkpoint
from rna_head_alignment.head2head.metrics import compute_head_metrics
from rna_head_alignment.io.paths import load_yaml, resolve_path
from rna_head_alignment.probes.finetune import (
    ReplaySubsetDataset,
    build_replay_subset_index,
    replay_collate,
)
from rna_head_alignment.probes.multitask import MultiTaskHead2HeadStudent

MT_ROOT = ROOT / "outputs/probes/multitask"
ROUTES = (6, 8, 12)


@torch.no_grad()
def _alignment_metrics(student, replay_loader, device, max_seqs: int) -> dict[str, float]:
    student.eval()
    accum: dict[str, list[float]] = {}
    counts = 0
    for batch in replay_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        lengths = batch["lengths"].to(device)
        out = student(input_ids, attention_mask, return_backbone=True)
        for depth in ROUTES:
            route_key = f"L{depth}"
            pred = out[f"pred_{route_key}"].float()
            teacher = batch[f"teacher_{route_key}"].to(device).float()
            per_seq = compute_head_metrics(
                pred, teacher, lengths,
                metrics=["mse", "rowwise_kl", "cosine"],
                include_per_sequence=True,
            )
            for metric in ("mse", "rowwise_kl", "cosine"):
                vals = per_seq[f"{metric}_per_seq"].mean(dim=1).detach().cpu().numpy()
                accum.setdefault(f"{route_key}_{metric}", []).extend(
                    float(v) for v in vals if not np.isnan(v)
                )
        counts += input_ids.shape[0]
        if counts >= max_seqs:
            break
    return {k: float(np.mean(v)) if v else float("nan") for k, v in accum.items()}


def _macro(metrics: dict[str, float], metric: str) -> float:
    vals = [metrics[f"L{d}_{metric}"] for d in ROUTES if not np.isnan(metrics.get(f"L{d}_{metric}", float("nan")))]
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--modes", nargs="*", default=["joint_no_replay", "joint_with_replay"])
    parser.add_argument("--seeds", nargs="*", type=int, default=[17, 41, 89])
    parser.add_argument("--max-seqs", type=int, default=1024)
    parser.add_argument("--replay-size", type=int, default=50_000)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, flush=True)

    registry = load_yaml(ROOT / "configs/ablation/experiment_registry.yaml")
    slim_cache_root = resolve_path(
        registry.get("base_data", {}).get("slim_cache_root",
                                          "outputs/teacher_attention_cache_slim/stage4b_head2head/full")
    )

    pre_student = _load_stage4b_student(_resolve_checkpoint(None), device)
    tokenizer = pre_student.backbone.tokenizer()

    rows: list[dict] = []
    for seed in args.seeds:
        # Rebuild the replay subset with the current code (adds the
        # sequence_manifest_index column the deployed F2-era parquets lack).
        index_path = MT_ROOT / f"replay_subset_index_fix_seed{seed}.parquet"
        if index_path.exists():
            index_df = pd.read_parquet(index_path)
        else:
            index_df = build_replay_subset_index(
                slim_cache_root=slim_cache_root,
                n_sequences=args.replay_size,
                seed=seed,
                output_path=index_path,
            )
        # Deduplicate to one row per sequence (expanded index is per seq x head).
        seq_df = index_df.drop_duplicates("seq_id").reset_index(drop=True)
        dataset = ReplaySubsetDataset(
            index_df=seq_df, slim_cache_root=slim_cache_root,
            tokenizer=tokenizer, max_len=args.max_len,
        )
        loader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=replay_collate,
                            shuffle=False, num_workers=0)

        pre_metrics = _alignment_metrics(pre_student, loader, device, args.max_seqs)
        print(f"[seed {seed}] pre: {pre_metrics}", flush=True)

        for mode in args.modes:
            ckpt_path = MT_ROOT / mode / f"seed_{seed}" / "best_checkpoint.pt"
            if not ckpt_path.exists():
                print(f"[skip] missing {ckpt_path}", flush=True)
                continue
            post_student = _load_stage4b_student(_resolve_checkpoint(None), device)
            mt_model = MultiTaskHead2HeadStudent(student=post_student, probes={})
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            mt_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            post_student.to(device)
            post_metrics = _alignment_metrics(post_student, loader, device, args.max_seqs)

            record = {
                "mode": mode,
                "seed": seed,
                "n_replay_seqs_evaluated": min(args.max_seqs, len(seq_df)),
                "replay_index": str(index_path),
                "pre": pre_metrics,
                "post": post_metrics,
                "delta": {k: post_metrics.get(k, float("nan")) - pre_metrics.get(k, float("nan"))
                          for k in pre_metrics},
            }
            record["macro"] = {
                m: {"pre": _macro(pre_metrics, m), "post": _macro(post_metrics, m),
                    "delta": _macro(post_metrics, m) - _macro(pre_metrics, m)}
                for m in ("cosine", "rowwise_kl", "mse")
            }
            out_path = MT_ROOT / mode / f"seed_{seed}" / "alignment_forgetting.json"
            out_path.write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
            print(f"[{mode}/seed_{seed}] post: {post_metrics}", flush=True)
            print(f"[{mode}/seed_{seed}] macro delta: {record['macro']}", flush=True)

            for metric in ("cosine", "rowwise_kl", "mse"):
                rows.append({
                    "mode": mode, "seed": seed, "metric": metric,
                    "pre": record["macro"][metric]["pre"],
                    "post": record["macro"][metric]["post"],
                    "delta": record["macro"][metric]["delta"],
                })
            del mt_model, post_student
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(MT_ROOT / "alignment_forgetting_summary.csv", index=False)
        print(df.groupby(["mode", "metric"])[["pre", "post", "delta"]].agg(["mean", "std"]), flush=True)
        print(f"wrote {MT_ROOT / 'alignment_forgetting_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
