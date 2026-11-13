#!/usr/bin/env python
"""Pairing-stability ablation (publication fix).

The original ``pairing_stability`` supplementary driver mis-wired the config
(no teacher heads selected) and produced empty runs.  The intended question
(experiment_registry.yaml): "How stable is the data-driven head pairing across
bootstrap subsets?" — 5 bootstrap subsets x 16 shards.

This script recomputes the Stage-4B data-driven head pairing on each existing
bootstrap subset and measures stability:

  1. For bootstrap b, read
     ``supplementary_indices/pairing_stability_bootstrap{b}_nshards16_seed17.parquet``
     (already on disk) to get the 16 source shard ids.
  2. Load teacher attention for sampled sequences of those shards from the
     full bucketed slim cache (``full/buckets/shard_{sid}_bucket_*.safetensors``).
  3. Run the *vanilla* rinalmo-micro student, capture all native head outputs
     at L6/L8/L12, score every (teacher, student-head) pair with the same
     metrics/combination as the original pair search (``score_head_pairs`` +
     ``solve_pairing``, bipartite).
  4. Write ``bootstrap{b}_seed17/pair_manifest.parquet`` (+ score matrices).

``--reduce`` then aggregates the 5 bootstrap pairings against the reference
``head_pair_manifest.parquet`` and writes ``pairing_stability_summary.parquet``
/ ``pairing_stability.md`` under ``outputs/ablation/alignment/pairing_stability``.

Usage:
    PYTHONPATH=src python scripts/ablation/run_pairing_stability.py --bootstrap 0
    PYTHONPATH=src python scripts/ablation/run_pairing_stability.py --reduce
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from torch.nn.utils.rnn import pad_sequence

from rna_head_alignment.head2head.head_manifest import PAIR_MANIFEST_COLUMNS, STUDENT_MODEL, load_manifest
from rna_head_alignment.head2head.pair_search import (
    DEFAULT_METRICS,
    DEFAULT_SPEARMAN_SAMPLE_SIZE,
    score_head_pairs,
    solve_pairing,
)
from rna_head_alignment.head2head.sequence_dataset import align_tokens_to_sequence
from rna_head_alignment.io.paths import resolve_path

ALIGNMENT_ROOT = Path("outputs/ablation/alignment")
PS_ROOT = ALIGNMENT_ROOT / "pairing_stability"
INDEX_ROOT = ALIGNMENT_ROOT / "supplementary_indices"
FULL_CACHE = Path("outputs/teacher_attention_cache_slim/stage4b_head2head/full")
MANIFEST_DIR = Path("outputs/teacher_attention_cache_slim/stage4b_head2head/manifests")
ROUTE_DEPTHS = (6, 8, 12)
STUDENT_MODEL_DIR = "assets/models/rinalmo-micro"


def _bootstrap_shard_ids(bootstrap_idx: int, seed: int, n_shards: int = 16) -> list[str]:
    path = INDEX_ROOT / f"pairing_stability_bootstrap{bootstrap_idx}_nshards{n_shards}_seed{seed}.parquet"
    df = pd.read_parquet(resolve_path(path))
    shard_ids = df["original_shard_id"].apply(
        lambda x: str(x[0]) if isinstance(x, (list, np.ndarray)) and len(x) else str(x)
    )
    return sorted(shard_ids.unique())


def _load_bucket(bucket_path: Path) -> dict:
    with safe_open(str(bucket_path), framework="pt") as f:
        meta = f.metadata() or {}
        out = {
            "lengths": f.get_tensor("lengths"),
            "manifest_indices": f.get_tensor("sequence_manifest_indices"),
            "bucket_size": int(meta.get("bucket_size", 0)),
        }
        for depth in ROUTE_DEPTHS:
            out[f"teacher_heads_L{depth}"] = f.get_tensor(f"teacher_heads_L{depth}")
            out[f"availability_L{depth}"] = f.get_tensor(f"head_availability_L{depth}")
            out[f"head_ids_L{depth}"] = json.loads(meta.get(f"route_heads_L{depth}", "[]"))
    return out


def _evenly_spaced(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return np.unique(np.linspace(0, n - 1, k, dtype=int)).tolist()


def run_bootstrap(
    bootstrap_idx: int,
    seed: int,
    max_samples_per_shard: int = 256,
    batch_size: int = 8,
    precision: str = "bf16",
    device: str = "auto",
) -> Path:
    from rna_head_alignment.head2head.student_backbone import RiNALMoStudentBackbone
    from rna_head_alignment.io.paths import get_device

    dev = torch.device(get_device(device))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(precision, torch.float32)

    shard_ids = _bootstrap_shard_ids(bootstrap_idx, seed)
    print(f"[pairing-stability] bootstrap {bootstrap_idx}: {len(shard_ids)} shards: {shard_ids}", flush=True)

    seq_manifest = pd.read_parquet(resolve_path(MANIFEST_DIR / "sequence_manifest.parquet"))
    seq_by_index = seq_manifest["sequence"].astype(str)

    student = RiNALMoStudentBackbone(model_dir=resolve_path(STUDENT_MODEL_DIR), routes=ROUTE_DEPTHS)
    student.eval()
    for p in student.parameters():
        p.requires_grad_(False)
    student = student.to(dev).to(dtype)
    candidate_heads = list(range(student.num_heads))
    H = len(candidate_heads)
    tokenizer = student.tokenizer()

    rng = torch.Generator(device=dev).manual_seed(seed + bootstrap_idx)

    score_acc: dict[int, dict[str, np.ndarray]] = {}
    weight_acc: dict[int, np.ndarray] = {}
    teacher_keys_by_route: dict[int, list[str]] = {}

    buckets_dir = resolve_path(FULL_CACHE / "buckets")
    for sid in shard_ids:
        buckets = sorted(buckets_dir.glob(f"shard_{int(sid):05d}_bucket_*.safetensors"))
        if not buckets:
            print(f"[pairing-stability] shard {sid}: no bucket files, skipping", flush=True)
            continue

        # Sample up to max_samples_per_shard sequences, spread across this
        # shard's length buckets (evenly spaced within each bucket).
        sampled: list[tuple[dict, int]] = []  # (bucket, row_idx)
        per_bucket = max(1, max_samples_per_shard // max(1, len(buckets)))
        for bp in buckets:
            bucket = _load_bucket(bp)
            n = int(bucket["lengths"].shape[0])
            for row_idx in _evenly_spaced(n, per_bucket):
                sampled.append((bucket, row_idx))
        if len(sampled) > max_samples_per_shard:
            sampled = [sampled[i] for i in _evenly_spaced(len(sampled), max_samples_per_shard)]

        # Group by bucket for batching (uniform spatial size within a batch).
        by_bucket: dict[int, list[tuple[dict, int]]] = {}
        for item in sampled:
            by_bucket.setdefault(id(item[0]), []).append(item)

        for group in by_bucket.values():
            bucket = group[0][0]
            bucket_size = int(bucket["bucket_size"])
            max_len = min(bucket_size, 512)
            for start in range(0, len(group), batch_size):
                chunk = group[start:start + batch_size]
                rows = [r for _, r in chunk]
                sequences = [seq_by_index.iloc[int(bucket["manifest_indices"][r])] for r in rows]
                lengths = torch.tensor(
                    [min(int(bucket["lengths"][r]), max_len) for r in rows], dtype=torch.long
                )
                ids_list, mask_list = [], []
                for seq in sequences:
                    ids, _ = align_tokens_to_sequence(tokenizer, seq, max_len)
                    ids_list.append(ids)
                    mask_list.append(torch.ones_like(ids))
                input_ids = pad_sequence(ids_list, batch_first=True, padding_value=0).to(dev)
                attention_mask = pad_sequence(mask_list, batch_first=True, padding_value=0).to(dev)

                with torch.no_grad():
                    exits = student(input_ids, attention_mask, return_all_heads=True)

                for depth in ROUTE_DEPTHS:
                    key = f"L{depth}"
                    head_ids = bucket[f"head_ids_{key}"]
                    if not head_ids:
                        continue
                    T = len(head_ids)
                    teacher_keys_by_route[depth] = list(head_ids)
                    A = bucket[f"teacher_heads_{key}"][rows].to(torch.float32)
                    avail = bucket[f"availability_{key}"][rows]
                    mini = {
                        f"teacher_heads_{key}": A,
                        f"teacher_head_ids_{key}": head_ids,
                        "lengths": lengths,
                    }
                    S_route = exits[depth]["all_head_outputs"].to(torch.float32)
                    scores_df = score_head_pairs(
                        mini, S_route, depth,
                        metrics=list(DEFAULT_METRICS),
                        candidate_student_heads=candidate_heads,
                        spearman_sample_size=DEFAULT_SPEARMAN_SAMPLE_SIZE,
                        rng=rng,
                    )
                    if depth not in score_acc:
                        score_acc[depth] = {m: np.zeros((T, H)) for m in DEFAULT_METRICS}
                        weight_acc[depth] = np.zeros((T, H))
                    # Weight by availability: broadcast per-sample availability
                    # of teacher t into the (T, H) accumulation.
                    w = float(len(rows))
                    avail_t = avail.to(torch.float32).mean(dim=0).numpy()  # [T]
                    for metric in DEFAULT_METRICS:
                        mat = scores_df.pivot(
                            index="teacher_idx", columns="student_head", values=metric
                        ).loc[range(T), candidate_heads].to_numpy(dtype=np.float64)
                        score_acc[depth][metric] += w * mat * avail_t[:, None]
                    weight_acc[depth] += w * avail_t[:, None] * np.ones((T, H))

                del exits, input_ids, attention_mask
                if dev.type == "cuda":
                    torch.cuda.empty_cache()

        del bucket
        gc.collect()
        print(f"[pairing-stability] bootstrap {bootstrap_idx} shard {sid} done", flush=True)

    # Finalize per-route score matrices and solve the bipartite pairing.
    target = load_manifest(resolve_path(MANIFEST_DIR / "target_head_manifest.parquet"))
    reference = load_manifest(resolve_path(MANIFEST_DIR / "head_pair_manifest.parquet"))

    out_rows: list[dict] = []
    score_dump: dict[str, np.ndarray] = {}
    for depth in ROUTE_DEPTHS:
        if depth not in score_acc:
            print(f"[pairing-stability] WARNING: no scores for route L{depth}", flush=True)
            continue
        T, H = weight_acc[depth].shape
        rows = []
        final: dict[str, np.ndarray] = {}
        for metric in DEFAULT_METRICS:
            final[metric] = score_acc[depth][metric] / np.maximum(weight_acc[depth], 1e-12)
            score_dump[f"L{depth}_{metric}"] = final[metric]
        for t in range(T):
            for h_idx, h in enumerate(candidate_heads):
                row = {"teacher_idx": t,
                       "teacher_head_key": teacher_keys_by_route[depth][t],
                       "student_head": h}
                for metric in DEFAULT_METRICS:
                    row[metric] = float(final[metric][t, h_idx])
                rows.append(row)
        solved = solve_pairing(pd.DataFrame(rows), method="bipartite")

        target_route = target[target["route_depth"] == depth].copy()
        solved = solved.copy()
        solved["head_key"] = solved["teacher_head_key"]
        merged = target_route.merge(
            solved[["head_key", "student_head", "matching_metric", "matching_score", "pairing_method"]],
            on="head_key", how="left",
        )
        ref_lookup = dict(zip(reference["task"] + "|" + reference["teacher_model"]
                              + "|L" + reference["teacher_layer"].astype(str)
                              + ":H" + reference["teacher_head"].astype(str),
                              reference["student_head"]))
        for _, row in merged.iterrows():
            if pd.isna(row.get("student_head")):
                continue
            ref_key = f"{row['task']}|{row['teacher_model']}|L{row['teacher_layer']}:H{row['teacher_head']}"
            out_rows.append({
                "task": row["task"],
                "route_depth": depth,
                "teacher_model": row["teacher_model"],
                "teacher_layer": row["teacher_layer"],
                "teacher_head": row["teacher_head"],
                "teacher_cache_channel": row["cache_channel"],
                "student_model": STUDENT_MODEL,
                "student_layer": depth,
                "student_head": int(row["student_head"]),
                "matching_metric": row.get("matching_metric", "combined"),
                "matching_score": float(row["matching_score"]) if pd.notna(row.get("matching_score")) else float("nan"),
                "pairing_method": row.get("pairing_method", "bipartite"),
                "keep": True,
                "adapter_key": f"L{depth}_H{int(row['student_head'])}",
                "reference_student_head": ref_lookup.get(ref_key, -1),
            })

    df = pd.DataFrame(out_rows)
    out_dir = resolve_path(PS_ROOT / f"bootstrap{bootstrap_idx}_seed{seed}")
    out_dir.mkdir(parents=True, exist_ok=True)
    pair_path = out_dir / "pair_manifest.parquet"
    df.to_parquet(pair_path, index=False)
    np.savez(out_dir / "score_matrices.npz", **score_dump)
    meta = {
        "bootstrap": bootstrap_idx, "seed": seed, "n_shards": len(shard_ids),
        "shard_ids": shard_ids, "max_samples_per_shard": max_samples_per_shard,
        "batch_size": batch_size, "precision": precision,
        "metrics": list(DEFAULT_METRICS), "pairing": "bipartite",
        "student": "vanilla rinalmo-micro (no adapters)",
        "teacher_source": str(FULL_CACHE / "buckets"),
    }
    (out_dir / "pair_search_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[pairing-stability] bootstrap {bootstrap_idx}: wrote {len(df)} pairs -> {pair_path}", flush=True)
    return pair_path


def reduce(seed: int = 17, n_bootstrap: int = 5) -> None:
    reference = load_manifest(resolve_path(MANIFEST_DIR / "head_pair_manifest.parquet"))
    ref_map = dict(zip(
        reference["teacher_model"] + ":L" + reference["teacher_layer"].astype(str)
        + ":H" + reference["teacher_head"].astype(str) + ":L" + reference["route_depth"].astype(str),
        reference["student_head"],
    ))

    long_rows: list[dict] = []
    for b in range(n_bootstrap):
        path = resolve_path(PS_ROOT / f"bootstrap{b}_seed{seed}" / "pair_manifest.parquet")
        if not path.exists():
            print(f"[reduce] missing {path}")
            continue
        df = pd.read_parquet(path)
        for _, row in df.iterrows():
            tkey = (f"{row['teacher_model']}:L{row['teacher_layer']}:H{row['teacher_head']}"
                    f":L{row['route_depth']}")
            long_rows.append({
                "bootstrap": b,
                "route_depth": row["route_depth"],
                "teacher_key": tkey,
                "student_head": int(row["student_head"]),
                "matching_score": row["matching_score"],
                "reference_student_head": int(ref_map.get(tkey, -1)),
            })
    long_df = pd.DataFrame(long_rows)
    if long_df.empty:
        raise SystemExit("[reduce] no bootstrap pairings found")

    # Per-teacher-head stability stats.
    head_rows: list[dict] = []
    for tkey, grp in long_df.groupby("teacher_key"):
        counts = grp["student_head"].value_counts()
        mode_head = int(counts.index[0])
        ref_head = int(grp["reference_student_head"].iloc[0])
        head_rows.append({
            "teacher_key": tkey,
            "route_depth": int(grp["route_depth"].iloc[0]),
            "n_bootstraps": len(grp),
            "mode_student_head": mode_head,
            "mode_frequency": float(counts.iloc[0] / len(grp)),
            "n_distinct_student_heads": int(counts.shape[0]),
            "reference_student_head": ref_head,
            "matches_reference_rate": float((grp["student_head"] == ref_head).mean()),
            "mean_matching_score": float(grp["matching_score"].mean()),
        })
    head_df = pd.DataFrame(head_rows).sort_values(["route_depth", "teacher_key"])

    summary = {
        "n_bootstraps": int(long_df["bootstrap"].nunique()),
        "n_teacher_heads": int(head_df.shape[0]),
        "mean_mode_frequency": float(head_df["mode_frequency"].mean()),
        "frac_heads_always_same_pair": float((head_df["n_distinct_student_heads"] == 1).mean()),
        "mean_matches_reference_rate": float(head_df["matches_reference_rate"].mean()),
        "exact_full_match_rate_vs_reference": float(
            long_df.groupby("bootstrap")
            .apply(lambda g: (g["student_head"] == g["reference_student_head"]).mean())
            .mean()
        ),
    }

    root = resolve_path(PS_ROOT)
    long_df.to_parquet(root / "pairing_stability_summary.parquet", index=False)
    head_df.to_parquet(root / "pairing_stability_per_head.parquet", index=False)

    lines = ["# Pairing stability (5 bootstrap subsets x 16 shards)", ""]
    for k, v in summary.items():
        lines.append(f"- {k}: {v:.4f}" if isinstance(v, float) else f"- {k}: {v}")
    lines += ["", "## Per-head stability", "",
              "| teacher head | route | mode student head | mode freq | #distinct | ref head | match ref rate |",
              "|---|---|---|---|---|---|---|"]
    for _, r in head_df.iterrows():
        lines.append(f"| {r['teacher_key']} | L{r['route_depth']} | H{r['mode_student_head']} | "
                     f"{r['mode_frequency']:.2f} | {r['n_distinct_student_heads']} | "
                     f"H{r['reference_student_head']} | {r['matches_reference_rate']:.2f} |")
    (root / "pairing_stability.md").write_text("\n".join(lines), encoding="utf-8")

    # Refresh the supplementary index parquet statuses.
    idx_path = resolve_path(ALIGNMENT_ROOT / "pairing_stability.parquet")
    if idx_path.exists():
        idx = pd.read_parquet(idx_path)
        done = set(long_df["bootstrap"].unique())
        idx["status"] = [
            "pairing_computed" if int(str(v).replace("bootstrap", "")) in done else "missing"
            for v in idx["variant"]
        ]
        idx.to_parquet(idx_path, index=False)

    print(json.dumps(summary, indent=2), flush=True)
    print(f"[reduce] wrote {root/'pairing_stability_summary.parquet'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bootstrap", type=int, default=None)
    parser.add_argument("--reduce", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--n-bootstrap", type=int, default=5)
    parser.add_argument("--max-samples-per-shard", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--precision", default="bf16")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.reduce:
        reduce(seed=args.seed, n_bootstrap=args.n_bootstrap)
    elif args.bootstrap is not None:
        run_bootstrap(args.bootstrap, args.seed, args.max_samples_per_shard,
                      args.batch_size, args.precision, args.device)
    else:
        parser.error("specify --bootstrap N or --reduce")


if __name__ == "__main__":
    main()
