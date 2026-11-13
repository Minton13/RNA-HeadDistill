#!/usr/bin/env python
"""Cache exact teacher attention heads for downstream sequences (P5 fill).

One Slurm array task handles one (route, model, shard range) triple from
``outputs/downstream_teacher_fill/p5_cache_jobs.csv``.  For every shard it
extracts only this model's selected heads at nucleotide resolution, pads the
maps to the shard bucket, and writes a partial file:

    outputs/downstream_teacher_fill/partials/L{depth}/model={model}/shard_{i:05d}.pt

Partials are later merged by ``merge_downstream_teacher_shards.py`` into the
``downstream-teacher-cache-v1`` schema consumed by
:class:`rna_head_alignment.probes.feature_extractor.TeacherCacheIndex`.

Canonical head order per route matches the existing route cache metadata
(``teacher_head_ids_L*``), see ``ROUTE_HEADS`` below.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import sys
from pathlib import Path
from typing import Any

import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rna_head_alignment.io.paths import get_device
from rna_head_alignment.models.attention_extractors import extract_attention_bundle_batch
from rna_head_alignment.models.loaders import load_frozen_model
from rna_head_alignment.models.registry import get_model_spec

FILL_DIR = _PROJECT_ROOT / "outputs" / "downstream_teacher_fill"
SHARD_SIZE = 2048
BUCKETS = (64, 128, 256, 512)
MAX_LEN = 512

# Canonical per-route heads in the exact order of the existing route cache
# ``teacher_head_ids_L{depth}``: (model, layer, head).
ROUTE_HEADS: dict[int, list[tuple[str, int, int]]] = {
    6: [
        ("nucleotide_transformer_v2_500m", 4, 3),
        ("plantrna_fm", 8, 18),
        ("rinalmo_mega", 18, 5),
        ("rnafm", 4, 3),
        ("splicebert", 5, 15),
    ],
    8: [
        ("plantrna_fm", 9, 9),
        ("plantrna_fm", 9, 7),
        ("plantrna_fm", 10, 7),
        ("splicebert", 2, 7),
        ("utrlm_mrl", 2, 10),
    ],
    12: [
        ("moderngena_base", 12, 11),
        ("mrnafm", 4, 9),
        ("plantrna_fm", 8, 2),
        ("rinalmo_giga", 31, 1),
        ("rinalmo_giga", 31, 13),
    ],
}

ROUTE_TASK_NAME: dict[int, str] = {
    6: "manual_L6_sequence_bank",
    8: "manual_L8_chemical_bank",
    12: "manual_L12_structure_bank",
}


def route_head_ids(depth: int) -> list[str]:
    task = ROUTE_TASK_NAME[depth]
    return [
        f"{model}:L{layer}:H{head}:L{depth}:{task}"
        for model, layer, head in ROUTE_HEADS[depth]
    ]


def bucket_for_length(length: int) -> int:
    for b in BUCKETS:
        if length <= b:
            return b
    raise ValueError(f"sequence length {length} exceeds max bucket {BUCKETS[-1]}")


def read_jsonl_slice(path: Path, start: int, end: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if i < start:
                continue
            if i >= end:
                break
            rows.append(json.loads(line))
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--route-depth", type=int, required=True, choices=(6, 8, 12))
    p.add_argument("--model", required=True)
    p.add_argument("--shard-start", type=int, required=True)
    p.add_argument("--shard-end", type=int, required=True, help="exclusive")
    p.add_argument("--models-yaml", default="configs/models.yaml")
    p.add_argument("--device", default="auto")
    p.add_argument("--max-tokens-per-batch", type=int, default=4096)
    return p.parse_args()


def _malloc_trim() -> None:
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def main() -> int:
    args = parse_args()
    depth = args.route_depth
    heads_all = ROUTE_HEADS[depth]
    my_heads = [
        {"layer": layer, "head": head, "model": args.model}
        for (model, layer, head) in heads_all
        if model == args.model
    ]
    if not my_heads:
        raise ValueError(f"model {args.model} has no heads in route L{depth}")
    head_slots = [i for i, h in enumerate(heads_all) if h[0] == args.model]

    sequences_path = FILL_DIR / f"sequences_L{depth}.jsonl"
    out_dir = FILL_DIR / "partials" / f"L{depth}" / f"model={args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.device)
    spec = get_model_spec(args.models_yaml, args.model)
    print(f"[p5-cache] loading {args.model} for route L{depth} on {device}", flush=True)
    tokenizer, model = load_frozen_model(spec, device)

    for shard_idx in range(args.shard_start, args.shard_end):
        out_path = out_dir / f"shard_{shard_idx:05d}.pt"
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"[p5-cache] {args.model} shard {shard_idx:05d} exists, skip", flush=True)
            continue
        rows = read_jsonl_slice(
            sequences_path, shard_idx * SHARD_SIZE, (shard_idx + 1) * SHARD_SIZE
        )
        if not rows:
            continue
        bucket = bucket_for_length(min(max(r["length"] for r in rows), MAX_LEN))
        n = len(rows)
        k = len(my_heads)
        attn_out = torch.zeros((n, k, bucket, bucket), dtype=torch.float16)
        avail = torch.zeros((n, k), dtype=torch.bool)

        bundles = extract_attention_bundle_batch(
            spec,
            tokenizer,
            model,
            rows,
            max_len=MAX_LEN,
            device=device,
            max_tokens_per_batch=args.max_tokens_per_batch,
            output_hidden_states=False,
            output_attentions=True,
            selected_heads=my_heads,
        )
        row_pos = {r["seq_id"]: i for i, r in enumerate(rows)}
        n_ok = 0
        for bundle in bundles:
            i = row_pos[bundle.seq_id]
            if bundle.attentions is None:
                print(f"[p5-cache] WARN no attention {args.model} {bundle.seq_id}", flush=True)
                continue
            a = bundle.attentions  # [k, L, L] nucleotide resolution
            L = min(int(a.shape[-1]), bucket)
            attn_out[i, :, :L, :L] = a[:, :L, :L].to(torch.float16).cpu()
            avail[i] = True
            n_ok += 1
        payload = {
            "model": args.model,
            "route_depth": depth,
            "shard_idx": shard_idx,
            "bucket": bucket,
            "seq_ids": [r["seq_id"] for r in rows],
            "head_slots": head_slots,
            "head_keys": [route_head_ids(depth)[s] for s in head_slots],
            "attention": attn_out,
            "availability": avail,
            "n_ok": n_ok,
        }
        torch.save(payload, out_path)
        print(
            f"[p5-cache] {args.model} shard {shard_idx:05d}: {n_ok}/{n} ok, "
            f"bucket={bucket}, -> {out_path}",
            flush=True,
        )
        del payload, attn_out, avail, bundles
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        _malloc_trim()

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _malloc_trim()
    print(f"[p5-cache] {args.model} L{depth} shards {args.shard_start}-{args.shard_end} done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
