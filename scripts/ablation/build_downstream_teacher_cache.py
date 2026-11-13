#!/usr/bin/env python
"""Extract per-route teacher-head caches from the Stage 4B full slim cache.

For every downstream task, this script creates a compact cache that keeps only
the teacher attention heads belonging to that task's route depth:

    sequence_classification -> L6
    token_regression        -> L8
    pairwise                -> L12

Output files follow the slim-cache ``shard_*.pt`` convention expected by
:class:`rna_head_alignment.probes.feature_extractor.TeacherCacheIndex`.

To avoid duplicating the large L6 cache for the four sequence-level tasks, a
single physical route cache is written under
``outputs/downstream_teacher_head_cache/.route_cache/L{depth}/`` and each task
directory contains symlinks to the relevant route files.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import load_file

from rna_head_alignment.head2head.head_manifest import ROUTE_DEPTHS
from rna_head_alignment.io.paths import load_yaml, resolve_path
from rna_head_alignment.probes.task_config import TASK_TO_ROUTE_DEPTH


SLIM_SCHEMA_VERSION = "downstream-teacher-cache-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build per-task downstream teacher-head caches from the full slim cache."
    )
    parser.add_argument("--config", required=True, help="Path to downstream_data.yaml.")
    parser.add_argument("--tasks", default=None, help="Comma-separated task subset.")
    parser.add_argument("--dry-run", action="store_true", help="Preview work without writing files.")
    parser.add_argument("--resume", action="store_true", help="Skip existing route shard files.")
    return parser.parse_args()


def read_bucket_metadata(bucket_path: Path) -> dict[str, str]:
    with safe_open(str(bucket_path), framework="pt") as f:
        return f.metadata() or {}


def extract_route_bucket(
    bucket_path: Path,
    depth: int,
    seq_manifest: pd.DataFrame,
) -> dict[str, Any]:
    """Return a dictionary ready to be torch.saved for one route."""
    route_key = f"L{depth}"
    metadata = read_bucket_metadata(bucket_path)
    tensors = load_file(str(bucket_path), device="cpu")

    teacher = tensors[f"teacher_heads_{route_key}"]
    availability = tensors[f"head_availability_{route_key}"]
    lengths = tensors["lengths"]
    indices = tensors["sequence_manifest_indices"].to(torch.int64).numpy()

    head_keys = json.loads(metadata.get(f"route_heads_{route_key}", "[]"))

    seq_ids = seq_manifest["seq_id"].iloc[indices].tolist()
    sequences = seq_manifest["sequence"].iloc[indices].tolist()

    return {
        "schema_version": SLIM_SCHEMA_VERSION,
        "shard_id": bucket_path.stem,
        "seq_ids": [str(s) for s in seq_ids],
        "sequences": [str(s) for s in sequences],
        "lengths": lengths.to(torch.int32),
        f"teacher_heads_{route_key}": teacher.to(torch.float16),
        f"teacher_head_ids_{route_key}": [str(k) for k in head_keys],
        "head_availability": availability.to(torch.bool),
        "metadata": {
            "source_bucket": str(bucket_path),
            "route_depth": depth,
            "route_key": route_key,
            "n_heads": int(teacher.shape[1]),
            "bucket_size": int(teacher.shape[2]),
            "n_seqs": int(teacher.shape[0]),
            "built_at": datetime.now(timezone.utc).isoformat(),
        },
    }


def build_route_cache(
    full_cache_root: Path,
    route_depth: int,
    seq_manifest: pd.DataFrame,
    route_cache_dir: Path,
    dry_run: bool,
    resume: bool,
) -> list[dict[str, Any]]:
    """Iterate full-cache buckets and write one ``.pt`` file per bucket per route."""
    route_key = f"L{route_depth}"
    manifest_path = full_cache_root / "manifests" / "slim_cache_manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Slim cache manifest not found: {manifest_path}")
    bucket_manifest = pd.read_parquet(manifest_path)

    if not dry_run:
        route_cache_dir.mkdir(parents=True, exist_ok=True)

    shard_reports: list[dict[str, Any]] = []
    for _, row in bucket_manifest.iterrows():
        bucket_path = Path(str(row["file_path"]))
        if not bucket_path.is_absolute():
            candidate = full_cache_root / bucket_path
            bucket_path = candidate if candidate.exists() else Path.cwd() / bucket_path
        if not bucket_path.exists():
            print(f"[teacher-cache] WARNING missing bucket {bucket_path}", flush=True)
            continue

        out_path = route_cache_dir / f"{bucket_path.stem}.pt"
        if resume and out_path.exists() and out_path.stat().st_size > 0:
            shard_reports.append({
                "shard_id": bucket_path.stem,
                "route": route_key,
                "out_path": str(out_path),
                "status": "skipped_existing",
            })
            continue

        if dry_run:
            shard_reports.append({
                "shard_id": bucket_path.stem,
                "route": route_key,
                "out_path": str(out_path),
                "status": "dry_run",
            })
            continue

        payload = extract_route_bucket(bucket_path, route_depth, seq_manifest)
        torch.save(payload, out_path)
        shard_reports.append({
            "shard_id": bucket_path.stem,
            "route": route_key,
            "out_path": str(out_path),
            "status": "ok",
            "n_seqs": int(payload["metadata"]["n_seqs"]),
            "n_heads": int(payload["metadata"]["n_heads"]),
            "bucket_size": int(payload["metadata"]["bucket_size"]),
        })
        del payload
        gc.collect()
    return shard_reports


def link_route_to_task(
    route_cache_dir: Path,
    task_cache_dir: Path,
    dry_run: bool,
) -> list[dict[str, Any]]:
    """Symlink route shard files into a per-task cache directory."""
    if not dry_run:
        task_cache_dir.mkdir(parents=True, exist_ok=True)
    links: list[dict[str, Any]] = []
    for shard_file in sorted(route_cache_dir.glob("shard_*.pt")):
        link_path = task_cache_dir / shard_file.name
        if dry_run:
            links.append({"source": str(shard_file), "link": str(link_path), "status": "dry_run"})
            continue
        if link_path.is_symlink() or link_path.exists():
            link_path.unlink(missing_ok=True)
        try:
            os.symlink(shard_file.resolve(), link_path)
            links.append({"source": str(shard_file), "link": str(link_path), "status": "symlink"})
        except OSError as exc:
            # Fallback to a copy if the filesystem does not support symlinks.
            import shutil
            shutil.copy2(shard_file, link_path)
            links.append({"source": str(shard_file), "link": str(link_path), "status": "copy", "error": str(exc)})
    return links


def main() -> int:
    args = parse_args()
    cfg = load_yaml(args.config)

    full_cache_root = resolve_path(cfg.get("full_slim_cache_root"))
    if not full_cache_root.exists():
        raise FileNotFoundError(f"Full slim cache root not found: {full_cache_root}")

    seq_manifest_path = resolve_path(cfg.get("sequence_manifest"))
    if not seq_manifest_path.exists():
        raise FileNotFoundError(f"Sequence manifest not found: {seq_manifest_path}")
    seq_manifest = pd.read_parquet(seq_manifest_path)
    print(f"[teacher-cache] loaded sequence manifest with {len(seq_manifest):,} rows", flush=True)

    out_root = resolve_path(cfg.get("downstream_teacher_cache_root", "outputs/downstream_teacher_head_cache"))

    task_cfgs = cfg.get("tasks", {})
    requested = args.tasks.split(",") if args.tasks else list(task_cfgs.keys())
    requested = [t.strip() for t in requested if t.strip()]
    unknown = [t for t in requested if t not in TASK_TO_ROUTE_DEPTH]
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}. Supported: {list(TASK_TO_ROUTE_DEPTH)}")

    # Build each required route depth exactly once.
    route_depths_needed = sorted({TASK_TO_ROUTE_DEPTH[t] for t in requested})
    route_cache_dirs: dict[int, Path] = {}
    for depth in route_depths_needed:
        route_dir = out_root / ".route_cache" / f"L{depth}"
        route_cache_dirs[depth] = route_dir
        print(f"[teacher-cache] building route L{depth} cache ...", flush=True)
        reports = build_route_cache(
            full_cache_root,
            depth,
            seq_manifest,
            route_dir,
            dry_run=args.dry_run,
            resume=args.resume,
        )
        ok = sum(1 for r in reports if r.get("status") == "ok")
        skipped = sum(1 for r in reports if r.get("status") == "skipped_existing")
        print(
            f"[teacher-cache] route L{depth}: {ok} written, {skipped} skipped, "
            f"{len(reports)} total shards",
            flush=True,
        )

    # Link each task to the appropriate route cache.
    for task in requested:
        depth = TASK_TO_ROUTE_DEPTH[task]
        task_dir = out_root / task
        print(f"[teacher-cache] linking {task} -> L{depth} ...", flush=True)
        links = link_route_to_task(route_cache_dirs[depth], task_dir, dry_run=args.dry_run)
        print(f"[teacher-cache] {task}: {len(links)} shard links created", flush=True)

    # Write a small route manifest for bookkeeping.
    if not args.dry_run:
        manifest_rows = []
        for task in requested:
            depth = TASK_TO_ROUTE_DEPTH[task]
            route_dir = route_cache_dirs[depth]
            n_shards = len(list(route_dir.glob("shard_*.pt")))
            size_bytes = sum(p.stat().st_size for p in route_dir.glob("shard_*.pt"))
            manifest_rows.append({
                "task": task,
                "route_depth": depth,
                "route_cache_dir": str(route_dir),
                "task_cache_dir": str(out_root / task),
                "n_shards": n_shards,
                "size_bytes": size_bytes,
                "size_gb": round(size_bytes / 1e9, 3),
            })
        manifest_df = pd.DataFrame(manifest_rows)
        manifest_path = out_root / "downstream_teacher_cache_manifest.parquet"
        manifest_df.to_parquet(manifest_path, index=False)
        print(f"[teacher-cache] wrote manifest {manifest_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
