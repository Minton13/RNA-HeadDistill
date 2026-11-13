"""Stage 4B bucketed full slim teacher attention cache.

Re-packs the 10-model teacher attention cache into bucketed safetensors shards
that keep only the 15 target teacher heads.  Sequences are grouped by length
into fixed-size buckets so the student dataloader can batch efficiently.  The
output contains no full sequences, hidden states, valid masks, or non-target
attention heads.

Schema of each bucket file ``shard_{idx:05d}_bucket_{size}.safetensors``:

- ``teacher_heads_L6`` / ``_L8`` / ``_L12``: fp16 [B, T_route, size, size]
- ``head_availability_L6`` / ``_L8`` / ``_L12``: bool [B, T_route]
- ``lengths``: int32 [B]
- ``sequence_manifest_indices``: int64 [B]

Metadata (safetensors header):

- ``schema_version``, ``shard_id``, ``bucket_size``, ``n_seqs``
- ``route_heads_L6`` / ``_L8`` / ``_L12``: JSON list of head_keys aligned with
  dim 1 of the matching tensor
- ``source_shards``: JSON dict ``{teacher_model: source_path}``
- ``source_Lmax``, ``built_at``
"""

from __future__ import annotations

import gc
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

from .head_manifest import ROUTE_DEPTHS, load_manifest

SLIM_SCHEMA_VERSION = "stage4b-bucketed-v1"

DEFAULT_BUCKETS = (64, 128, 256, 512, 1024)

MAX_ABS_ERROR_TOL = 1e-3
MIN_COSINE_TOL = 0.9999


def _load_target_manifest(target_manifest: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(target_manifest, (str, Path)):
        return load_manifest(target_manifest)
    return target_manifest


def _load_sequence_manifest(sequence_manifest: str | Path | pd.DataFrame) -> pd.DataFrame:
    if isinstance(sequence_manifest, (str, Path)):
        return pd.read_parquet(sequence_manifest)
    return sequence_manifest


def _default_buckets(max_position: int) -> tuple[int, ...]:
    """Return fixed power-of-two buckets up to ``max_position``."""
    buckets = [b for b in DEFAULT_BUCKETS if b <= max_position]
    if not buckets or buckets[-1] < max_position:
        buckets.append(max_position)
    return tuple(sorted(set(buckets)))


def _discover_models(cache_root: str | Path) -> list[str]:
    cache_root = Path(cache_root)
    return sorted(
        p.name.split("=", 1)[1]
        for p in cache_root.glob("model=*")
        if p.is_dir()
    )


def _shard_path(cache_root: Path, model: str, shard_idx: int) -> Path:
    return cache_root / f"model={model}" / f"shard_{shard_idx:05d}.pt"


def _load_source_shard(path: Path) -> dict[str, Any]:
    return torch.load(path, map_location="cpu", weights_only=False)


def _bucket_for_length(
    length: int,
    buckets: tuple[int, ...],
    max_position: int,
) -> tuple[int | None, bool]:
    """Return (bucket_size, is_holdout)."""
    if length > max_position:
        return None, True
    for b in buckets:
        if length <= b:
            return b, False
    # This should not happen if buckets cover max_position.
    return buckets[-1], False


def _pad_route_bucket(
    route_attn: torch.Tensor,
    lengths: list[int],
    bucket_size: int,
) -> torch.Tensor:
    """Crop/pad per-sequence attention to a fixed bucket size.

    Args:
        route_attn: [B, T, Lmax, Lmax] aligned tensor for one route.
        lengths: per-sequence true lengths (``<= bucket_size``).
        bucket_size: target spatial size.

    Returns:
        fp16 tensor [B, T, bucket_size, bucket_size].
    """
    B, T, Lmax, _ = route_attn.shape
    spatial = min(Lmax, bucket_size)
    # First crop or pad to the bucket spatial size.
    out = torch.zeros(B, T, bucket_size, bucket_size, dtype=torch.float16)
    out[:, :, :spatial, :spatial] = route_attn[:, :, :spatial, :spatial].to(torch.float16)
    # Zero out positions beyond each sequence's true length.
    lengths_t = torch.as_tensor(lengths, dtype=torch.int64).view(B, 1, 1, 1)
    idx = torch.arange(bucket_size, dtype=torch.int64).view(1, 1, bucket_size, 1)
    row_mask = idx < lengths_t
    col_mask = idx.transpose(-2, -1) < lengths_t
    mask = row_mask & col_mask
    return out * mask


def _validate_fp16_equivalence(
    source_attn: torch.Tensor,
    gathered: torch.Tensor,
    ref_rows: torch.Tensor,
    available: torch.Tensor,
    target_channels: torch.Tensor,
    ref_lengths: list[int],
    sample_count: int = 5,
) -> dict[str, float]:
    """Compare gathered slices against the original source attention.

    Both ``source_attn`` and ``gathered`` are fp16.  The function reports the
    round-trip fp32->fp16 error floor as well as the actual max/mean absolute
    error and cosine similarity on the real ``[:L, :L]`` region.
    """
    avail_idx = torch.where(available)[0]
    n_avail = int(available.sum().item())
    if n_avail == 0 or len(target_channels) == 0:
        return {
            "n_samples": 0,
            "max_abs_error": 0.0,
            "mean_abs_error": 0.0,
            "min_cosine_similarity": 1.0,
            "fp32_roundtrip_max_error": 0.0,
        }

    n_samples = min(sample_count, n_avail)
    sample_pos = avail_idx[:n_samples]

    max_err = 0.0
    mean_err_sum = 0.0
    min_cos = 1.0
    n_metrics = 0

    for pos in sample_pos:
        pos = int(pos.item())
        src_row = int(ref_rows[pos].item())
        L = int(ref_lengths[pos])
        if L <= 0:
            continue
        for hi, ch in enumerate(target_channels):
            ch = int(ch.item())
            orig = source_attn[src_row, ch, :L, :L].to(torch.float32)
            stored = gathered[pos, hi, :L, :L].to(torch.float32)
            diff = (orig - stored).abs()
            max_err = max(max_err, float(diff.max().item()))
            mean_err_sum += float(diff.mean().item())
            # Cosine similarity; guard against zero-norm slices.
            orig_v = orig.reshape(1, -1)
            stored_v = stored.reshape(1, -1)
            norm = orig_v.norm(2) * stored_v.norm(2)
            if norm.item() > 0:
                cos = float(F.cosine_similarity(orig_v, stored_v).item())
            else:
                cos = 1.0
            min_cos = min(min_cos, cos)
            n_metrics += 1

    # Round-trip fp32->fp16 error floor for reference.
    sample_src_rows = [int(ref_rows[p].item()) for p in sample_pos if ref_rows[p].item() >= 0]
    if sample_src_rows:
        sample_src = source_attn[sample_src_rows][:, target_channels]
        fp32 = sample_src.to(torch.float32)
        roundtrip = fp32.to(torch.float16).to(torch.float32)
        rt_err = float((fp32 - roundtrip).abs().max().item())
    else:
        rt_err = 0.0

    return {
        "n_samples": n_metrics,
        "max_abs_error": max_err,
        "mean_abs_error": mean_err_sum / max(n_metrics, 1),
        "min_cosine_similarity": min_cos,
        "fp32_roundtrip_max_error": rt_err,
    }


def _check_validation(metrics: dict[str, float]) -> None:
    """Raise if gathered fp16 attention is not equivalent to source."""
    max_err = metrics.get("max_abs_error", 0.0)
    min_cos = metrics.get("min_cosine_similarity", 1.0)
    rt_err = metrics.get("fp32_roundtrip_max_error", 0.0)
    if max_err > MAX_ABS_ERROR_TOL:
        raise ValueError(
            f"fp16 equivalence check failed: max_abs_error={max_err:.6g} "
            f"(tol={MAX_ABS_ERROR_TOL})"
        )
    if min_cos < MIN_COSINE_TOL:
        raise ValueError(
            f"fp16 equivalence check failed: min_cosine_similarity={min_cos:.6f} "
            f"(tol={MIN_COSINE_TOL})"
        )
    # The stored tensor should match the source at least as well as a fp32
    # round-trip would allow.
    if max_err > rt_err + 1e-5:
        raise ValueError(
            f"fp16 equivalence check failed: max_abs_error={max_err:.6g} exceeds "
            f"fp32 round-trip floor={rt_err:.6g}"
        )


def _compute_file_checksum(path: Path, algorithm: str = "md5") -> str:
    """Return a hex checksum for a file."""
    if algorithm != "md5":
        raise ValueError(f"unsupported checksum algorithm: {algorithm}")
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _serialize_for_parquet(value: Any) -> Any:
    """Recursively coerce nested structures to JSON strings for parquet."""
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False, default=str)
    if isinstance(value, np.ndarray):
        return json.dumps(value.tolist(), ensure_ascii=False, default=str)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return json.dumps(value, ensure_ascii=False, default=str)


def _write_partial_manifest(report: dict[str, Any], partial_dir: str | Path) -> Path:
    """Write a one-row parquet report for a single shard."""
    partial_dir = Path(partial_dir)
    partial_dir.mkdir(parents=True, exist_ok=True)
    row = {k: _serialize_for_parquet(v) for k, v in report.items()}
    path = partial_dir / f"shard_{report['shard_id']}_partial.parquet"
    pd.DataFrame([row]).to_parquet(path, index=False)
    return path


def build_bucketed_slim_shard(
    shard_idx: int,
    cache_root: str | Path,
    target_manifest: str | Path | pd.DataFrame,
    sequence_manifest: str | Path | pd.DataFrame,
    out_root: str | Path,
    buckets: tuple[int, ...] | list[int] | None = None,
    max_position: int = 1024,
    strict: bool = True,
    resume: bool = True,
    validation_sample_count: int = 5,
    checksum_algorithm: str = "md5",
) -> dict[str, Any]:
    """Build one bucketed slim shard for the full 1033-shard Stage 4B cache.

    The function loads each original teacher-model shard exactly once, gathers
    all target heads for that model, aligns everything to the reference order
    in ``sequence_manifest``, and writes per-bucket safetensors files.

    Sequences longer than ``max_position`` are counted as holdout and are not
    written.  Per-head availability is stored compactly as a bool tensor for
    each route so partially-missing model shards (e.g. ``rinalmo_mega`` shard
    936) do not force the whole shard to be dropped.

    Returns a per-shard report dict.  On validation failures or data problems
    the function raises when ``strict=True`` and returns ``ok=False`` when
    ``strict=False``.
    """
    start_time = time.time()
    cache_root = Path(cache_root)
    out_root = Path(out_root)
    buckets = (
        _default_buckets(max_position) if buckets is None else tuple(sorted(set(int(b) for b in buckets)))
    )
    if buckets[-1] < max_position:
        buckets = tuple(sorted(set(buckets + (max_position,))))

    manifest = _load_target_manifest(target_manifest)
    seq_manifest = _load_sequence_manifest(sequence_manifest)
    shard_id = f"{shard_idx:05d}"

    report: dict[str, Any] = {
        "shard_idx": shard_idx,
        "shard_id": shard_id,
        "ok": False,
        "error": None,
        "n_seqs": 0,
        "n_trainable": 0,
        "n_holdout": 0,
        "n_available_any_head": 0,
        "source_bytes": 0,
        "out_bytes": 0,
        "models_used": [],
        "missing_models": [],
        "skipped_heads": [],
        "bucket_files": [],
        "bucket_sizes": [],
        "bucket_seq_counts": [],
        "checksums": {},
        "head_coverage": {},
        "route_coverage": {},
        "validation_metrics": {},
        "warnings": [],
        "build_time_sec": 0.0,
    }

    try:
        # ---- Reference order for this shard from the sequence manifest. ------
        shard_idx_str = shard_id

        def _shard_matches(ids: Any) -> bool:
            if ids is None:
                return False
            if isinstance(ids, np.ndarray):
                ids = ids.tolist()
            return shard_idx_str in {str(x) for x in ids}

        ref_mask = seq_manifest["original_shard_id"].apply(_shard_matches)
        ref_df = seq_manifest[ref_mask].copy()
        if len(ref_df) == 0:
            raise ValueError(f"no sequences in sequence_manifest for shard {shard_idx}")

        # Preserve sequence_manifest order.
        ref_order = ref_df.index.to_numpy()
        ref_seq_ids = ref_df["seq_id"].tolist()
        ref_lengths = ref_df["length"].astype(int).tolist()
        report["n_seqs"] = len(ref_seq_ids)

        # ---- Trainability / holdout / bucket assignment. ---------------------
        bucket_assignments: list[int | None] = []
        trainable_mask: list[bool] = []
        for L in ref_lengths:
            b, holdout = _bucket_for_length(L, buckets, max_position)
            bucket_assignments.append(b)
            trainable_mask.append(not holdout)

        report["n_trainable"] = int(sum(trainable_mask))
        report["n_holdout"] = len(ref_lengths) - report["n_trainable"]

        # ---- Load source shards, grouped by teacher model. -------------------
        manifest_by_model = {
            m: grp for m, grp in manifest.groupby("teacher_model", sort=False)
        }
        models = sorted(manifest_by_model.keys())

        model_data: dict[str, dict[str, Any]] = {}
        source_lmax: int | None = None
        for model in models:
            path = _shard_path(cache_root, model, shard_idx)
            if not path.exists():
                msg = f"missing source shard: {path}"
                if strict:
                    raise FileNotFoundError(msg)
                report["warnings"].append(msg)
                report["missing_models"].append(model)
                continue
            shard = _load_source_shard(path)
            report["source_bytes"] += path.stat().st_size
            seq_ids = list(shard["seq_ids"])
            head_keys = list(shard["head_keys"])
            attention = shard["attention"]
            Lmax = int(attention.shape[-1])
            if source_lmax is None:
                source_lmax = Lmax
            elif Lmax != source_lmax:
                raise ValueError(
                    f"Lmax mismatch in shard {shard_idx}: {model} has {Lmax}, "
                    f"expected {source_lmax}"
                )
            source_lengths = shard.get("lengths")
            if isinstance(source_lengths, torch.Tensor):
                source_lengths = source_lengths.to(torch.int64).tolist()
            else:
                source_lengths = [int(x) for x in source_lengths] if source_lengths else []

            model_data[model] = {
                "path": path,
                "shard": shard,
                "attention": attention,
                "seq_id_to_row": {s: i for i, s in enumerate(seq_ids)},
                "channel_map": {k: i for i, k in enumerate(head_keys)},
                "source_lengths": source_lengths,
            }
            report["models_used"].append(model)

        if not model_data:
            raise RuntimeError(f"no source model shards loaded for shard {shard_idx}")

        # ---- Gather target heads per model, validate, track availability. ----
        route_slices: dict[int, list[torch.Tensor]] = {d: [] for d in ROUTE_DEPTHS}
        route_avails: dict[int, list[torch.Tensor]] = {d: [] for d in ROUTE_DEPTHS}
        route_head_keys: dict[int, list[str]] = {d: [] for d in ROUTE_DEPTHS}
        channel_provenance: list[dict[str, Any]] = []

        for model, data in model_data.items():
            rows = manifest_by_model[model]
            attention = data["attention"]
            seq_id_to_row = data["seq_id_to_row"]
            channel_map = data["channel_map"]
            source_lengths = data["source_lengths"]

            ref_rows = torch.tensor(
                [seq_id_to_row.get(s, -1) for s in ref_seq_ids],
                dtype=torch.long,
            )
            available = ref_rows >= 0

            # Validate source lengths against sequence manifest for available seqs.
            if source_lengths:
                for pos, (ok, src_row, L) in enumerate(zip(available, ref_rows, ref_lengths)):
                    if ok and 0 <= src_row.item() < len(source_lengths):
                        src_L = int(source_lengths[src_row.item()])
                        if src_L != L:
                            msg = (
                                f"{model} shard {shard_id}: seq_id {ref_seq_ids[pos]} "
                                f"source length {src_L} != manifest length {L}"
                            )
                            if strict:
                                raise ValueError(msg)
                            report["warnings"].append(msg)

            target_channels: list[int] = []
            route_indices: dict[int, list[int]] = {d: [] for d in ROUTE_DEPTHS}
            for _, row in rows.iterrows():
                head_key = row["head_key"]
                depth = int(row["route_depth"])
                channel = channel_map.get(head_key)
                if channel is None:
                    msg = (
                        f"head_key {head_key!r} not in {data['path']} "
                        f"head_keys {list(channel_map.keys())}"
                    )
                    if strict:
                        raise KeyError(msg)
                    report["warnings"].append(msg)
                    report["skipped_heads"].append(head_key)
                    continue
                if int(row["cache_channel"]) != channel:
                    msg = (
                        f"cache_channel mismatch for {head_key}: manifest "
                        f"{int(row['cache_channel'])} vs shard {channel}"
                    )
                    if strict:
                        raise ValueError(msg)
                    report["warnings"].append(msg)

                idx = len(target_channels)
                target_channels.append(channel)
                route_indices[depth].append(idx)
                route_head_keys[depth].append(head_key)
                channel_provenance.append(
                    {
                        "head_key": head_key,
                        "route_depth": depth,
                        "model": model,
                        "channel": channel,
                        "source_shard": str(data["path"]),
                    }
                )

            if not target_channels:
                continue

            target_channels_t = torch.tensor(target_channels, dtype=torch.long)
            T_target = len(target_channels)
            gathered = torch.zeros(
                len(ref_seq_ids), T_target, source_lmax, source_lmax, dtype=attention.dtype
            )
            if available.any():
                gathered[available] = attention[ref_rows[available]][:, target_channels_t]

            # Validate fp16 equivalence before padding.
            metrics = _validate_fp16_equivalence(
                attention,
                gathered,
                ref_rows,
                available,
                target_channels_t,
                ref_lengths,
                sample_count=validation_sample_count,
            )
            report["validation_metrics"][model] = metrics
            _check_validation(metrics)

            avail_repeat = available.unsqueeze(1).expand(-1, T_target)
            for depth, idxs in route_indices.items():
                if not idxs:
                    continue
                idxs_t = torch.tensor(idxs, dtype=torch.long)
                route_slices[depth].append(gathered[:, idxs_t])
                route_avails[depth].append(avail_repeat[:, idxs_t])

            # Release the heavy source attention tensor and shard dict.
            del attention, gathered, data["attention"], data["shard"]
            gc.collect()

        # ---- Concatenate routes across models. -------------------------------
        route_tensors: dict[int, torch.Tensor] = {}
        route_availability: dict[int, torch.Tensor] = {}
        for depth in ROUTE_DEPTHS:
            if route_slices[depth]:
                route_tensors[depth] = torch.cat(route_slices[depth], dim=1)
                route_availability[depth] = torch.cat(route_avails[depth], dim=1)
            else:
                route_tensors[depth] = torch.zeros(
                    len(ref_seq_ids), 0, source_lmax, source_lmax, dtype=torch.float16
                )
                route_availability[depth] = torch.zeros(
                    len(ref_seq_ids), 0, dtype=torch.bool
                )

        # Per-head / per-route availability counts (over the whole shard).
        any_available = torch.zeros(len(ref_seq_ids), dtype=torch.bool)
        for depth in ROUTE_DEPTHS:
            avail = route_availability[depth]
            if avail.numel() == 0:
                continue
            route_sum = int(avail.any(dim=1).sum().item())
            report["route_coverage"][f"L{depth}"] = route_sum
            any_available |= avail.any(dim=1)
            for i, head_key in enumerate(route_head_keys[depth]):
                report["head_coverage"][head_key] = int(avail[:, i].sum().item())
        report["n_available_any_head"] = int(any_available.sum().item())

        # ---- Write per-bucket safetensors. -----------------------------------
        buckets_dir = out_root / "buckets"
        buckets_dir.mkdir(parents=True, exist_ok=True)

        # Pre-compute trainable positions for each bucket.
        train_positions = [i for i, ok in enumerate(trainable_mask) if ok]
        bucket_of_position = {pos: bucket_assignments[pos] for pos in train_positions}

        for bucket_size in buckets:
            positions = [p for p, b in bucket_of_position.items() if b == bucket_size]
            if not positions:
                continue

            lengths = [ref_lengths[p] for p in positions]
            indices = [int(ref_order[p]) for p in positions]

            tensors: dict[str, torch.Tensor] = {}
            metadata: dict[str, str] = {
                "schema_version": SLIM_SCHEMA_VERSION,
                "shard_id": shard_id,
                "bucket_size": str(bucket_size),
                "n_seqs": str(len(positions)),
                "source_Lmax": str(source_lmax),
                "built_at": datetime.now(timezone.utc).isoformat(),
                "route_heads_L6": json.dumps(route_head_keys[6]),
                "route_heads_L8": json.dumps(route_head_keys[8]),
                "route_heads_L12": json.dumps(route_head_keys[12]),
                "source_shards": json.dumps(
                    {m: str(data["path"]) for m, data in model_data.items()}
                ),
                "channel_provenance": json.dumps(channel_provenance),
            }

            for depth in ROUTE_DEPTHS:
                route_attn = route_tensors[depth][positions]
                padded = _pad_route_bucket(route_attn, lengths, bucket_size)
                tensors[f"teacher_heads_L{depth}"] = padded
                tensors[f"head_availability_L{depth}"] = route_availability[depth][positions]

            tensors["lengths"] = torch.tensor(lengths, dtype=torch.int32)
            tensors["sequence_manifest_indices"] = torch.tensor(indices, dtype=torch.int64)

            bucket_name = f"shard_{shard_id}_bucket_{bucket_size}.safetensors"
            bucket_path = buckets_dir / bucket_name

            # Resume: skip if file exists and is non-empty.
            if resume and bucket_path.exists() and bucket_path.stat().st_size > 0:
                report["warnings"].append(f"skipped existing bucket file: {bucket_name}")
            else:
                save_file(tensors, bucket_path, metadata=metadata)

            checksum = _compute_file_checksum(bucket_path, checksum_algorithm)
            report["bucket_files"].append(bucket_name)
            report["bucket_sizes"].append(bucket_size)
            report["bucket_seq_counts"].append(len(positions))
            report["checksums"][bucket_name] = checksum
            report["out_bytes"] += bucket_path.stat().st_size

        report["ok"] = True
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"{type(exc).__name__}: {exc}"
        if strict:
            raise
    finally:
        report["build_time_sec"] = time.time() - start_time
        gc.collect()

    return report


def write_partial_manifest(report: dict[str, Any], partial_dir: str | Path) -> Path:
    """Thin wrapper exposed for scripts."""
    return _write_partial_manifest(report, partial_dir)
