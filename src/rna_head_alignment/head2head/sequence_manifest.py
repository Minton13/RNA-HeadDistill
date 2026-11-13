"""Stage 4B sequence manifest builder.

Builds a per-sequence manifest that joins the authoritative sequence source
(``outputs/task_samples/rinalmo_pretrain_10pct_len_sorted.jsonl``) with the
slim teacher attention cache shards.

The output parquet is written to
``outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/sequence_manifest.parquet``.
"""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from .slim_cache import ROUTE_DEPTHS, _load_shard


REQUIRED_SOURCE_FIELDS = {"seq_id", "sequence", "length", "source"}


SequenceManifestRow = dict[str, Any]


def _read_source_jsonl(path: Path) -> tuple[dict[str, dict[str, Any]], list[str], list[str]]:
    """Read the authoritative sequence source into a dict keyed by seq_id.

    Returns ``(records, order, warnings)``.  Warnings are emitted for duplicate
    seq_ids with mismatched fields.
    """
    records: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    warnings: list[str] = []

    with path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, 1):
            raw = raw.strip()
            if not raw:
                continue
            rec = json.loads(raw)
            missing = REQUIRED_SOURCE_FIELDS - set(rec.keys())
            if missing:
                raise ValueError(f"{path}:{line_no} missing fields {sorted(missing)}")

            seq_id = rec["seq_id"]
            entry = {
                "seq_id": seq_id,
                "sequence": str(rec["sequence"]),
                "length": int(rec["length"]),
                "source": rec["source"],
            }

            if seq_id in records:
                old = records[seq_id]
                if (
                    old["sequence"] != entry["sequence"]
                    or old["length"] != entry["length"]
                    or old["source"] != entry["source"]
                ):
                    warnings.append(
                        f"{path}:{line_no}: duplicate seq_id {seq_id!r} with mismatched fields"
                    )
            else:
                records[seq_id] = entry
                order.append(seq_id)

    return records, order, warnings


def _available_routes(shard: dict[str, Any]) -> list[int]:
    """Return route depths with at least one preserved teacher head in ``shard``."""
    routes: list[int] = []
    for depth in ROUTE_DEPTHS:
        if shard.get(f"teacher_head_ids_L{depth}"):
            routes.append(depth)
    return routes


def _lengths_list(shard: dict[str, Any]) -> list[int]:
    """Coerce the ``lengths`` tensor/list to a plain Python list of ints."""
    lengths = shard.get("lengths")
    if lengths is None:
        return []
    if isinstance(lengths, torch.Tensor):
        return lengths.tolist()
    return [int(x) for x in lengths]


def build_sequence_manifest(
    source_jsonl: str | Path,
    slim_cache_root: str | Path,
    target_manifest: str | Path,
) -> dict[str, Any]:
    """Build and write the Stage 4B sequence manifest parquet.

    Args:
        source_jsonl: Path to the authoritative JSONL with fields
            ``seq_id, sequence, length, source``.
        slim_cache_root: Directory containing ``shard_*.pt`` slim shards.
        target_manifest: Output parquet path.

    Returns:
        A report dict with ``n_source``, ``n_slim_seqs``, ``n_output_rows``,
        ``n_collisions``, ``collisions``, and ``warnings``.
    """
    source_jsonl = Path(source_jsonl)
    slim_cache_root = Path(slim_cache_root)
    target_manifest = Path(target_manifest)

    source_records, source_order, warnings = _read_source_jsonl(source_jsonl)

    # ---- Walk slim shards and aggregate metadata per seq_id. -----------------
    slim_records: dict[str, dict[str, Any]] = {}
    shard_paths = sorted(slim_cache_root.glob("shard_*.pt"))

    for shard_path in shard_paths:
        shard = _load_shard(shard_path)
        shard_id = str(shard.get("shard_id", shard_path.stem.split("_", 1)[1]))
        seq_ids = list(shard.get("seq_ids", []))
        sequences = list(shard.get("sequences", []))
        lengths = _lengths_list(shard)
        routes = _available_routes(shard)

        for idx, seq_id in enumerate(seq_ids):
            if idx >= len(sequences):
                warnings.append(f"{shard_path.name}: missing sequence for {seq_id}")
                continue
            if idx >= len(lengths):
                warnings.append(f"{shard_path.name}: missing length for {seq_id}")
                continue

            sequence = sequences[idx]
            teacher_l = int(lengths[idx])
            seq_len = len(sequence)

            if seq_len != teacher_l:
                warnings.append(
                    f"{shard_path.name}: seq_id {seq_id} len={seq_len} != "
                    f"teacher attention L={teacher_l}"
                )

            entry = slim_records.setdefault(
                seq_id,
                {"sequences": set(), "lengths": set(), "shard_ids": set(), "routes": set()},
            )
            entry["sequences"].add(sequence)
            entry["lengths"].add(teacher_l)
            entry["shard_ids"].add(shard_id)
            entry["routes"].update(routes)

        del shard
        gc.collect()

    # ---- Build rows, validate against source, and detect hash collisions. ----
    rows: list[SequenceManifestRow] = []
    hash_to_seq_id: dict[str, str] = {}
    collisions: list[dict[str, Any]] = []

    def _sequence_hash(sequence: str) -> str:
        return hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest()

    for seq_id in source_order:
        src = source_records[seq_id]
        sequence = src["sequence"]
        src_len = src["length"]
        seq_hash = _sequence_hash(sequence)

        # Stable hash collision detection.
        if seq_hash in hash_to_seq_id:
            prev_id = hash_to_seq_id[seq_hash]
            if prev_id != seq_id:
                collisions.append(
                    {"hash": seq_hash, "seq_ids": sorted({prev_id, seq_id})}
                )
        else:
            hash_to_seq_id[seq_hash] = seq_id

        if "T" in sequence.upper():
            warnings.append(f"seq_id {seq_id}: DNA thymine (T) present in RNA sequence")

        if len(sequence) != src_len:
            warnings.append(
                f"seq_id {seq_id}: source length={src_len} != len(sequence)={len(sequence)}"
            )

        slim = slim_records.get(seq_id)
        if slim is None:
            warnings.append(f"seq_id {seq_id}: present in source but missing from slim cache")
            rows.append(
                {
                    "seq_id": seq_id,
                    "sequence": sequence,
                    "length": src_len,
                    "original_dataset": src["source"],
                    "original_shard_id": [],
                    "sequence_hash": seq_hash,
                    "available_teacher_routes": [],
                }
            )
            continue

        if len(slim["sequences"]) > 1:
            warnings.append(f"seq_id {seq_id}: sequence differs across slim shards")
        if src_len not in slim["lengths"]:
            warnings.append(
                f"seq_id {seq_id}: source length={src_len} not in slim lengths "
                f"{sorted(slim['lengths'])}"
            )

        rows.append(
            {
                "seq_id": seq_id,
                "sequence": sequence,
                "length": src_len,
                "original_dataset": src["source"],
                "original_shard_id": sorted(slim["shard_ids"]),
                "sequence_hash": seq_hash,
                "available_teacher_routes": sorted(slim["routes"]),
            }
        )

    # Sequences present only in the slim cache (should be rare).
    for seq_id in sorted(slim_records):
        if seq_id in source_records:
            continue
        warnings.append(f"seq_id {seq_id}: present in slim cache but missing from source")
        entry = slim_records[seq_id]
        sequence = next(iter(entry["sequences"]))
        rows.append(
            {
                "seq_id": seq_id,
                "sequence": sequence,
                "length": next(iter(entry["lengths"])),
                "original_dataset": None,
                "original_shard_id": sorted(entry["shard_ids"]),
                "sequence_hash": _sequence_hash(sequence),
                "available_teacher_routes": sorted(entry["routes"]),
            }
        )

    df = pd.DataFrame(
        rows,
        columns=[
            "seq_id",
            "sequence",
            "length",
            "original_dataset",
            "original_shard_id",
            "sequence_hash",
            "available_teacher_routes",
        ],
    )

    target_manifest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target_manifest, index=False)

    return {
        "n_source": len(source_records),
        "n_slim_seqs": len(slim_records),
        "n_output_rows": len(df),
        "n_collisions": len(collisions),
        "collisions": collisions,
        "warnings": warnings,
    }
