"""Dataset and collation for the Stage 4B bucketed full slim cache.

The full slim cache reorganises the original teacher attention cache into
bucketed safetensors files that contain only the 15 target teacher heads.
Each bucket file holds a fixed spatial size (e.g. 128, 256) and the collate
function pads to the real maximum length in a mini-batch.

Public exports
--------------
* :class:`FullSlimBucketDataset` — lazy dataset over bucketed safetensors.
* :class:`LengthBucketedSampler` — DDP-aware, dynamic pair-token budget sampler.
* :func:`full_variable_length_collate` — pads teacher maps and builds masks.
"""

from __future__ import annotations

import json
import math
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch.utils.data import Dataset, Sampler

from .head_manifest import ROUTE_DEPTHS
from .sequence_dataset import align_tokens_to_sequence

ROUTE_KEYS = tuple(f"L{d}" for d in ROUTE_DEPTHS)


def _read_bucket_metadata_route_heads(path: str | Path) -> dict[int, list[str]]:
    """Read the ``route_heads_L{depth}`` JSON metadata from a bucket file."""
    with safe_open(str(path), framework="pt") as f:
        metadata = f.metadata() or {}
    return {
        depth: json.loads(metadata.get(f"route_heads_L{depth}", "[]"))
        for depth in ROUTE_DEPTHS
    }


class FullSlimBucketDataset(Dataset):
    """Lazy dataset over the bucketed full slim cache.

    Parameters
    ----------
    full_cache_root:
        Root of the full cache (contains ``buckets/`` and ``manifests/``).
    split_index_path:
        Parquet file with a ``seq_id`` column selecting sequences for this split.
    sequence_manifest_path:
        ``sequence_manifest.parquet``; must have ``seq_id``, ``sequence``,
        ``length`` and use its row index as the manifest index.
    tokenizer:
        Local rinalmo-micro tokenizer (or any tokenizer exposing
        ``get_special_tokens_mask``).
    max_len:
        Maximum nucleotide length kept per sequence.
    cache_size:
        Number of full bucket files kept in an in-memory LRU cache.
        The default is 1; with ``num_workers=0`` this keeps host memory
        bounded while avoiding repeated per-sample random reads.
    route_depths:
        Teacher route depths to expose (default 6/8/12).
    """

    def __init__(
        self,
        full_cache_root: str | Path,
        split_index_path: str | Path,
        sequence_manifest_path: str | Path,
        tokenizer: Any,
        max_len: int = 1024,
        cache_size: int = 1,
        route_depths: tuple[int, ...] = ROUTE_DEPTHS,
    ) -> None:
        self.full_cache_root = Path(full_cache_root)
        self.split_index_path = Path(split_index_path)
        self.sequence_manifest = pd.read_parquet(sequence_manifest_path)
        self._seq_arr = self.sequence_manifest["sequence"].astype(str).to_numpy()
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.cache_size = int(cache_size)
        self.route_depths = tuple(route_depths)

        # LRU cache of loaded bucket tensors.  Loading the whole bucket once
        # and serving consecutive samples from RAM is much faster than
        # opening the file and slicing per sample on this shared-storage setup.
        self._cache: OrderedDict[str, dict[str, torch.Tensor]] = OrderedDict()
        self.route_head_keys: dict[int, list[str]] | None = None

        split_df = pd.read_parquet(self.split_index_path)
        if "seq_id" not in split_df.columns:
            raise ValueError(f"split index {self.split_index_path} missing seq_id column")
        allowed_seq_ids = set(split_df["seq_id"].astype(str).tolist())

        bucket_manifest_path = self.full_cache_root / "manifests" / "slim_cache_manifest.parquet"
        if not bucket_manifest_path.exists():
            raise FileNotFoundError(f"bucket manifest not found: {bucket_manifest_path}")
        bucket_df = pd.read_parquet(bucket_manifest_path).dropna(subset=["file_path"])

        # Pre-compute vectorised lookups over the full sequence manifest.
        seq_ids_arr = self.sequence_manifest["seq_id"].astype(str).to_numpy()
        lengths_arr = self.sequence_manifest["length"].to_numpy()
        n_seqs = len(self.sequence_manifest)
        allowed_arr = self.sequence_manifest["seq_id"].astype(str).isin(allowed_seq_ids).to_numpy()

        # manifest index -> (bucket_path, within-bucket position)
        self._items: list[dict[str, Any]] = []
        for _, row in bucket_df.iterrows():
            bucket_path = Path(str(row["file_path"]))
            if not bucket_path.is_absolute():
                # Manifest may store the path relative to cwd or relative to the cache root.
                candidate_cwd = Path.cwd() / bucket_path
                candidate_root = self.full_cache_root / bucket_path
                bucket_path = candidate_cwd if candidate_cwd.exists() else candidate_root
            if not bucket_path.exists():
                warnings.warn(f"missing bucket file: {bucket_path}", stacklevel=2)
                continue
            try:
                with safe_open(str(bucket_path), framework="pt") as f:
                    manifest_indices = f.get_tensor("sequence_manifest_indices").to(torch.int64).numpy()
            except Exception as exc:  # noqa: BLE001
                warnings.warn(f"cannot read {bucket_path}: {exc}", stacklevel=2)
                continue

            valid = (
                (manifest_indices >= 0)
                & (manifest_indices < n_seqs)
                & allowed_arr[manifest_indices]
            )
            manifest_indices = manifest_indices[valid]
            positions = np.nonzero(valid)[0]
            if manifest_indices.size == 0:
                continue
            for manifest_idx, position in zip(manifest_indices.tolist(), positions.tolist()):
                self._items.append(
                    {
                        "manifest_idx": int(manifest_idx),
                        "seq_id": str(seq_ids_arr[manifest_idx]),
                        "length": min(int(lengths_arr[manifest_idx]), self.max_len),
                        "bucket_path": bucket_path,
                        "position": int(position),
                    }
                )

        if not self._items:
            raise ValueError(
                f"no sequences matched between split {self.split_index_path} "
                f"and bucket files in {self.full_cache_root}"
            )

        # Sort by length so the length-bucketed sampler can pack efficiently.
        self._items.sort(key=lambda it: (it["length"], it["seq_id"]))

        # Peek route head keys from the first bucket for consistency checks.
        first_path = self._items[0]["bucket_path"]
        self.route_head_keys = _read_bucket_metadata_route_heads(first_path)

    def __len__(self) -> int:
        return len(self._items)

    def lengths(self) -> list[int]:
        """Return per-sample nucleotide lengths for the batch sampler."""
        return [it["length"] for it in self._items]

    def _load_bucket(self, path: Path) -> dict[str, torch.Tensor]:
        """Load a full bucket file into memory, keeping a small LRU cache.

        With ``cache_size=1`` and ``num_workers=0`` each rank holds at most
        one bucket in RAM.  The largest bucket is ~16 GB, so six ranks need
        ~96 GB of host memory, well inside the node's 360 GB.
        """
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        tensors = load_file(str(path), device="cpu")
        self._cache[key] = tensors
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return tensors

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item = self._items[idx]
        bucket = self._load_bucket(item["bucket_path"])
        position = item["position"]

        seq = str(self._seq_arr[item["manifest_idx"]])
        input_ids, seq_len = align_tokens_to_sequence(self.tokenizer, seq, self.max_len)

        out: dict[str, Any] = {
            "seq_id": item["seq_id"],
            "input_ids": input_ids,
            "attention_mask": torch.ones(input_ids.numel(), dtype=torch.long),
            "length": seq_len,
            "bucket_path": str(item["bucket_path"]),
        }

        for depth in self.route_depths:
            route_key = f"L{depth}"
            teacher = bucket[f"teacher_heads_{route_key}"][position, :, :seq_len, :seq_len]
            availability = bucket[f"head_availability_{route_key}"][position, :]
            out[f"teacher_{route_key}"] = teacher
            out[f"head_availability_{route_key}"] = availability

        return out


def full_variable_length_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad a list of :class:`FullSlimBucketDataset` items into a batch.

    Returns ``input_ids`` ``[B, Lmax_tok]``, ``attention_mask`` ``[B, Lmax_tok]``,
    ``lengths`` ``[B]``, ``teacher_L6/L8/L12`` ``[B, T, Lmax_seq, Lmax_seq]``,
    ``head_availability_L6/L8/L12`` ``[B, T]``, a temporary pair mask
    ``pair_mask`` ``[B, 1, Lmax_seq, Lmax_seq]``, ``seq_ids`` and ``bucket_paths``.
    """
    if not items:
        raise ValueError("cannot collate an empty item list")

    batch_size = len(items)
    max_tok = max(int(item["input_ids"].numel()) for item in items)
    max_seq = max(int(item["length"]) for item in items)
    pad_id = 0

    input_ids = torch.zeros(batch_size, max_tok, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_tok, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    bucket_paths: list[str] = []

    teachers: dict[str, torch.Tensor] = {}
    availabilities: dict[str, torch.Tensor] = {}

    for b, item in enumerate(items):
        ids = item["input_ids"]
        l_tok = ids.numel()
        input_ids[b, :l_tok] = ids
        attention_mask[b, :l_tok] = 1

        l_seq = int(item["length"])
        lengths[b] = l_seq
        bucket_paths.append(str(item.get("bucket_path", "")))

        for route_key in ROUTE_KEYS:
            teacher = item[f"teacher_{route_key}"]
            availability = item[f"head_availability_{route_key}"]
            if route_key not in teachers:
                teachers[route_key] = torch.zeros(
                    batch_size,
                    teacher.shape[0],
                    max_seq,
                    max_seq,
                    dtype=teacher.dtype,
                )
                availabilities[route_key] = torch.zeros(
                    batch_size,
                    availability.shape[0],
                    dtype=torch.bool,
                )
            teachers[route_key][b, :, :l_seq, :l_seq] = teacher
            availabilities[route_key][b, :] = availability

    # Pair mask derived from real lengths (temporary, not persisted).
    idx = torch.arange(max_seq)
    valid = idx.unsqueeze(0) < lengths.unsqueeze(1)  # [B, L]
    pair_mask = (valid.unsqueeze(2) & valid.unsqueeze(1)).unsqueeze(1)  # [B, 1, L, L]

    batch: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "pair_mask": pair_mask,
        "seq_ids": [item["seq_id"] for item in items],
        "shard_ids": bucket_paths,
    }
    for route_key in ROUTE_KEYS:
        batch[f"teacher_{route_key}"] = teachers[route_key]
        batch[f"head_availability_{route_key}"] = availabilities[route_key]

    return batch


class LengthBucketedSampler(Sampler[list[int]]):
    """Length-bucketed sampler with a dynamic pair-token budget.

    Indices are sorted by length and grouped into ``bucket_size_samples``
    buckets.  Within each bucket, batches are built greedily so that
    ``sum(length_i ** 2)`` does not exceed ``max_pair_tokens`` and the number
    of sequences does not exceed ``max_sequences_per_batch``.

    For DDP, batches are split across ranks
    (``batches[rank::world_size]``).  Call :meth:`set_epoch` each epoch for
    reproducible shuffling.

    Parameters
    ----------
    lengths:
        Per-sample sequence lengths.
    max_pair_tokens:
        Pair-token budget per batch (``sum(L_i ** 2)``).
    max_sequences_per_batch:
        Hard cap on sequences per batch.
    bucket_size_samples:
        Number of samples in each length bucket used for batch construction.
        Default is ``100 * max_sequences_per_batch``.
    rank, world_size:
        DDP split.
    seed, shuffle, drop_last:
        Reproducibility / ordering controls.
    """

    def __init__(
        self,
        lengths: list[int],
        max_pair_tokens: int,
        max_sequences_per_batch: int,
        bucket_size_samples: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.lengths = list(lengths)
        self.max_pair_tokens = int(max_pair_tokens)
        self.max_sequences_per_batch = int(max_sequences_per_batch)
        self.bucket_size_samples = int(
            bucket_size_samples if bucket_size_samples is not None else 100 * self.max_sequences_per_batch
        )
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _make_batches(self) -> list[list[int]]:
        order = sorted(range(len(self.lengths)), key=lambda i: self.lengths[i])
        buckets = [
            order[i : i + self.bucket_size_samples]
            for i in range(0, len(order), self.bucket_size_samples)
        ]

        g = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.shuffle:
            perm = torch.randperm(len(buckets), generator=g).tolist()
            buckets = [buckets[j] for j in perm]
            for bucket in buckets:
                sub_perm = torch.randperm(len(bucket), generator=g).tolist()
                bucket[:] = [bucket[j] for j in sub_perm]

        batches: list[list[int]] = []
        for bucket in buckets:
            current: list[int] = []
            current_cost = 0
            for idx in bucket:
                cost = self.lengths[idx] * self.lengths[idx]
                if current and (
                    current_cost + cost > self.max_pair_tokens
                    or len(current) >= self.max_sequences_per_batch
                ):
                    batches.append(current)
                    current = [idx]
                    current_cost = cost
                else:
                    current.append(idx)
                    current_cost += cost
            if current and (len(current) == self.max_sequences_per_batch or not self.drop_last):
                batches.append(current)

        if self.shuffle and batches:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[j] for j in perm]

        # Ensure every rank receives the same number of batches to keep DDP
        # all_reduce calls aligned across processes.
        if self.world_size > 1:
            keep = (len(batches) // self.world_size) * self.world_size
            if keep < len(batches):
                batches = batches[:keep]

        return batches[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._make_batches())

    def __len__(self) -> int:
        # Approximate upper bound; exact count requires building all batches.
        n = math.ceil(len(self.lengths) / self.max_sequences_per_batch)
        if self.drop_last:
            n = len(self.lengths) // self.max_sequences_per_batch
        return (n - self.rank + self.world_size - 1) // self.world_size
