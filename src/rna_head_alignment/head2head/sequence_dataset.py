"""Slim teacher-attention shard dataset with variable-length collation.

Slim shard format (``shard_{idx:05d}.pt``, produced by the slim-cache stage)::

    {
        "schema_version": ...,
        "shard_id": str,
        "seq_ids": list[str],
        "sequences": list[str],
        "lengths": int32 tensor [B],
        "teacher_heads_L6/L8/L12": fp16 tensor [B, T, Lmax, Lmax],
        "teacher_head_ids_L6/L8/L12": list[str],
        "metadata": dict,
    }

Teacher map coordinates are **nucleotide coordinates**. The student tokenizer
adds special tokens (``<cls>``/``<eos>``), so :func:`align_tokens_to_sequence`
strips them and returns token ids that line up 1:1 with nucleotide positions
(same idea as ``token_mapping.select_token_positions``): batch element ``i`` has
``input_ids[i, :L_i]`` = the nucleotide tokens and ``teacher_Lx[i, :, :L_i, :L_i]``
is the corresponding teacher region. Model pred maps and teacher maps therefore
share the ``[:L_i, :L_i]`` loss region directly.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import Dataset, Sampler

ROUTES = ("L6", "L8", "L12")


def align_tokens_to_sequence(
    tokenizer: Any,
    sequence: str,
    max_len: int,
) -> tuple[torch.Tensor, int]:
    """Tokenize ``sequence`` and return nucleotide-aligned token ids.

    Special tokens are stripped: the returned ``input_ids`` tensor has length
    ``min(len(sequence), max_len)`` and element ``i`` is the token for
    nucleotide ``i``. For the rinalmo 1-nt-per-token tokenizer this is the
    raw nucleotide token ids; the generic implementation selects non-special
    token positions via ``get_special_tokens_mask`` (cf.
    ``token_mapping.select_token_positions``).

    Returns ``(input_ids [L], seq_len)``.
    """
    encoded = tokenizer(sequence, add_special_tokens=True, return_tensors="pt")
    ids = encoded["input_ids"][0]
    special_mask = torch.tensor(
        tokenizer.get_special_tokens_mask(ids.tolist(), already_has_special_tokens=True),
        dtype=torch.bool,
    )
    nuc_ids = ids[~special_mask][:max_len]
    return nuc_ids, int(nuc_ids.numel())


class SlimShardDataset(Dataset):
    """Lazy dataset over slim teacher-attention shards with an LRU shard cache.

    Parameters
    ----------
    shard_paths:
        Paths to ``shard_*.pt`` files.
    tokenizer:
        Local rinalmo-micro tokenizer (special tokens stripped on load).
    max_len:
        Maximum nucleotide length kept per sequence (longer sequences and
        their teacher maps are truncated to ``max_len``).
    cache_size:
        Number of shards kept in the in-memory LRU cache.
    """

    def __init__(
        self,
        shard_paths: list[str | Path],
        tokenizer: Any,
        max_len: int = 512,
        cache_size: int = 2,
    ) -> None:
        self.shard_paths = [Path(p) for p in shard_paths]
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[int, dict] = OrderedDict()

        # Global index -> (shard_idx, within-shard index); lengths peeked from
        # each shard's small "lengths" tensor without loading teacher maps.
        self._index: list[tuple[int, int]] = []
        self._lengths: list[int] = []
        for shard_idx, path in enumerate(self.shard_paths):
            shard = torch.load(path, map_location="cpu", weights_only=False)
            n = len(shard["seq_ids"])
            lengths = shard["lengths"]
            for i in range(n):
                self._index.append((shard_idx, i))
                self._lengths.append(int(lengths[i]))
            del shard

    def __len__(self) -> int:
        return len(self._index)

    def sequence_lengths(self) -> list[int]:
        """Nucleotide lengths (post-truncation) for the batch sampler."""
        return [min(l, self.max_len) for l in self._lengths]

    def _load_shard(self, shard_idx: int) -> dict:
        if shard_idx in self._cache:
            self._cache.move_to_end(shard_idx)
            return self._cache[shard_idx]
        shard = torch.load(self.shard_paths[shard_idx], map_location="cpu", weights_only=False)
        self._cache[shard_idx] = shard
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return shard

    def __getitem__(self, idx: int) -> dict[str, Any]:
        shard_idx, i = self._index[idx]
        shard = self._load_shard(shard_idx)
        seq = shard["sequences"][i]
        input_ids, seq_len = align_tokens_to_sequence(self.tokenizer, seq, self.max_len)
        item: dict[str, Any] = {
            "seq_id": shard["seq_ids"][i],
            "input_ids": input_ids,
            "length": seq_len,
            "shard_id": str(shard["shard_id"]),
        }
        for route in ROUTES:
            key = f"teacher_heads_{route}"
            # [T, Lmax, Lmax] fp16 -> crop to [T, L_i, L_i]; kept fp16, the
            # collate/loss path upcasts as needed.
            item[f"teacher_{route}"] = shard[key][i, :, :seq_len, :seq_len]
        return item


def variable_length_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad a list of dataset items into a batch.

    Returns ``input_ids`` ``[B, Lmax_tok]`` (nucleotide tokens only, padded with
    the tokenizer pad id), ``attention_mask`` ``[B, Lmax_tok]``, ``lengths``
    ``[B]`` (nucleotide lengths — the region shared by model pred maps and
    teacher maps), ``teacher_L6/L8/L12`` ``[B, T, Lmax_seq, Lmax_seq]`` zero-padded,
    plus ``seq_lengths``, ``seq_ids`` and ``shard_ids``.
    """
    if not items:
        raise ValueError("cannot collate an empty item list")
    batch_size = len(items)
    max_tok = max(item["input_ids"].numel() for item in items)
    max_seq = max(item["length"] for item in items)
    pad_id = 0

    input_ids = torch.zeros(batch_size, max_tok, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_tok, dtype=torch.long)
    lengths = torch.zeros(batch_size, dtype=torch.long)
    teachers: dict[str, torch.Tensor] = {}
    for route in ROUTES:
        ref = items[0][f"teacher_{route}"]
        teachers[route] = torch.zeros(
            batch_size, ref.shape[0], max_seq, max_seq, dtype=ref.dtype
        )

    for b, item in enumerate(items):
        ids = item["input_ids"]
        l_tok = ids.numel()
        input_ids[b, :l_tok] = ids
        attention_mask[b, :l_tok] = 1
        l_seq = item["length"]
        lengths[b] = l_seq
        for route in ROUTES:
            t = item[f"teacher_{route}"]
            teachers[route][b, :, :l_seq, :l_seq] = t

    batch: dict[str, Any] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
        "seq_lengths": lengths.clone(),
        "seq_ids": [item["seq_id"] for item in items],
        "shard_ids": [item["shard_id"] for item in items],
    }
    for route in ROUTES:
        batch[f"teacher_{route}"] = teachers[route]
    return batch


class LengthBucketedBatchSampler(Sampler[list[int]]):
    """Length-bucketed batch sampler that minimizes padding.

    Indices are sorted by sequence length, chunked into buckets, and each epoch
    (see :meth:`set_epoch`) bucket order and batch composition are reshuffled
    with ``seed + epoch``. For DDP, batches are split across ranks
    (``batches[rank::world_size]``) so no two ranks see the same batch.

    Parameters
    ----------
    lengths:
        Per-sample sequence lengths (e.g. ``dataset.sequence_lengths()``).
    batch_size, bucket_size:
        Batch size and bucket size (defaults to 100 × batch_size).
    rank, world_size:
        DDP rank split; defaults to single process.
    seed, shuffle, drop_last:
        Reproducibility / ordering controls.
    """

    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        bucket_size: int | None = None,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        shuffle: bool = True,
        drop_last: bool = False,
    ) -> None:
        self.lengths = list(lengths)
        self.batch_size = int(batch_size)
        self.bucket_size = int(bucket_size) if bucket_size else 100 * self.batch_size
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
            order[i : i + self.bucket_size] for i in range(0, len(order), self.bucket_size)
        ]
        g = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.shuffle:
            perm = torch.randperm(len(buckets), generator=g).tolist()
            buckets = [buckets[j] for j in perm]
            for bucket in buckets:
                perm = torch.randperm(len(bucket), generator=g).tolist()
                bucket[:] = [bucket[j] for j in perm]
        batches: list[list[int]] = []
        for bucket in buckets:
            for i in range(0, len(bucket), self.batch_size):
                batch = bucket[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                batches.append(batch)
        if self.shuffle:
            perm = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[j] for j in perm]
        return batches[self.rank :: self.world_size]

    def __iter__(self) -> Iterator[list[int]]:
        return iter(self._make_batches())

    def __len__(self) -> int:
        n_batches = math.ceil(len(self.lengths) / self.batch_size)
        if self.drop_last:
            n_batches = len(self.lengths) // self.batch_size
        return (n_batches - self.rank + self.world_size - 1) // self.world_size
