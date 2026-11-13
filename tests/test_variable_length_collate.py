"""Tests for the slim-shard dataset, variable-length collate and batch sampler."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from rna_head_alignment.head2head.sequence_dataset import (
    LengthBucketedBatchSampler,
    SlimShardDataset,
    align_tokens_to_sequence,
    variable_length_collate,
)


@pytest.fixture()
def rinalmo_tokenizer(tiny_rinalmo_model_dir):
    import multimolecule  # noqa: F401  (registers the rinalmo tokenizer)

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(
        str(tiny_rinalmo_model_dir), local_files_only=True, use_fast=False
    )


def _make_shard(path, shard_id: str, seqs: list[str], n_heads: int = 5, lmax: int = 16):
    lengths = torch.tensor([min(len(s), lmax) for s in seqs], dtype=torch.int32)
    g = torch.Generator().manual_seed(hash(shard_id) % 2**31)
    shard = {
        "schema_version": "slim-1",
        "shard_id": shard_id,
        "seq_ids": [f"{shard_id}_seq{i}" for i in range(len(seqs))],
        "sequences": seqs,
        "lengths": lengths,
        "metadata": {"routes": ["L6", "L8", "L12"]},
    }
    for route in ("L6", "L8", "L12"):
        t = torch.rand(len(seqs), n_heads, lmax, lmax, generator=g)
        t = t / t.sum(dim=-1, keepdim=True)
        shard[f"teacher_heads_{route}"] = t.half()
        shard[f"teacher_head_ids_{route}"] = [f"{route}_t{i}" for i in range(n_heads)]
    torch.save(shard, path)


@pytest.fixture()
def shards(tmp_path):
    seqs_a = ["ACGUACGUAC", "GGGAAACCC"]          # lengths 10, 9
    seqs_b = ["ACGU", "UUUUCCCCAAAAGGGG", "AC"]  # lengths 4, 16, 2
    p1 = tmp_path / "shard_00000.pt"
    p2 = tmp_path / "shard_00001.pt"
    _make_shard(p1, "s0", seqs_a, lmax=16)
    _make_shard(p2, "s1", seqs_b, lmax=16)
    return [p1, p2]


def test_align_tokens_strips_special_tokens(rinalmo_tokenizer):
    tokenizer = rinalmo_tokenizer
    ids, seq_len = align_tokens_to_sequence(tokenizer, "ACGUACGU", max_len=512)
    assert seq_len == 8
    assert ids.tolist() == [6, 7, 8, 9, 6, 7, 8, 9]  # no <cls>=1 / <eos>=2
    ids, seq_len = align_tokens_to_sequence(tokenizer, "ACGUACGUACGU", max_len=5)
    assert seq_len == 5 and ids.numel() == 5


def test_dataset_getitem(shards, rinalmo_tokenizer):
    tokenizer = rinalmo_tokenizer
    ds = SlimShardDataset(shards, tokenizer, max_len=512, cache_size=1)
    assert len(ds) == 5
    item = ds[0]
    assert item["seq_id"] == "s0_seq0"
    assert item["length"] == 10
    assert item["input_ids"].numel() == 10
    for route in ("L6", "L8", "L12"):
        t = item[f"teacher_{route}"]
        assert t.shape == (5, 10, 10)
        assert t.dtype == torch.float16
    # teacher maps are in nucleotide coordinates: teacher[i, j] pairs the
    # nucleotides at positions i and j, which are exactly input_ids[i], input_ids[j]
    assert item["input_ids"][0].item() == 6  # 'A', not <cls>


def test_collate_shapes_and_padding(shards, rinalmo_tokenizer):
    tokenizer = rinalmo_tokenizer
    ds = SlimShardDataset(shards, tokenizer, max_len=512)
    items = [ds[0], ds[3], ds[4]]  # lengths 10, 16, 2
    batch = variable_length_collate(items)
    b = 3
    assert batch["input_ids"].shape == (b, 16)
    assert batch["attention_mask"].shape == (b, 16)
    assert batch["lengths"].tolist() == [10, 16, 2]
    assert batch["seq_lengths"].tolist() == [10, 16, 2]
    assert batch["attention_mask"].sum(dim=1).tolist() == [10, 16, 2]
    assert batch["seq_ids"] == ["s0_seq0", "s1_seq1", "s1_seq2"]
    for route in ("L6", "L8", "L12"):
        t = batch[f"teacher_{route}"]
        assert t.shape == (b, 5, 16, 16)
        # padded rows/cols of the short sequences are exactly zero
        assert t[0, :, 10:, :].abs().sum() == 0
        assert t[0, :, :, 10:].abs().sum() == 0
        assert t[2, :, 2:, :].abs().sum() == 0
        assert t[2, :, :, 2:].abs().sum() == 0
        # valid region matches the dataset item
        assert torch.equal(t[0, :, :10, :10], items[0][f"teacher_{route}"])


def test_max_len_truncates_teacher_region(shards, rinalmo_tokenizer):
    tokenizer = rinalmo_tokenizer
    ds = SlimShardDataset(shards, tokenizer, max_len=6)
    item = ds[3]  # 16-nt sequence
    assert item["length"] == 6
    assert item["teacher_L6"].shape == (5, 6, 6)


def test_length_bucketed_batch_sampler():
    lengths = [10, 201, 11, 202, 12, 203, 13, 200, 14, 15]
    sampler = LengthBucketedBatchSampler(lengths, batch_size=2, bucket_size=6, seed=0)
    batches = list(iter(sampler))
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(10))
    # within a batch, lengths come from the same bucket (bounded spread)
    for b in batches:
        bl = [lengths[i] for i in b]
        assert max(bl) - min(bl) <= 5
    # DDP split: ranks see disjoint batches
    s0 = list(iter(LengthBucketedBatchSampler(lengths, 2, bucket_size=6, seed=0, rank=0, world_size=2)))
    s1 = list(iter(LengthBucketedBatchSampler(lengths, 2, bucket_size=6, seed=0, rank=1, world_size=2)))
    assert not ({tuple(sorted(b)) for b in s0} & {tuple(sorted(b)) for b in s1})
    assert sorted(i for b in s0 + s1 for i in b) == list(range(10))
    # epoch changes reshuffle
    sampler.set_epoch(1)
    assert list(iter(sampler)) != batches
