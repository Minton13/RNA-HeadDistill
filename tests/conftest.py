"""Shared fixtures for head2head slim-cache tests.

Builds a fake 2-model teacher attention cache in a tmp directory that mimics
the real on-disk schema (see outputs/teacher_attention_cache/), including one
shard where a model is missing sequences (like rinalmo_mega/shard_00936).
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import torch
import yaml

from rna_head_alignment.head2head.head_manifest import build_target_head_manifest

# ---------------------------------------------------------------- tiny model
# The public repo does not ship assets/models/ (model weights live on the HF
# hub). The tiny-CPU tests only need the rinalmo-micro HF config/tokenizer
# files to build randomly-initialised tiny models (weights come from
# config_overrides), so a minimal snapshot is materialised here. File contents
# are verbatim copies of the upstream multimolecule/rinalmo-micro snapshot
# (config.json / tokenizer_config.json / vocab.txt).

RINALMO_MICRO_CONFIG = {
    "add_cross_attention": False,
    "architectures": ["RiNALMoForPreTraining"],
    "attention_dropout": 0.1,
    "bos_token_id": 1,
    "dtype": "float32",
    "emb_layer_norm_before": True,
    "eos_token_id": 2,
    "head": None,
    "hidden_act": "gelu",
    "hidden_dropout": 0.1,
    "hidden_size": 480,
    "id2label": None,
    "initializer_range": 0.02,
    "intermediate_size": 1920,
    "is_decoder": False,
    "label2id": None,
    "layer_norm_eps": 1e-05,
    "learnable_beta": True,
    "lm_head": None,
    "mask_token_id": 4,
    "max_position_embeddings": 1024,
    "model_type": "rinalmo",
    "null_token_id": 5,
    "num_attention_heads": 20,
    "num_hidden_layers": 12,
    "num_labels": 1,
    "pad_token_id": 0,
    "position_embedding_type": "rotary",
    "tie_word_embeddings": True,
    "token_dropout": True,
    "unk_token_id": 3,
    "use_cache": True,
    "vocab_size": 26,
}

_RINALMO_MICRO_SPECIALS = ["<pad>", "<cls>", "<eos>", "<unk>", "<mask>", "<null>"]
RINALMO_MICRO_TOKENIZER_CONFIG = {
    "added_tokens_decoder": {
        str(i): {
            "content": tok,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
        for i, tok in enumerate(_RINALMO_MICRO_SPECIALS)
    },
    "backend": "custom",
    "bos_token": "<cls>",
    "clean_up_tokenization_spaces": True,
    "cls_token": "<cls>",
    "codon": False,
    "eos_token": "<eos>",
    "extra_special_tokens": ["<null>"],
    "mask_token": "<mask>",
    "model_max_length": 1000000000000000019884624838656,
    "nmers": 1,
    "pad_token": "<pad>",
    "replace_T_with_U": True,
    "sep_token": "<eos>",
    "tokenizer_class": "RnaTokenizer",
    "unk_token": "<unk>",
}

RINALMO_MICRO_VOCAB = "\n".join(
    _RINALMO_MICRO_SPECIALS
    + ["A", "C", "G", "U", "N", "R", "Y", "S", "W", "K", "M", "B", "D", "H",
       "V", ".", "X", "*", "-", "I"]
)


@pytest.fixture(scope="session")
def tiny_rinalmo_model_dir(tmp_path_factory):
    """Minimal rinalmo-micro HF snapshot (config + tokenizer + random weights).

    The config/tokenizer files are verbatim upstream; weights are randomly
    initialised from the config (the real checkpoint is not redistributed).
    """
    d = tmp_path_factory.mktemp("rinalmo-micro")
    (d / "config.json").write_text(json.dumps(RINALMO_MICRO_CONFIG, indent=2),
                                   encoding="utf-8")
    (d / "tokenizer_config.json").write_text(
        json.dumps(RINALMO_MICRO_TOKENIZER_CONFIG, indent=2), encoding="utf-8")
    (d / "vocab.txt").write_text(RINALMO_MICRO_VOCAB, encoding="utf-8")
    try:
        import multimolecule  # noqa: F401  (registers the rinalmo model type)
        from safetensors.torch import save_file
        from transformers import AutoConfig, AutoModel

        cfg = AutoConfig.from_pretrained(str(d), local_files_only=True)
        model = AutoModel.from_config(cfg, trust_remote_code=True)
        save_file(model.state_dict(), str(d / "model.safetensors"))
    except Exception:
        pass  # config/tokenizer-only snapshot; weight-loading tests will fail
    return d

FAKE_BANK = {
    "banks": {
        "L6": {
            "student_layer": 6,
            "bank_name": "fake_seq_bank",
            "tasks": {
                "manual_L6_fake": {
                    "teacher_heads": [
                        {"model": "alpha", "layer": 1, "head": 0, "metric": None, "source": "manual"},
                    ]
                }
            },
        },
        "L8": {
            "student_layer": 8,
            "bank_name": "fake_chem_bank",
            "tasks": {
                "manual_L8_fake": {
                    "teacher_heads": [
                        {"model": "beta", "layer": 2, "head": 1, "metric": None, "source": "manual"},
                    ]
                }
            },
        },
        "L12": {
            "student_layer": 12,
            "bank_name": "fake_struct_bank",
            "tasks": {
                "manual_L12_fake": {
                    "teacher_heads": [
                        {"model": "alpha", "layer": 3, "head": 2, "metric": None, "source": "manual"},
                    ]
                }
            },
        },
    }
}

FAKE_HEAD_KEYS = {
    "alpha": [
        "alpha:L1:H0:L6:manual_L6_fake",
        "alpha:L3:H2:L12:manual_L12_fake",
    ],
    "beta": ["beta:L2:H1:L8:manual_L8_fake"],
}


def _write_fake_shard(cache_root, model, shard_idx, seq_ids, lmax, seed):
    gen = torch.Generator().manual_seed(seed)
    head_keys = FAKE_HEAD_KEYS[model]
    batch, channels = len(seq_ids), len(head_keys)
    attention = torch.randn(batch, channels, lmax, lmax, generator=gen).to(torch.float16)
    lengths = torch.randint(1, lmax + 1, (batch,), generator=gen, dtype=torch.int64)
    shard = {
        "model": model,
        "task_or_bank": [k.split(":", 3)[3] for k in head_keys],
        "selected_heads": [],
        "head_keys": head_keys,
        "seq_ids": list(seq_ids),
        "sequences": ["ACGUACGUAC"[: max(1, min(10, lmax))]] * batch,
        "lengths": lengths,
        "attention": attention,
        "valid_mask": torch.ones(batch, lmax, dtype=torch.bool),
        "attention_dtype": torch.float16,
        "errors": [],
        "attention_layout": "attention[b, k, i, j] where k follows selected_heads/head_keys",
        "source_sequences": list(seq_ids),
        "sequence_start": shard_idx * batch,
        "sequence_end": shard_idx * batch + batch,
    }
    path = cache_root / f"model={model}" / f"shard_{shard_idx:05d}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(shard, path)
    return path


@pytest.fixture
def fake_teacher_cache(tmp_path):
    """Fake 2-model x 2-shard cache; shard 1 has 'beta' missing 2 seqs."""
    bank_path = tmp_path / "bank.yaml"
    bank_path.write_text(yaml.safe_dump(FAKE_BANK), encoding="utf-8")

    cache_root = tmp_path / "cache"
    lmax = 12
    shard0_ids = [f"s{i}" for i in range(6)]
    shard1_ids = [f"t{i}" for i in range(6)]
    # beta's shard 1 misses t1 and t3 (mirrors rinalmo_mega/shard_00936)
    shard1_beta_ids = ["t0", "t2", "t4", "t5"]
    dropped = ["t1", "t3"]

    for model in ("alpha", "beta"):
        _write_fake_shard(cache_root, model, 0, shard0_ids, lmax, seed=100 + len(model))
    _write_fake_shard(cache_root, "alpha", 1, shard1_ids, lmax, seed=201)
    _write_fake_shard(cache_root, "beta", 1, shard1_beta_ids, lmax, seed=202)

    manifest = build_target_head_manifest(bank_path)
    return SimpleNamespace(
        bank_path=bank_path,
        cache_root=cache_root,
        manifest=manifest,
        lmax=lmax,
        shard0_ids=shard0_ids,
        shard1_ids=shard1_ids,
        dropped=dropped,
    )
