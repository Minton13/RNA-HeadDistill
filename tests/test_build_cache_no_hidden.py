from __future__ import annotations

import json
import tempfile
from pathlib import Path

import torch

from rna_head_alignment.cache.build_teacher_attention_cache import build_teacher_attention_cache


class _FakeTokenizer:
    pad_token_id = 0
    all_special_ids = [0]

    def __call__(self, sequences, **kwargs):
        if isinstance(sequences, str):
            sequences = [sequences]
        max_len = max(len(s) for s in sequences) + 2
        input_ids = []
        attention_mask = []
        special_tokens_mask = []
        for seq in sequences:
            ids = [0] + [ord(c) % 10 + 1 for c in seq] + [0]
            ids = ids + [0] * (max_len - len(ids))
            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
            special_tokens_mask.append([1] + [0] * (len(ids) - 2) + [1])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "special_tokens_mask": torch.tensor(special_tokens_mask, dtype=torch.long),
        }


class _FakeModel:
    def __init__(self, num_layers: int = 3, num_heads: int = 2, hidden_size: int = 8):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_size = hidden_size

    def __call__(self, **kwargs):
        B, T = kwargs["input_ids"].shape
        hidden = [
            torch.randn(B, T, self.hidden_size)
            for _ in range(self.num_layers + 1)
        ]
        attn = [
            torch.softmax(torch.randn(B, self.num_heads, T, T), dim=-1)
            for _ in range(self.num_layers)
        ]
        return type("Out", (), {"hidden_states": hidden, "attentions": attn})()


class _FakeSpec:
    name = "fake_model"
    model_type = "RNA"

    def normalize_sequence(self, seq: str) -> str:
        return seq.upper().replace("T", "U")


def test_build_cache_runs_without_hidden_states():
    with tempfile.TemporaryDirectory() as tmpdir:
        seqs_path = Path(tmpdir) / "seqs.jsonl"
        out_path = Path(tmpdir) / "out"
        bank_path = Path(tmpdir) / "bank.yaml"
        models_path = Path(tmpdir) / "models.yaml"

        with seqs_path.open("w") as f:
            for i in range(12):
                f.write(json.dumps({"seq_id": f"s{i}", "sequence": "ACGU" * (i % 3 + 2)}) + "\n")

        bank_path.write_text(
            "banks:\n"
            "  b1:\n"
            "    tasks:\n"
            "      t1:\n"
            "        teacher_heads:\n"
            "          - {model: fake_model, layer: 1, head: 0}\n"
        )

        models_path.write_text(
            "models:\n"
            "  fake_model:\n"
            "    type: RNA\n"
            "    local_dir: /tmp/fake_model\n"
            "    attention_teacher: true\n"
        )

        # Monkeypatch loader and spec resolution for the test
        import rna_head_alignment.cache.build_teacher_attention_cache as bc
        import rna_head_alignment.models.attention_extractors as ae
        import rna_head_alignment.models.loaders as loaders
        import rna_head_alignment.models.registry as registry

        orig_load = loaders.load_frozen_model
        orig_get_spec = registry.get_model_spec

        def fake_load(spec, device):
            return _FakeTokenizer(), _FakeModel()

        def fake_get_spec(path, name):
            return _FakeSpec()

        bc.load_frozen_model = fake_load
        registry.get_model_spec = fake_get_spec

        try:
            build_teacher_attention_cache(
                teacher_bank=bank_path,
                sequences=seqs_path,
                output=out_path,
                models_yaml=models_path,
                max_len=64,
                shard_size=5,
                device="cpu",
                dtype="fp32",
                max_tokens_per_batch=1024,
                sub_batch_size=5,
            )
            shards = sorted((out_path / "model=fake_model").glob("shard_*.pt"))
            assert len(shards) == 3, f"expected 3 shards, got {len(shards)}"
            payload = torch.load(shards[0], map_location="cpu")
            assert payload["attention"].ndim == 4
            assert payload["attention"].shape[1] == 1
        finally:
            bc.load_frozen_model = orig_load
            registry.get_model_spec = orig_get_spec


if __name__ == "__main__":
    test_build_cache_runs_without_hidden_states()
    print("OK")
