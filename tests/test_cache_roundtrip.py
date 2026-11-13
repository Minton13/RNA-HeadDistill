import pandas as pd
import torch

from rna_head_alignment.cache.validate_attention_cache import validate_attention_cache


def test_cache_roundtrip_validation(tmp_path):
    root = tmp_path / "cache"
    model_dir = root / "model=fake"
    model_dir.mkdir(parents=True)
    shard = model_dir / "shard_00000.pt"
    torch.save(
        {
            "model": "fake",
            "selected_heads": [{"layer": 0, "head": 0, "task": "x"}],
            "seq_ids": ["s1"],
            "sequences": ["AUGC"],
            "lengths": torch.tensor([4]),
            "attention": torch.full((1, 1, 4, 4), 0.25),
            "valid_mask": torch.ones(1, 4, dtype=torch.bool),
        },
        shard,
    )
    pd.DataFrame([{"model": "fake", "shard": 0, "path": str(shard), "status": "ok"}]).to_csv(root / "manifest.csv", index=False)
    report = validate_attention_cache(root, tmp_path / "report.md")
    assert report["ok"]
    assert report["n_maps"] == 1
