from __future__ import annotations

import csv
from pathlib import Path

import torch
from torch.optim import AdamW

from rna_head_alignment.alignment.aligners import LayerHeadAlignerBank
from rna_head_alignment.alignment.attention_losses import mask_normalized_mse
from rna_head_alignment.io.cache_io import load_pt
from rna_head_alignment.io.paths import ensure_parent, load_yaml, resolve_path


def train_alignment_debug(config: str, teacher_cache: str, output: str, steps: int | None = None) -> dict:
    cfg = load_yaml(config)
    train_cfg = cfg.get("training", {})
    steps = int(steps or train_cfg.get("steps", 100))
    out = resolve_path(output)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    shards = sorted(resolve_path(teacher_cache).glob("model=*/shard_*.pt"))
    if not shards:
        raise FileNotFoundError(f"No cache shards found in {teacher_cache}")
    mapping = {"L6": [0], "L8": [0], "L12": [0]}
    aligners = LayerHeadAlignerBank(mapping)
    opt = AdamW(aligners.parameters(), lr=float(train_cfg.get("lr", 1e-4)))
    log_path = ensure_parent(out / "train_log.csv")
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "loss_total", "cache_shard"])
        writer.writeheader()
        for step in range(steps):
            payload = load_pt(shards[step % len(shards)])
            teacher = payload["attention"][:, :1]
            valid = payload["valid_mask"]
            student = torch.rand_like(teacher, requires_grad=True)
            pred = aligners.forward_one("L6", 0, student[:, 0]).unsqueeze(1)
            loss = mask_normalized_mse(pred, teacher, valid)
            opt.zero_grad()
            loss.backward()
            opt.step()
            writer.writerow({"step": step, "loss_total": float(loss.item()), "cache_shard": str(shards[step % len(shards)])})
    torch.save({"aligners": aligners.state_dict(), "config": cfg}, out / "checkpoints" / "debug_last.pt")
    (out / "README.md").write_text(
        "# Student Alignment Debug Run\n\n"
        "This debug path validates per-head aligner optimization against cached teacher maps. "
        "Full RiNALMo-micro DDP training is wired through the same CLI but layer-restricted "
        "gradient updates remain a TODO in the first implementation.\n",
        encoding="utf-8",
    )
    return {"ok": True, "steps": steps, "log": str(log_path)}
