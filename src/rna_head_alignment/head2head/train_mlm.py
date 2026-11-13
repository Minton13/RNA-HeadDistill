"""Standalone MLM pretraining worker for the from-scratch small student (S0).

Trains a randomly-initialised RiNALMo backbone (built via
``RiNALMoStudentBackbone`` ``config_overrides``) with masked-language
modelling only — no teacher targets, no slim cache. Sequences come straight
from a split-index parquet (``seq_id``/``sequence``/``length`` columns).

Checkpoint payload::

    {"step", "model_state_dict": {"backbone.*": ..., "mlm_head.*": ...},
     "optimizer_state_dict", "scheduler_state_dict", "metrics", "config"}

``backbone.*`` keys map directly onto :class:`RiNALMoHeadToHeadStudent` via
the ``init_checkpoint`` mechanism in ``train_ddp._make_student_full``.
"""

from __future__ import annotations

import math
import os
import time
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from rna_head_alignment.head2head.full_sequence_dataset import LengthBucketedSampler
from rna_head_alignment.head2head.mlm import (
    RiNALMoMLMHead,
    mask_tokens,
    mlm_loss_from_logits,
)
from rna_head_alignment.head2head.student_backbone import RiNALMoStudentBackbone
from rna_head_alignment.head2head.train_ddp import (
    _save_run_metadata,
    all_reduce_dict,
    cleanup_ddp,
    setup_ddp,
)


class MLMSequenceDataset(Dataset):
    """Sequence-only dataset over a split-index parquet."""

    def __init__(
        self,
        split_index_path: str | Path,
        tokenizer: Any,
        max_len: int = 1024,
    ) -> None:
        df = pd.read_parquet(split_index_path, columns=["sequence", "length"])
        self._sequences: list[str] = df["sequence"].tolist()
        self._lengths: list[int] = df["length"].astype(int).tolist()
        self._tokenizer = tokenizer
        self.max_len = int(max_len)

    def __len__(self) -> int:
        return len(self._sequences)

    def lengths(self) -> list[int]:
        return self._lengths

    def __getitem__(self, idx: int) -> dict[str, Any]:
        seq = self._sequences[idx]
        enc = self._tokenizer(seq[: self.max_len])
        input_ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        return {"input_ids": input_ids, "length": int(min(len(seq), self.max_len))}


def mlm_collate(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    """Pad to the max token length in the batch."""
    max_tok = max(int(x["input_ids"].shape[0]) for x in items)
    bsz = len(items)
    input_ids = torch.zeros(bsz, max_tok, dtype=torch.long)  # pad_token_id == 0
    attention_mask = torch.zeros(bsz, max_tok, dtype=torch.long)
    for i, x in enumerate(items):
        n = int(x["input_ids"].shape[0])
        input_ids[i, :n] = x["input_ids"]
        attention_mask[i, :n] = 1
    lengths = torch.tensor([x["length"] for x in items], dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "lengths": lengths}


class SmallMLMModel(nn.Module):
    """Backbone + tied MLM head. State-dict keys: ``backbone.*`` / ``mlm_head.*``."""

    def __init__(self, backbone: RiNALMoStudentBackbone) -> None:
        super().__init__()
        self.backbone = backbone
        self.mlm_head = RiNALMoMLMHead(backbone)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        out = self.backbone.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
        )
        return self.mlm_head(out.last_hidden_state)


def _save_mlm_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    metrics: dict[str, float],
    path: Path,
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config,
        },
        path,
    )


def run_mlm_worker(
    rank: int,
    world_size: int,
    config: dict[str, Any],
    run_dir: Path,
    is_ddp: bool,
) -> None:
    runtime_cfg = config.get("runtime", {})
    training_cfg = config.get("training", {})
    optimizer_cfg = config.get("optimizer", {})
    scheduler_cfg = config.get("scheduler", {})
    dynamic_cfg = config.get("dynamic_batch", {})
    student_cfg = config.get("student", {})
    seed = int(runtime_cfg.get("seed", 42))
    torch.manual_seed(seed + rank)

    if is_ddp:
        timeout = int(training_cfg.get("nccl_timeout", 1800))
        setup_ddp(rank, world_size, timeout_sec=timeout)

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model_dir = Path(student_cfg.get("model_dir", "assets/models/rinalmo-micro"))
    tokenizer = RiNALMoStudentBackbone(model_dir=model_dir).tokenizer()

    backbone = RiNALMoStudentBackbone(
        model_dir=model_dir,
        config_overrides=student_cfg.get("config_overrides") or None,
    )
    model = SmallMLMModel(backbone).to(device)

    if is_ddp:
        # The pooler (and any other never-used submodule) receives no gradient
        # under MLM — enable unused-parameter detection.
        model = DDP(
            model,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=True,
        )
    unwrapped = model.module if is_ddp else model

    if rank == 0:
        n_params = sum(p.numel() for p in unwrapped.parameters())
        print(f"[rank 0] small-student MLM worker on {device}; params={n_params/1e6:.2f}M", flush=True)

    lr = float(optimizer_cfg.get("lr", 5.0e-4))
    wd = float(optimizer_cfg.get("weight_decay", 0.01))
    fused = bool(optimizer_cfg.get("fused", True)) and device.type == "cuda"
    try:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd, fused=fused)
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    max_steps = int(training_cfg.get("max_steps", 40000))
    warmup_steps = int(scheduler_cfg.get("warmup_steps", 2000))
    min_lr_ratio = float(scheduler_cfg.get("min_lr_ratio", 0.05))

    def lr_lambda(step: int) -> float:
        step = min(step, max_steps)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress)) * (1.0 - min_lr_ratio) + min_lr_ratio

    scheduler = LambdaLR(optimizer, lr_lambda)

    resume = bool(training_cfg.get("resume", True))
    start_step = 0
    latest_path = run_dir / "checkpoints" / "latest.pt"
    if resume and latest_path.exists():
        ckpt = torch.load(latest_path, map_location=device, weights_only=False)
        unwrapped.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if ckpt.get("scheduler_state_dict"):
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = int(ckpt.get("step", 0)) + 1
        if rank == 0:
            print(f"[rank 0] resumed from {latest_path} at step {start_step}", flush=True)

    max_len = int(runtime_cfg.get("max_len", 1024))
    train_dataset = MLMSequenceDataset(config["train_split_index"], tokenizer, max_len=max_len)
    val_dataset = MLMSequenceDataset(config["val_split_index"], tokenizer, max_len=max_len)

    max_pair_tokens = int(dynamic_cfg.get("max_pair_tokens", 16 * 512 * 512))
    max_sequences = int(dynamic_cfg.get("max_sequences_per_batch", training_cfg.get("batch_size", 64)))
    bucket_size_samples = dynamic_cfg.get("bucket_size_samples")

    def _make_loader(ds: MLMSequenceDataset, shuffle: bool) -> tuple[DataLoader, LengthBucketedSampler]:
        sampler = LengthBucketedSampler(
            ds.lengths(),
            max_pair_tokens=max_pair_tokens,
            max_sequences_per_batch=max_sequences,
            bucket_size_samples=bucket_size_samples,
            rank=rank,
            world_size=world_size,
            seed=seed,
            shuffle=shuffle,
            drop_last=False,
        )
        loader = DataLoader(
            ds,
            batch_sampler=sampler,
            collate_fn=mlm_collate,
            num_workers=int(runtime_cfg.get("num_workers", 0)),
            pin_memory=device.type == "cuda",
        )
        return loader, sampler

    train_loader, train_sampler = _make_loader(train_dataset, shuffle=True)
    val_loader, _ = _make_loader(val_dataset, shuffle=False)

    precision = str(training_cfg.get("precision", "bf16"))
    use_amp = precision in ("bf16", "fp16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if precision == "bf16" else (torch.float16 if precision == "fp16" else torch.float32)

    mlm_prob = float(training_cfg.get("mlm_prob", 0.15))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    log_every = int(training_cfg.get("log_every", 50))
    save_every = int(training_cfg.get("save_every", 2000))
    eval_every = int(training_cfg.get("eval_every", 2000))
    eval_batches = int(training_cfg.get("eval_batches", 50))

    if rank == 0:
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        _save_run_metadata(run_dir, config)
    if is_ddp:
        dist.barrier()

    mlm_gen = torch.Generator(device=device)
    mlm_gen.manual_seed(seed * 1000 + rank)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []

    step = start_step
    epoch = 0
    train_iter = iter(train_loader)
    model.train()

    while step < max_steps:
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step_start = time.perf_counter()
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            masked_ids, labels = mask_tokens(
                input_ids, attention_mask, mlm_prob=mlm_prob, generator=mlm_gen
            )
            logits = model(masked_ids, attention_mask)
            loss = mlm_loss_from_logits(logits, labels)
        loss.backward()
        loss_val = float(loss.detach().item())
        if not math.isfinite(loss_val):
            raise RuntimeError(f"Non-finite MLM loss at step {step}: {loss_val}")

        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        metrics = all_reduce_dict(
            {"loss/mlm": loss_val, "lr": optimizer.param_groups[0]["lr"],
             "step_time_s": time.perf_counter() - step_start},
            device,
        )

        if rank == 0 and (step % log_every == 0 or step == 0):
            print(
                f"[mlm] step {step}/{max_steps} loss={metrics['loss/mlm']:.4f} "
                f"lr={metrics['lr']:.2e} step_time={metrics['step_time_s']:.2f}s",
                flush=True,
            )
            train_rows.append({"step": step, **metrics})

        if rank == 0 and ((step + 1) % save_every == 0 or step + 1 == max_steps):
            _save_mlm_checkpoint(
                unwrapped, optimizer, scheduler, step, metrics,
                run_dir / "checkpoints" / "latest.pt", config,
            )
            _save_mlm_checkpoint(
                unwrapped, optimizer, scheduler, step, metrics,
                run_dir / "checkpoints" / f"step_{step + 1:05d}.pt", config,
            )
        if is_ddp and ((step + 1) % save_every == 0):
            dist.barrier()

        if (step + 1) % eval_every == 0 or step + 1 == max_steps:
            model.eval()
            val_loss_sum, val_n = 0.0, 0
            with torch.no_grad():
                for bi, vbatch in enumerate(val_loader):
                    if bi >= eval_batches:
                        break
                    vids = vbatch["input_ids"].to(device)
                    vmask = vbatch["attention_mask"].to(device)
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        mids, vlabels = mask_tokens(vids, vmask, mlm_prob=mlm_prob, generator=mlm_gen)
                        vlogits = model(mids, vmask)
                        vloss = mlm_loss_from_logits(vlogits, vlabels)
                    val_loss_sum += float(vloss.item())
                    val_n += 1
            model.train()
            val_metrics = all_reduce_dict({"val/mlm": val_loss_sum / max(1, val_n)}, device)
            if rank == 0:
                print(f"[mlm] step {step + 1} val_loss={val_metrics['val/mlm']:.4f}", flush=True)
                val_rows.append({"step": step + 1, **val_metrics})

        step += 1

    if rank == 0:
        if train_rows:
            pd.DataFrame(train_rows).to_csv(run_dir / "train_metrics.csv", index=False)
        if val_rows:
            pd.DataFrame(val_rows).to_csv(run_dir / "val_metrics.csv", index=False)
        print(f"[mlm] done at step {step}", flush=True)

    cleanup_ddp()


def launch_mlm_training(
    config: dict[str, Any],
    run_dir: Path,
    world_size: int | None = None,
) -> None:
    run_dir = Path(run_dir)
    if world_size is None:
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        else:
            world_size = int(config.get("world_size", 1))

    if (
        dist.is_available()
        and "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and world_size > 1
    ):
        rank = int(os.environ["RANK"])
        run_mlm_worker(rank, world_size, config, run_dir, is_ddp=True)
    elif world_size > 1:
        mp.spawn(run_mlm_worker, args=(world_size, config, run_dir, True), nprocs=world_size, join=True)
    else:
        run_mlm_worker(0, 1, config, run_dir, is_ddp=False)
