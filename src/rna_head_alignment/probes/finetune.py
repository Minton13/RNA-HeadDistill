"""Route-specific fine-tuning for Stage 4B head-to-head student on downstream tasks.

Implements feature modes F0 (frozen probe baseline), F1 (fine-tune without
alignment replay), and F2 (fine-tune with alignment replay).  The fine-tuner
loads the Stage 4B checkpoint, freezes backbone blocks above the task route,
and trains the route's adapters together with a task-specific probe head.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, Sampler

from rna_head_alignment.head2head.losses import route_alignment_loss, total_route_loss
from rna_head_alignment.head2head.metrics import compute_head_metrics
from rna_head_alignment.head2head.model import RiNALMoHeadToHeadStudent
from rna_head_alignment.head2head.sequence_dataset import (
    LengthBucketedBatchSampler,
    SlimShardDataset,
    variable_length_collate,
)
from rna_head_alignment.probes.eval import evaluate_probe
from rna_head_alignment.probes.task_config import TASK_TO_ROUTE_DEPTH, TASK_TO_TYPE
from rna_head_alignment.probes.train import _is_better, _metric_name_for_task


# Default optimizer grid for fine-tuning probes (not for the backbone).
PROBE_OPTIMIZER_GRID: list[tuple[float, float]] = [
    (1.0e-4, 1.0e-5),
    (3.0e-4, 1.0e-5),
    (1.0e-3, 1.0e-5),
]


def _sequence_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    if num_classes == 2:
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    return F.cross_entropy(logits, labels)


def _token_loss(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Padded label positions are NaN; mask them out before computing MSE.
    labels = torch.nan_to_num(labels, nan=0.0)
    loss = F.mse_loss(preds, labels, reduction="none") * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def _pair_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def _task_loss(
    probe: torch.nn.Module,
    features: torch.Tensor,
    batch: dict[str, torch.Tensor],
    task_type: str,
    num_classes: int | None,
) -> torch.Tensor:
    if task_type == "sequence_classification":
        logits = probe(features, mask=batch["attention_mask"])
        return _sequence_loss(logits, batch["labels"], num_classes or 2)
    if task_type == "token_regression":
        preds = probe(features)
        return _token_loss(preds, batch["labels"], batch["label_mask"])
    if task_type == "pairwise":
        logits = probe(features)
        return _pair_loss(logits, batch["labels"], batch["label_mask"])
    raise ValueError(task_type)


def _freeze_backbone_above_route(student: RiNALMoHeadToHeadStudent, route_depth: int) -> None:
    """Freeze all backbone parameters except embeddings/layers up to route_depth."""
    # First freeze everything.
    for p in student.parameters():
        p.requires_grad_(False)

    # Unfreeze embeddings.
    for p in student.backbone.model.embeddings.parameters():
        p.requires_grad_(True)

    # Unfreeze encoder layers 0 .. route_depth-1 (1-based 1 .. route_depth).
    for layer_idx in range(route_depth):
        for p in student.backbone.model.encoder.layer[layer_idx].parameters():
            p.requires_grad_(True)


def _unfreeze_route_adapters(student: RiNALMoHeadToHeadStudent, route_depth: int) -> None:
    """Unfreeze adapters belonging to the specified route."""
    route_key = f"L{route_depth}"
    for key, adapter in student.adapters.items():
        if key.startswith(route_key + "_"):
            for p in adapter.parameters():
                p.requires_grad_(True)


def build_replay_subset_index(
    slim_cache_root: str | Path,
    n_sequences: int = 50_000,
    seed: int = 42,
    output_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build a deterministic stratified replay subset from the full slim cache.

    Stratifies by length bin and source shard, sampling sequences proportionally
    to the available sequences in each stratum.  Each sampled sequence carries
    all route/head annotations from its shard, so downstream replay batches can
    draw all L6/L8/L12 teacher heads.  Returns a DataFrame with columns:
    seq_id, sequence, length, shard_id, route_depth, head_key.
    """
    slim_cache_root = Path(slim_cache_root)
    manifest_path = slim_cache_root / "manifests" / "slim_cache_manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Slim cache manifest not found: {manifest_path}")

    bucket_manifest = pd.read_parquet(manifest_path)
    seq_manifest_path = slim_cache_root.parent / "manifests" / "sequence_manifest.parquet"
    if not seq_manifest_path.exists():
        seq_manifest_path = slim_cache_root / "manifests" / "sequence_manifest.parquet"
    seq_manifest = None
    seq_lookup: dict[str, str] = {}
    if seq_manifest_path.exists():
        seq_manifest = pd.read_parquet(seq_manifest_path)
        seq_lookup = dict(zip(seq_manifest["seq_id"], seq_manifest["sequence"]))

    rng = np.random.default_rng(seed)
    seq_rows: list[dict[str, Any]] = []
    head_keys_by_shard: dict[str, dict[int, list[str]]] = {}

    for _, row in bucket_manifest.iterrows():
        bucket_path = Path(str(row["file_path"]))
        if not bucket_path.is_absolute():
            candidate = slim_cache_root / bucket_path
            bucket_path = candidate if candidate.exists() else Path.cwd() / bucket_path
        if not bucket_path.exists():
            continue

        if bucket_path.suffix == ".safetensors":
            from safetensors import safe_open

            with safe_open(str(bucket_path), framework="pt") as f:
                meta = f.metadata() or {}
                shard_id = str(meta.get("shard_id", bucket_path.stem))
                lengths = f.get_tensor("lengths").numpy()
                manifest_indices = f.get_tensor("sequence_manifest_indices").numpy()
                if seq_manifest is not None:
                    seq_ids = seq_manifest["seq_id"].astype(str).iloc[manifest_indices].tolist()
                    seqs = seq_manifest["sequence"].astype(str).iloc[manifest_indices].tolist()
                    for sid, seq in zip(seq_ids, seqs):
                        seq_lookup.setdefault(sid, seq)
                else:
                    seq_ids = [f"seq_{i}" for i in manifest_indices]
                shard_heads: dict[int, list[str]] = {}
                for depth in (6, 8, 12):
                    head_ids = json.loads(meta.get(f"route_heads_L{depth}", "[]"))
                    if head_ids:
                        shard_heads[depth] = [str(h) for h in head_ids]
        else:
            shard = torch.load(bucket_path, map_location="cpu", weights_only=False)
            seq_ids = list(shard.get("seq_ids", []))
            lengths = shard["lengths"].numpy()
            manifest_indices = shard.get("sequence_manifest_indices")
            if manifest_indices is not None:
                manifest_indices = manifest_indices.numpy()
            shard_id = str(shard.get("shard_id", bucket_path.stem))
            shard_heads = {}
            for depth in (6, 8, 12):
                head_ids = shard.get(f"teacher_head_ids_L{depth}", [])
                if head_ids:
                    shard_heads[depth] = [str(h) for h in head_ids]
            del shard
        head_keys_by_shard[shard_id] = shard_heads

        for s_idx, seq_id in enumerate(seq_ids):
            length = int(lengths[s_idx])
            seq_rows.append({
                "seq_id": str(seq_id),
                "sequence": seq_lookup.get(str(seq_id), ""),
                "length": length,
                "length_bin": f"{(length // 64) * 64}-{((length // 64) + 1) * 64}",
                "shard_id": shard_id,
                "sequence_manifest_index": int(manifest_indices[s_idx]) if manifest_indices is not None else -1,
            })

    seq_df = pd.DataFrame(seq_rows)
    if seq_df.empty:
        return seq_df

    # Stratified sampling without replacement, capped at n_sequences.
    seq_df["_stratum"] = seq_df["length_bin"].astype(str) + "_" + seq_df["shard_id"].astype(str)
    strat_counts = seq_df["_stratum"].value_counts()
    total = len(seq_df)
    samples_per_stratum = (
        (strat_counts / total * n_sequences).round().astype(int).clip(lower=1)
    )

    sampled: list[pd.DataFrame] = []
    for stratum, n in samples_per_stratum.items():
        sub = seq_df[seq_df["_stratum"] == stratum]
        n = min(n, len(sub))
        sampled.append(sub.sample(n=n, random_state=rng))

    sampled_seqs = pd.concat(sampled, ignore_index=True).drop(columns=["_stratum"])
    if len(sampled_seqs) > n_sequences:
        sampled_seqs = sampled_seqs.sample(n=n_sequences, random_state=rng).reset_index(drop=True)

    # Expand each sampled sequence to one row per (route, head).
    out_rows: list[dict[str, Any]] = []
    for _, seq_row in sampled_seqs.iterrows():
        shard_heads = head_keys_by_shard.get(seq_row["shard_id"], {})
        for depth, heads in shard_heads.items():
            for head_key in heads:
                out_rows.append({
                    "seq_id": seq_row["seq_id"],
                    "sequence": seq_row["sequence"],
                    "length": seq_row["length"],
                    "shard_id": seq_row["shard_id"],
                    "route_depth": depth,
                    "head_key": head_key,
                })

    out = pd.DataFrame(out_rows)
    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
    return out
    if len(out) > n_sequences:
        out = out.sample(n=n_sequences, random_state=seed).reset_index(drop=True)

    if output_path is not None:
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(out_path, index=False)
    return out


class ReplaySubsetDataset(Dataset):
    """Dataset that samples sequences from the slim cache by index DataFrame."""

    def __init__(
        self,
        index_df: pd.DataFrame,
        slim_cache_root: str | Path,
        tokenizer: Any,
        max_len: int = 512,
    ) -> None:
        self.index_df = index_df.reset_index(drop=True)
        self.slim_cache_root = Path(slim_cache_root)
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self._cache: dict[str, dict] = {}

    def __len__(self) -> int:
        return len(self.index_df)

    def _load_bucket(self, shard_id: str, length: int) -> dict[str, torch.Tensor]:
        cache_key = f"{shard_id}_{length}"
        if cache_key in self._cache:
            return self._cache[cache_key]
        buckets_dir = self.slim_cache_root / "buckets"
        candidates = sorted(buckets_dir.glob(f"shard_{shard_id}_*.safetensors"))
        if not candidates:
            candidates = sorted(self.slim_cache_root.glob(f"shard_{shard_id}_*.safetensors"))
        if not candidates:
            raise FileNotFoundError(f"No bucket for shard {shard_id} under {self.slim_cache_root}")
        # Pick the bucket whose size covers the sequence length.
        def _bucket_size(path: Path) -> int:
            stem = path.stem
            for tag in ("_bucket_1024", "_bucket_768", "_bucket_512", "_bucket_256", "_bucket_128", "_bucket_64"):
                if tag in stem:
                    return int(tag.rsplit("_", 1)[-1])
            return 64
        chosen = candidates[0]
        for cand in candidates:
            if _bucket_size(cand) >= length:
                chosen = cand
                break
        from safetensors.torch import load_file

        tensors = load_file(str(chosen), device="cpu")
        self._cache[cache_key] = tensors
        return tensors

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.index_df.iloc[idx]
        length = int(row.get("length", 0))
        tensors = self._load_bucket(str(row["shard_id"]), length)
        seq_id = str(row["seq_id"])
        # Locate the sequence within the bucket via sequence_manifest_indices.
        manifest_idx = int(row.get("sequence_manifest_index", -1))
        if manifest_idx >= 0:
            positions = (tensors["sequence_manifest_indices"] == manifest_idx).nonzero(as_tuple=True)[0]
            local_idx = int(positions[0].item()) if len(positions) else 0
        else:
            local_idx = 0
        seq = str(row.get("sequence", ""))
        length = int(tensors["lengths"][local_idx].item())
        from rna_head_alignment.head2head.sequence_dataset import align_tokens_to_sequence

        input_ids, tok_len = align_tokens_to_sequence(self.tokenizer, seq, self.max_len)
        item: dict[str, Any] = {
            "seq_id": seq_id,
            "input_ids": input_ids,
            "length": tok_len,
            "attention_mask": torch.ones(tok_len, dtype=torch.long),
            "shard_id": str(row["shard_id"]),
        }
        for route in ("L6", "L8", "L12"):
            key = f"teacher_{route}"
            t = tensors[f"teacher_heads_{route}"][local_idx, :, :tok_len, :tok_len]
            item[key] = t
        return item


def replay_collate(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate replay items analogously to variable_length_collate."""
    if not items:
        raise ValueError("cannot collate empty list")
    batch = variable_length_collate(items)
    return batch


class RouteSpecificFineTuner:
    """Fine-tune a Stage 4B student for a single downstream task.

    Parameters
    ----------
    student:
        Stage 4B head-to-head student (will be modified in place).
    task:
        Downstream task name.
    route_depth:
        1-based route layer (6/8/12).
    feature_mode:
        ``F0`` (frozen), ``F1`` (fine-tune, no replay), or ``F2`` (fine-tune with
        alignment replay).
    device:
        torch device.
    output_dir:
        Directory for checkpoints and logs.
    replay_dataset:
        Optional replay dataset for F2 mode.
    lambda_align:
        Weight for alignment replay loss in F2 mode.
    backbone_lr, adapter_lr, probe_lr:
        Learning rates per component group.
    max_steps:
        Maximum fine-tuning steps.
    eval_every:
        Evaluate on validation set every N steps.
    save_every:
        Save a checkpoint every N steps.
    grad_clip:
        Gradient clipping value.
    mixed_precision:
        Use AMP bf16 on CUDA.
    """

    def __init__(
        self,
        student: RiNALMoHeadToHeadStudent,
        task: str,
        route_depth: int,
        feature_mode: str,
        device: torch.device,
        output_dir: str | Path,
        replay_dataset: Dataset | None = None,
        lambda_align: float = 0.1,
        backbone_lr: float = 2.0e-6,
        adapter_lr: float = 3.0e-5,
        probe_lr: float = 1.0e-4,
        max_steps: int = 2_000,
        eval_every: int = 200,
        save_every: int = 500,
        grad_clip: float = 1.0,
        mixed_precision: bool = False,
        optimizer_grid: list[tuple[float, float]] | None = None,
    ) -> None:
        self.student = student
        self.task = task
        self.task_type = TASK_TO_TYPE[task]
        self.route_depth = route_depth
        self.feature_mode = feature_mode
        self.device = device
        self.output_dir = Path(output_dir)
        self.replay_dataset = replay_dataset
        self.lambda_align = float(lambda_align)
        self.backbone_lr = backbone_lr
        self.adapter_lr = adapter_lr
        self.probe_lr = probe_lr
        self.max_steps = int(max_steps)
        self.eval_every = int(eval_every)
        self.save_every = int(save_every)
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.optimizer_grid = optimizer_grid or PROBE_OPTIMIZER_GRID
        self.metric_name = _metric_name_for_task(self.task_type)
        self.amp_dtype = torch.bfloat16 if mixed_precision else torch.float32

        self.history: list[dict[str, Any]] = []
        self.best_state: dict[str, Any] | None = None
        self.best_metric = -float("inf")

        self._setup_trainability()

    def _setup_trainability(self) -> None:
        if self.feature_mode == "F0":
            # Everything stays frozen; only the probe will be added externally.
            for p in self.student.parameters():
                p.requires_grad_(False)
            return

        # F1 / F2: unfreeze backbone up to route and route adapters.
        _freeze_backbone_above_route(self.student, self.route_depth)
        _unfreeze_route_adapters(self.student, self.route_depth)

    def _parameter_groups(self, probe: torch.nn.Module) -> list[dict[str, Any]]:
        backbone_params: list[torch.nn.Parameter] = []
        adapter_params: list[torch.nn.Parameter] = []
        for name, p in self.student.named_parameters():
            if not p.requires_grad:
                continue
            if "adapters." in name:
                adapter_params.append(p)
            else:
                backbone_params.append(p)
        probe_params = [p for p in probe.parameters() if p.requires_grad]
        groups = []
        if backbone_params:
            groups.append({"params": backbone_params, "lr": self.backbone_lr})
        if adapter_params:
            groups.append({"params": adapter_params, "lr": self.adapter_lr})
        if probe_params:
            groups.append({"params": probe_params, "lr": self.probe_lr})
        return groups

    def _train_step(
        self,
        probe: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler | None,
        task_batch: dict[str, Any],
        replay_batch: dict[str, Any] | None,
    ) -> dict[str, float]:
        self.student.train()
        probe.train()
        optimizer.zero_grad(set_to_none=True)

        use_amp = self.mixed_precision and self.device.type == "cuda"

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=use_amp):
            # Downstream task loss.
            input_ids = task_batch["input_ids"].to(self.device)
            attention_mask = task_batch["attention_mask"].to(self.device)
            out = self.student(input_ids, attention_mask, return_backbone=True)
            hidden = out["backbone"][self.route_depth]["hidden"]
            if torch.isnan(hidden).any():
                print(f"[nan-debug] hidden NaN at step, route={self.route_depth}, input_ids shape={input_ids.shape}", flush=True)
            task_loss = _task_loss(
                probe,
                hidden,
                {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in task_batch.items()},
                self.task_type,
                self.num_classes,
            )
            if torch.isnan(task_loss).any():
                print(f"[nan-debug] task_loss NaN, hidden_nan={torch.isnan(hidden).any().item()}, labels_nan={torch.isnan(task_batch['labels']).any().item()}, mask_sum={task_batch['label_mask'].sum().item()}", flush=True)

            total_loss = task_loss
            align_loss_val = 0.0
            if self.feature_mode == "F2" and replay_batch is not None and self.lambda_align > 0:
                align_loss = self._alignment_replay_loss(replay_batch)
                total_loss = total_loss + self.lambda_align * align_loss
                align_loss_val = float(align_loss.detach().item())

        if use_amp and scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self._all_trainable(probe), self.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self._all_trainable(probe), self.grad_clip)
            optimizer.step()

        return {
            "task_loss": float(task_loss.detach().item()),
            "align_loss": align_loss_val,
            "total_loss": float(total_loss.detach().item()),
        }

    def _all_trainable(self, probe: torch.nn.Module) -> list[torch.nn.Parameter]:
        return [p for p in list(self.student.parameters()) + list(probe.parameters()) if p.requires_grad]

    @torch.no_grad()
    def _evaluate_downstream(
        self,
        probe: torch.nn.Module,
        val_loader: DataLoader,
    ) -> dict[str, Any]:
        from rna_head_alignment.probes.feature_extractor import ProbeFeatureExtractor

        self.student.eval()
        probe.eval()
        extractor = ProbeFeatureExtractor(
            model=self.student,
            feature_mode="distilled_hidden",
            task=self.task,
            route_depth=self.route_depth,
        )
        extractor.to(self.device)
        return evaluate_probe(
            probe,
            extractor,
            val_loader,
            self.task,
            self.device,
            num_classes=self.num_classes,
            mixed_precision=self.mixed_precision,
        )

    @torch.no_grad()
    def _evaluate_alignment(self, replay_loader: DataLoader) -> dict[str, float]:
        """Mean teacher-head alignment metrics over a replay sample."""
        self.student.eval()
        metrics_accum: dict[str, list[float]] = {}
        counts = 0
        for batch in replay_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            lengths = batch["lengths"].to(self.device)
            key_padding_mask = attention_mask == 0
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.mixed_precision and self.device.type == "cuda"):
                out = self.student(input_ids, attention_mask, return_backbone=True)
            for depth in (6, 8, 12):
                route_key = f"L{depth}"
                pred = out[f"pred_{route_key}"].float()
                teacher = batch[f"teacher_{route_key}"].to(self.device).float()
                per_seq = compute_head_metrics(
                    pred,
                    teacher,
                    lengths,
                    metrics=["mse", "rowwise_kl", "cosine"],
                    include_per_sequence=True,
                )
                for metric in ("mse", "rowwise_kl", "cosine"):
                    key = f"{route_key}_{metric}"
                    vals = per_seq[f"{metric}_per_seq"].mean(dim=1).detach().cpu().numpy()
                    metrics_accum.setdefault(key, []).extend(float(v) for v in vals if not np.isnan(v))
            counts += input_ids.shape[0]
            if counts >= 1_024:
                break
        return {k: float(np.mean(v)) if v else float("nan") for k, v in metrics_accum.items()}

    def _alignment_replay_loss(self, replay_batch: dict[str, Any]) -> torch.Tensor:
        input_ids = replay_batch["input_ids"].to(self.device)
        attention_mask = replay_batch["attention_mask"].to(self.device)
        lengths = replay_batch["lengths"].to(self.device)
        out = self.student(input_ids, attention_mask, return_backbone=True)
        route_losses: dict[str, torch.Tensor] = {}
        for depth in (6, 8, 12):
            route_key = f"L{depth}"
            pred = out[f"pred_{route_key}"]
            teacher = replay_batch[f"teacher_{route_key}"].to(self.device)
            if torch.isnan(pred).any() or torch.isnan(teacher).any():
                print(f"[nan-debug] align replay {route_key}: pred_nan={torch.isnan(pred).any().item()} teacher_nan={torch.isnan(teacher).any().item()}", flush=True)
            loss, _ = route_alignment_loss(
                pred,
                teacher,
                lengths,
                lambdas=(1.0, 1.0, 0.1),
                normalize_within_route=True,
            )
            if torch.isnan(loss).any():
                print(f"[nan-debug] align replay {route_key} loss NaN", flush=True)
            route_losses[route_key] = loss
        return total_route_loss(route_losses)

    def fit(
        self,
        probe: torch.nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int | None = None,
    ) -> dict[str, Any]:
        """Run route-specific fine-tuning and return a summary."""
        self.num_classes = num_classes
        start = time.time()

        # Build replay loader if needed.
        replay_loader: DataLoader | None = None
        if self.feature_mode == "F2" and self.replay_dataset is not None:
            replay_loader = DataLoader(
                self.replay_dataset,
                batch_size=train_loader.batch_size or 8,
                collate_fn=replay_collate,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
            )
            replay_iter = iter(replay_loader)
        else:
            replay_iter = None

        best_overall = -float("inf")
        best_state: dict[str, Any] | None = None
        best_step = 0
        train_iter = iter(train_loader)
        step = 0
        scaler = torch.cuda.amp.GradScaler() if self.mixed_precision and self.device.type == "cuda" else None

        # Use a single optimizer for the jointly trained components.
        optimizer = AdamW(self._parameter_groups(probe))

        # Pre-fine-tune alignment snapshot.
        pre_align = {}
        if replay_loader is not None:
            pre_align = self._evaluate_alignment(replay_loader)

        while step < self.max_steps:
            try:
                task_batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                task_batch = next(train_iter)

            replay_batch = None
            if replay_iter is not None and step % 5 == 4:  # 4 task batches, then 1 replay
                try:
                    replay_batch = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(replay_loader)
                    replay_batch = next(replay_iter)

            loss_vals = self._train_step(probe, optimizer, scaler, task_batch, replay_batch)
            step += 1

            if step % self.eval_every == 0 or step == self.max_steps:
                val_result = self._evaluate_downstream(probe, val_loader)
                val_metric = val_result["metrics"].get(self.metric_name, float("nan"))
                row = {"step": step, **loss_vals, f"val_{self.metric_name}": float(val_metric)}
                row.update({f"val_{k}": v for k, v in val_result["metrics"].items()})
                self.history.append(row)
                print(
                    f"[route-finetune] task={self.task} mode={self.feature_mode} "
                    f"step={step}/{self.max_steps} loss={loss_vals.get('total_loss', float('nan')):.4f} "
                    f"val_metric={val_metric:.4f} best={best_overall:.4f}",
                    flush=True,
                )

                if _is_better(self.metric_name, val_metric, best_overall):
                    best_overall = val_metric
                    best_step = step
                    best_state = {
                        "student": copy.deepcopy(self.student.state_dict()),
                        "probe": copy.deepcopy(probe.state_dict()),
                        "step": step,
                    }

            if step % self.save_every == 0:
                self._save_checkpoint(probe, step)

        if best_state is not None:
            self.student.load_state_dict(best_state["student"])
            probe.load_state_dict(best_state["probe"])

        # Post-fine-tune alignment snapshot.
        post_align = {}
        if replay_loader is not None:
            post_align = self._evaluate_alignment(replay_loader)

        summary = {
            "task": self.task,
            "feature_mode": self.feature_mode,
            "route_depth": self.route_depth,
            "best_val_metric": best_overall,
            "best_step": best_step,
            "trainable_params": sum(p.numel() for p in self._all_trainable(probe)),
            "wall_time_sec": time.time() - start,
            "pre_alignment": pre_align,
            "post_alignment": post_align,
        }
        self._save_outputs(probe, summary)
        return summary

    def _save_checkpoint(self, probe: torch.nn.Module, step: int) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "student_state_dict": self.student.state_dict(),
                "probe_state_dict": probe.state_dict(),
                "step": step,
            },
            self.output_dir / f"checkpoint_step_{step}.pt",
        )

    def _save_outputs(self, probe: torch.nn.Module, summary: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "training_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        with open(self.output_dir / "training_history.json", "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2, default=str)
        torch.save(
            {
                "student_state_dict": self.student.state_dict(),
                "probe_state_dict": probe.state_dict(),
                "summary": summary,
            },
            self.output_dir / "best_checkpoint.pt",
        )
