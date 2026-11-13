"""Multi-task joint model for Stage 5 downstream probes.

The :class:`MultiTaskHead2HeadStudent` shares the Stage 4B backbone and aligned
adapters across all tasks, but keeps a separate probe head per task.  Training
uses a task-balanced batch sampler and route-specific forward passes (adapters
for deeper routes are skipped when the current task does not need them).
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

from rna_head_alignment.head2head.losses import route_alignment_loss, total_route_loss
from rna_head_alignment.head2head.metrics import compute_head_metrics
from rna_head_alignment.head2head.model import RiNALMoHeadToHeadStudent
from rna_head_alignment.probes.eval import evaluate_probe
from rna_head_alignment.probes.finetune import (
    ReplaySubsetDataset,
    _freeze_backbone_above_route,
    _unfreeze_route_adapters,
    replay_collate,
)
from rna_head_alignment.probes.task_config import TASK_TO_ROUTE_DEPTH, TASK_TO_TYPE
from rna_head_alignment.probes.train import _is_better, _metric_name_for_task


def _has_nan_inf(t: torch.Tensor) -> bool:
    return bool(torch.isnan(t).any() or torch.isinf(t).any())


def _sequence_loss(logits: torch.Tensor, labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    labels = labels.float() if num_classes == 2 else labels.long()
    if _has_nan_inf(logits) or _has_nan_inf(labels):
        raise RuntimeError(f"NaN/Inf inputs to sequence loss: logits={logits.isnan().any().item()}, labels={labels.isnan().any().item()}")
    if num_classes == 2:
        return F.binary_cross_entropy_with_logits(logits, labels)
    return F.cross_entropy(logits, labels)


def _token_loss(preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
    loss = F.mse_loss(preds, labels, reduction="none") * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def _pair_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    labels = torch.nan_to_num(labels, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def _task_loss(
    probe: torch.nn.Module,
    features: torch.Tensor,
    batch: dict[str, Any],
    task_type: str,
    num_classes: int | None,
) -> torch.Tensor:
    if _has_nan_inf(features):
        raise RuntimeError("NaN/Inf in task features")
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


class MultiTaskHead2HeadStudent(nn.Module):
    """Shared Stage 4B student with per-task probe heads.

    Parameters
    ----------
    student:
        The shared :class:`RiNALMoHeadToHeadStudent`.
    probes:
        Mapping task name -> probe module.
    """

    def __init__(
        self,
        student: RiNALMoHeadToHeadStudent,
        probes: dict[str, torch.nn.Module],
    ) -> None:
        super().__init__()
        self.student = student
        self.probes = nn.ModuleDict(probes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task: str,
        return_backbone: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Route-specific forward: only adapters up to the task route are used."""
        route_depth = TASK_TO_ROUTE_DEPTH[task]
        exits = self.student.backbone(input_ids, attention_mask)
        key_padding_mask = attention_mask == 0
        out: dict[str, torch.Tensor] = {}
        for route, specs in self.student.adapter_specs.items():
            if route > route_depth:
                continue
            head_outputs = exits[route]["head_outputs"]
            preds = []
            for k, (key, _head_idx) in enumerate(specs):
                preds.append(
                    self.student.adapters[key](head_outputs[:, k], key_padding_mask)["pred_map"]
                )
            out[f"pred_L{route}"] = torch.stack(preds, dim=1)
        features = exits[route_depth]["hidden"]
        out["features"] = features
        if return_backbone:
            out["backbone"] = exits
        return out


class TaskBalancedSampler:
    """Yield task names in a balanced round-robin order.

    If ``size_proportional`` is ``True``, tasks are sampled with probability
    proportional to their dataset size; otherwise strict round-robin.
    """

    def __init__(
        self,
        tasks: list[str],
        sizes: dict[str, int] | None = None,
        size_proportional: bool = False,
        seed: int = 42,
    ) -> None:
        self.tasks = list(tasks)
        self.size_proportional = size_proportional
        self.rng = np.random.default_rng(seed)
        if size_proportional and sizes:
            total = sum(sizes[t] for t in tasks)
            self.probs = np.array([sizes[t] / total for t in tasks])
        else:
            self.probs = np.ones(len(tasks)) / len(tasks)

    def __iter__(self) -> Iterator[str]:
        while True:
            if self.size_proportional:
                yield self.rng.choice(self.tasks, p=self.probs)
            else:
                for task in self.tasks:
                    yield task


class MultiTaskTrainer:
    """Train a multi-task joint model over all Stage 5 downstream tasks.

    Each training step samples one task, runs a single batch from that task's
    dataloader through the shared student, and updates both the task-specific
    probe and the shared route components.  Optional alignment replay batches
    are inserted every ``replay_every`` steps.
    """

    def __init__(
        self,
        multi_task_model: MultiTaskHead2HeadStudent,
        task_loaders: dict[str, DataLoader],
        task_num_classes: dict[str, int | None],
        device: torch.device,
        output_dir: str | Path,
        replay_loader: DataLoader | None = None,
        lambda_align: float = 0.1,
        backbone_lr: float = 2.0e-6,
        adapter_lr: float = 3.0e-5,
        probe_lr: float = 1.0e-4,
        max_steps: int = 10_000,
        eval_every: int = 500,
        save_every: int = 1_000,
        replay_every: int = 5,
        grad_clip: float = 1.0,
        mixed_precision: bool = False,
        size_proportional: bool = False,
        warmup_steps: int = 500,
        debug_steps: int = 10,
        eval_subset_size: int | None = None,
    ) -> None:
        self.model = multi_task_model
        self.task_loaders = task_loaders
        self.task_num_classes = task_num_classes
        self.device = device
        self.output_dir = Path(output_dir)
        self.replay_loader = replay_loader
        self.lambda_align = float(lambda_align)
        self.backbone_lr = backbone_lr
        self.adapter_lr = adapter_lr
        self.probe_lr = probe_lr
        self.max_steps = int(max_steps)
        self.eval_every = int(eval_every)
        self.save_every = int(save_every)
        self.replay_every = int(replay_every)
        self.grad_clip = grad_clip
        self.mixed_precision = mixed_precision
        self.amp_dtype = torch.bfloat16 if mixed_precision else torch.float32
        self.size_proportional = size_proportional
        self.warmup_steps = int(warmup_steps)
        self.debug_steps = int(debug_steps)
        self.eval_subset_size = int(eval_subset_size) if eval_subset_size else None

        self.history: list[dict[str, Any]] = []
        self.best_state: dict[str, Any] | None = None
        self.best_metric = -float("inf")

        self._setup_trainability()

    def _setup_trainability(self) -> None:
        # Maximum route depth across all tasks; unfreeze backbone up to that depth.
        max_route = max(TASK_TO_ROUTE_DEPTH[t] for t in self.task_loaders)
        _freeze_backbone_above_route(self.model.student, max_route)
        for depth in set(TASK_TO_ROUTE_DEPTH[t] for t in self.task_loaders):
            _unfreeze_route_adapters(self.model.student, depth)
        # Probes are always trainable.
        for p in self.model.probes.parameters():
            p.requires_grad_(True)

    def _parameter_groups(self) -> list[dict[str, Any]]:
        backbone_params: list[torch.nn.Parameter] = []
        adapter_params: list[torch.nn.Parameter] = []
        for name, p in self.model.student.named_parameters():
            if not p.requires_grad:
                continue
            if "adapters." in name:
                adapter_params.append(p)
            else:
                backbone_params.append(p)
        probe_params = [p for p in self.model.probes.parameters() if p.requires_grad]
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
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler | None,
        task: str,
        task_batch: dict[str, Any],
        replay_batch: dict[str, Any] | None,
    ) -> dict[str, float]:
        self.model.train()
        optimizer.zero_grad(set_to_none=True)

        use_amp = self.mixed_precision and self.device.type == "cuda"
        task_type = TASK_TO_TYPE[task]
        num_classes = self.task_num_classes.get(task)

        with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=use_amp):
            input_ids = task_batch["input_ids"].to(self.device)
            attention_mask = task_batch["attention_mask"].to(self.device)
            out = self.model(input_ids, attention_mask, task=task)
            features = out["features"]
            if _has_nan_inf(features):
                raise RuntimeError(f"[multitask-step] NaN/Inf in backbone features for task={task}")
            device_batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in task_batch.items()}
            task_loss = _task_loss(self.model.probes[task], features, device_batch, task_type, num_classes)
            if _has_nan_inf(task_loss):
                raise RuntimeError(f"[multitask-step] NaN/Inf task_loss for task={task}")

            total_loss = task_loss
            align_loss_val = 0.0
            if replay_batch is not None:
                align_loss = self._alignment_replay_loss(replay_batch)
                if _has_nan_inf(align_loss):
                    raise RuntimeError("[multitask-step] NaN/Inf alignment replay loss")
                total_loss = total_loss + self.lambda_align * align_loss
                align_loss_val = float(align_loss.detach().item())

        if use_amp and scaler is not None:
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self._all_trainable(), self.grad_clip)
            if _has_nan_inf(grad_norm):
                raise RuntimeError("[multitask-step] NaN/Inf gradient norm before step")
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self._all_trainable(), self.grad_clip)
            if _has_nan_inf(grad_norm):
                raise RuntimeError("[multitask-step] NaN/Inf gradient norm before step")
            optimizer.step()

        # Verify parameters did not become NaN.
        for name, p in self.model.named_parameters():
            if p.requires_grad and _has_nan_inf(p):
                raise RuntimeError(f"[multitask-step] NaN/Inf in parameter {name} after optimizer step")

        return {
            "task": task,
            "task_loss": float(task_loss.detach().item()),
            "align_loss": align_loss_val,
            "total_loss": float(total_loss.detach().item()),
            "grad_norm": float(grad_norm.item()) if torch.isfinite(grad_norm).all() else float("nan"),
        }

    def _all_trainable(self) -> list[torch.nn.Parameter]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def _alignment_replay_loss(self, replay_batch: dict[str, Any]) -> torch.Tensor:
        input_ids = replay_batch["input_ids"].to(self.device)
        attention_mask = replay_batch["attention_mask"].to(self.device)
        lengths = replay_batch["lengths"].to(self.device)
        out = self.model.student(input_ids, attention_mask, return_backbone=True)
        route_losses: dict[str, torch.Tensor] = {}
        for depth in (6, 8, 12):
            route_key = f"L{depth}"
            pred = out[f"pred_{route_key}"]
            teacher = replay_batch[f"teacher_{route_key}"].to(self.device)
            loss, _ = route_alignment_loss(
                pred,
                teacher,
                lengths,
                lambdas=(1.0, 1.0, 0.1),
                normalize_within_route=True,
            )
            route_losses[route_key] = loss
        return total_route_loss(route_losses)

    @torch.no_grad()
    def _evaluate(self, val_loaders: dict[str, DataLoader]) -> dict[str, Any]:
        from rna_head_alignment.probes.feature_extractor import ProbeFeatureExtractor

        self.model.eval()
        per_task: dict[str, dict[str, Any]] = {}
        for task, loader in val_loaders.items():
            extractor = ProbeFeatureExtractor(
                model=self.model.student,
                feature_mode="distilled_hidden",
                task=task,
                route_depth=TASK_TO_ROUTE_DEPTH[task],
            )
            extractor.to(self.device)
            eval_loader = loader
            if self.eval_subset_size is not None and hasattr(loader, "dataset"):
                n = min(self.eval_subset_size, len(loader.dataset))
                subset = torch.utils.data.Subset(loader.dataset, range(n))
                eval_loader = DataLoader(
                    subset,
                    batch_size=loader.batch_size or 8,
                    collate_fn=loader.collate_fn,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=False,
                )
            result = evaluate_probe(
                self.model.probes[task],
                extractor,
                eval_loader,
                task,
                self.device,
                num_classes=self.task_num_classes.get(task),
                mixed_precision=self.mixed_precision,
            )
            metric_name = _metric_name_for_task(TASK_TO_TYPE[task])
            per_task[task] = {
                "metrics": result["metrics"],
                "primary_metric": result["metrics"].get(metric_name, float("nan")),
            }
        return per_task

    def fit(self, val_loaders: dict[str, DataLoader]) -> dict[str, Any]:
        start = time.time()

        task_iters = {task: iter(loader) for task, loader in self.task_loaders.items()}
        sizes = {task: len(getattr(loader, "dataset", loader)) for task, loader in self.task_loaders.items()}
        sampler = TaskBalancedSampler(
            list(self.task_loaders.keys()),
            sizes=sizes,
            size_proportional=self.size_proportional,
        )
        task_stream = iter(sampler)

        replay_iter: Iterator | None = None
        if self.replay_loader is not None:
            replay_iter = iter(self.replay_loader)

        optimizer = AdamW(self._parameter_groups())
        scaler = torch.cuda.amp.GradScaler() if self.mixed_precision and self.device.type == "cuda" else None
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: min(1.0, (step + 1) / max(1, self.warmup_steps)),
        )

        step = 0
        best_overall = -float("inf")
        best_state: dict[str, Any] | None = None
        best_step = 0

        while step < self.max_steps:
            task = next(task_stream)
            try:
                task_batch = next(task_iters[task])
            except StopIteration:
                task_iters[task] = iter(self.task_loaders[task])
                task_batch = next(task_iters[task])

            replay_batch = None
            if replay_iter is not None and step % self.replay_every == (self.replay_every - 1):
                try:
                    replay_batch = next(replay_iter)
                except StopIteration:
                    replay_iter = iter(self.replay_loader)
                    replay_batch = next(replay_iter)

            loss_vals = self._train_step(optimizer, scaler, task, task_batch, replay_batch)
            scheduler.step()
            step += 1

            if step <= self.debug_steps:
                print(f"[multitask-debug] step={step} task={task} loss={loss_vals['task_loss']:.6f} total={loss_vals['total_loss']:.6f} grad_norm={loss_vals.get('grad_norm', float('nan')):.4f}", flush=True)

            if step % self.eval_every == 0 or step == self.max_steps:
                per_task = self._evaluate(val_loaders)
                primary_scores = [v["primary_metric"] for v in per_task.values() if not np.isnan(v["primary_metric"])]
                mean_primary = float(np.mean(primary_scores)) if primary_scores else float("nan")
                row = {"step": step, **loss_vals, "mean_val_primary": mean_primary}
                for task_name, res in per_task.items():
                    metric_name = _metric_name_for_task(TASK_TO_TYPE[task_name])
                    row[f"{task_name}_val_{metric_name}"] = res["primary_metric"]
                self.history.append(row)

                if _is_better("auroc", mean_primary, best_overall) or (
                    np.isnan(best_overall) and not np.isnan(mean_primary)
                ):
                    best_overall = mean_primary
                    best_step = step
                    best_state = {
                        "model": copy.deepcopy(self.model.state_dict()),
                        "step": step,
                    }

            if step % self.save_every == 0:
                self._save_checkpoint(step)

        if best_state is not None:
            self.model.load_state_dict(best_state["model"])

        summary = {
            "best_mean_val_primary": best_overall,
            "best_step": best_step,
            "trainable_params": sum(p.numel() for p in self._all_trainable()),
            "wall_time_sec": time.time() - start,
        }
        self._save_outputs(summary)
        return summary

    def _save_checkpoint(self, step: int) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"model_state_dict": self.model.state_dict(), "step": step},
            self.output_dir / f"checkpoint_step_{step}.pt",
        )

    def _save_outputs(self, summary: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "training_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        with open(self.output_dir / "training_history.json", "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2, default=str)
        torch.save(
            {"model_state_dict": self.model.state_dict(), "summary": summary},
            self.output_dir / "best_checkpoint.pt",
        )
