"""Single-task frozen probe trainer with early stopping and optimizer search."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from rna_head_alignment.probes.eval import evaluate_probe
from rna_head_alignment.probes.task_config import TASK_TO_TYPE


# Same optimizer search space for all feature modes.
OPTIMIZER_GRID: list[tuple[float, float]] = [
    (1.0e-4, 1.0e-5),
    (3.0e-4, 1.0e-5),
    (1.0e-3, 1.0e-5),
    (1.0e-4, 1.0e-4),
    (3.0e-4, 1.0e-4),
    (1.0e-3, 1.0e-4),
]


def _sequence_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    if num_classes == 2:
        return F.binary_cross_entropy_with_logits(logits, labels.float())
    return F.cross_entropy(logits, labels)


def _token_loss(
    preds: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    # Replace padded/invalid NaN labels so masked positions do not poison the
    # mean (NaN * 0 == NaN in PyTorch).
    labels = torch.nan_to_num(labels, nan=0.0)
    loss = F.mse_loss(preds, labels, reduction="none") * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def _pair_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * mask
    return loss.sum() / mask.sum().clamp_min(1.0)


def _metric_name_for_task(task_type: str) -> str:
    if task_type == "sequence_classification":
        return "auroc"
    if task_type == "token_regression":
        return "pearson"
    return "top_l_precision"


def _is_better(metric: str, new: float, old: float) -> bool:
    # All chosen metrics are higher-is-better.
    return not np.isnan(new) and new > old


class ProbeTrainer:
    """Train a frozen probe with a small built-in hyperparameter search.

    The pretrained feature extractor is kept frozen.  Only the task-specific
    gates, optional projection, and the probe head are trained.  For every
    ``(lr, weight_decay)`` candidate we train until convergence with early
    stopping and keep the best candidate according to the task primary metric.
    """

    def __init__(
        self,
        feature_extractor: torch.nn.Module,
        probe: torch.nn.Module,
        task: str,
        device: torch.device,
        output_dir: str | Path,
        max_epochs: int = 50,
        patience: int = 10,
        mixed_precision: bool = False,
        optimizer_grid: list[tuple[float, float]] | None = None,
        max_training_hours: float = 0.0,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.probe = probe
        self.task = task
        self.task_type = TASK_TO_TYPE[task]
        self.device = device
        self.output_dir = Path(output_dir)
        self.max_epochs = max_epochs
        self.patience = patience
        self.mixed_precision = mixed_precision
        self.optimizer_grid = optimizer_grid or OPTIMIZER_GRID
        self.metric_name = _metric_name_for_task(self.task_type)
        self.amp_dtype = torch.float16 if mixed_precision else torch.float32
        self.max_training_hours = float(max_training_hours)
        self._fit_start_time: float | None = None

        self.history: list[dict[str, Any]] = []

    def _time_exceeded(self) -> bool:
        if self.max_training_hours <= 0 or self._fit_start_time is None:
            return False
        elapsed_hours = (time.time() - self._fit_start_time) / 3600.0
        return elapsed_hours >= self.max_training_hours

    def _trainable_parameters(self) -> list[torch.nn.Parameter]:
        return [
            p for p in list(self.feature_extractor.parameters()) + list(self.probe.parameters())
            if p.requires_grad
        ]

    def _reset_trainable(self) -> None:
        """Re-initialize gates/projection/probe for a fresh grid trial."""
        if hasattr(self.feature_extractor, "reset_trainable_parameters"):
            self.feature_extractor.reset_trainable_parameters()

        def _init(m: torch.nn.Module) -> None:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.probe.apply(_init)

    def _train_epoch(self, loader: DataLoader) -> float:
        self.feature_extractor.train()
        self.probe.train()
        total_loss = 0.0
        total_samples = 0
        use_amp = self.mixed_precision and self.device.type == "cuda"

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            seq_ids = batch.get("seq_ids", [])

            self.optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=self.device.type, dtype=self.amp_dtype, enabled=use_amp
            ):
                features = self.feature_extractor(input_ids, attention_mask, seq_ids=seq_ids)

                if self.task_type == "sequence_classification":
                    logits = self.probe(features, mask=attention_mask)
                    loss = _sequence_loss(logits, batch["labels"].to(self.device), self.num_classes)
                elif self.task_type == "token_regression":
                    preds = self.probe(features)
                    loss = _token_loss(
                        preds,
                        batch["labels"].to(self.device),
                        batch["label_mask"].to(self.device),
                    )
                elif self.task_type == "pairwise":
                    logits = self.probe(features)
                    loss = _pair_loss(
                        logits,
                        batch["labels"].to(self.device),
                        batch["label_mask"].to(self.device),
                    )
                else:
                    raise ValueError(self.task_type)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._trainable_parameters(), 1.0)
            self.optimizer.step()

            n = input_ids.shape[0]
            total_loss += float(loss.detach().item()) * n
            total_samples += n

        return total_loss / max(total_samples, 1)

    def _run_one_config(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float,
        weight_decay: float,
    ) -> tuple[float, dict[str, Any]]:
        self._reset_trainable()
        self.optimizer = AdamW(
            self._trainable_parameters(), lr=lr, weight_decay=weight_decay
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.max_epochs)

        best_metric = -float("inf")
        best_state: dict[str, Any] = {}
        epochs_no_improve = 0
        train_log: list[dict[str, Any]] = []

        for epoch in range(self.max_epochs):
            if self._time_exceeded():
                print(
                    f"[probe] time budget exceeded after {epoch} epochs for lr={lr} wd={weight_decay}, stopping config early",
                    flush=True,
                )
                break
            train_loss = self._train_epoch(train_loader)
            val_result = evaluate_probe(
                self.probe,
                self.feature_extractor,
                val_loader,
                self.task,
                self.device,
                num_classes=self.num_classes,
                mixed_precision=self.mixed_precision,
            )
            val_metric = val_result["metrics"].get(self.metric_name, float("nan"))

            row = {
                "epoch": epoch + 1,
                "lr": float(self.optimizer.param_groups[0]["lr"]),
                "train_loss": train_loss,
                "val_metric": float(val_metric),
            }
            row.update({f"val_{k}": v for k, v in val_result["metrics"].items()})
            train_log.append(row)
            print(
                f"[probe] task={self.task} lr={lr} wd={weight_decay} epoch={epoch + 1}/{self.max_epochs} "
                f"train_loss={train_loss:.4f} val_metric={val_metric:.4f} best={best_metric:.4f}",
                flush=True,
            )

            if _is_better(self.metric_name, val_metric, best_metric):
                best_metric = val_metric
                best_state = {
                    "feature_extractor": copy.deepcopy(self.feature_extractor.state_dict()),
                    "probe": copy.deepcopy(self.probe.state_dict()),
                    "epoch": epoch + 1,
                }
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1

            self.scheduler.step()

            if epochs_no_improve >= self.patience:
                break

        return best_metric, {"lr": lr, "weight_decay": weight_decay, "log": train_log, "best_state": best_state}

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_classes: int | None = None,
    ) -> dict[str, Any]:
        """Run the optimizer grid and return the best model/state."""
        self.num_classes = num_classes
        self._fit_start_time = time.time()

        best_overall = -float("inf")
        best_cfg: dict[str, Any] | None = None
        best_state: dict[str, Any] | None = None

        for lr, wd in self.optimizer_grid:
            if self._time_exceeded():
                print(
                    f"[probe] time budget exceeded before lr={lr} wd={wd}; stopping grid search early",
                    flush=True,
                )
                break
            metric, info = self._run_one_config(train_loader, val_loader, lr, wd)
            info["best_val_metric"] = float(metric)
            self.history.append(info)
            if _is_better(self.metric_name, metric, best_overall):
                best_overall = metric
                best_cfg = {"lr": lr, "weight_decay": wd}
                best_state = info["best_state"]

        if best_state is None:
            raise RuntimeError("No successful training run found")

        self.feature_extractor.load_state_dict(best_state["feature_extractor"])
        self.probe.load_state_dict(best_state["probe"])

        summary = {
            "task": self.task,
            "metric_name": self.metric_name,
            "best_val_metric": best_overall,
            "best_cfg": best_cfg,
            "trainable_params": sum(p.numel() for p in self._trainable_parameters()),
            "wall_time_sec": time.time() - self._fit_start_time,
        }
        self._save_outputs(summary)
        return summary

    def _save_outputs(self, summary: dict[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "training_summary.json", "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, default=str)
        with open(self.output_dir / "training_history.json", "w", encoding="utf-8") as fh:
            json.dump(self.history, fh, indent=2, default=str)
        torch.save(
            {
                "feature_extractor_state_dict": self.feature_extractor.state_dict(),
                "probe_state_dict": self.probe.state_dict(),
                "summary": summary,
            },
            self.output_dir / "best_checkpoint.pt",
        )
