"""Frozen probe evaluator."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from rna_head_alignment.probes.feature_extractor import ProbeFeatureExtractor
from rna_head_alignment.probes.metrics import (
    pairwise_metrics,
    pairwise_metrics_streaming,
    sequence_classification_metrics,
    token_regression_metrics,
)
from rna_head_alignment.probes.task_config import TASK_TO_TYPE


def _top_k_overlap_per_sample(
    pred: torch.Tensor,
    target: torch.Tensor,
    length: int,
    k: int,
) -> float:
    """Fraction of top-k target pairs that are also in top-k predicted pairs."""
    pred = pred[:length, :length]
    target = target[:length, :length]
    valid_mask = torch.ones(length, length, dtype=torch.bool, device=pred.device)
    diag = torch.eye(length, dtype=torch.bool, device=pred.device)
    valid_mask = valid_mask & ~diag
    pred = pred.masked_fill(~valid_mask, float("-inf"))
    target = target.masked_fill(~valid_mask, float("-inf"))
    flat_pred = pred.view(-1)
    flat_target = target.view(-1)
    n_valid = int(valid_mask.sum().item())
    kk = min(k, n_valid)
    if kk <= 0:
        return float("nan")
    _, idx_pred = torch.topk(flat_pred, kk)
    _, idx_tgt = torch.topk(flat_target, kk)
    idx_pred = idx_pred.sort().values
    idx_tgt = idx_tgt.sort().values
    overlap = (idx_tgt.unsqueeze(0) == idx_pred.unsqueeze(1)).any(dim=1).sum().item()
    return overlap / kk


@torch.no_grad()
def evaluate_probe(
    probe: torch.nn.Module,
    feature_extractor: ProbeFeatureExtractor,
    loader: DataLoader,
    task: str,
    device: torch.device,
    num_classes: int | None = None,
    mixed_precision: bool = False,
    return_predictions: bool = False,
) -> dict[str, Any]:
    """Evaluate a frozen probe on a split and return metrics + predictions.

    Parameters
    ----------
    probe:
        Task-specific probe head.
    feature_extractor:
        Frozen feature extractor (P0-P6).
    loader:
        DataLoader for the split.
    task:
        Downstream task name.
    device:
        torch device.
    num_classes:
        Required for sequence classification tasks.
    mixed_precision:
        Use AMP on CUDA.
    return_predictions:
        If True, also return padded predictions/labels.  This increases memory
        use and is disabled by default.

    Returns
    -------
    Dict with ``metrics`` and optionally ``predictions``/``labels`` numpy arrays.
    """
    feature_extractor.eval()
    probe.eval()
    task_type = TASK_TO_TYPE[task]

    amp_dtype = torch.float16 if mixed_precision else torch.float32
    use_amp = mixed_precision and device.type == "cuda"

    if task_type == "sequence_classification":
        all_logits: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            seq_ids = batch.get("seq_ids", [])
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                features = feature_extractor(input_ids, attention_mask, seq_ids=seq_ids)
                logits = probe(features, mask=attention_mask)
            all_logits.append(logits.detach().cpu().numpy())
            all_labels.append(batch["labels"].numpy())
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        metrics = sequence_classification_metrics(labels, logits, num_classes or 2)
        out: dict[str, Any] = {"metrics": metrics}
        if return_predictions:
            out["predictions"] = logits
            out["labels"] = labels
        return out

    if task_type == "token_regression":
        flat_preds_list: list[np.ndarray] = []
        flat_labels_list: list[np.ndarray] = []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            seq_ids = batch.get("seq_ids", [])
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                features = feature_extractor(input_ids, attention_mask, seq_ids=seq_ids)
                preds = probe(features)
            mask = batch["label_mask"].to(device)
            valid = mask.detach().cpu().numpy().astype(bool)
            flat_preds_list.append(preds.detach().cpu().numpy()[valid])
            flat_labels_list.append(batch["labels"].numpy()[valid])
        flat_preds = np.concatenate(flat_preds_list, axis=0)
        flat_labels = np.concatenate(flat_labels_list, axis=0)
        # Dummy full-array args for the existing metrics API.
        metrics = token_regression_metrics(
            flat_labels,
            flat_preds,
            np.ones_like(flat_labels, dtype=bool),
        )
        out = {"metrics": metrics}
        if return_predictions:
            out["predictions"] = flat_preds
            out["labels"] = flat_labels
        return out

    if task_type == "pairwise":
        flat_true_list: list[np.ndarray] = []
        flat_probs_list: list[np.ndarray] = []
        top_l_vals: list[float] = []
        top_2l_vals: list[float] = []
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            seq_ids = batch.get("seq_ids", [])
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                features = feature_extractor(input_ids, attention_mask, seq_ids=seq_ids)
                logits = probe(features)
            mask = batch["label_mask"].to(device)
            labels = batch["labels"].to(device)
            lengths = batch["lengths"].to(device)
            probs = torch.sigmoid(logits)

            valid = mask.detach().cpu().numpy().astype(bool)
            flat_true_list.append(labels.detach().cpu().numpy()[valid])
            flat_probs_list.append(probs.detach().cpu().numpy()[valid])

            pred_t = probs
            target_t = labels
            for b in range(pred_t.shape[0]):
                L = int(lengths[b].item())
                if L <= 1:
                    continue
                try:
                    top_l_vals.append(
                        _top_k_overlap_per_sample(pred_t[b], target_t[b], L, k=L)
                    )
                    top_2l_vals.append(
                        _top_k_overlap_per_sample(pred_t[b], target_t[b], L, k=2 * L)
                    )
                except Exception:
                    continue

        flat_true = np.concatenate(flat_true_list, axis=0)
        flat_probs = np.concatenate(flat_probs_list, axis=0)
        metrics = pairwise_metrics_streaming(flat_true, flat_probs, top_l_vals, top_2l_vals)
        out = {"metrics": metrics}
        if return_predictions:
            out["predictions"] = flat_probs
            out["labels"] = flat_true
        return out

    raise ValueError(f"Unsupported task type: {task_type}")


def write_eval_outputs(
    output_dir: str | Path,
    split: str,
    result: dict[str, Any],
) -> Path:
    """Write ``eval_{split}.json`` and optionally ``predictions_{split}.npz``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"eval_{split}.json"
    metrics_path.write_text(
        json.dumps(result["metrics"], indent=2, default=str), encoding="utf-8"
    )
    if "predictions" in result and "labels" in result:
        np.savez(
            output_dir / f"predictions_{split}.npz",
            predictions=result["predictions"],
            labels=result["labels"],
        )
    return metrics_path
