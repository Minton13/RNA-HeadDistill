"""Task metrics for frozen downstream probes."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)


def _to_numpy(x: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def sequence_classification_metrics(
    y_true: torch.Tensor | np.ndarray,
    y_score: torch.Tensor | np.ndarray,
    num_classes: int,
) -> dict[str, float]:
    """AUROC, AUPRC, F1, MCC for sequence classification.

    For binary tasks ``y_score`` is a 1-D logit/probability.  For multiclass
    tasks it is ``[N, C]`` logits/probabilities.
    """
    y_true = _to_numpy(y_true)
    y_score = _to_numpy(y_score)

    out: dict[str, float] = {}
    try:
        if num_classes == 2:
            probs = 1.0 / (1.0 + np.exp(-y_score))
            out["auroc"] = float(roc_auc_score(y_true, probs))
            out["auprc"] = float(average_precision_score(y_true, probs))
            y_pred = (y_score >= 0).astype(int)
            out["f1"] = float(f1_score(y_true, y_pred))
            out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
        else:
            # One-vs-rest AUROC / AUPRC averaged over classes present in y_true.
            y_score = np.nan_to_num(y_score, nan=0.0, posinf=1e6, neginf=-1e6)
            from scipy.special import softmax

            probs = softmax(y_score, axis=1)
            present_labels = sorted(np.unique(y_true).tolist())

            # Compute per-class binary metrics only for classes that appear in
            # this split. This avoids sklearn errors when the validation split
            # does not contain every class.
            auroc_vals: list[float] = []
            auprc_vals: list[float] = []
            for lab in present_labels:
                binary_true = (y_true == lab).astype(int)
                binary_score = probs[:, lab]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        auroc_vals.append(float(roc_auc_score(binary_true, binary_score)))
                    except Exception:
                        pass
                    try:
                        auprc_vals.append(float(average_precision_score(binary_true, binary_score)))
                    except Exception:
                        pass

            out["auroc"] = float(np.mean(auroc_vals)) if auroc_vals else float("nan")
            out["auprc"] = float(np.mean(auprc_vals)) if auprc_vals else float("nan")

            y_pred = y_score.argmax(axis=1)
            out["f1"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    except Exception as exc:
        out["metric_error"] = str(exc)
        for k in ("auroc", "auprc", "f1", "mcc"):
            out.setdefault(k, float("nan"))
    return out


def token_regression_metrics(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Pearson, Spearman, MSE, MAE over masked token positions."""
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    mask = _to_numpy(mask).astype(bool)

    valid = mask.ravel()
    yt = y_true.ravel()[valid]
    yp = y_pred.ravel()[valid]
    out: dict[str, float] = {}
    if len(yt) < 2:
        out.update({"pearson": float("nan"), "spearman": float("nan"), "mse": float("nan"), "mae": float("nan")})
        return out
    out["mse"] = float(np.mean((yt - yp) ** 2))
    out["mae"] = float(np.mean(np.abs(yt - yp)))
    try:
        out["pearson"] = float(np.corrcoef(yt, yp)[0, 1])
    except Exception:
        out["pearson"] = float("nan")
    try:
        out["spearman"] = float(spearmanr(yt, yp)[0])
    except Exception:
        out["spearman"] = float("nan")
    return out


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
    # Exclude the diagonal.
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


def pairwise_metrics(
    y_true: torch.Tensor | np.ndarray,
    y_pred: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    lengths: torch.Tensor | np.ndarray | None = None,
    precomputed_top_l: list[float] | None = None,
    precomputed_top_2l: list[float] | None = None,
) -> dict[str, float]:
    """Top-L / top-2L precision, F1, MCC, AUPRC for pairwise contact maps.

    ``y_pred`` should be logits; they are converted to probabilities with a
    sigmoid.  ``mask`` selects the valid pair positions (length mask and
    minimum separation).

    If ``precomputed_top_l`` / ``precomputed_top_2l`` are provided they are
    used directly; otherwise they are computed from the full 3D arrays.
    """
    y_true = _to_numpy(y_true)
    y_pred = _to_numpy(y_pred)
    mask = _to_numpy(mask).astype(bool)
    probs = 1.0 / (1.0 + np.exp(-y_pred))

    out: dict[str, float] = {}
    flat_true = y_true[mask]
    flat_probs = probs[mask]
    flat_pred = (flat_probs >= 0.5).astype(int)

    try:
        out["auprc"] = float(average_precision_score(flat_true, flat_probs))
    except Exception:
        out["auprc"] = float("nan")
    try:
        out["f1"] = float(f1_score(flat_true, flat_pred, zero_division=0))
    except Exception:
        out["f1"] = float("nan")
    try:
        out["mcc"] = float(matthews_corrcoef(flat_true, flat_pred))
    except Exception:
        out["mcc"] = float("nan")

    # Top-L / top-2L precision.
    if precomputed_top_l is not None:
        out["top_l_precision"] = float(np.nanmean(precomputed_top_l)) if precomputed_top_l else float("nan")
        out["top_2l_precision"] = float(np.nanmean(precomputed_top_2l)) if precomputed_top_2l else float("nan")
        return out

    if lengths is None:
        lengths = [int(mask[b].any(axis=1).sum()) for b in range(mask.shape[0])]
    lengths = _to_numpy(lengths)

    pred_t = torch.as_tensor(probs, dtype=torch.float32)
    target_t = torch.as_tensor(y_true, dtype=torch.float32)
    top_l_vals: list[float] = []
    top_2l_vals: list[float] = []
    for b in range(pred_t.shape[0]):
        L = int(lengths[b])
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
    out["top_l_precision"] = float(np.nanmean(top_l_vals)) if top_l_vals else float("nan")
    out["top_2l_precision"] = float(np.nanmean(top_2l_vals)) if top_2l_vals else float("nan")
    return out


def pairwise_metrics_streaming(
    flat_true: np.ndarray,
    flat_probs: np.ndarray,
    top_l_vals: list[float],
    top_2l_vals: list[float],
) -> dict[str, float]:
    """Compute pairwise metrics from pre-aggregated flat arrays and top-k lists.

    This avoids materialising padded ``[N, Lmax, Lmax]`` arrays during eval.
    """
    out: dict[str, float] = {}
    flat_pred = (flat_probs >= 0.5).astype(int)
    try:
        out["auprc"] = float(average_precision_score(flat_true, flat_probs))
    except Exception:
        out["auprc"] = float("nan")
    try:
        out["f1"] = float(f1_score(flat_true, flat_pred, zero_division=0))
    except Exception:
        out["f1"] = float("nan")
    try:
        out["mcc"] = float(matthews_corrcoef(flat_true, flat_pred))
    except Exception:
        out["mcc"] = float("nan")
    out["top_l_precision"] = float(np.nanmean(top_l_vals)) if top_l_vals else float("nan")
    out["top_2l_precision"] = float(np.nanmean(top_2l_vals)) if top_2l_vals else float("nan")
    return out
