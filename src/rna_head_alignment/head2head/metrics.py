"""Per-head evaluation metrics for Stage 4B head-to-head alignment.

All metrics are computed only over the real-length region ``[:L_i, :L_i]`` of
each sequence in a padded batch.  Inputs are predicted and teacher attention
maps; teacher maps are row-normalized and prediction maps are softmax outputs,
so both can be treated as row-wise probability distributions where needed.

Every public function accepts either a single head pair ``[B, L, L]`` or, via
:func:`compute_head_metrics`, a batched route ``[B, T, L, L]`` and returns
per-head scalar tensors.  When ``per_sequence=True`` the single-head helpers
return ``[B]`` per-sequence values; the batched helper stores them under the
``{metric}_per_seq`` key as ``[B, T]``.
"""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

DEFAULT_METRICS = (
    "mse",
    "mae",
    "rowwise_kl",
    "cosine",
    "pearson",
    "spearman",
    "top_l_overlap",
    "top_2l_overlap",
)


def _pair_mask(lengths: torch.Tensor, size: int) -> torch.Tensor:
    """Return ``[B, L, L]`` bool mask: True where row and col are within length."""
    idx = torch.arange(size, device=lengths.device)
    valid = idx.unsqueeze(0) < lengths.unsqueeze(1)  # [B, L]
    return valid.unsqueeze(2) & valid.unsqueeze(1)  # [B, L, L]


def _normalize_rows(x: torch.Tensor, mask: torch.Tensor, eps: float) -> torch.Tensor:
    """Row-normalize ``x`` over the valid columns indicated by ``mask``."""
    x_pos = x.clamp_min(eps)
    norm = (x_pos * mask).sum(dim=-1, keepdim=True).clamp_min(eps)
    return x_pos / norm


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    per_sequence: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean squared error over valid pair positions."""
    se = (pred - target) ** 2
    if per_sequence:
        denom = mask.sum(dim=(-2, -1)).clamp_min(eps)
        return (se * mask).sum(dim=(-2, -1)) / denom
    return (se * mask).sum() / mask.sum().clamp_min(eps)


def masked_mae(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    per_sequence: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Mean absolute error over valid pair positions."""
    ae = (pred - target).abs()
    if per_sequence:
        denom = mask.sum(dim=(-2, -1)).clamp_min(eps)
        return (ae * mask).sum(dim=(-2, -1)) / denom
    return (ae * mask).sum() / mask.sum().clamp_min(eps)


def rowwise_kl(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    per_sequence: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Row-wise KL(target || pred) averaged over valid query rows.

    Both distributions are re-normalized over their valid columns to stay
    robust to numerical drift in the inputs.
    """
    B, L = pred.shape[:2]
    mask = _pair_mask(lengths, L)
    pred_row = _normalize_rows(pred, mask, eps)
    target_row = _normalize_rows(target, mask, eps)
    kl_elem = target_row * (torch.log(target_row + eps) - torch.log(pred_row + eps))
    kl_per_row = (kl_elem * mask).sum(dim=-1)  # [B, L]
    row_valid = mask.any(dim=-1)  # [B, L]
    if per_sequence:
        per_seq = kl_per_row.sum(dim=-1) / row_valid.sum(dim=-1).clamp_min(1.0)
        return per_seq
    if row_valid.any():
        return kl_per_row[row_valid].mean()
    return pred.sum() * 0.0


def masked_cosine_similarity(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    per_sequence: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Cosine similarity between flattened valid pair positions."""
    if per_sequence:
        sims = []
        for b in range(pred.shape[0]):
            m = mask[b]
            if not m.any():
                sims.append(torch.tensor(0.0, device=pred.device))
                continue
            p = pred[b][m]
            t = target[b][m]
            sims.append(F.cosine_similarity(p, t, dim=0).clamp(-1.0, 1.0))
        return torch.stack(sims)
    p = pred[mask]
    t = target[mask]
    if p.numel() == 0:
        return torch.tensor(0.0, device=pred.device)
    return F.cosine_similarity(p, t, dim=0).clamp(-1.0, 1.0)


def _rank_correlation(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    method: str,
    per_sequence: bool,
    eps: float,
) -> torch.Tensor:
    if per_sequence:
        rs = []
        for b in range(pred.shape[0]):
            m = mask[b]
            n_valid = int(m.sum().item())
            if n_valid < 2:
                rs.append(torch.tensor(float("nan"), device=pred.device))
                continue
            x = pred[b][m]
            y = target[b][m]
            if method == "spearman":
                # approximate ranks via argsort-of-argsort
                x = torch.argsort(torch.argsort(x)).float()
                y = torch.argsort(torch.argsort(y)).float()
            x = x - x.mean()
            y = y - y.mean()
            denom = ((x**2).sum().sqrt() * (y**2).sum().sqrt()).clamp_min(eps)
            rs.append((x * y).sum() / denom)
        return torch.stack(rs)
    p = pred[mask]
    t = target[mask]
    if p.numel() < 2:
        return torch.tensor(float("nan"), device=pred.device)
    if method == "spearman":
        p = torch.argsort(torch.argsort(p)).float()
        t = torch.argsort(torch.argsort(t)).float()
    p = p - p.mean()
    t = t - t.mean()
    denom = ((p**2).sum().sqrt() * (t**2).sum().sqrt()).clamp_min(eps)
    return (p * t).sum() / denom


def masked_pearson_r(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    per_sequence: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Pearson correlation over valid pair positions."""
    return _rank_correlation(pred, target, mask, "pearson", per_sequence, eps)


def masked_spearman_rho(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    per_sequence: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Spearman rank correlation over valid pair positions (approximated)."""
    return _rank_correlation(pred, target, mask, "spearman", per_sequence, eps)


def top_k_overlap(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    k: int | str,
    per_sequence: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Fraction of top-k target pairs that are also in top-k predicted pairs.

    ``k`` may be an integer or one of ``"L"`` / ``"2L"``.  For variable-length
    sequences the per-sequence ``k_i`` is capped at the number of valid pairs
    ``L_i * L_i``.
    """
    B, L = pred.shape[:2]
    mask = _pair_mask(lengths, L)
    flat_pred = pred.view(B, -1)
    flat_target = target.view(B, -1)
    flat_mask = mask.view(B, -1)
    n_valid = flat_mask.sum(dim=-1).long()  # [B]

    if k == "L":
        kk = lengths.clamp_max(n_valid).clamp_min(0)
    elif k == "2L":
        kk = (lengths * 2).clamp_max(n_valid).clamp_min(0)
    else:
        kk = torch.full((B,), int(k), device=pred.device).clamp_max(n_valid).clamp_min(0)

    pred_sorted = flat_pred.clone().masked_fill(flat_mask == 0, float("-inf"))
    target_sorted = flat_target.clone().masked_fill(flat_mask == 0, float("-inf"))

    kmax = int(kk.max().item()) if kk.numel() > 0 else 0
    if kmax <= 0:
        val = torch.zeros(B, device=pred.device)
        return val if per_sequence else val.mean()

    _, idx_pred = torch.topk(pred_sorted, kmax, dim=-1)  # [B, kmax]
    _, idx_tgt = torch.topk(target_sorted, kmax, dim=-1)  # [B, kmax]
    idx_pred = idx_pred.sort(dim=-1).values
    idx_tgt = idx_tgt.sort(dim=-1).values

    overlap = torch.zeros(B, device=pred.device)
    for i in range(kmax):
        pred_i = idx_pred[:, i]  # [B]
        exists = (idx_tgt == pred_i.unsqueeze(1)).any(dim=-1)  # [B]
        overlap += exists * (i < kk).float()
    overlap = overlap / kk.clamp_min(1).float()
    return overlap if per_sequence else overlap.mean()


def compute_head_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    lengths: torch.Tensor,
    metrics: Iterable[str] | None = None,
    include_per_sequence: bool = False,
    eps: float = 1e-8,
) -> dict[str, torch.Tensor]:
    """Compute all requested metrics for a route ``[B, T, L, L]``.

    Returns a dict mapping metric name to a ``[T]`` tensor of per-head scalar
    values.  When ``include_per_sequence=True`` an additional
    ``{metric}_per_seq`` entry with shape ``[B, T]`` is included.
    """
    metrics = tuple(metrics or DEFAULT_METRICS)
    B, T, L = pred.shape[:3]
    mask = _pair_mask(lengths, L).unsqueeze(1).expand(B, T, L, L)
    results: dict[str, torch.Tensor] = {}

    for metric in metrics:
        head_values = []
        per_seq_values = []
        for t in range(T):
            p = pred[:, t]
            tt = target[:, t]
            m = mask[:, t]
            if metric == "mse":
                v = masked_mse(p, tt, m, per_sequence=include_per_sequence, eps=eps)
            elif metric == "mae":
                v = masked_mae(p, tt, m, per_sequence=include_per_sequence, eps=eps)
            elif metric == "rowwise_kl":
                v = rowwise_kl(p, tt, lengths, per_sequence=include_per_sequence, eps=eps)
            elif metric == "cosine":
                v = masked_cosine_similarity(p, tt, m, per_sequence=include_per_sequence, eps=eps)
            elif metric == "pearson":
                v = masked_pearson_r(p, tt, m, per_sequence=include_per_sequence, eps=eps)
            elif metric == "spearman":
                v = masked_spearman_rho(p, tt, m, per_sequence=include_per_sequence, eps=eps)
            elif metric == "top_l_overlap":
                v = top_k_overlap(p, tt, lengths, "L", per_sequence=include_per_sequence, eps=eps)
            elif metric == "top_2l_overlap":
                v = top_k_overlap(p, tt, lengths, "2L", per_sequence=include_per_sequence, eps=eps)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            if include_per_sequence:
                per_seq_values.append(v)
                head_values.append(v.nanmean())
            else:
                head_values.append(v)
        results[metric] = torch.stack(head_values)
        if include_per_sequence:
            results[f"{metric}_per_seq"] = torch.stack(per_seq_values, dim=1)

    return results
