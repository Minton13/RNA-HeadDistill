"""Head-to-head alignment losses.

For every preserved teacher head::

    L_head = λ_kl * rowwise_KL(teacher || pred)
           + λ_mse * MSE
           + λ_cos * (1 - cosine)

computed only on the real-length region ``[:L_i, :L_i]`` of each sequence
(padding is excluded via a pair mask built from ``lengths`` — the mask is a
local implementation detail, never persisted). No symmetrization, no top-k BCE,
no min-separation, no diagonal exclusion. Teacher and predicted maps are
row-normalized over the *valid* positions inside ``[:L_i, :L_i]`` before the KL
term; MSE and cosine are computed on the raw attention values. KL uses
``eps=1e-8`` for stability.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_LAMBDAS = (1.0, 1.0, 0.1)


def _pair_mask(lengths: torch.Tensor, size: int) -> torch.Tensor:
    """[B, 1, L, L] bool mask: True where both positions are within length."""
    idx = torch.arange(size, device=lengths.device)
    valid = idx.unsqueeze(0) < lengths.unsqueeze(1)  # [B, L]
    return (valid.unsqueeze(2) & valid.unsqueeze(1)).unsqueeze(1)  # [B, 1, L, L]


def _normalize_rows(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-normalize ``x`` over the valid columns indicated by ``mask``.

    Teacher attention maps are row-normalized over the full cached square, but
    after masking to the real-length region ``[:L_i, :L_i]`` the row sum may no
    longer be 1.  KL therefore uses this renormalized distribution.
    """
    x_pos = x.clamp_min(eps)
    norm = (x_pos * mask).sum(dim=-1, keepdim=True).clamp_min(eps)
    return x_pos / norm


def head_alignment_loss(
    pred: torch.Tensor,
    teacher: torch.Tensor,
    lengths: torch.Tensor,
    lambdas: tuple[float, float, float] = DEFAULT_LAMBDAS,
    eps: float = 1e-8,
    sample_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Alignment loss for a single head.

    Parameters
    ----------
    pred, teacher:
        ``[B, L, L]`` attention maps (pred = softmax output, teacher =
        row-normalized cached map).
    lengths:
        ``[B]`` real sequence lengths; only ``[:L_i, :L_i]`` contributes.
    lambdas:
        ``(λ_kl, λ_mse, λ_cos)``.
    sample_mask:
        Optional ``[B]`` bool mask selecting which batch elements contribute to
        the loss.  Used to ignore heads that are unavailable for some sequences.

    Returns
    -------
    ``(total, components)`` with scalar tensors for ``kl``, ``mse``, ``cos`` and
    ``total``. The cosine term is computed per sequence on its flattened valid
    pairs, then averaged over the selected batch elements.
    """
    lam_kl, lam_mse, lam_cos = lambdas
    pred = pred.float()
    teacher = teacher.float()
    B = pred.shape[0]
    mask = _pair_mask(lengths, pred.shape[-1])  # [B, 1, L, L]
    mask2 = mask.squeeze(1)  # [B, L, L]

    if sample_mask is None:
        sample_mask = torch.ones(B, dtype=torch.bool, device=pred.device)
    else:
        sample_mask = sample_mask.to(device=pred.device, dtype=torch.bool)

    # Row-wise KL(teacher || pred), averaged over valid query rows.
    # Both distributions are re-normalized over the valid columns of each row
    # so the cropped [:L_i, :L_i] region remains a proper probability
    # distribution even if the original cache was normalized over Lmax.
    teacher_row = _normalize_rows(teacher, mask2, eps)
    pred_row = _normalize_rows(pred, mask2, eps)
    kl_elem = teacher_row * (torch.log(teacher_row + eps) - torch.log(pred_row + eps))
    row_mask = mask2.any(dim=-1) & sample_mask.unsqueeze(1)  # [B, L]
    kl_per_row = (kl_elem * mask2).sum(dim=-1)  # [B, L]
    kl = kl_per_row[row_mask].mean() if row_mask.any() else pred.sum() * 0.0

    # MSE over valid pairs, averaged per-sample then over selected samples.
    se = (pred - teacher) ** 2
    denom_per_sample = mask2.sum(dim=(-2, -1)).clamp_min(1)  # [B]
    mse_per_sample = (se * mask2).sum(dim=(-2, -1)) / denom_per_sample  # [B]
    mse = mse_per_sample[sample_mask].mean() if sample_mask.any() else pred.sum() * 0.0

    # Cosine per sequence on flattened valid pairs, then mean over selected.
    cos_terms = []
    for b in range(B):
        if not sample_mask[b]:
            continue
        m = mask2[b]
        if not m.any():
            continue
        p_flat = pred[b][m]
        t_flat = teacher[b][m]
        cos_terms.append(F.cosine_similarity(p_flat, t_flat, dim=0))
    if cos_terms:
        cos_sim = torch.stack(cos_terms).mean()
        cos_sim = cos_sim.clamp(-1.0, 1.0)
    else:
        cos_sim = pred.sum() * 0.0 + 1.0
    cos_loss = 1.0 - cos_sim

    total = lam_kl * kl + lam_mse * mse + lam_cos * cos_loss
    components = {"kl": kl, "mse": mse, "cos": cos_loss, "total": total}
    return total, components


def route_alignment_loss(
    pred: torch.Tensor,
    teacher: torch.Tensor,
    lengths: torch.Tensor,
    head_weights: list[float] | None = None,
    lambdas: tuple[float, float, float] = DEFAULT_LAMBDAS,
    head_availability: torch.Tensor | None = None,
    normalize_within_route: bool = True,
) -> tuple[torch.Tensor, list[dict[str, torch.Tensor]]]:
    """Alignment loss for one route over all its heads.

    ``pred``/``teacher``: ``[B, T, L, L]``. Returns ``(route_loss, per_head)``
    where ``route_loss`` is the (optionally weighted) mean or sum over heads and
    ``per_head`` holds each head's component dict.

    ``head_availability`` may be ``[B, T]`` or ``[T]`` bool.  When provided,
    unavailable heads are masked out on a per-sequence basis before averaging.
    """
    B, T = pred.shape[:2]
    if head_weights is None:
        head_weights = [1.0] * T
    if head_availability is not None:
        if head_availability.dim() == 1:
            head_availability = head_availability.unsqueeze(0).expand(B, -1)
        elif head_availability.shape != (B, T):
            raise ValueError(
                f"head_availability shape {head_availability.shape} does not match (B={B}, T={T})"
            )

    per_head: list[dict[str, torch.Tensor]] = []
    losses = []
    for t in range(T):
        mask_t = head_availability[:, t] if head_availability is not None else None
        loss_t, comps = head_alignment_loss(
            pred[:, t], teacher[:, t], lengths, lambdas, sample_mask=mask_t
        )
        per_head.append(comps)
        losses.append(head_weights[t] * loss_t)

    total = torch.stack(losses).sum()
    if normalize_within_route:
        total = total / max(sum(head_weights), 1e-12)
    return total, per_head


def total_route_loss(
    route_losses: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """``w6 * L6 + w8 * L8 + w12 * L12`` — one backward gives accumulated grads."""
    weights = weights or {"L6": 1.0, "L8": 1.0, "L12": 1.0}
    total = None
    for key, loss in route_losses.items():
        weighted = weights.get(key, 1.0) * loss
        total = weighted if total is None else total + weighted
    if total is None:
        raise ValueError("route_losses is empty")
    return total
