from __future__ import annotations

import torch
import torch.nn.functional as F


def valid_pair_mask(valid_mask: torch.Tensor) -> torch.Tensor:
    return valid_mask[:, :, None].bool() & valid_mask[:, None, :].bool()


def row_normalize(attn: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    attn = attn.clamp_min(0)
    return attn / attn.sum(dim=-1, keepdim=True).clamp_min(eps)


def mask_normalized_mse(student: torch.Tensor, teacher: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_pair_mask(valid_mask).to(student.device)
    loss = (row_normalize(student) - row_normalize(teacher.to(student.device))).pow(2)
    while mask.ndim < loss.ndim:
        mask = mask.unsqueeze(1)
    denom = mask.sum().clamp_min(1)
    return (loss * mask).sum() / denom


def mask_normalized_kl(student: torch.Tensor, teacher: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_pair_mask(valid_mask).to(student.device)
    s = row_normalize(student).clamp_min(1e-8).log()
    t = row_normalize(teacher.to(student.device)).clamp_min(1e-8)
    loss = F.kl_div(s, t, reduction="none").sum(dim=-1)
    while mask.ndim - 1 < loss.ndim:
        mask = mask.unsqueeze(1)
    row_mask = valid_mask[:, None, :].to(student.device).bool()
    denom = row_mask.sum().clamp_min(1)
    return (loss * row_mask).sum() / denom


def mask_normalized_cosine(student: torch.Tensor, teacher: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    mask = valid_pair_mask(valid_mask).to(student.device)
    while mask.ndim < student.ndim:
        mask = mask.unsqueeze(1)
    s = row_normalize(student) * mask
    t = row_normalize(teacher.to(student.device)) * mask
    sim = F.cosine_similarity(s.flatten(2), t.flatten(2), dim=-1)
    return (1.0 - sim).mean()
