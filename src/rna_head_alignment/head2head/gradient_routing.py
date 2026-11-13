"""Gradient routing helpers for the three-exit head-to-head student.

The student forward runs all 12 backbone layers and produces three exits
(L6/L8/L12). Route losses are combined as ``w6*L6 + w8*L8 + w12*L12`` and a
single ``backward()`` accumulates gradients naturally: L6 touches backbone
layers 1-6, L8 layers 1-8, L12 layers 1-12 (1-based), so earlier layers receive
the sum of all routes that pass through them. Nothing is detached.
"""

from __future__ import annotations

import torch
from torch import nn

from .losses import route_alignment_loss, total_route_loss


def compute_route_losses(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    loss_fn=route_alignment_loss,
    normalize_within_route: bool = True,
) -> dict[str, tuple[torch.Tensor, list[dict[str, torch.Tensor]]]]:
    """Compute per-route losses from model outputs and a collated batch.

    ``pred`` carries ``pred_L6/pred_L8/pred_L12`` ``[B, T, L, L]``; ``batch``
    carries ``teacher_L6/...``, ``lengths`` and optionally
    ``head_availability_L6/...``. Returns ``{'L6': (loss, per_head), ...}``.
    """
    out: dict[str, tuple[torch.Tensor, list[dict[str, torch.Tensor]]]] = {}
    for route_key in ("L6", "L8", "L12"):
        if f"pred_{route_key}" not in pred or f"teacher_{route_key}" not in batch:
            continue
        kwargs: dict[str, Any] = {"normalize_within_route": normalize_within_route}
        avail_key = f"head_availability_{route_key}"
        if avail_key in batch:
            kwargs["head_availability"] = batch[avail_key]
        out[route_key] = loss_fn(
            pred[f"pred_{route_key}"], batch[f"teacher_{route_key}"], batch["lengths"], **kwargs
        )
    return out


def backward_routes(
    losses: dict[str, torch.Tensor],
    weights: dict[str, float] | None = None,
) -> torch.Tensor:
    """Combine route losses with weights and run a single backward pass."""
    total = total_route_loss(losses, weights)
    total.backward()
    return total


def layer_grad_norms(model: nn.Module) -> dict[int, float]:
    """Total grad norm per backbone layer (0-based layer index; -1 = embeddings,
    num_layers = final norm/pooler)."""
    backbone = getattr(model, "backbone", model)
    norms: dict[int, float] = {}
    for name, param in backbone.model.named_parameters():
        if name.startswith("encoder.layer."):
            layer_idx = int(name.split(".")[2])
        elif name.startswith("embeddings"):
            layer_idx = -1
        else:
            layer_idx = backbone.num_layers
        if param.grad is not None:
            norms[layer_idx] = norms.get(layer_idx, 0.0) + float(param.grad.norm() ** 2)
    return {k: v**0.5 for k, v in sorted(norms.items())}


def route_grad_norms(model: nn.Module) -> dict[str, float]:
    """Total grad norm of each route's adapters (keys 'L6'/'L8'/'L12')."""
    norms: dict[str, float] = {}
    for route, specs in model.adapter_specs.items():
        total = 0.0
        for key, _ in specs:
            for param in model.adapters[key].parameters():
                if param.grad is not None:
                    total += float(param.grad.norm() ** 2)
        norms[f"L{route}"] = total**0.5
    return norms


def gradient_diagnostics(model: nn.Module) -> dict[str, float]:
    """Return aggregate gradient statistics for logging and fail-fast checks.

    Includes total grad norm, per-route adapter grad norms, per-backbone-layer
    grad norms, the number of parameters with ``grad is None`` and the number
    of gradient elements that are NaN or Inf.
    """
    total_norm_sq = 0.0
    zero_grad_params = 0
    nan_inf_grad_elements = 0
    total_params = 0
    for p in model.parameters():
        n = p.numel()
        total_params += n
        if p.grad is None:
            zero_grad_params += n
            continue
        total_norm_sq += float(p.grad.norm() ** 2)
        finite = torch.isfinite(p.grad)
        nan_inf_grad_elements += int(n - finite.sum().item())

    diag: dict[str, float] = {
        "total_grad_norm": total_norm_sq**0.5,
        "zero_grad_params": float(zero_grad_params),
        "nan_inf_grad_params": float(nan_inf_grad_elements),
        "total_params": float(total_params),
    }
    diag.update(route_grad_norms(model))
    diag.update({f"layer_{k}_grad_norm": v for k, v in layer_grad_norms(model).items()})
    return diag


def assert_route_isolation(model: nn.Module, route: str | None = None) -> dict:
    """Test helper: report which backbone layers / adapter groups have nonzero grads.

    Intended to be called after zero_grad + backward of a single route loss.
    ``route`` is accepted for API symmetry but only documents intent; the
    function reports whatever grads are present on the model.

    Returns ``{'layers': {layer_idx: bool}, 'adapters': {'L6': bool, ...}}``
    where ``True`` means at least one parameter in the group has a nonzero grad.
    """
    backbone = getattr(model, "backbone", model)
    layers: dict[int, bool] = {}
    for name, param in backbone.model.named_parameters():
        if name.startswith("encoder.layer."):
            layer_idx = int(name.split(".")[2])
        elif name.startswith("embeddings"):
            layer_idx = -1
        else:
            layer_idx = backbone.num_layers
        nonzero = param.grad is not None and bool(param.grad.abs().sum() > 0)
        layers[layer_idx] = layers.get(layer_idx, False) or nonzero

    adapters: dict[str, bool] = {}
    if hasattr(model, "adapter_specs"):
        for r, specs in model.adapter_specs.items():
            any_grad = False
            for key, _ in specs:
                for param in model.adapters[key].parameters():
                    if param.grad is not None and bool(param.grad.abs().sum() > 0):
                        any_grad = True
            adapters[f"L{r}"] = any_grad
    return {"layers": layers, "adapters": adapters}
