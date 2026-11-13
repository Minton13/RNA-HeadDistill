"""Gradient-routing tests: route losses must only update the layers below their exit.

L6 loss -> backbone layers 1-6 (0-based 0-5) + L6 adapters;
L8 loss -> layers 1-8; L12 loss -> all 12. The weighted total
``w6*L6 + w8*L8 + w12*L12`` backward accumulates: layers 0-5 get strictly
larger grads than from L12 alone, while layers 8-11 equal the L12-only
contribution.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("multimolecule")

from rna_head_alignment.head2head.gradient_routing import (
    assert_route_isolation,
    backward_routes,
    compute_route_losses,
    layer_grad_norms,
    route_grad_norms,
)
from rna_head_alignment.head2head.model import RiNALMoHeadToHeadStudent
from rna_head_alignment.head2head.student_backbone import RiNALMoStudentBackbone

N_LAYERS = 12


def _make_model(model_dir) -> RiNALMoHeadToHeadStudent:
    torch.manual_seed(0)
    backbone = RiNALMoStudentBackbone.from_config(model_dir=model_dir)
    return RiNALMoHeadToHeadStudent(backbone, adapter_dim=32).train()


def _make_batch() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(7)
    input_ids = torch.tensor(
        [[6, 7, 8, 9, 6, 7, 0, 0], [8, 9, 6, 7, 0, 0, 0, 0]], dtype=torch.long
    )
    attention_mask = (input_ids != 0).long()
    lengths = attention_mask.sum(dim=1)
    b, l = input_ids.shape
    batch: dict[str, torch.Tensor] = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "lengths": lengths,
    }
    for route in ("L6", "L8", "L12"):
        teacher = torch.rand(b, 5, l, l, generator=g)
        teacher = teacher / teacher.sum(dim=-1, keepdim=True)
        # zero out the padding region (collate would do the same)
        pair = (torch.arange(l).unsqueeze(0) < lengths.unsqueeze(1))
        pair = pair.unsqueeze(2) & pair.unsqueeze(1)
        batch[f"teacher_{route}"] = teacher * pair.unsqueeze(1)
    return batch


def _layer_param_grads(model: RiNALMoHeadToHeadStudent) -> dict[int, torch.Tensor]:
    """Sum of |grad| per backbone encoder layer (attention/MLP trunk params)."""
    out: dict[int, torch.Tensor] = {}
    for name, param in model.backbone.model.named_parameters():
        if not name.startswith("encoder.layer."):
            continue
        idx = int(name.split(".")[2])
        grad = param.grad
        val = torch.zeros(()) if grad is None else grad.abs().sum().detach()
        out[idx] = out.get(idx, torch.zeros(())) + val
    return out


def _backward(model: RiNALMoHeadToHeadStudent, batch: dict, which: str) -> torch.Tensor:
    model.zero_grad(set_to_none=True)
    out = model(batch["input_ids"], batch["attention_mask"])
    losses = {k: v[0] for k, v in compute_route_losses(out, batch).items()}
    if which == "total":
        return backward_routes(losses)
    losses[which].backward()
    return losses[which]


@pytest.mark.parametrize(
    "route,expect_nonzero,expect_zero",
    [
        ("L6", range(0, 6), range(6, 12)),
        ("L8", range(0, 8), range(8, 12)),
        ("L12", range(0, 12), range(0, 0)),
    ],
)
def test_route_isolation(route, expect_nonzero, expect_zero, tiny_rinalmo_model_dir):
    model = _make_model(tiny_rinalmo_model_dir)
    batch = _make_batch()
    _backward(model, batch, route)
    report = assert_route_isolation(model, route)
    for idx in expect_nonzero:
        assert report["layers"][idx], f"{route}: layer {idx} expected nonzero grad"
    for idx in expect_zero:
        assert not report["layers"].get(idx, False), f"{route}: layer {idx} must have zero grad"
    assert report["adapters"][route]
    for other in ("L6", "L8", "L12"):
        if other != route:
            assert not report["adapters"][other], f"{route}: {other} adapters must stay clean"


def test_total_backward_accumulates(tiny_rinalmo_model_dir):
    batch = _make_batch()
    # L12-only reference.
    model_ref = _make_model(tiny_rinalmo_model_dir)
    _backward(model_ref, batch, "L12")
    grads_ref = _layer_param_grads(model_ref)
    norms_ref = layer_grad_norms(model_ref)
    assert route_grad_norms(model_ref)["L12"] > 0

    # Weighted total (all weights 1).
    model = _make_model(tiny_rinalmo_model_dir)
    _backward(model, batch, "total")
    grads_total = _layer_param_grads(model)

    # Layers 0-5 carry L6+L8+L12: strictly more gradient than L12 alone.
    for idx in range(0, 6):
        assert grads_total[idx] > grads_ref[idx] + 1e-6, (
            f"layer {idx}: total {grads_total[idx]:.4e} not > L12-only {grads_ref[idx]:.4e}"
        )
    # Layers 8-11 are only reachable from the L12 exit: total == L12-only.
    for idx in range(8, 12):
        assert torch.allclose(grads_total[idx], grads_ref[idx], rtol=1e-4, atol=1e-8), (
            f"layer {idx}: total {grads_total[idx]:.6e} != L12-only {grads_ref[idx]:.6e}"
        )
    # L12 reaches embeddings (-1) and all encoder layers; pooler/final norm
    # sit after the captured exit and stay grad-free.
    assert set(norms_ref) == {-1, *range(12)}
