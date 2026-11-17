"""Shape tests for the Stage 4B head-to-head student (tiny CPU config)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("multimolecule")

from rna_head_alignment.head2head.model import RiNALMoHeadToHeadStudent
from rna_head_alignment.head2head.student_backbone import RiNALMoStudentBackbone


def _tiny_student(model_dir) -> RiNALMoHeadToHeadStudent:
    torch.manual_seed(0)
    backbone = RiNALMoStudentBackbone.from_config(model_dir=model_dir)
    return RiNALMoHeadToHeadStudent(backbone, adapter_dim=32)


def _batch() -> dict[str, torch.Tensor]:
    # Two sequences of different lengths (nucleotide-token space, vocab 26).
    input_ids = torch.tensor(
        [
            [6, 7, 8, 9, 6, 7, 0, 0],
            [8, 9, 6, 7, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = (input_ids != 0).long()
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_pred_shapes_and_row_normalization(tiny_rinalmo_model_dir):
    model = _tiny_student(tiny_rinalmo_model_dir).eval()
    batch = _batch()
    with torch.no_grad():
        out = model(batch["input_ids"], batch["attention_mask"])
    b, l = batch["input_ids"].shape
    for route in (6, 8, 12):
        pred = out[f"pred_L{route}"]
        assert pred.shape == (b, 5, l, l), f"pred_L{route} shape {tuple(pred.shape)}"
        row_sums = pred.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5), (
            f"pred_L{route} rows not normalized"
        )


def test_exit_hidden_and_head_output_shapes(tiny_rinalmo_model_dir):
    model = _tiny_student(tiny_rinalmo_model_dir).eval()
    batch = _batch()
    with torch.no_grad():
        out = model(batch["input_ids"], batch["attention_mask"], return_backbone=True)
    exits = out["backbone"]
    for route in (6, 8, 12):
        assert exits[route]["hidden"].shape == (2, 8, 48)
        assert exits[route]["head_outputs"].shape == (2, 5, 8, 8)  # [B, K, L, head_dim]


@pytest.mark.parametrize("tuple_output", [False, True])
def test_layer_hook_preserves_batch_and_gradients(tuple_output):
    backbone = RiNALMoStudentBackbone.__new__(RiNALMoStudentBackbone)
    torch.nn.Module.__init__(backbone)
    backbone._captured = {}
    hidden = torch.randn(2, 8, 48, requires_grad=True)
    output = (hidden,) if tuple_output else hidden

    backbone._make_layer_hook(6)(torch.nn.Identity(), (), output)

    captured = backbone._captured[6]["hidden"]
    assert captured.shape == (2, 8, 48)
    captured.sum().backward()
    assert torch.equal(hidden.grad, torch.ones_like(hidden))


def test_tiny_parameter_count(tiny_rinalmo_model_dir):
    model = _tiny_student(tiny_rinalmo_model_dir)
    counts = model.count_parameters()
    assert counts["total"] < 5_000_000, counts
    assert counts["backbone"] > 0 and counts["adapters"] > 0
    groups = model.route_parameter_groups()
    for route in ("L6", "L8", "L12"):
        assert len(groups[route]) > 0
    assert sorted(groups["backbone"].keys()) == [-1, *range(12), 12]
