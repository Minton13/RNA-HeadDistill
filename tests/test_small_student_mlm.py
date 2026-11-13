"""Test init_checkpoint loading + MLM aux on the small-student path (CPU, tiny config)."""

from __future__ import annotations

import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from rna_head_alignment.head2head.head_manifest import make_head_key  # noqa: E402
from rna_head_alignment.head2head.mlm import mask_tokens, mlm_loss_from_logits  # noqa: E402
from rna_head_alignment.head2head.student_backbone import (  # noqa: E402
    RiNALMoStudentBackbone,
    _TINY_CONFIG_DEFAULTS,
)
from rna_head_alignment.head2head.train_ddp import _make_student_full  # noqa: E402


def _write_pair_manifest(path) -> None:
    rows = []
    for route in (6, 8, 12):
        for k in range(5):
            rows.append(
                {
                    "task": f"manual_L{route}",
                    "route_depth": route,
                    "teacher_model": "t",
                    "teacher_layer": 0,
                    "teacher_head": k,
                    "teacher_cache_channel": k,
                    "student_model": "small",
                    "student_layer": route,
                    "student_head": k,
                    "matching_metric": "combined",
                    "matching_score": 0.0,
                    "pairing_method": "bipartite",
                    "keep": True,
                    "adapter_key": f"L{route}_H{k}",
                }
            )
    pd.DataFrame(rows).to_parquet(path, index=False)


def _route_head_keys():
    return {
        route: [make_head_key("t", 0, k, f"L{route}", f"manual_L{route}") for k in range(5)]
        for route in (6, 8, 12)
    }


def _config(init_checkpoint=None, mlm_weight=0.0, model_dir="assets/models/rinalmo-micro"):
    student_cfg = {
        "model_dir": str(model_dir),
        "config_overrides": dict(_TINY_CONFIG_DEFAULTS),
    }
    if init_checkpoint:
        student_cfg["init_checkpoint"] = str(init_checkpoint)
    return {
        "student": student_cfg,
        "model": {"adapter_dim": 16, "adapter_ffn_mult": 2, "train_backbone": True},
        "training": {"mlm_weight": mlm_weight},
    }


def test_init_checkpoint_loads_backbone(tmp_path, tiny_rinalmo_model_dir):
    pm = tmp_path / "pair.parquet"
    _write_pair_manifest(pm)

    # Save a "S0-like" checkpoint from a randomly initialised backbone.
    donor = RiNALMoStudentBackbone.from_config(model_dir=tiny_rinalmo_model_dir)
    ckpt_path = tmp_path / "s0.pt"
    torch.save(
        {"model_state_dict": {f"backbone.{k}": v for k, v in donor.state_dict().items()}},
        ckpt_path,
    )

    config = _config(init_checkpoint=ckpt_path, mlm_weight=1.0,
                     model_dir=tiny_rinalmo_model_dir)
    student, _ = _make_student_full(config, pm, _route_head_keys(), torch.device("cpu"))

    # Backbone weights equal the donor's.
    donor_state = donor.state_dict()
    for k, v in student.backbone.state_dict().items():
        assert torch.equal(v, donor_state[k]), f"mismatch at {k}"
    # MLM head attached.
    assert hasattr(student, "mlm_head")


def test_mlm_aux_forward_backward_all_params(tmp_path, tiny_rinalmo_model_dir):
    pm = tmp_path / "pair.parquet"
    _write_pair_manifest(pm)
    config = _config(mlm_weight=1.0, model_dir=tiny_rinalmo_model_dir)
    student, _ = _make_student_full(config, pm, _route_head_keys(), torch.device("cpu"))

    torch.manual_seed(0)
    input_ids = torch.randint(6, 26, (2, 10))
    attention_mask = torch.ones(2, 10, dtype=torch.long)

    # Alignment branch (unmasked): use the predicted L12 maps for a loss.
    pred = student(input_ids, attention_mask)
    assert "pred_L12" in pred and torch.is_tensor(pred["pred_L12"])
    align_loss = pred["pred_L12"].float().square().mean()

    # MLM branch (masked, separate forward).
    gen = torch.Generator().manual_seed(0)
    masked, labels = mask_tokens(input_ids, attention_mask, generator=gen)
    out = student.backbone.model(input_ids=masked, attention_mask=attention_mask, return_dict=True)
    mlm_loss = mlm_loss_from_logits(student.mlm_head(out.last_hidden_state), labels)

    total = align_loss + 1.0 * mlm_loss
    total.backward()

    # Backbone, adapters and MLM head all receive gradients.
    assert any(p.grad is not None for p in student.backbone.parameters())
    assert any(p.grad is not None for p in student.adapters.parameters())
    assert student.mlm_head.decoder.bias.grad is not None
