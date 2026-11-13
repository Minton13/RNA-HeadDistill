from rna_head_alignment.models.loaders import verify_one_model
from rna_head_alignment.models.registry import ModelSpec


def test_rinalmo_micro_loading(tiny_rinalmo_model_dir):
    """Load + forward + attention extraction on a locally staged snapshot.

    Uses the tiny random-weight snapshot built by the fixture (the real
    checkpoint is not redistributed with the public repo).
    """
    spec = ModelSpec("rinalmo_micro", "RNA", tiny_rinalmo_model_dir,
                     attention_teacher=True)
    row = verify_one_model(spec, device="cpu", max_len=32)
    assert row["load_ok"], row["error"]
    assert row["forward_ok"], row["error"]
    assert row["has_attentions"], row["error"]
