import json

import pandas as pd
import pytest

from rna_head_alignment.data.task_loaders import load_task_samples


def test_task_loaders_pairwise_and_seqcls(tmp_path):
    pair_path = tmp_path / "secondary.jsonl"
    pair_path.write_text(
        json.dumps({"seq_id": "x", "sequence": "GGGAAACCC", "pairs": [[0, 8], [1, 7]], "split": "train"})
        + "\n",
        encoding="utf-8",
    )
    pair_samples = load_task_samples(
        "secondary_structure_pair",
        {"task_type": "pairwise", "path": str(pair_path)},
        limit=5,
    )
    assert pair_samples[0].pair_matrix.shape == (9, 9)
    assert pair_samples[0].pair_matrix[0, 8] == 1

    csv_path = tmp_path / "seqcls.csv"
    pd.DataFrame({"sequence": ["AUGC", "UUUU"], "label": [0, 1], "split": ["train", "val"]}).to_csv(csv_path, index=False)
    seq_samples = load_task_samples(
        "rbp_binding_seqcls",
        {"task_type": "sequence_classification", "path": str(csv_path)},
        limit=5,
    )
    assert len(seq_samples) == 2
    assert seq_samples[1].label == 1


def test_task_loaders_token_regression_json_and_sequence_regression(tmp_path):
    token_path = tmp_path / "chemical.json"
    token_path.write_text(
        json.dumps(
            [
                {
                    "id": "chem_0",
                    "sequence": "AUGCU",
                    "reactivity": [0.1, None, 0.3, 0.4, 0.5],
                    "split": "train",
                }
            ]
        ),
        encoding="utf-8",
    )
    token_samples = load_task_samples(
        "chemical_reactivity_rnascope",
        {"task_type": "token_regression", "path": str(token_path), "target_column": "reactivity"},
        limit=5,
    )
    assert token_samples[0].y_token.shape == (5,)
    assert token_samples[0].y_token[0] == pytest.approx(0.1)
    assert pd.isna(token_samples[0].y_token[1])

    reg_path = tmp_path / "mrna_stability.csv"
    pd.DataFrame({"seq": ["AUGC", "UUUU"], "y": [0.25, -0.5], "split": ["train", "test"]}).to_csv(
        reg_path, index=False
    )
    reg_samples = load_task_samples(
        "mrna_stability",
        {"task_type": "sequence_regression", "path": str(reg_path), "sequence_column": "seq", "target_column": "y"},
        limit=5,
    )
    assert len(reg_samples) == 2
    assert reg_samples[0].label == 0.25
