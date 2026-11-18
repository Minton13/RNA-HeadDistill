"""Regression tests for the v1.0.1 screening fixes in head_scan.py.

Background: v1.0.0 shipped ``attn_np = bundle.attentions.numpy()`` which
raises TypeError on CUDA tensors; the per-sample ``except Exception:
continue`` in screen_heads then silently skipped every sample, so any GPU
screening run produced an empty atlas (observed: Slurm job 65391, 37/37
models -> ``no_attention_or_forward_failed``). v1.0.1 converts with
``.detach().cpu().numpy()`` instead.

These tests also cover the new optional ``split_filter`` / ``post_filter_limit``
config keys (used by the train-only secondary re-screening, job 65397) and
verify the legacy default path is unchanged.

Runs fully offline on CPU in <1 min. The CUDA variant skips when no GPU is
available.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

import rna_head_alignment.screening.head_scan as hs
from rna_head_alignment.data.task_loaders import TaskSample

L = 8
N_LAYERS, N_HEADS = 2, 2


def _sample(i: int, split: str) -> TaskSample:
    rng = np.random.default_rng(i)
    pair = (rng.random((L, L)) > 0.75).astype(np.float32)
    pair = np.triu(pair + pair.T).astype(np.float32)
    return TaskSample(
        seq_id=f"s{i}", sequence="AUGCAUGC", task="toy_pair",
        task_type="pairwise", split=split,
        pair_matrix=pair, valid_pair_mask=np.ones((L, L), dtype=bool),
    )


class _CudaLikeTensor:
    """Wrapper simulating a CUDA tensor: plain .numpy() raises TypeError.

    The v1.0.0 code path (bundle.attentions.numpy()) dies on this object;
    the fixed path (.detach().cpu().numpy()) works.
    """

    def __init__(self, t: torch.Tensor):
        self._t = t

    def numpy(self):
        raise TypeError("can't convert cuda:0 device type tensor to numpy. "
                            "Use .cpu() to copy the tensor to host memory first.")

    def detach(self):
        return self

    def cpu(self):
        return self._t

    def __getattr__(self, name):
        return getattr(self._t, name)


@pytest.fixture()
def fake_pipeline(monkeypatch):
    samples = [_sample(0, "train"), _sample(1, "train"), _sample(2, "train"),
               _sample(3, "train"), _sample(4, "val"), _sample(5, "test")]
    monkeypatch.setattr(hs, "load_task_samples",
                        lambda *a, **k: list(samples))
    monkeypatch.setattr(hs, "load_frozen_model", lambda *a, **k: (None, None))

    def fake_extract(spec, tokenizer, model, seq, seq_id=None, max_len=None,
                     device="cpu"):
        torch.manual_seed(hash(seq_id) % (2**31))
        attn = torch.rand(N_LAYERS, N_HEADS, L, L)
        return SimpleNamespace(attentions=_CudaLikeTensor(attn))

    monkeypatch.setattr(hs, "extract_attention_bundle", fake_extract)
    return samples


def _write_yamls(tmp_path, extra_cfg=""):
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(
        "seed: 42\npairwise:\n  max_pairs_per_sample: 32\n" + extra_cfg)
    models = tmp_path / "models.yaml"
    models.write_text("models:\n  fake_model:\n    type: RNA\n"
                      "    local_dir: assets/models/fake\n")
    tasks = tmp_path / "tasks.yaml"
    tasks.write_text("tasks:\n  toy_pair:\n    task_type: pairwise\n"
                     "    path: unused.jsonl\n    metric: auprc\n")
    return cfg, models, tasks


def _run(tmp_path, extra_cfg=""):
    cfg, models, tasks = _write_yamls(tmp_path, extra_cfg)
    out = tmp_path / "out" / "head_atlas.csv"
    rows = hs.screen_heads("fake_model", "toy_pair", cfg, out,
                           models_yaml=models, tasks_yaml=tasks,
                           limit=None, device="cpu")
    return pd.DataFrame(rows), out


def test_cuda_like_tensor_no_longer_yields_empty_atlas(fake_pipeline, tmp_path):
    df, out = _run(tmp_path)
    assert out.exists()
    assert len(df) == N_LAYERS * N_HEADS
    assert set(df["layer"]) == {0, 1}
    assert df["metric_primary"].notna().all()


def test_split_filter_train_only_with_cap(fake_pipeline, tmp_path):
    df, _ = _run(tmp_path, 'split_filter: ["train"]\npost_filter_limit: 3\n')
    assert df["n_samples"].unique().tolist() == [3]


def test_legacy_default_uses_all_splits(fake_pipeline, tmp_path):
    df, _ = _run(tmp_path)
    # no split_filter: all 6 samples (train/val/test) are screened
    assert df["n_samples"].unique().tolist() == [6]


def test_cpu_tensor_with_gradients(fake_pipeline, tmp_path, monkeypatch):
    def extract(*args, **kwargs):
        return SimpleNamespace(attentions=torch.rand(N_LAYERS, N_HEADS, L, L, requires_grad=True))
    monkeypatch.setattr(hs, "extract_attention_bundle", extract)
    df, _ = _run(tmp_path)
    assert len(df) == N_LAYERS * N_HEADS
    assert df["n_samples"].unique().tolist() == [6]


def test_forward_failures_are_logged(fake_pipeline, tmp_path, monkeypatch, caplog):
    def extract(*args, **kwargs):
        raise RuntimeError("deliberate forward failure")
    monkeypatch.setattr(hs, "extract_attention_bundle", extract)
    df, _ = _run(tmp_path)
    assert df["notes"].iloc[0] == "no_attention_or_forward_failed"
    assert sum("Screening failed" in record.message for record in caplog.records) == 6
    assert "deliberate forward failure" in caplog.text


def test_split_filter_precedes_loading_limit(tmp_path, monkeypatch):
    import json
    from rna_head_alignment.data.task_loaders import load_task_samples
    path = tmp_path / "samples.jsonl"
    records = [{"seq_id": str(i), "sequence": "AUGCAUGC", "split": split, "pairs": [[0, 7]]}
               for i, split in enumerate(["val", "test", "val", "train", "train", "train"])]
    path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    task = {"task_type": "pairwise", "path": str(path)}
    # Legacy prefix sampling remains unchanged.
    assert [s.split for s in load_task_samples("toy", task, limit=2)] == ["val", "test"]
    samples = load_task_samples("toy", task, limit=2, split_filter={"train"})
    assert [s.seq_id for s in samples] == ["3", "4"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
def test_real_cuda_tensor(fake_pipeline, tmp_path, monkeypatch):
    def fake_extract_cuda(spec, tokenizer, model, seq, seq_id=None,
                          max_len=None, device="cuda"):
        torch.manual_seed(hash(seq_id) % (2**31))
        attn = torch.rand(N_LAYERS, N_HEADS, L, L, device="cuda")
        return SimpleNamespace(attentions=attn)
    monkeypatch.setattr(hs, "extract_attention_bundle", fake_extract_cuda)
    df, _ = _run(tmp_path)
    assert len(df) == N_LAYERS * N_HEADS
