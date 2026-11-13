from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rna_head_alignment.data.task_loaders import TaskSample

from .metrics import regression_metrics


def _row_entropy(attn: np.ndarray) -> np.ndarray:
    p = np.clip(attn, 1e-8, 1.0)
    return -(p * np.log(p)).sum(axis=-1)


def _token_features(attn: np.ndarray) -> np.ndarray:
    """Return per-token attention features [L, F] for one head.

    Features are deliberately small so local screening stays cheap:
    row entropy, row max, self-attention, received attention, local-window mass,
    and expected attention distance.
    """
    L = int(attn.shape[0])
    p = np.clip(attn.astype(float), 1e-8, 1.0)
    row_entropy = -(p * np.log(p)).sum(axis=-1)
    row_max = p.max(axis=-1)
    diag = np.diag(p) if p.ndim == 2 else np.zeros(L)
    received = p.sum(axis=0)
    idx = np.arange(L)
    dist = np.abs(idx[:, None] - idx[None, :]).astype(float)
    local = (dist <= 3).astype(float)
    local_mass = (p * local).sum(axis=-1)
    expected_dist = (p * dist).sum(axis=-1) / max(L - 1, 1)
    return np.stack([row_entropy, row_max, diag, received, local_mass, expected_dist], axis=1).astype(np.float32)


def token_feature_stack(attentions: np.ndarray) -> list[list[np.ndarray]]:
    """Compress one attention stack to per-head token features.

    Returned shape is a nested list [layer][head] with arrays [L, 6].
    A list keeps variable sequence lengths cheap and avoids retaining full
    [layers, heads, L, L] maps for token-probe tasks.
    """
    n_layers, n_heads = attentions.shape[:2]
    return [
        [_token_features(attentions[layer, head]) for head in range(n_heads)]
        for layer in range(n_layers)
    ]


def _fit_token_probe(
    samples: list[TaskSample],
    attentions: list[np.ndarray],
    layer: int,
    head: int,
) -> dict[str, float] | None:
    valid_indices = []
    for i, (sample, attn) in enumerate(zip(samples, attentions)):
        if sample.y_token is None:
            continue
        pred_len = int(attn[layer, head].shape[0])
        y = sample.y_token[:pred_len]
        if np.isfinite(y).sum() >= 3:
            valid_indices.append(i)
    if len(valid_indices) < 8:
        return None
    train_idx, val_idx = train_test_split(valid_indices, test_size=0.35, random_state=42)

    def build(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for i in indices:
            sample = samples[i]
            attn = attentions[i][layer, head]
            x = _token_features(attn)
            y = sample.y_token[: x.shape[0]]
            ok = np.isfinite(y)
            if ok.sum() >= 3:
                xs.append(x[ok])
                ys.append(y[ok])
        if not xs:
            return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0).astype(np.float32)

    x_train, y_train = build(list(train_idx))
    x_val, y_val = build(list(val_idx))
    if x_train.shape[0] < 10 or x_val.shape[0] < 10 or np.nanstd(y_train) == 0:
        return None
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    reg = Ridge(alpha=1.0)
    reg.fit(x_train, y_train)
    pred = reg.predict(x_val)
    return regression_metrics(y_val, pred)


def _fit_token_feature_probe(
    samples: list[TaskSample],
    features: list[list[list[np.ndarray]]],
    layer: int,
    head: int,
) -> dict[str, float] | None:
    valid_indices = []
    for i, sample in enumerate(samples[: len(features)]):
        if sample.y_token is None:
            continue
        pred_len = int(features[i][layer][head].shape[0])
        y = sample.y_token[:pred_len]
        if np.isfinite(y).sum() >= 3:
            valid_indices.append(i)
    if len(valid_indices) < 8:
        return None
    train_idx, val_idx = train_test_split(valid_indices, test_size=0.35, random_state=42)

    def build(indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        xs = []
        ys = []
        for i in indices:
            sample = samples[i]
            x = features[i][layer][head]
            y = sample.y_token[: x.shape[0]]
            ok = np.isfinite(y)
            if ok.sum() >= 3:
                xs.append(x[ok])
                ys.append(y[ok])
        if not xs:
            return np.empty((0, 6), dtype=np.float32), np.empty((0,), dtype=np.float32)
        return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0).astype(np.float32)

    x_train, y_train = build(list(train_idx))
    x_val, y_val = build(list(val_idx))
    if x_train.shape[0] < 10 or x_val.shape[0] < 10 or np.nanstd(y_train) == 0:
        return None
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    reg = Ridge(alpha=1.0)
    reg.fit(x_train, y_train)
    pred = reg.predict(x_val)
    return regression_metrics(y_val, pred)


def score_token_head_features(
    model: str,
    model_type: str,
    task: str,
    samples: list[TaskSample],
    features: list[list[list[np.ndarray]]],
    primary: str = "pearson",
) -> list[dict]:
    if not features:
        return []
    n_layers = len(features[0])
    n_heads = len(features[0][0]) if n_layers else 0
    rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            try:
                m = _fit_token_feature_probe(samples, features, layer, head)
                note = "attention_token_ridge_probe_streamed"
            except Exception as e:
                m = None
                note = repr(e)
            if m is None:
                y_all = []
                p_all = []
                for sample, sample_features in zip(samples, features):
                    if sample.y_token is None:
                        continue
                    pred = sample_features[layer][head][:, 0]
                    y = sample.y_token[: pred.shape[0]]
                    ok = np.isfinite(y)
                    if ok.sum() >= 3:
                        y_all.append(y[ok])
                        p_all.append(pred[ok])
                if not y_all:
                    continue
                y_cat = np.concatenate(y_all)
                p_cat = np.concatenate(p_all)
                m = regression_metrics(y_cat, p_cat)
                note = "attention_row_entropy_fallback_streamed"
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "model_type": model_type,
                    "layer": layer,
                    "head": head,
                    "metric_primary": m.get(primary, float("nan")),
                    "auroc": float("nan"),
                    "auprc": float("nan"),
                    "macro_f1": float("nan"),
                    "pearson": m["pearson"],
                    "mcc": float("nan"),
                    "f1": float("nan"),
                    "n_samples": len(samples),
                    "n_pos": 0,
                    "n_neg": 0,
                    "split": "train_val_probe" if note == "attention_token_ridge_probe_streamed" else "mixed",
                    "notes": note,
                }
            )
    return rows


def score_token_heads(model: str, model_type: str, task: str, samples: list[TaskSample], attentions: list[np.ndarray], primary: str = "pearson") -> list[dict]:
    if not attentions:
        return []
    n_layers, n_heads = attentions[0].shape[:2]
    rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            try:
                m = _fit_token_probe(samples, attentions, layer, head)
                note = "attention_token_ridge_probe"
            except Exception as e:
                m = None
                note = repr(e)
            if m is None:
                y_all = []
                p_all = []
                for sample, attn in zip(samples, attentions):
                    if sample.y_token is None:
                        continue
                    pred = _row_entropy(attn[layer, head])
                    y = sample.y_token[: pred.shape[0]]
                    ok = np.isfinite(y)
                    if ok.sum() >= 3:
                        y_all.append(y[ok])
                        p_all.append(pred[ok])
                if not y_all:
                    continue
                y_cat = np.concatenate(y_all)
                p_cat = np.concatenate(p_all)
                m = regression_metrics(y_cat, p_cat)
                note = "attention_row_entropy_fallback"
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "model_type": model_type,
                    "layer": layer,
                    "head": head,
                    "metric_primary": m.get(primary, float("nan")),
                    "auroc": float("nan"),
                    "auprc": float("nan"),
                    "macro_f1": float("nan"),
                    "pearson": m["pearson"],
                    "mcc": float("nan"),
                    "f1": float("nan"),
                    "n_samples": len(samples),
                    "n_pos": 0,
                    "n_neg": 0,
                    "split": "train_val_probe" if note == "attention_token_ridge_probe" else "mixed",
                    "notes": note,
                }
            )
    return rows
