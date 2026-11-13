from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from rna_head_alignment.data.task_loaders import TaskSample

from .metrics import regression_metrics
from .seqcls_screen import _head_features


def score_seqreg_head_features(
    model: str,
    model_type: str,
    task: str,
    samples: list[TaskSample],
    features: np.ndarray,
    primary: str = "pearson",
) -> list[dict]:
    y = np.asarray([float(s.label) for s in samples if s.label is not None], dtype=np.float32)
    if features.size == 0 or y.size < 8 or np.nanstd(y) == 0:
        return []
    n = min(features.shape[0], y.size)
    y = y[:n]
    features = features[:n]
    n_layers, n_heads = features.shape[1:3]
    rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            X = features[:, layer, head, :]
            try:
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.35, random_state=42)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                reg = Ridge(alpha=1.0)
                reg.fit(X_train, y_train)
                pred = reg.predict(X_val)
                m = regression_metrics(y_val, pred)
                note = "attention_stats_ridge_probe_streamed"
            except Exception as e:
                m = {"pearson": float("nan"), "spearman": float("nan"), "mae": float("nan")}
                note = repr(e)
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
                    "accuracy": float("nan"),
                    "pearson": m["pearson"],
                    "spearman": m["spearman"],
                    "mae": m["mae"],
                    "mcc": float("nan"),
                    "f1": float("nan"),
                    "n_samples": int(n),
                    "n_pos": 0,
                    "n_neg": 0,
                    "split": "train_val_probe",
                    "notes": note,
                }
            )
    return rows


def score_seqreg_heads(
    model: str,
    model_type: str,
    task: str,
    samples: list[TaskSample],
    attentions: list[np.ndarray],
    primary: str = "pearson",
) -> list[dict]:
    y = np.asarray([float(s.label) for s in samples if s.label is not None], dtype=np.float32)
    if not attentions or y.size < 8 or np.nanstd(y) == 0:
        return []
    n = min(len(attentions), y.size)
    y = y[:n]
    n_layers, n_heads = attentions[0].shape[:2]
    rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            X = np.stack([_head_features(attn[layer, head]) for attn in attentions[:n]])
            try:
                X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.35, random_state=42)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                reg = Ridge(alpha=1.0)
                reg.fit(X_train, y_train)
                pred = reg.predict(X_val)
                m = regression_metrics(y_val, pred)
                note = "attention_stats_ridge_probe"
            except Exception as e:
                m = {"pearson": float("nan"), "spearman": float("nan"), "mae": float("nan")}
                note = repr(e)
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
                    "accuracy": float("nan"),
                    "pearson": m["pearson"],
                    "spearman": m["spearman"],
                    "mae": m["mae"],
                    "mcc": float("nan"),
                    "f1": float("nan"),
                    "n_samples": int(n),
                    "n_pos": 0,
                    "n_neg": 0,
                    "split": "train_val_probe",
                    "notes": note,
                }
            )
    return rows
