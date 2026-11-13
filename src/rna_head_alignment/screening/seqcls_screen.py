from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from rna_head_alignment.data.task_loaders import TaskSample

from .metrics import classification_metrics


def _head_features(attn: np.ndarray) -> np.ndarray:
    p = np.clip(attn, 1e-8, 1.0)
    entropy = -(p * np.log(p)).sum(axis=-1)
    diag = np.diag(attn)
    return np.asarray(
        [
            float(attn.mean()),
            float(attn.std()),
            float(attn.max()),
            float(entropy.mean()),
            float(entropy.std()),
            float(diag.mean()) if diag.size else 0.0,
        ],
        dtype=np.float32,
    )


def head_feature_stack(attentions: np.ndarray) -> np.ndarray:
    """Compress one sample's attention stack to [layers, heads, features]."""
    n_layers, n_heads = attentions.shape[:2]
    feats = np.empty((n_layers, n_heads, 6), dtype=np.float32)
    for layer in range(n_layers):
        for head in range(n_heads):
            feats[layer, head] = _head_features(attentions[layer, head])
    return feats


def score_seqcls_head_features(
    model: str,
    model_type: str,
    task: str,
    samples: list[TaskSample],
    features: np.ndarray,
    primary: str = "auroc",
) -> list[dict]:
    labels = [s.label for s in samples]
    if features.size == 0 or len(set(map(str, labels))) < 2:
        return []
    encoded_y = LabelEncoder().fit_transform([str(x) for x in labels])
    n_layers, n_heads = features.shape[1:3]
    rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            X = features[:, layer, head, :]
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    encoded_y,
                    test_size=0.35,
                    random_state=42,
                    stratify=encoded_y if min(np.bincount(encoded_y)) >= 2 else None,
                )
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                clf = LogisticRegression(max_iter=200, class_weight="balanced")
                clf.fit(X_train, y_train)
                pred = clf.predict(X_val)
                score = None
                if len(np.unique(encoded_y)) == 2 and hasattr(clf, "predict_proba"):
                    score = clf.predict_proba(X_val)[:, 1]
                m = classification_metrics(y_val, pred, score)
            except Exception as e:
                m = {"accuracy": float("nan"), "macro_f1": float("nan"), "auroc": float("nan")}
                note = repr(e)
            else:
                note = "attention_stats_logistic_probe_streamed"
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "model_type": model_type,
                    "layer": layer,
                    "head": head,
                    "metric_primary": m.get(primary, m.get("macro_f1", float("nan"))),
                    "auroc": m.get("auroc", float("nan")),
                    "auprc": float("nan"),
                    "macro_f1": m.get("macro_f1", float("nan")),
                    "accuracy": m.get("accuracy", float("nan")),
                    "pearson": float("nan"),
                    "mcc": float("nan"),
                    "f1": float("nan"),
                    "n_samples": len(samples),
                    "n_pos": int((encoded_y == 1).sum()) if len(np.unique(encoded_y)) == 2 else 0,
                    "n_neg": int((encoded_y == 0).sum()) if len(np.unique(encoded_y)) == 2 else 0,
                    "split": "train_val_probe",
                    "notes": note,
                }
            )
    return rows


def score_seqcls_heads(model: str, model_type: str, task: str, samples: list[TaskSample], attentions: list[np.ndarray], primary: str = "auroc") -> list[dict]:
    labels = [s.label for s in samples]
    if not attentions or len(set(map(str, labels))) < 2:
        return []
    encoded_y = LabelEncoder().fit_transform([str(x) for x in labels])
    n_layers, n_heads = attentions[0].shape[:2]
    rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            X = np.stack([_head_features(attn[layer, head]) for attn in attentions])
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X,
                    encoded_y,
                    test_size=0.35,
                    random_state=42,
                    stratify=encoded_y if min(np.bincount(encoded_y)) >= 2 else None,
                )
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_val = scaler.transform(X_val)
                clf = LogisticRegression(max_iter=200, class_weight="balanced")
                clf.fit(X_train, y_train)
                pred = clf.predict(X_val)
                score = None
                if len(np.unique(encoded_y)) == 2 and hasattr(clf, "predict_proba"):
                    score = clf.predict_proba(X_val)[:, 1]
                m = classification_metrics(y_val, pred, score)
            except Exception as e:
                m = {"accuracy": float("nan"), "macro_f1": float("nan"), "auroc": float("nan")}
                note = repr(e)
            else:
                note = "attention_stats_logistic_probe"
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "model_type": model_type,
                    "layer": layer,
                    "head": head,
                    "metric_primary": m.get(primary, m.get("macro_f1", float("nan"))),
                    "auroc": m.get("auroc", float("nan")),
                    "auprc": float("nan"),
                    "macro_f1": m.get("macro_f1", float("nan")),
                    "accuracy": m.get("accuracy", float("nan")),
                    "pearson": float("nan"),
                    "mcc": float("nan"),
                    "f1": float("nan"),
                    "n_samples": len(samples),
                    "n_pos": int((encoded_y == 1).sum()) if len(np.unique(encoded_y)) == 2 else 0,
                    "n_neg": int((encoded_y == 0).sum()) if len(np.unique(encoded_y)) == 2 else 0,
                    "split": "train_val_probe",
                    "notes": note,
                }
            )
    return rows
