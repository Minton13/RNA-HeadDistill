from __future__ import annotations

import numpy as np

from rna_head_alignment.data.task_loaders import TaskSample

from .metrics import binary_metrics


def select_pair_indices(
    sample: TaskSample,
    length: int,
    max_pairs: int = 1024,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray] | None:
    if sample.pair_matrix is None:
        return None
    pair = sample.pair_matrix[:length, :length]
    mask = sample.valid_pair_mask[:length, :length] if sample.valid_pair_mask is not None else np.ones_like(pair, dtype=bool)
    flat_mask = mask.reshape(-1)
    valid_idx = np.flatnonzero(flat_mask)
    if valid_idx.size == 0:
        return None
    y_valid = pair.reshape(-1)[valid_idx].astype(int)
    if max_pairs <= 0 or valid_idx.size <= max_pairs:
        return valid_idx, y_valid
    pos_local = np.flatnonzero(y_valid == 1)
    neg_local = np.flatnonzero(y_valid == 0)
    rng = np.random.default_rng(seed)
    max_pos = min(pos_local.size, max(max_pairs // 2, 1))
    if pos_local.size > max_pos:
        pos_local = rng.choice(pos_local, size=max_pos, replace=False)
    remaining = max(max_pairs - pos_local.size, 0)
    if neg_local.size > remaining:
        neg_local = rng.choice(neg_local, size=remaining, replace=False)
    chosen_local = np.concatenate([pos_local, neg_local])
    if chosen_local.size == 0:
        return None
    rng.shuffle(chosen_local)
    return valid_idx[chosen_local], y_valid[chosen_local]


def init_pairwise_score_parts(n_layers: int, n_heads: int) -> list[list[list[np.ndarray]]]:
    return [[[] for _ in range(n_heads)] for _ in range(n_layers)]


def add_pairwise_score_parts(
    score_parts: list[list[list[np.ndarray]]],
    attentions: np.ndarray,
    flat_indices: np.ndarray,
) -> None:
    n_layers, n_heads = attentions.shape[:2]
    for layer in range(n_layers):
        for head in range(n_heads):
            score = attentions[layer, head]
            score = (score + score.T) / 2.0
            score_parts[layer][head].append(score.reshape(-1)[flat_indices].astype(np.float32, copy=False))


def score_pairwise_streamed(
    model: str,
    model_type: str,
    task: str,
    y_parts: list[np.ndarray],
    score_parts: list[list[list[np.ndarray]]],
    n_samples: int,
    primary: str = "auprc",
    note: str = "streamed_sampled_pairs",
) -> list[dict]:
    if not y_parts or not score_parts:
        return []
    y_cat = np.concatenate(y_parts).astype(int)
    n_pos = int(y_cat.sum())
    n_neg = int((1 - y_cat).sum())
    rows = []
    for layer, layer_parts in enumerate(score_parts):
        for head, parts in enumerate(layer_parts):
            if not parts:
                continue
            p_cat = np.concatenate(parts).astype(float)
            m = binary_metrics(y_cat, p_cat)
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "model_type": model_type,
                    "layer": layer,
                    "head": head,
                    "metric_primary": m.get(primary, float("nan")),
                    "auroc": m["auroc"],
                    "auprc": m["auprc"],
                    "macro_f1": float("nan"),
                    "pearson": float("nan"),
                    "mcc": m["mcc"],
                    "f1": m["f1"],
                    "n_samples": n_samples,
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "split": "mixed",
                    "notes": note,
                }
            )
    return rows


def score_pairwise_heads(model: str, model_type: str, task: str, samples: list[TaskSample], attentions: list[np.ndarray], primary: str = "auprc") -> list[dict]:
    if not attentions:
        return []
    n_layers, n_heads = attentions[0].shape[:2]
    rows = []
    for layer in range(n_layers):
        for head in range(n_heads):
            y_all = []
            p_all = []
            n_pos = 0
            n_neg = 0
            for sample, attn in zip(samples, attentions):
                if sample.pair_matrix is None:
                    continue
                score = attn[layer, head]
                score = (score + score.T) / 2.0
                mask = sample.valid_pair_mask if sample.valid_pair_mask is not None else np.ones_like(sample.pair_matrix, dtype=bool)
                y = sample.pair_matrix[mask].astype(int)
                p = score[mask].astype(float)
                y_all.append(y)
                p_all.append(p)
                n_pos += int(y.sum())
                n_neg += int((1 - y).sum())
            if not y_all:
                continue
            y_cat = np.concatenate(y_all)
            p_cat = np.concatenate(p_all)
            m = binary_metrics(y_cat, p_cat)
            rows.append(
                {
                    "task": task,
                    "model": model,
                    "model_type": model_type,
                    "layer": layer,
                    "head": head,
                    "metric_primary": m.get(primary, float("nan")),
                    "auroc": m["auroc"],
                    "auprc": m["auprc"],
                    "macro_f1": float("nan"),
                    "pearson": float("nan"),
                    "mcc": m["mcc"],
                    "f1": m["f1"],
                    "n_samples": len(samples),
                    "n_pos": n_pos,
                    "n_neg": n_neg,
                    "split": "mixed",
                    "notes": "",
                }
            )
    return rows
