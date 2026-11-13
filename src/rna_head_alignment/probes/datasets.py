"""Downstream task datasets and collators for frozen probes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from rna_head_alignment.data.task_loaders import valid_pair_mask as _valid_pair_mask
from rna_head_alignment.head2head.sequence_dataset import align_tokens_to_sequence
from rna_head_alignment.probes.task_config import TASK_TO_TYPE


def _normalize_sequence(seq: Any) -> str:
    seq = str(seq).upper().replace("T", "U")
    return "".join(ch for ch in seq if ch in "ACGUN")


def _pad_1d(tensors: list[torch.Tensor], pad_value: int = 0) -> torch.Tensor:
    max_len = max(int(t.numel()) for t in tensors)
    out = torch.full((len(tensors), max_len), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        out[i, : t.numel()] = t
    return out


def _pad_1d_float(
    tensors: list[torch.Tensor], pad_value: float = float("nan")
) -> torch.Tensor:
    max_len = max(int(t.numel()) for t in tensors)
    out = torch.full((len(tensors), max_len), pad_value, dtype=torch.float32)
    for i, t in enumerate(tensors):
        out[i, : t.numel()] = t
    return out


def _pad_2d(
    tensors: list[torch.Tensor], pad_value: float = 0.0
) -> torch.Tensor:
    max_h = max(int(t.shape[0]) for t in tensors)
    max_w = max(int(t.shape[1]) for t in tensors)
    out = torch.full((len(tensors), max_h, max_w), pad_value, dtype=tensors[0].dtype)
    for i, t in enumerate(tensors):
        h, w = t.shape
        out[i, :h, :w] = t
    return out


def _pair_matrix_from_label(
    label: Any, length: int
) -> torch.Tensor:
    """Convert a label to an ``[L, L]`` float target matrix."""
    arr = np.asarray(label, dtype=object)
    if arr.ndim == 2:
        mat = np.asarray(label, dtype=np.float32)
    elif arr.ndim == 1 and arr.size > 0 and isinstance(arr[0], (np.ndarray, list)):
        # Stored as a list of rows (object array); stack into a 2D matrix.
        mat = np.stack([np.asarray(row, dtype=np.float32) for row in label], axis=0)
    elif isinstance(label, str):
        try:
            obj = json.loads(label)
            if isinstance(obj, list) and len(obj) == 2 and isinstance(obj[0], int):
                obj = [obj]
            mat = _pair_matrix_from_pairs(length, obj)
        except Exception:
            mat = np.zeros((length, length), dtype=np.float32)
    else:
        mat = np.zeros((length, length), dtype=np.float32)

    if mat.shape != (length, length):
        padded = np.zeros((length, length), dtype=np.float32)
        h, w = mat.shape[:2]
        copy_h, copy_w = min(h, length), min(w, length)
        padded[:copy_h, :copy_w] = mat[:copy_h, :copy_w]
        mat = padded
    return torch.from_numpy(mat)


def _pair_matrix_from_pairs(length: int, pairs: Any) -> np.ndarray:
    mat = np.zeros((length, length), dtype=np.float32)
    if not pairs:
        return mat
    for pair in pairs:
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            continue
        i, j = int(pair[0]), int(pair[1])
        if 0 <= i < length and 0 <= j < length and i != j:
            mat[i, j] = 1.0
            mat[j, i] = 1.0
    return mat


def _token_values_from_label(label: Any, length: int) -> torch.Tensor:
    """Convert a token label to ``[L]`` float tensor."""
    if isinstance(label, (np.ndarray, list)):
        vals = np.asarray(label, dtype=np.float32)
    elif isinstance(label, str):
        try:
            vals = np.asarray(json.loads(label), dtype=np.float32)
        except Exception:
            parts = [x.strip() for x in label.replace(";", ",").split(",") if x.strip()]
            vals = np.asarray([float(x) if x else float("nan") for x in parts], dtype=np.float32)
    else:
        vals = np.full(length, float("nan"), dtype=np.float32)

    if vals.shape[0] != length:
        padded = np.full(length, float("nan"), dtype=np.float32)
        copy_len = min(int(vals.shape[0]), length)
        padded[:copy_len] = vals[:copy_len]
        vals = padded
    return torch.from_numpy(vals)


def _token_mask_from_label(label: Any, length: int) -> torch.Tensor:
    """Best-effort per-token validity mask from a label_mask column."""
    if label is None:
        return torch.ones(length, dtype=torch.bool)
    if isinstance(label, (np.ndarray, list)):
        mask = np.asarray(label, dtype=bool)
    elif isinstance(label, str):
        try:
            mask = np.asarray(json.loads(label), dtype=bool)
        except Exception:
            mask = np.ones(length, dtype=bool)
    else:
        mask = np.ones(length, dtype=bool)

    if mask.shape[0] != length:
        padded = np.zeros(length, dtype=bool)
        copy_len = min(int(mask.shape[0]), length)
        padded[:copy_len] = mask[:copy_len]
        mask = padded
    return torch.from_numpy(mask)


class DownstreamDataset(Dataset):
    """Parquet-backed downstream dataset with on-the-fly tokenization.

    Expects ``{split}.parquet`` in ``data_dir`` with at least ``seq_id``,
    ``sequence`` and ``label`` columns.  Optional columns ``label_mask``,
    ``family``, ``cluster``, and ``source`` are preserved as metadata.

    Memory-efficient: rows are converted to tensors on demand in ``__getitem__``
    rather than pre-building a full item list.  For very large parquets the
    DataFrame itself is kept, which is far smaller than the expanded tensor
    representation.
    """

    def __init__(
        self,
        task: str,
        split: str,
        data_dir: str | Path,
        tokenizer: Any,
        max_len: int = 512,
        label_map: dict[Any, int] | None = None,
    ) -> None:
        super().__init__()
        self.task = task
        self.split = split
        self.task_type = TASK_TO_TYPE[task]
        self.data_dir = Path(data_dir)
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.path = self.data_dir / f"{split}.parquet"
        if not self.path.exists():
            raise FileNotFoundError(f"Downstream split not found: {self.path}")
        self.df = pd.read_parquet(self.path)
        self.label_map = label_map
        # Pre-compute valid row indices (skip empty sequences) without
        # materializing full tensors.
        self._valid_indices: list[int] = []
        for idx, row in self.df.iterrows():
            seq = _normalize_sequence(row.get("sequence", ""))
            if seq:
                self._valid_indices.append(idx)

    def __len__(self) -> int:
        return len(self._valid_indices)

    def _row_to_item(self, row: pd.Series) -> dict[str, Any]:
        seq = _normalize_sequence(row.get("sequence", ""))
        length = min(len(seq), self.max_len)
        seq = seq[: self.max_len]
        item: dict[str, Any] = {
            "seq_id": str(row.get("seq_id", f"{self.task}_{length}")),
            "sequence": seq,
            "length": length,
            "metadata": {
                "source": row.get("source"),
                "family": row.get("family"),
                "cluster": row.get("cluster"),
                "sequence_hash": row.get("sequence_hash"),
            },
        }

        raw_label = row.get("label")
        if self.task_type == "sequence_classification":
            item["label"] = (
                self.label_map[raw_label]
                if self.label_map is not None
                else raw_label
            )
        elif self.task_type == "token_regression":
            item["label"] = _token_values_from_label(raw_label, length)
            item["label_mask"] = _token_mask_from_label(
                row.get("label_mask"), length
            )
        elif self.task_type == "pairwise":
            # Prefer an explicit pair_matrix column; fall back to the generic label.
            pair_label = row.get("pair_matrix", raw_label)
            item["label"] = _pair_matrix_from_label(pair_label, length)
            valid = row.get("valid_pair_mask")
            if valid is None:
                valid = _valid_pair_mask(length)
            else:
                valid = np.asarray(valid, dtype=bool)
                if valid.shape != (length, length):
                    padded = np.zeros((length, length), dtype=bool)
                    h, w = valid.shape[:2]
                    copy_h, copy_w = min(h, length), min(w, length)
                    padded[:copy_h, :copy_w] = valid[:copy_h, :copy_w]
                    valid = padded
            item["label_mask"] = torch.from_numpy(valid)
        return item

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row_idx = self._valid_indices[idx]
        row = self.df.iloc[row_idx]
        item = self._row_to_item(row)
        input_ids, length = align_tokens_to_sequence(
            self.tokenizer, item["sequence"], self.max_len
        )
        out: dict[str, Any] = {
            "seq_id": item["seq_id"],
            "input_ids": input_ids,
            "length": length,
            "attention_mask": torch.ones(length, dtype=torch.long),
            "metadata": item["metadata"],
        }
        if self.task_type == "sequence_classification":
            out["label"] = item["label"]
        else:
            out["label"] = item["label"]
            out["label_mask"] = item["label_mask"]
        return out


def _sequence_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    input_ids = _pad_1d([b["input_ids"] for b in batch], pad_value=0)
    lengths = torch.tensor([b["length"] for b in batch], dtype=torch.long)
    labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != 0).long(),
        "lengths": lengths,
        "labels": labels,
        "seq_ids": [b["seq_id"] for b in batch],
    }


def _token_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    input_ids = _pad_1d([b["input_ids"] for b in batch], pad_value=0)
    labels = _pad_1d_float([b["label"] for b in batch], pad_value=float("nan"))
    label_mask = _pad_1d([b["label_mask"].long() for b in batch], pad_value=0).bool()
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != 0).long(),
        "lengths": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "labels": labels,
        "label_mask": label_mask,
        "seq_ids": [b["seq_id"] for b in batch],
    }


def _pair_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    input_ids = _pad_1d([b["input_ids"] for b in batch], pad_value=0)
    labels = _pad_2d([b["label"] for b in batch], pad_value=0.0)
    label_mask = _pad_2d(
        [b["label_mask"].float() for b in batch], pad_value=0.0
    ).bool()
    return {
        "input_ids": input_ids,
        "attention_mask": (input_ids != 0).long(),
        "lengths": torch.tensor([b["length"] for b in batch], dtype=torch.long),
        "labels": labels,
        "label_mask": label_mask,
        "seq_ids": [b["seq_id"] for b in batch],
    }


def get_collator(task_type: str) -> Any:
    if task_type == "sequence_classification":
        return _sequence_collate
    if task_type == "token_regression":
        return _token_collate
    if task_type == "pairwise":
        return _pair_collate
    raise ValueError(f"Unknown task type {task_type!r}")


def build_label_map(
    train_dataset: DownstreamDataset,
    data_dir: str | Path | None = None,
    task: str | None = None,
    tokenizer: Any = None,
    max_len: int = 512,
) -> dict[Any, int]:
    """Build a stable integer label map from the training split.

    When ``data_dir``/``task``/``tokenizer`` are provided, val/test labels are
    also scanned so unseen classes in those splits still map consistently.
    """
    if train_dataset.task_type != "sequence_classification":
        return {}
    # Read labels directly from the DataFrame to avoid tokenizing every item.
    labels = set(train_dataset.df["label"].tolist())
    if data_dir is not None and task is not None and tokenizer is not None:
        for split in ("val", "test"):
            try:
                ds = DownstreamDataset(task, split, data_dir, tokenizer, max_len=max_len)
                labels.update(ds.df["label"].tolist())
            except FileNotFoundError:
                continue
    labels = sorted(labels)
    return {label: i for i, label in enumerate(labels)}


def get_downstream_dataloaders(
    task: str,
    data_dir: str | Path,
    tokenizer: Any,
    batch_size: int = 32,
    max_len: int = 512,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[Any, int] | None]:
    """Return train/val/test dataloaders and a label map (if classification)."""
    train_dataset = DownstreamDataset(
        task, "train", data_dir, tokenizer, max_len=max_len
    )
    label_map = build_label_map(
        train_dataset,
        data_dir=data_dir,
        task=task,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    if label_map:
        # Reload with the resolved label map so all splits use the same encoding.
        train_dataset = DownstreamDataset(
            task, "train", data_dir, tokenizer, max_len=max_len, label_map=label_map
        )
        val_dataset = DownstreamDataset(
            task, "val", data_dir, tokenizer, max_len=max_len, label_map=label_map
        )
        test_dataset = DownstreamDataset(
            task, "test", data_dir, tokenizer, max_len=max_len, label_map=label_map
        )
    else:
        val_dataset = DownstreamDataset(
            task, "val", data_dir, tokenizer, max_len=max_len
        )
        test_dataset = DownstreamDataset(
            task, "test", data_dir, tokenizer, max_len=max_len
        )

    collator = get_collator(TASK_TO_TYPE[task])
    common = dict(
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=False,
    )
    return (
        DataLoader(train_dataset, shuffle=True, **common),
        DataLoader(val_dataset, shuffle=False, **common),
        DataLoader(test_dataset, shuffle=False, **common),
        label_map or None,
    )
