from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from rna_head_alignment.io.manifests import write_dataframe, write_json, write_markdown_table
from rna_head_alignment.io.paths import load_yaml, resolve_path

from .rna3d_loaders import iter_contact_lmdb_dir, iter_rnaglib_dir
from .split_utils import stable_split


@dataclass
class TaskSample:
    seq_id: str
    sequence: str
    task: str
    task_type: str
    split: str = "train"
    label: Any = None
    y_token: np.ndarray | None = None
    pair_matrix: np.ndarray | None = None
    valid_pair_mask: np.ndarray | None = None
    metadata: dict[str, Any] | None = None


def normalize_rna(seq: str) -> str:
    return "".join(ch for ch in str(seq).upper().replace("T", "U") if ch in "ACGUN")


def dot_bracket_pairs(dot: str) -> list[tuple[int, int]]:
    stacks: dict[str, list[int]] = {"(": [], "[": [], "{": [], "<": []}
    close_to_open = {")": "(", "]": "[", "}": "{", ">": "<"}
    pairs = []
    for i, ch in enumerate(dot):
        if ch in stacks:
            stacks[ch].append(i)
        elif ch in close_to_open:
            stack = stacks[close_to_open[ch]]
            if stack:
                j = stack.pop()
                pairs.append((j, i))
    return pairs


def pair_matrix_from_pairs(length: int, pairs: list[tuple[int, int]]) -> np.ndarray:
    mat = np.zeros((length, length), dtype=np.float32)
    for i, j in pairs:
        if 0 <= i < length and 0 <= j < length and i != j:
            mat[i, j] = 1.0
            mat[j, i] = 1.0
    return mat


def valid_pair_mask(length: int, min_separation: int = 3) -> np.ndarray:
    idx = np.arange(length)
    return (np.abs(idx[:, None] - idx[None, :]) > min_separation).astype(bool)


def _infer_sequence_column(df: pd.DataFrame) -> str | None:
    for col in ("sequence", "seq", "Sequence", "rna_sequence", "target_sequence", "input"):
        if col in df.columns:
            return col
    for col in df.columns:
        if "seq" in col.lower():
            return col
    return None


def _infer_label_column(df: pd.DataFrame, sequence_col: str | None) -> str | None:
    candidates = [
        c
        for c in df.columns
        if c != sequence_col
        and c.lower()
        in {
            "label",
            "target",
            "class",
            "family",
            "y",
            "is_bound",
            "binding",
            "splice",
            "half_life",
            "halflife",
            "decay_rate",
            "stability",
            "mrl",
        }
    ]
    if candidates:
        return candidates[0]
    for col in df.columns:
        lc = col.lower()
        if col != sequence_col and any(x in lc for x in ("label", "target", "family", "class")):
            return col
    non_seq = [c for c in df.columns if c != sequence_col]
    return non_seq[0] if non_seq else None


def _read_json_table(path: Path, limit: int | None) -> pd.DataFrame:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        for key in ("data", "records", "train", "items"):
            if isinstance(data.get(key), list):
                records = data[key]
                break
        else:
            if data and all(isinstance(v, (list, tuple)) for v in data.values()):
                lengths = {len(v) for v in data.values()}
                if len(lengths) == 1:
                    df = pd.DataFrame(data)
                    return df.head(limit) if limit else df
            if data and all(isinstance(v, dict) for v in data.values()):
                row_key_sets = [set(v.keys()) for v in data.values()]
                if row_key_sets and all(keys == row_key_sets[0] for keys in row_key_sets[1:]):
                    df = pd.DataFrame(data)
                    return df.head(limit) if limit else df
            records = list(data.values()) if all(isinstance(v, dict) for v in data.values()) else [data]
    else:
        raise ValueError(f"Unsupported JSON table shape: {path}")
    return pd.DataFrame.from_records(records[:limit] if limit else records)


def _read_table(path: Path, limit: int | None) -> pd.DataFrame:
    if path.is_dir():
        frames = []
        remaining = limit
        files = sorted(
            [p for ext in ("*.parquet", "*.csv", "*.tsv", "*.json") for p in path.rglob(ext)]
        )
        for file_path in files:
            frame = _read_table(file_path, limit=remaining)
            if "split" not in frame.columns:
                stem = file_path.stem.lower()
                split = "train"
                for candidate in ("train", "valid", "validation", "val", "test"):
                    if stem.startswith(candidate) or f"-{candidate}" in stem or f"_{candidate}" in stem:
                        split = "validation" if candidate in {"valid", "val"} else candidate
                        break
                frame["split"] = split
            frames.append(frame)
            if remaining is not None:
                remaining -= len(frame)
                if remaining <= 0:
                    break
        if not frames:
            raise ValueError(f"No supported table files found in directory: {path}")
        return pd.concat(frames, ignore_index=True)
    suffix = path.suffix.lower()
    if suffix == ".csv" or suffix == ".tsv":
        sep = "\t" if suffix == ".tsv" else ","
        return pd.read_csv(path, nrows=limit, sep=sep)
    if suffix == ".json":
        return _read_json_table(path, limit=limit)
    if suffix == ".parquet":
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            table = pf.read_row_group(0)
            df = table.to_pandas()
            return df.head(limit) if limit else df
        except Exception:
            return pd.read_parquet(path).head(limit) if limit else pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def _matches_filters(row: pd.Series, filters: dict[str, Any]) -> bool:
    for col, expected in filters.items():
        if col not in row:
            return False
        value = row[col]
        if isinstance(expected, (list, tuple, set)):
            if value not in expected:
                return False
        elif value != expected:
            return False
    return True


def _parse_token_values(raw: Any) -> list[float]:
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                raw = json.loads(text)
            except Exception:
                raw = []
        else:
            raw = [x for x in text.replace(";", ",").split(",") if x.strip()]
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if not isinstance(raw, (list, tuple)):
        return []
    vals = []
    for value in raw:
        if value is None:
            vals.append(float("nan"))
            continue
        try:
            vals.append(float(value))
        except Exception:
            vals.append(float("nan"))
    return vals


def _token_columns(df: pd.DataFrame, task_cfg: dict[str, Any]) -> list[str]:
    configured = task_cfg.get("token_columns")
    if configured:
        return [str(c) for c in configured if str(c) in df.columns]
    prefix = task_cfg.get("token_value_prefix")
    if prefix:
        return [c for c in df.columns if str(c).startswith(str(prefix))]
    return [
        c
        for c in df.columns
        if any(x in c.lower() for x in ("reactivity", "signal", "dms", "shape", "deg_"))
        and "error" not in c.lower()
    ]


def _load_jsonl_pairwise(task: str, task_type: str, path: Path, limit: int | None, max_len: int | None,
                         split_filter: set[str] | None = None) -> list[TaskSample]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if limit and len(rows) >= limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            if split_filter and str(obj.get("split", "")) not in split_filter:
                continue
            seq = normalize_rna(obj.get("sequence", ""))
            if not seq or (max_len and len(seq) > max_len):
                continue
            pairs = obj.get("pairs")
            if pairs is None and obj.get("dot_bracket"):
                pairs = dot_bracket_pairs(str(obj["dot_bracket"]))
            pairs = [(int(i), int(j)) for i, j in (pairs or [])]
            pair = pair_matrix_from_pairs(len(seq), pairs)
            rows.append(
                TaskSample(
                    seq_id=str(obj.get("seq_id") or f"{task}_{len(rows)}"),
                    sequence=seq,
                    task=task,
                    task_type=task_type,
                    split=str(obj.get("split") or stable_split(str(obj.get("seq_id") or len(rows)))),
                    pair_matrix=pair,
                    valid_pair_mask=valid_pair_mask(len(seq)),
                    metadata={"source_path": str(path)},
                )
            )
    return rows


def _load_table_task(
    task: str,
    task_type: str,
    path: Path,
    limit: int | None,
    max_len: int | None,
    task_cfg: dict[str, Any],
) -> list[TaskSample]:
    read_limit = limit
    if task_type in {"sequence_classification", "sequence_regression"} and limit and path.suffix.lower() != ".json":
        read_limit = max(limit * 100, limit)
    if task_cfg.get("filters") and limit and path.suffix.lower() != ".json":
        read_limit = max(limit * 100, limit)
    df = _read_table(path, limit=read_limit)
    filters = task_cfg.get("filters") or {}
    if filters:
        df = df[df.apply(lambda row: _matches_filters(row, filters), axis=1)]
    if limit and path.suffix.lower() == ".json":
        df = df.head(limit)
    seq_col = task_cfg.get("sequence_column") or _infer_sequence_column(df)
    if seq_col is None:
        raise ValueError(f"Could not infer sequence column for {path}")
    seq_col = str(seq_col)
    label_col = task_cfg.get("label_column") or task_cfg.get("target_column") or _infer_label_column(df, seq_col)
    target_col = task_cfg.get("target_column")
    split_col = str(task_cfg.get("split_column") or "split") if (task_cfg.get("split_column") or "split") in df.columns else None
    seq_id_col = str(task_cfg.get("seq_id_column") or "seq_id") if (task_cfg.get("seq_id_column") or "seq_id") in df.columns else None
    token_cols = _token_columns(df, task_cfg)
    rows: list[TaskSample] = []
    for i, row in df.iterrows():
        seq = normalize_rna(row[seq_col])
        if not seq or (max_len and len(seq) > max_len):
            continue
        split = str(row[split_col]) if split_col else stable_split(str(row.get("seq_id", i)))
        seq_id = str(row[seq_id_col]) if seq_id_col else str(row.get("id", f"{task}_{i}"))
        if task_type == "token_regression":
            vals: list[float] = []
            if target_col and target_col in row:
                vals = _parse_token_values(row[target_col])
            if not vals:
                for c in token_cols:
                    try:
                        vals.append(float(row[c]))
                    except Exception:
                        vals.append(float("nan"))
            y_token = np.full(len(seq), np.nan, dtype=np.float32)
            if vals:
                raw_values = np.asarray(vals, dtype=np.float32)
                n_copy = min(len(seq), raw_values.size)
                y_token[:n_copy] = raw_values[:n_copy]
            rows.append(TaskSample(seq_id, seq, task, task_type, split=split, y_token=y_token, metadata={"source_path": str(path)}))
        elif task_type == "sequence_regression":
            if not label_col:
                continue
            try:
                label = float(row[label_col])
            except Exception:
                continue
            if not np.isfinite(label):
                continue
            rows.append(TaskSample(seq_id, seq, task, task_type, split=split, label=label, metadata={"source_path": str(path)}))
        else:
            label = row[label_col] if label_col else 0
            rows.append(TaskSample(seq_id, seq, task, task_type, split=split, label=label, metadata={"source_path": str(path)}))
    if task_type == "sequence_classification" and limit and len(rows) > limit:
        by_label: dict[str, list[TaskSample]] = {}
        for sample in rows:
            by_label.setdefault(str(sample.label), []).append(sample)
        if len(by_label) > 1:
            per_label = max(1, limit // len(by_label))
            balanced: list[TaskSample] = []
            for label in sorted(by_label):
                balanced.extend(by_label[label][:per_label])
            if len(balanced) < limit:
                used = {id(x) for x in balanced}
                for sample in rows:
                    if id(sample) not in used:
                        balanced.append(sample)
                    if len(balanced) >= limit:
                        break
            rows = balanced[:limit]
        else:
            rows = rows[:limit]
    elif limit and len(rows) > limit:
        rows = rows[:limit]
    return rows


def load_task_samples(
    task_name: str,
    task_cfg: dict[str, Any],
    limit: int | None = None,
    max_len: int | None = None,
    split_filter: set[str] | None = None,
) -> list[TaskSample]:
    task_type = str(task_cfg["task_type"])
    path = resolve_path(task_cfg["path"])
    if split_filter:
        if task_type == "pairwise" or (task_type == "rna3d_contact" and path.suffix.lower() == ".jsonl"):
            return _load_jsonl_pairwise(task_name, task_type, path, limit, max_len, split_filter)
        # Preserve existing loaders when no filter is requested; with a
        # filter, apply the cap to the retained sample pool.
        samples = load_task_samples(task_name, task_cfg, limit=None, max_len=max_len)
        samples = [sample for sample in samples if str(sample.split) in split_filter]
        return samples[:limit] if limit is not None else samples
    if task_type in {"pairwise"}:
        return _load_jsonl_pairwise(task_name, task_type, path, limit, max_len)
    if task_type == "rna3d_contact":
        if path.is_dir():
            has_lmdb = any(x.is_dir() and x.suffix == ".lmdb" for x in path.iterdir())
            raw = iter_contact_lmdb_dir(path, limit=limit) if has_lmdb else iter_rnaglib_dir(path, limit=limit)
            samples = []
            for item in raw:
                seq = normalize_rna(item["sequence"])
                if not seq or (max_len and len(seq) > max_len):
                    continue
                pair = item["pair_matrix"].astype(np.float32)
                samples.append(
                    TaskSample(
                        seq_id=item["seq_id"],
                        sequence=seq,
                        task=task_name,
                        task_type=task_type,
                        split=item.get("split", "train"),
                        pair_matrix=pair,
                        valid_pair_mask=valid_pair_mask(len(seq)),
                        metadata={"source": item.get("source")},
                    )
                )
            return samples
        if path.suffix.lower() == ".jsonl":
            return _load_jsonl_pairwise(task_name, task_type, path, limit, max_len)
        raise ValueError(f"Unsupported rna3d path: {path}")
    return _load_table_task(task_name, task_type, path, limit, max_len, task_cfg)


def load_tasks_config(path: str | Path) -> dict[str, dict[str, Any]]:
    cfg = load_yaml(path)
    tasks = cfg.get("tasks") or {}
    if not isinstance(tasks, dict):
        raise ValueError(f"Expected tasks mapping in {path}")
    return tasks


def verify_data(tasks_yaml: str | Path, output_csv: str | Path, limit: int | None = 5) -> list[dict[str, Any]]:
    tasks = load_tasks_config(tasks_yaml)
    rows = []
    for name, cfg in tasks.items():
        path = resolve_path(cfg["path"])
        row: dict[str, Any] = {
            "task": name,
            "task_type": cfg.get("task_type"),
            "path": str(path),
            "exists": path.exists(),
            "read_ok": False,
            "sample_count_read": 0,
            "seq_min_len": None,
            "seq_max_len": None,
            "has_split": False,
            "has_label": False,
            "has_pair_matrix": False,
            "illegal_sequence_count": 0,
            "error": "",
        }
        if not path.exists():
            row["error"] = "missing_path"
            rows.append(row)
            continue
        try:
            samples = load_task_samples(name, cfg, limit=limit)
            row["read_ok"] = True
            row["sample_count_read"] = len(samples)
            if samples:
                lengths = [len(s.sequence) for s in samples]
                row["seq_min_len"] = min(lengths)
                row["seq_max_len"] = max(lengths)
                row["has_split"] = any(bool(s.split) for s in samples)
                row["has_label"] = any(s.label is not None or s.y_token is not None for s in samples)
                row["has_pair_matrix"] = any(s.pair_matrix is not None for s in samples)
                row["illegal_sequence_count"] = sum(
                    1 for s in samples if any(ch not in "ACGUN" for ch in s.sequence)
                )
        except Exception as e:
            row["error"] = repr(e)
        rows.append(row)
    df = pd.DataFrame(rows)
    out = resolve_path(output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_dataframe(df, out)
    write_json(rows, out.with_suffix(".json"))
    write_markdown_table(rows, out.with_suffix(".md"), "Data Check")
    return rows
