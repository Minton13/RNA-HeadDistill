"""Downstream dataset preparation and decontamination helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from rna_head_alignment.data.split_utils import stable_split
from rna_head_alignment.data.task_loaders import (
    dot_bracket_pairs,
    normalize_rna,
    pair_matrix_from_pairs,
    valid_pair_mask,
)
from rna_head_alignment.io.paths import resolve_path


REQUIRED_COLUMNS = (
    "seq_id",
    "sequence",
    "length",
    "label",
    "label_mask",
    "sequence_hash",
    "source_dataset",
    "family",
    "cluster",
)


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def sequence_hash(sequence: str) -> str:
    """SHA-256 of the normalized, upper-cased sequence."""
    return hashlib.sha256(sequence.upper().encode("utf-8")).hexdigest()


def normalize_split(value: Any) -> str:
    s = str(value).strip().lower()
    if s in {"train", "training"}:
        return "train"
    if s in {"val", "valid", "validation", "dev"}:
        return "val"
    if s in {"test", "testing"}:
        return "test"
    return ""


def ensure_seq_id(df: pd.DataFrame, task_name: str) -> pd.DataFrame:
    """Guarantee a ``seq_id`` column exists."""
    if "seq_id" in df.columns:
        df["seq_id"] = df["seq_id"].astype(str)
        return df
    for col in ("id", "sequence_id", "ID"):
        if col in df.columns:
            df = df.rename(columns={col: "seq_id"})
            df["seq_id"] = df["seq_id"].astype(str)
            return df
    df = df.reset_index(drop=True)
    df["seq_id"] = [f"{task_name}_{i}" for i in range(len(df))]
    return df


def add_core_columns(
    df: pd.DataFrame,
    task_name: str,
    task_type: str,
    source_dataset: str,
    family_col: str | None = None,
    cluster_col: str | None = None,
) -> pd.DataFrame:
    """Add canonical columns required by every downstream task file."""
    df = df.copy()
    df["task"] = task_name
    df["task_type"] = task_type
    df["source_dataset"] = source_dataset

    if family_col and family_col in df.columns:
        df["family"] = df[family_col].astype(str)
    else:
        df["family"] = None

    if cluster_col and cluster_col in df.columns:
        df["cluster"] = df[cluster_col].astype(str)
    else:
        df["cluster"] = df["family"]

    df["length"] = df["sequence"].str.len().astype(np.int32)
    df["sequence_hash"] = df["sequence"].apply(sequence_hash)
    return df


def to_nested_list(value: Any) -> Any:
    """Recursively convert numpy arrays to nested Python lists for parquet."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def listify_array_column(series: pd.Series) -> pd.Series:
    """Convert an object column of arrays to an object column of nested lists."""
    return series.apply(to_nested_list)


# ---------------------------------------------------------------------------
# Splitting
# ---------------------------------------------------------------------------


def apply_source_split(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize an existing source ``split`` column if present."""
    if "split" not in df.columns:
        return df
    df = df.copy()
    normalized = df["split"].apply(normalize_split)
    df["split"] = normalized.replace("", np.nan)
    return df


def group_aware_split(
    df: pd.DataFrame,
    group_col: str | None,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> pd.DataFrame:
    """Assign whole groups to train/val/test while balancing split sizes.

    Groups are kept intact (no related sequences are split across sets).  A
    greedy load-balancing pass respects the requested train/val/test ratios as
    closely as possible given whole-group assignment.
    """
    df = df.copy()
    if group_col and group_col in df.columns and df[group_col].notna().any():
        group_sizes = df[group_col].value_counts()
        groups = group_sizes.index.tolist()
        rng = np.random.default_rng(seed)
        rng.shuffle(groups)

        target_total = len(df)
        targets = {
            "train": train_ratio * target_total,
            "val": val_ratio * target_total,
            "test": test_ratio * target_total,
        }
        current = {"train": 0, "val": 0, "test": 0}
        split_by_group: dict[Any, str] = {}

        for g in groups:
            size = int(group_sizes[g])
            # Pick the split that is most under its target ratio, with a small
            # random tie-break to avoid deterministic order bias.
            deficits = {
                split: targets[split] - current[split]
                for split in ("train", "val", "test")
            }
            best = max(deficits, key=lambda s: (deficits[s], rng.random()))
            split_by_group[g] = best
            current[best] += size

        df["split"] = df[group_col].map(split_by_group)
    else:
        # Deterministic hash-based split when no grouping column is available.
        df["split"] = df["seq_id"].apply(
            lambda sid: stable_split(sid, train_ratio, val_ratio)
        )
    return df


def stratified_hash_split(
    df: pd.DataFrame,
    label_col: str,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> pd.DataFrame:
    """Deterministic split stratified on ``label_col`` using sequence IDs."""
    df = df.copy()

    def _split(row: pd.Series) -> str:
        label = row[label_col]
        sid = str(row["seq_id"])
        # Append label to seq_id so each class is split independently.
        return stable_split(f"{sid}_label{label}", train_ratio, val_ratio)

    df["split"] = df.apply(_split, axis=1)
    return df


# ---------------------------------------------------------------------------
# Decontamination
# ---------------------------------------------------------------------------


def load_decontamination_hashes(manifest_path: str | Path) -> set[str]:
    """Load the Stage 4B sequence hash set used for exact-match decontamination."""
    df = pd.read_parquet(resolve_path(manifest_path))
    return set(df["sequence_hash"].astype(str).unique())


def decontaminate(
    df: pd.DataFrame, hashes: set[str]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove rows whose ``sequence_hash`` overlaps the Stage 4B corpus."""
    mask = ~df["sequence_hash"].isin(hashes)
    removed = int((~mask).sum())
    removed_by_source = (
        df.loc[~mask, "source_dataset"].value_counts().to_dict() if removed else {}
    )
    clean = df[mask].reset_index(drop=True)
    summary = {
        "total_before": int(len(df)),
        "removed": removed,
        "remaining": int(len(clean)),
        "contamination_rate": removed / max(len(df), 1),
        "removed_by_source": removed_by_source,
    }
    return clean, summary


# ---------------------------------------------------------------------------
# Per-task builders
# ---------------------------------------------------------------------------


def build_splice_dataset(
    task_name: str,
    cfg: dict[str, Any],
    split_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Binary classification for donor (class 1) or acceptor (class 2) sites."""
    source_path = resolve_path(cfg["source_path"])
    df = pd.read_csv(source_path)
    df = ensure_seq_id(df, task_name)
    df["sequence"] = df["sequence"].apply(normalize_rna)
    max_len = int(cfg.get("max_len", 512))
    df = df[df["sequence"].str.len() <= max_len]
    positive_class = int(cfg["positive_class"])
    df["label"] = (df[cfg["label_column"]].astype(int) == positive_class).astype(int)
    df = add_core_columns(
        df,
        task_name,
        "sequence_classification",
        cfg.get("source_dataset_name", source_path.stem),
        family_col=cfg.get("family_column"),
    )
    df["label_mask"] = [[True]] * len(df)
    df = stratified_hash_split(
        df,
        "label",
        split_cfg["train"],
        split_cfg["val"],
        split_cfg["test"],
    )
    return df


def build_sequence_classification_dataset(
    task_name: str,
    cfg: dict[str, Any],
    split_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Generic sequence classification builder used for RBP and ncRNA."""
    source_path = resolve_path(cfg["source_path"])
    if source_path.suffix == ".parquet":
        df = pd.read_parquet(source_path)
    else:
        df = pd.read_csv(source_path)
    df = ensure_seq_id(df, task_name)
    df["sequence"] = df["sequence"].apply(normalize_rna)
    max_len = int(cfg.get("max_len", 512))
    df = df[df["sequence"].str.len() <= max_len]

    label_col = cfg["label_column"]
    if df[label_col].dtype == object:
        # Encode string labels as integer class ids while preserving the string in ``family``.
        label_map = {v: i for i, v in enumerate(sorted(df[label_col].unique()))}
        df["label"] = df[label_col].map(label_map).astype(int)
    else:
        df["label"] = df[label_col].astype(int)

    df = add_core_columns(
        df,
        task_name,
        "sequence_classification",
        cfg.get("source_dataset_name", source_path.stem),
        family_col=cfg.get("family_column"),
        cluster_col=cfg.get("cluster_column"),
    )
    df["label_mask"] = [[True]] * len(df)

    df = apply_source_split(df)
    missing = df["split"].isna()
    if missing.any():
        group_col = cfg.get("family_column") if cfg.get("family_column") in df.columns else None
        df_missing = group_aware_split(
            df[missing].copy(),
            group_col,
            split_cfg["train"],
            split_cfg["val"],
            split_cfg["test"],
            int(split_cfg.get("seed", 42)),
        )
        df.loc[missing, "split"] = df_missing["split"]
    return df


def build_chemical_reactivity_dataset(
    task_name: str,
    cfg: dict[str, Any],
    split_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Token regression from Ribonanza reactivity columns.

    Rows for the same ``sequence_id`` and multiple ``experiment_type`` values
    are aggregated by nan-mean so each sequence yields one per-nucleotide label.
    """
    source_path = resolve_path(cfg["source_path"])
    experiment_types = cfg.get("experiment_types")
    token_prefix = cfg.get("token_value_prefix", "reactivity_")
    max_len = int(cfg.get("max_len", 512))
    group_col = cfg.get("group_column", "dataset_name")

    # Determine the reactivity columns from the header, then read only what is needed.
    header = pd.read_csv(source_path, nrows=0)
    reactivity_cols = [
        c
        for c in header.columns
        if str(c).startswith(token_prefix) and "error" not in str(c).lower()
    ]
    if not reactivity_cols:
        raise ValueError(f"No reactivity columns with prefix {token_prefix!r} in {source_path}")

    usecols = ["sequence_id", "sequence", "experiment_type", group_col] + reactivity_cols
    dtype_map = {c: np.float32 for c in reactivity_cols}
    df = pd.read_csv(source_path, usecols=usecols, dtype=dtype_map, low_memory=False)

    if experiment_types:
        df = df[df["experiment_type"].isin(experiment_types)].copy()

    # Aggregate by sequence.  pandas mean skips NaNs by default.
    agg_map: dict[str, Any] = {
        "sequence": ("sequence", "first"),
        group_col: (group_col, "first"),
    }
    agg_map.update({c: (c, "mean") for c in reactivity_cols})
    grouped = df.groupby("sequence_id").agg(**agg_map).reset_index()

    grouped["sequence"] = grouped["sequence"].astype(str).apply(normalize_rna)
    grouped = grouped[
        (grouped["sequence"].str.len() > 0) & (grouped["sequence"].str.len() <= max_len)
    ].copy()

    reac_arr = grouped[reactivity_cols].to_numpy(dtype=np.float32)
    mask = np.isfinite(reac_arr)
    label = np.where(mask, reac_arr, 0.0).astype(np.float32)
    grouped["label"] = [arr.tolist() for arr in label]
    grouped["label_mask"] = [arr.tolist() for arr in mask]
    grouped = grouped.drop(columns=reactivity_cols)

    grouped = grouped.rename(columns={"sequence_id": "seq_id", group_col: "cluster"})
    grouped = add_core_columns(
        grouped,
        task_name,
        "token_regression",
        cfg.get("source_dataset_name", source_path.stem),
        cluster_col="cluster",
    )
    grouped = group_aware_split(
        grouped,
        "cluster",
        split_cfg["train"],
        split_cfg["val"],
        split_cfg["test"],
        int(split_cfg.get("seed", 42)),
    )
    return grouped


def build_secondary_structure_dataset(
    task_name: str,
    cfg: dict[str, Any],
    split_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Pairwise secondary-structure contact map builder."""
    source_path = resolve_path(cfg["source_path"])
    max_len = int(cfg.get("max_len", 512))
    rows: list[dict[str, Any]] = []
    with source_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seq = normalize_rna(obj.get("sequence", ""))
            if not seq or len(seq) > max_len:
                continue
            pairs = obj.get("pairs")
            if pairs is None and obj.get("dot_bracket"):
                pairs = dot_bracket_pairs(str(obj["dot_bracket"]))
            pairs = [(int(i), int(j)) for i, j in (pairs or [])]
            pair_mat = pair_matrix_from_pairs(len(seq), pairs)
            mask = valid_pair_mask(len(seq))
            rows.append(
                {
                    "seq_id": str(obj.get("seq_id", f"{task_name}_{len(rows)}")),
                    "sequence": seq,
                    "label": pair_mat,
                    "label_mask": mask,
                    "split": obj.get("split"),
                    "source": obj.get("source"),
                }
            )

    df = pd.DataFrame(rows)
    df = add_core_columns(
        df,
        task_name,
        "pairwise",
        cfg.get("source_dataset_name", source_path.stem),
        family_col=cfg.get("family_column", "source"),
        cluster_col=cfg.get("family_column", "source"),
    )
    df["label"] = listify_array_column(df["label"])
    df["label_mask"] = listify_array_column(df["label_mask"])

    df = apply_source_split(df)
    missing = df["split"].isna()
    if missing.any():
        group_col = cfg.get("family_column", "source")
        df_missing = group_aware_split(
            df[missing].copy(),
            group_col if group_col in df.columns else None,
            split_cfg["train"],
            split_cfg["val"],
            split_cfg["test"],
            int(split_cfg.get("seed", 42)),
        )
        df.loc[missing, "split"] = df_missing["split"]
    return df


def prepare_secondary_structure_streaming(
    task_name: str,
    cfg: dict[str, Any],
    split_cfg: dict[str, Any],
    hashes: set[str],
    output_root: Path,
) -> dict[str, Any]:
    """Memory-efficient secondary-structure builder that writes splits while streaming.

    Dense ``[L, L]`` contact matrices and valid-pair masks are written as
    variable-size nested lists directly to parquet.  This avoids holding the
    entire dataset in memory.
    """
    source_path = resolve_path(cfg["source_path"])
    max_len = int(cfg.get("max_len", 512))
    source_dataset = cfg.get("source_dataset_name", source_path.stem)
    family_col = cfg.get("family_column", "source")
    batch_size = 1024

    output_dir = Path(output_root) / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    schema = pa.schema(
        [
            ("seq_id", pa.string()),
            ("sequence", pa.string()),
            ("length", pa.int32()),
            ("label", pa.list_(pa.list_(pa.float32()))),
            ("label_mask", pa.list_(pa.list_(pa.bool_()))),
            ("sequence_hash", pa.string()),
            ("source_dataset", pa.string()),
            ("family", pa.string()),
            ("cluster", pa.string()),
            ("split", pa.string()),
            ("task", pa.string()),
            ("task_type", pa.string()),
        ]
    )

    writers: dict[str, pq.ParquetWriter] = {}
    buffers: dict[str, list[dict[str, Any]]] = {}
    for split in ("train", "val", "test"):
        writers[split] = pq.ParquetWriter(output_dir / f"{split}.parquet", schema)
        buffers[split] = []

    total_before = 0
    removed = 0
    removed_by_source: dict[str, int] = {}

    def _flush(split: str) -> None:
        buf = buffers[split]
        if not buf:
            return
        arrays = [pa.array([row[c] for row in buf], type=schema.field(c).type) for c in schema.names]
        writers[split].write_table(pa.Table.from_arrays(arrays, names=schema.names))
        buf.clear()

    def _split_for(source: Any, seq_id: str) -> str:
        s = normalize_split(source)
        if s:
            return s
        return stable_split(seq_id, split_cfg["train"], split_cfg["val"])

    with source_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            seq = normalize_rna(obj.get("sequence", ""))
            if not seq or len(seq) > max_len:
                continue
            seq_id = str(obj.get("seq_id", f"{task_name}_{total_before}"))
            source = obj.get("source")
            seq_hash = sequence_hash(seq)
            total_before += 1
            if seq_hash in hashes:
                removed += 1
                removed_by_source[str(source_dataset)] = removed_by_source.get(str(source_dataset), 0) + 1
                continue

            pairs = obj.get("pairs")
            if pairs is None and obj.get("dot_bracket"):
                pairs = dot_bracket_pairs(str(obj["dot_bracket"]))
            pairs = [(int(i), int(j)) for i, j in (pairs or [])]
            pair_mat = pair_matrix_from_pairs(len(seq), pairs).tolist()
            mask = valid_pair_mask(len(seq)).tolist()

            split = _split_for(source, seq_id)
            row = {
                "seq_id": seq_id,
                "sequence": seq,
                "length": np.int32(len(seq)),
                "label": pair_mat,
                "label_mask": mask,
                "sequence_hash": seq_hash,
                "source_dataset": source_dataset,
                "family": str(source) if source else None,
                "cluster": str(source) if source else None,
                "split": split,
                "task": task_name,
                "task_type": "pairwise",
            }
            buffers[split].append(row)
            if len(buffers[split]) >= batch_size:
                _flush(split)

    for split in writers:
        _flush(split)
        writers[split].close()

    split_counts = {split: int(len(buffers[split])) for split in writers}
    # Buffers were cleared on flush; re-count from files.
    split_counts = {}
    for split in writers:
        df = pd.read_parquet(output_dir / f"{split}.parquet")
        split_counts[split] = int(len(df))

    return {
        "task": task_name,
        "task_type": "pairwise",
        "total_before": total_before,
        "removed": removed,
        "remaining": total_before - removed,
        "contamination_rate": removed / max(total_before, 1),
        "split_counts": split_counts,
        "removed_by_source": removed_by_source,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_task_dataset(
    task_name: str,
    cfg: dict[str, Any],
    split_cfg: dict[str, Any],
) -> pd.DataFrame:
    """Dispatch to the correct builder for ``task_name``."""
    task_type = cfg.get("task_type", "sequence_classification")
    if task_name.startswith("splice_") and task_type == "sequence_classification":
        return build_splice_dataset(task_name, cfg, split_cfg)
    if task_type == "sequence_classification":
        return build_sequence_classification_dataset(task_name, cfg, split_cfg)
    if task_type == "token_regression":
        return build_chemical_reactivity_dataset(task_name, cfg, split_cfg)
    if task_type in {"pairwise", "rna3d_contact"}:
        return build_secondary_structure_dataset(task_name, cfg, split_cfg)
    raise ValueError(f"Unsupported task type {task_type!r} for {task_name}")


def write_task_splits(
    df: pd.DataFrame,
    output_dir: Path,
    task_name: str,
) -> dict[str, int]:
    """Write train/val/test parquet files and return per-split counts."""
    output_dir = Path(output_dir) / task_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Ensure all required columns are present.
    for col in REQUIRED_COLUMNS:
        if col not in df.columns:
            df[col] = None

    # Keep a stable column ordering.
    keep_cols = list(REQUIRED_COLUMNS) + [
        c for c in df.columns if c not in REQUIRED_COLUMNS
    ]
    df = df[keep_cols]

    split_counts = {}
    for split in ("train", "val", "test"):
        split_df = df[df["split"] == split].reset_index(drop=True)
        out_path = output_dir / f"{split}.parquet"
        split_df.to_parquet(out_path, index=False)
        split_counts[split] = int(len(split_df))
    return split_counts


def build_summary(
    task_name: str,
    decon: dict[str, Any],
    split_counts: dict[str, int],
    df: pd.DataFrame,
) -> dict[str, Any]:
    """Build a per-task decontamination / split summary dict."""
    summary = {
        "task": task_name,
        "task_type": str(df["task_type"].iloc[0]) if "task_type" in df.columns else None,
        "total_before": decon["total_before"],
        "removed": decon["removed"],
        "remaining": decon["remaining"],
        "contamination_rate": decon["removed"] / max(decon["total_before"], 1),
        "split_counts": split_counts,
        "label_summary": {},
    }
    if "label" in df.columns:
        try:
            first_label = df["label"].iloc[0]
            if isinstance(first_label, (list, np.ndarray)):
                summary["label_summary"] = {"n_samples": int(len(df))}
            else:
                summary["label_summary"] = df["label"].value_counts().to_dict()
        except Exception:
            summary["label_summary"] = {"n_samples": int(len(df))}
    return summary


def write_decontamination_report(
    summaries: list[dict[str, Any]],
    report_dir: Path,
) -> None:
    """Write aggregate markdown report and per-task JSON summaries."""
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    for summary in summaries:
        task_dir = report_dir / summary["task"]
        task_dir.mkdir(parents=True, exist_ok=True)
        json_path = task_dir / "decontamination_summary.json"
        json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    md_path = report_dir / "decontamination_report.md"
    lines = [
        "# Downstream Dataset Decontamination Report",
        "",
        "Decontaminated against Stage 4B sequence manifest by exact ``sequence_hash`` overlap.",
        "",
        "| task | total before | removed | remaining | contamination rate | train | val | test |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for s in summaries:
        sc = s.get("split_counts", {})
        lines.append(
            f"| {s['task']} | {s['total_before']} | {s['removed']} | {s['remaining']} "
            f"| {s['contamination_rate']:.4%} | {sc.get('train', 0)} | {sc.get('val', 0)} | {sc.get('test', 0)} |"
        )
    lines.append("")
    lines.append("## Per-task summaries")
    lines.append("")
    for s in summaries:
        lines.append(f"- **{s['task']}**: {s['remaining']} remaining rows "
                     f"({s['removed']} removed, {s['contamination_rate']:.4%})")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
