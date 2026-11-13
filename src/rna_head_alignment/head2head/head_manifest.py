"""Stage 4B head manifests: target teacher heads and default head pairs.

Builds two parquet manifests from ``configs/teacher_bank_selected.yaml``:

- ``target_head_manifest``: one row per preserved teacher head (15 rows,
  5 per route depth 6/8/12), with the cache channel index that locates the
  head inside its teacher model's attention-cache shard.
- ``head_pair_manifest``: one row per manual student-head <-> teacher-head
  pairing from ``student_teacher_mapping``.

Channel indices and ``head_key`` strings must match the on-disk teacher
attention cache exactly (see :func:`compute_model_channel_map`).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROUTE_DEPTHS = (6, 8, 12)

TARGET_MANIFEST_COLUMNS = [
    "task",
    "route_depth",
    "teacher_model",
    "teacher_layer",
    "teacher_head",
    "cache_channel",
    "selection_score",
    "selection_metric",
    "rank",
    "keep",
    "loss_weight",
    "head_key",
    "student_route_layer",
]

PAIR_MANIFEST_COLUMNS = [
    "task",
    "route_depth",
    "teacher_model",
    "teacher_layer",
    "teacher_head",
    "teacher_cache_channel",
    "student_model",
    "student_layer",
    "student_head",
    "matching_metric",
    "matching_score",
    "pairing_method",
    "keep",
    "adapter_key",
]

STUDENT_MODEL = "rinalmo_micro"


def route_key_to_depth(route_key: str) -> int:
    """Map a bank route key like ``"L6"`` to its depth ``6``."""
    key = str(route_key).strip()
    if key.upper().startswith("L"):
        key = key[1:]
    depth = int(key)
    if depth not in ROUTE_DEPTHS:
        raise ValueError(f"unknown route depth {depth} (from {route_key!r})")
    return depth


def make_head_key(model: str, layer: int, head: int, route_key: str, task: str) -> str:
    """Canonical head key, e.g. ``'rnafm:L4:H3:L6:manual_L6_sequence_bank'``.

    This is the same format used in the ``head_keys`` field of the teacher
    attention cache shards; do not change it without rebuilding the cache.
    """
    return f"{model}:L{int(layer)}:H{int(head)}:{route_key}:{task}"


def load_teacher_bank(bank_path: str | Path) -> dict:
    """Parse the teacher-bank YAML config."""
    with open(bank_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _iter_bank_heads(bank: dict):
    """Yield (route_key, route_cfg, task_name, head_entry, bank_order) in the
    canonical cache-builder traversal order: banks L6 -> L8 -> L12, then tasks
    in YAML order, then teacher_heads in YAML order."""
    banks = bank.get("banks", {})
    order = 0
    for depth in ROUTE_DEPTHS:
        route_key = f"L{depth}"
        route_cfg = banks.get(route_key)
        if route_cfg is None:
            continue
        for task_name, task_cfg in (route_cfg.get("tasks") or {}).items():
            for head_entry in task_cfg.get("teacher_heads", []):
                yield route_key, route_cfg, task_name, head_entry, order
                order += 1


def compute_model_channel_map(bank_path: str | Path) -> dict[str, list[str]]:
    """Map each teacher model to its ordered list of channel head_keys.

    The traversal order (banks L6 -> L8 -> L12, tasks in YAML order, heads in
    YAML order, deduplicated per model by (layer, head)) MUST stay identical
    to the cache builder (``build_teacher_attention_cache``): channel ``k`` of
    the ``attention`` tensor in a shard corresponds to
    ``channel_map[model][k]``. Slim-cache code never trusts position blindly —
    it re-validates against each shard's ``head_keys`` — but this map defines
    the expected layout and the ``cache_channel`` column of the manifests.
    """
    bank = load_teacher_bank(bank_path)
    channel_map: dict[str, list[str]] = {}
    seen: dict[str, set[tuple[int, int]]] = {}
    for route_key, _route_cfg, task_name, head_entry, _order in _iter_bank_heads(bank):
        model = head_entry["model"]
        layer = int(head_entry["layer"])
        head = int(head_entry["head"])
        key = make_head_key(model, layer, head, route_key, task_name)
        seen.setdefault(model, set())
        if (layer, head) in seen[model]:
            continue
        seen[model].add((layer, head))
        channel_map.setdefault(model, []).append(key)
    return channel_map


def _backfill_scores_from_atlas(df: pd.DataFrame, atlas_csv: str | Path) -> pd.DataFrame:
    """Fill selection_score / selection_metric from the head-screening atlas.

    For each target head, match atlas rows by (model, layer, head) and keep
    the maximum of the atlas ``metric_primary`` column (the per-task primary
    score value; the atlas stores the value itself, not a metric name).
    Backfilled rows get selection_metric "metric_primary"; heads without any
    usable atlas score keep NaN / "manual".
    """
    atlas = pd.read_csv(atlas_csv)
    primary = pd.to_numeric(atlas["metric_primary"], errors="coerce")
    scores: list[float] = []
    metrics: list[str] = []
    for _, row in df.iterrows():
        mask = (
            (atlas["model"] == row["teacher_model"])
            & (atlas["layer"] == row["teacher_layer"])
            & (atlas["head"] == row["teacher_head"])
        )
        values = primary[mask].dropna()
        if len(values):
            scores.append(float(values.max()))
            metrics.append("metric_primary")
        else:
            scores.append(float("nan"))
            metrics.append("manual")
    df = df.copy()
    df["selection_score"] = scores
    df["selection_metric"] = metrics
    return df


def build_target_head_manifest(
    bank_path: str | Path,
    atlas_csv: str | Path | None = None,
) -> pd.DataFrame:
    """Build the 15-row target teacher-head manifest.

    Columns: task, route_depth, teacher_model, teacher_layer, teacher_head,
    cache_channel, selection_score, selection_metric, rank, keep, loss_weight,
    head_key, student_route_layer. ``cache_channel`` is the head's channel
    index inside its teacher model's attention-cache shard (see
    :func:`compute_model_channel_map`). ``rank`` orders heads within each
    route by selection_score descending; heads without a score keep bank
    order after the scored heads.
    """
    bank = load_teacher_bank(bank_path)
    channel_map = compute_model_channel_map(bank_path)
    channel_of = {
        key: idx for model, keys in channel_map.items() for idx, key in enumerate(keys)
    }

    rows = []
    for route_key, route_cfg, task_name, head_entry, bank_order in _iter_bank_heads(bank):
        model = head_entry["model"]
        layer = int(head_entry["layer"])
        head = int(head_entry["head"])
        head_key = make_head_key(model, layer, head, route_key, task_name)
        rows.append(
            {
                "task": task_name,
                "route_depth": route_key_to_depth(route_key),
                "teacher_model": model,
                "teacher_layer": layer,
                "teacher_head": head,
                "cache_channel": channel_of[head_key],
                "selection_score": float("nan"),
                "selection_metric": "manual",
                "rank": 0,
                "keep": True,
                "loss_weight": 1.0,
                "head_key": head_key,
                "student_route_layer": int(route_cfg["student_layer"]),
                "_bank_order": bank_order,
            }
        )
    df = pd.DataFrame(rows)

    if atlas_csv is not None:
        df = _backfill_scores_from_atlas(df, atlas_csv)

    # Rank within each route: scored heads first (score desc), then unscored
    # heads in bank order.
    df["_score_sort"] = df["selection_score"].fillna(-np.inf)
    df = df.sort_values(
        ["route_depth", "_score_sort", "_bank_order"],
        ascending=[True, False, True],
    )
    df["rank"] = df.groupby("route_depth").cumcount() + 1
    df = df.sort_values("_bank_order").drop(columns=["_bank_order", "_score_sort"])
    return df[TARGET_MANIFEST_COLUMNS].reset_index(drop=True)


def build_default_head_pair_manifest(bank_path: str | Path) -> pd.DataFrame:
    """Build the student-head <-> teacher-head pairing manifest from
    ``student_teacher_mapping`` (the manual seed pairs)."""
    bank = load_teacher_bank(bank_path)
    channel_map = compute_model_channel_map(bank_path)
    channel_of: dict[tuple[str, int, int], int] = {}
    for model, keys in channel_map.items():
        for idx, key in enumerate(keys):
            parts = key.split(":")
            layer = int(parts[1][1:])
            head = int(parts[2][1:])
            channel_of[(model, layer, head)] = idx

    rows = []
    mapping = bank.get("student_teacher_mapping", {})
    for depth in ROUTE_DEPTHS:
        route_key = f"L{depth}"
        for entry in mapping.get(route_key, []):
            for teacher in entry.get("teacher_group", []):
                model = teacher["model"]
                layer = int(teacher["layer"])
                head = int(teacher["head"])
                rows.append(
                    {
                        "task": teacher["task"],
                        "route_depth": depth,
                        "teacher_model": model,
                        "teacher_layer": layer,
                        "teacher_head": head,
                        "teacher_cache_channel": channel_of[(model, layer, head)],
                        "student_model": STUDENT_MODEL,
                        "student_layer": depth,
                        "student_head": int(entry["student_head"]),
                        "matching_metric": None,
                        "matching_score": float("nan"),
                        "pairing_method": "manual_seed",
                        "keep": True,
                        "adapter_key": entry["aligner"],
                    }
                )
    return pd.DataFrame(rows, columns=PAIR_MANIFEST_COLUMNS)


def save_manifest(df: pd.DataFrame, path: str | Path) -> Path:
    """Save a manifest as parquet, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def load_manifest(path: str | Path) -> pd.DataFrame:
    """Load a manifest parquet file."""
    return pd.read_parquet(path)
