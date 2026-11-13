"""Stage 4B head-to-head evaluation.

:func:`evaluate_head2head_student` loads a trained
:class:`~rna_head_alignment.head2head.model.RiNALMoHeadToHeadStudent`, scores
it on val/test slim shards, evaluates native-head and linear-projection
baselines, and writes per-head and grouped-summary parquets.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from rna_head_alignment.head2head.head_manifest import ROUTE_DEPTHS, make_head_key
from rna_head_alignment.head2head.metrics import DEFAULT_METRICS, compute_head_metrics
from rna_head_alignment.head2head.model import RiNALMoHeadToHeadStudent
from rna_head_alignment.head2head.sequence_dataset import (
    SlimShardDataset,
    variable_length_collate,
)
from rna_head_alignment.head2head.student_backbone import RiNALMoStudentBackbone
from rna_head_alignment.io.paths import get_device, load_yaml, resolve_path


def _head_key_to_identity(head_key: str) -> dict[str, Any]:
    """Parse ``model:L{layer}:H{head}:L{depth}:task`` into named fields."""
    parts = head_key.split(":")
    if len(parts) != 5:
        return {
            "task": "unknown",
            "teacher_model": "unknown",
            "teacher_layer": -1,
            "teacher_head": -1,
        }
    model = parts[0]
    layer = int(parts[1][1:]) if parts[1].startswith("L") else -1
    head = int(parts[2][1:]) if parts[2].startswith("H") else -1
    # parts[3] is the route key, e.g. L6
    task = parts[4]
    return {
        "task": task,
        "teacher_model": model,
        "teacher_layer": layer,
        "teacher_head": head,
    }


def _length_bin(length: int, bins: list[int] | None) -> str:
    """Bucket ``length`` into the configured bin edges."""
    bins = list(bins or [0, 64, 128, 256, 384, 512])
    if len(bins) < 2:
        return "all"
    for i in range(len(bins) - 1):
        if bins[i] <= length < bins[i + 1]:
            return f"{bins[i]}-{bins[i + 1]}"
    return f"{bins[-1]}+"


def _load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> dict[str, Any]:
    ckpt = torch.load(resolve_path(checkpoint_path), map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict):
        raise ValueError(f"Checkpoint {checkpoint_path} is not a dict")
    return ckpt


def _build_adapter_specs(
    pair_manifest_path: str | Path,
    teacher_head_ids: dict[int, list[str]],
) -> tuple[dict[int, list[tuple[str, int]]], dict[int, list[int]]]:
    """Build adapter specs and paired native student-head lists from manifests.

    Mirrors the construction used in training (see ``train_ddp.py``) so that
    evaluation reconstructs the exact same adapter keys and student-head
    pairings.
    """
    df = pd.read_parquet(resolve_path(pair_manifest_path))
    df = df[df["keep"].fillna(True)].copy()

    def _row_head_key(row: pd.Series) -> str:
        route_key = f"L{int(row['route_depth'])}"
        return make_head_key(
            str(row["teacher_model"]),
            int(row["teacher_layer"]),
            int(row["teacher_head"]),
            route_key,
            str(row["task"]),
        )

    df["head_key"] = df.apply(_row_head_key, axis=1)
    row_by_key = {row["head_key"]: row for _, row in df.iterrows()}

    adapter_specs: dict[int, list[tuple[str, int]]] = {}
    student_heads: dict[int, list[int]] = {}
    for depth in ROUTE_DEPTHS:
        ids = teacher_head_ids.get(depth, [])
        specs: list[tuple[str, int]] = []
        heads: list[int] = []
        for head_key in ids:
            row = row_by_key.get(head_key)
            if row is None:
                raise KeyError(
                    f"head_pair_manifest missing head_key {head_key!r}; "
                    "cannot align adapters to slim shard order"
                )
            specs.append((str(row["adapter_key"]), int(row["student_head"])))
            heads.append(int(row["student_head"]))
        adapter_specs[depth] = specs
        student_heads[depth] = heads
    return adapter_specs, student_heads


def _build_model(
    cfg: dict[str, Any],
    device: torch.device,
    teacher_head_ids: dict[int, list[str]] | None = None,
    route_head_keys: dict[int, list[str]] | None = None,
) -> RiNALMoHeadToHeadStudent:
    """Build a fresh student model from config (state is loaded separately).

    If ``head_pair_manifest`` is present in ``cfg`` and teacher-head identity
    lists are supplied, the model is reconstructed with the exact adapter keys
    and native student-head pairings used during training.
    """
    student_cfg = cfg.get("student", {})
    model_dir = student_cfg.get("model_dir", "assets/models/rinalmo-micro")
    routes = tuple(student_cfg.get("routes", ROUTE_DEPTHS))
    # Training runs may store backbone config overrides under student or model.
    config_overrides = student_cfg.get("config_overrides") or cfg.get("model", {}).get("config_overrides") or None

    adapter_specs = None
    student_heads = student_cfg.get("student_heads") or None
    pair_manifest_path = cfg.get("head_pair_manifest")
    if cfg.get("_force_empty_adapters"):
        # Alignment-free checkpoint (MLM-only): no adapters were ever trained.
        adapter_specs = {}
        student_heads = None
    elif pair_manifest_path is not None:
        ids = teacher_head_ids if teacher_head_ids is not None else route_head_keys
        if ids is not None:
            adapter_specs, student_heads = _build_adapter_specs(pair_manifest_path, ids)

    backbone = RiNALMoStudentBackbone(
        model_dir=model_dir,
        routes=routes,
        student_heads=student_heads,
        config_overrides=config_overrides,
    )
    adapter_dim = cfg.get("model", {}).get("adapter_dim", 64)
    model = RiNALMoHeadToHeadStudent(
        backbone=backbone,
        adapter_specs=adapter_specs,
        adapter_dim=adapter_dim,
    )
    model.to(device)
    return model


def _load_model_for_eval(
    checkpoint_path: str | Path,
    cfg: dict[str, Any],
    device: torch.device,
    teacher_head_ids: dict[int, list[str]] | None = None,
    route_head_keys: dict[int, list[str]] | None = None,
) -> tuple[RiNALMoHeadToHeadStudent, dict[str, Any]]:
    """Load a trained student and its checkpoint dict."""
    ckpt = _load_checkpoint(checkpoint_path, device)
    checkpoint_config = ckpt.get("config") if isinstance(ckpt, dict) else None

    # Merge any architecture information stored in the checkpoint over the
    # eval config so tiny test models can be reconstructed exactly.
    effective_cfg = dict(cfg)
    if checkpoint_config is not None:
        effective_cfg["student"] = {**effective_cfg.get("student", {}), **checkpoint_config.get("student", {})}
        effective_cfg["model"] = {**effective_cfg.get("model", {}), **checkpoint_config.get("model", {})}
        # Carry forward the manifests that define adapter pairing.
        for key in ("head_pair_manifest", "target_head_manifest"):
            if key in checkpoint_config and key not in effective_cfg:
                effective_cfg[key] = checkpoint_config[key]

    state = None
    for key in ("model_state_dict", "model", "state_dict", "aligners"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            state = ckpt[key]
            break
    if state is None:
        if isinstance(ckpt, dict):
            state = ckpt
        else:
            raise ValueError("Checkpoint does not contain a state_dict")

    # Checkpoints from alignment-free runs (e.g. MLM-only pretraining) contain
    # no adapter weights; rebuild the model with an empty adapter set so the
    # state dict matches instead of inventing default adapters.
    if not any(k.startswith("adapters.") for k in state):
        effective_cfg["_force_empty_adapters"] = True

    model = _build_model(
        effective_cfg,
        device,
        teacher_head_ids=teacher_head_ids,
        route_head_keys=route_head_keys,
    )

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(
            f"Checkpoint load missing keys ({len(missing)}): {missing[:5]}..."
        )
    model.eval()
    return model, ckpt


def _resolve_split_shards(
    cfg: dict[str, Any], checkpoint_config: dict[str, Any] | None
) -> dict[str, list[int]]:
    """Return ``{"val": [...], "test": [...]}`` shard index lists."""
    splits_cfg = cfg.get("splits", {})
    val = list(splits_cfg.get("val_shards") or [])
    test = list(splits_cfg.get("test_shards") or [])
    if not val and not test and checkpoint_config is not None:
        val = list(checkpoint_config.get("val_shards") or checkpoint_config.get("splits", {}).get("val_shards") or [])
        test = list(checkpoint_config.get("test_shards") or checkpoint_config.get("splits", {}).get("test_shards") or [])
    if not val and not test:
        raise ValueError(
            "No val/test shards configured and none found in checkpoint config"
        )
    return {"val": val, "test": test}


def _load_head_ids(slim_cache_root: str | Path, shard_idx: int) -> dict[int, list[str]]:
    path = resolve_path(slim_cache_root) / f"shard_{shard_idx:05d}.pt"
    shard = torch.load(path, map_location="cpu", weights_only=False)
    return {depth: shard[f"teacher_head_ids_L{depth}"] for depth in ROUTE_DEPTHS}


def _build_linear_projections(model: RiNALMoHeadToHeadStudent) -> dict[int, nn.ModuleList]:
    """Random per-head linear projections for the linear-projection baseline."""
    projs: dict[int, nn.ModuleList] = {}
    device = next(model.parameters()).device
    for depth, specs in model.adapter_specs.items():
        route_projs = nn.ModuleList(
            nn.Linear(model.backbone.head_dim, model.backbone.head_dim, bias=False)
            for _ in specs
        )
        for proj in route_projs:
            nn.init.xavier_uniform_(proj.weight)
            proj.requires_grad_(False)
        projs[depth] = route_projs.to(device)
    return projs


def _native_head_maps(
    head_outputs: torch.Tensor, key_padding_mask: torch.Tensor | None, eps: float = 1e-8
) -> torch.Tensor:
    """Self-attention maps from paired native head outputs ``[B, K, L, D]``."""
    B, K, L, D = head_outputs.shape
    scores = torch.einsum("bkld,bked->bkle", head_outputs, head_outputs) / math.sqrt(D)
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
    probs = F.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return probs


def _linear_head_maps(
    head_outputs: torch.Tensor,
    projections: nn.ModuleList,
    key_padding_mask: torch.Tensor | None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Linearly-projected maps from paired native head outputs."""
    B, K, L, D = head_outputs.shape
    transformed = torch.stack(
        [projections[k](head_outputs[:, k]) for k in range(K)], dim=1
    )
    scores = torch.einsum("bkld,bked->bkle", transformed, head_outputs) / math.sqrt(D)
    if key_padding_mask is not None:
        scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf"))
    probs = F.softmax(scores, dim=-1)
    probs = torch.nan_to_num(probs, nan=0.0)
    return probs


def _random_head_maps(
    head_outputs: torch.Tensor,
    model: RiNALMoHeadToHeadStudent,
    depth: int,
    key_padding_mask: torch.Tensor | None,
) -> torch.Tensor:
    """Run each teacher-head adapter on a different (deterministic) student head."""
    B, K, L, D = head_outputs.shape
    specs = model.adapter_specs[depth]
    preds = []
    for key, paired_head in specs:
        random_head = (paired_head + 1) % K
        pred_map = model.adapters[key](head_outputs[:, random_head], key_padding_mask)["pred_map"]
        preds.append(pred_map)
    return torch.stack(preds, dim=1)


def _evaluate_batch(
    model: RiNALMoHeadToHeadStudent,
    batch: dict[str, Any],
    head_ids_by_route: dict[int, list[str]],
    baseline_projections: dict[int, nn.ModuleList],
    metric_names: list[str],
    baselines: list[str],
    split: str,
    length_bins: list[int] | None,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> list[dict[str, Any]]:
    """Return one metric row per (sequence, head, metric, baseline)."""
    input_ids = batch["input_ids"].to(device)
    attention_mask = batch["attention_mask"].to(device)
    lengths = batch["lengths"].to(device)
    key_padding_mask = attention_mask == 0

    with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp, dtype=amp_dtype):
        out = model(input_ids, attention_mask, return_backbone=True)
        backbone = out["backbone"]

    rows: list[dict[str, Any]] = []
    for depth in ROUTE_DEPTHS:
        route_key = f"L{depth}"
        if f"teacher_{route_key}" not in batch:
            continue
        if f"pred_{route_key}" not in out:
            continue
        teacher = batch[f"teacher_{route_key}"].to(device).float()
        student_pred = out[f"pred_{route_key}"].float()
        head_outputs = backbone[depth]["head_outputs"].float()
        K = head_outputs.shape[1]
        head_ids = head_ids_by_route[depth]
        T = student_pred.shape[1]
        if T != len(head_ids):
            raise ValueError(
                f"Route {route_key}: prediction heads {T} != manifest head_ids {len(head_ids)}"
            )

        # Pre-compute baseline maps once per route.
        baseline_maps: dict[str, torch.Tensor] = {}
        if "native_head" in baselines:
            baseline_maps["native_head"] = _native_head_maps(head_outputs, key_padding_mask)
        if "linear_projection" in baselines:
            baseline_maps["linear_projection"] = _linear_head_maps(
                head_outputs, baseline_projections[depth], key_padding_mask
            )
        if "random_student_head" in baselines:
            baseline_maps["random_student_head"] = _random_head_maps(
                head_outputs, model, depth, key_padding_mask
            )

        for baseline_name in [None] + baselines:
            if baseline_name is None:
                pred_maps = student_pred
            else:
                pred_maps = baseline_maps[baseline_name]

            metrics = compute_head_metrics(
                pred_maps,
                teacher,
                lengths,
                metrics=metric_names,
                include_per_sequence=True,
            )

            for t in range(T):
                head_key = head_ids[t]
                identity = _head_key_to_identity(head_key)
                layer_head = f"L{identity['teacher_layer']}:H{identity['teacher_head']}"
                for metric in metric_names:
                    per_seq = metrics[f"{metric}_per_seq"][:, t].detach().cpu().numpy()
                    for b, val in enumerate(per_seq):
                        rows.append(
                            {
                                "seq_id": batch["seq_ids"][b],
                                "shard_id": batch["shard_ids"][b],
                                "split": split,
                                "route_depth": depth,
                                "head_idx": t,
                                "head_key": head_key,
                                "task": identity["task"],
                                "teacher_model": identity["teacher_model"],
                                "teacher_layer": identity["teacher_layer"],
                                "teacher_head": identity["teacher_head"],
                                "teacher_layer_head": layer_head,
                                "length": int(lengths[b].item()),
                                "length_bin": _length_bin(int(lengths[b].item()), length_bins),
                                "metric": metric,
                                "value": float("nan") if np.isnan(val) else float(val),
                                "baseline": baseline_name or "student",
                            }
                        )
    return rows


def _summarize(
    df: pd.DataFrame, group_by: list[str] | None, length_bins: list[int] | None
) -> pd.DataFrame:
    """Grouped summary of per-head metrics."""
    group_by = list(group_by or ["task", "route_depth", "teacher_model", "teacher_layer_head", "length_bin"])
    for col in group_by:
        if col not in df.columns:
            raise ValueError(f"Group-by column {col!r} not in metrics dataframe")
    agg = (
        df.groupby(group_by + ["metric", "baseline"], dropna=False)["value"]
        .agg(["mean", "std", "median", "count"])
        .reset_index()
    )
    return agg


def evaluate_head2head_student(
    config_path: str | Path,
    checkpoint_path: str | Path,
    run_name: str | None = None,
    output_dir: str | Path | None = None,
    device: str | torch.device = "auto",
    splits: list[str] | None = None,
    max_shards: int | None = None,
) -> dict[str, Any]:
    """Evaluate a trained head-to-head student and write parquets.

    Parameters
    ----------
    config_path:
        Path to ``eval_head2head.yaml``.
    checkpoint_path:
        Path to trained model checkpoint.
    run_name:
        Run identifier; defaults to ``run_name`` in config.
    output_dir:
        Root output directory; defaults to ``output_dir`` in config.
    device:
        ``auto`` selects CUDA if available.
    splits:
        Optional list of splits to run (e.g. ``["val"]``).
    max_shards:
        Limit number of shards per split for quick smoke tests.

    Returns
    -------
    Dict mapping split name to paths and row counts.
    """
    cfg = load_yaml(config_path)
    run_name = run_name or cfg.get("run_name")
    if not run_name:
        raise ValueError("run_name must be provided or set in config")

    if output_dir is not None:
        out_root = Path(output_dir) / run_name
    else:
        out_root = resolve_path(cfg.get("output_dir", "outputs/alignment/stage4b_head2head/eval")) / run_name
    out_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(get_device(device) if str(device).lower() == "auto" else device)
    precision = cfg.get("precision", "bf16")
    if precision == "bf16":
        amp_dtype = torch.bfloat16
    elif precision == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32
    use_amp = precision in ("bf16", "fp16") and device.type == "cuda"

    # Resolve teacher-head identities before building the model so the exact
    # adapter keys and native-head pairings used during training are restored.
    temp_ckpt = _load_checkpoint(checkpoint_path, device)
    checkpoint_config = temp_ckpt.get("config") if isinstance(temp_ckpt, dict) else None
    shard_lists = _resolve_split_shards(cfg, checkpoint_config)
    first_shard_idx = next(
        (idx for indices in shard_lists.values() for idx in indices), None
    )
    if first_shard_idx is None:
        raise ValueError("No shards to evaluate")
    head_ids_by_route = _load_head_ids(cfg["slim_cache_root"], first_shard_idx)

    model, ckpt = _load_model_for_eval(
        checkpoint_path, cfg, device, teacher_head_ids=head_ids_by_route
    )
    tokenizer = model.backbone.tokenizer()

    if splits is not None:
        shard_lists = {k: v for k, v in shard_lists.items() if k in splits}

    baseline_projections = _build_linear_projections(model)
    metric_names = list(cfg.get("metrics", list(DEFAULT_METRICS)))
    baselines = list(cfg.get("baselines", ["native_head", "linear_projection", "random_student_head"]))
    length_bins = list(cfg.get("length_bins", [0, 64, 128, 256, 384, 512]))
    max_len = int(cfg.get("max_len", 512))
    batch_size = int(cfg.get("batch_size", 8))

    report: dict[str, Any] = {"run_name": run_name, "output_root": str(out_root), "splits": {}}

    for split_name, shard_indices in shard_lists.items():
        if max_shards is not None:
            shard_indices = shard_indices[:max_shards]
        if not shard_indices:
            continue

        rows: list[dict[str, Any]] = []
        for shard_idx in shard_indices:
            shard_path = resolve_path(cfg["slim_cache_root"]) / f"shard_{shard_idx:05d}.pt"
            dataset = SlimShardDataset(
                [str(shard_path)], tokenizer, max_len=max_len, cache_size=1
            )
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                collate_fn=variable_length_collate,
                shuffle=False,
                num_workers=0,
            )
            for batch in loader:
                rows.extend(
                    _evaluate_batch(
                        model,
                        batch,
                        head_ids_by_route,
                        baseline_projections,
                        metric_names,
                        baselines,
                        split_name,
                        length_bins,
                        device,
                        use_amp,
                        amp_dtype,
                    )
                )

        split_dir = out_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(rows)
        per_head_path = split_dir / "per_head_metrics.parquet"
        df.to_parquet(per_head_path, index=False)

        summary = _summarize(df, cfg.get("group_by"), length_bins)
        summary_path = split_dir / "summary.parquet"
        summary.to_parquet(summary_path, index=False)

        report["splits"][split_name] = {
            "per_head_metrics": str(per_head_path),
            "summary": str(summary_path),
            "n_rows": len(df),
            "n_shards": len(shard_indices),
        }

    return report
