from __future__ import annotations

import json
import math
import os
import socket
import subprocess
import time
from contextlib import nullcontext
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, DistributedSampler

try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore[misc,assignment]

from rna_head_alignment.head2head.full_sequence_dataset import (
    ROUTE_KEYS as FULL_ROUTE_KEYS,
    FullSlimBucketDataset,
    LengthBucketedSampler,
    full_variable_length_collate,
)
from rna_head_alignment.head2head.gradient_routing import (
    compute_route_losses,
    gradient_diagnostics,
    layer_grad_norms,
    route_grad_norms,
)
from rna_head_alignment.head2head.head_manifest import ROUTE_DEPTHS, make_head_key
from rna_head_alignment.head2head.losses import route_alignment_loss, total_route_loss
from rna_head_alignment.head2head.mlm import (
    RiNALMoMLMHead,
    mask_tokens,
    mlm_loss_from_logits,
)
from rna_head_alignment.head2head.model import RiNALMoHeadToHeadStudent
from rna_head_alignment.head2head.sequence_dataset import (
    LengthBucketedBatchSampler,
    SlimShardDataset,
    variable_length_collate,
)
from rna_head_alignment.head2head.student_backbone import (
    DEFAULT_ROUTES,
    RiNALMoStudentBackbone,
)


ROUTE_KEYS = tuple(f"L{d}" for d in DEFAULT_ROUTES)


def setup_ddp(rank: int, world_size: int, timeout_sec: int = 1800) -> None:
    os.environ.setdefault("MASTER_ADDR", "localhost")
    os.environ.setdefault("MASTER_PORT", "29504")
    if not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(
            backend,
            rank=rank,
            world_size=world_size,
            timeout=timedelta(seconds=timeout_sec),
        )


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def all_reduce_dict(values: dict[str, float], device: torch.device) -> dict[str, float]:
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return values
    keys = sorted(values.keys())
    tensor = torch.tensor(
        [values.get(k, 0.0) for k in keys], dtype=torch.float32, device=device
    )
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return {k: float(tensor[i]) for i, k in enumerate(keys)}


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


def _resolve_shard_paths(
    slim_cache_root: str | Path,
    shards_cfg: Any,
    n_shards: int | None,
    shard_stride: int | None,
) -> list[Path]:
    root = Path(slim_cache_root)

    if isinstance(shards_cfg, str) and shards_cfg.lower() == "all":
        indices = None
    elif isinstance(shards_cfg, (list, tuple)) and shards_cfg:
        indices = [int(x) for x in shards_cfg]
    elif n_shards is not None and shard_stride is not None:
        indices = [int(i * shard_stride) for i in range(int(n_shards))]
    else:
        raise ValueError(
            "set shards to 'all', a list, or provide n_shards+shard_stride"
        )

    if indices is not None:
        paths = [root / f"shard_{idx:05d}.pt" for idx in indices]
        missing = [p for p in paths if not p.exists()]
        if missing:
            raise FileNotFoundError(f"missing slim shards: {missing}")
        return paths

    paths = sorted(root.glob("shard_*.pt"), key=lambda p: int(p.stem.split("_")[1]))
    if not paths:
        raise FileNotFoundError(f"no shard_*.pt files under {root}")
    return paths


def _load_first_slim_shard(paths: list[Path]) -> dict[str, Any]:
    for p in paths:
        if p.exists():
            return torch.load(p, map_location="cpu", weights_only=False)
    raise FileNotFoundError(f"no slim shards found in {paths[:3]}...")


def _build_adapter_specs(
    pair_manifest_path: str | Path,
    teacher_head_ids: dict[int, list[str]],
) -> tuple[dict[int, list[tuple[str, int]]], dict[int, list[int]]]:
    df = pd.read_parquet(pair_manifest_path)
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


def _make_student(
    config: dict[str, Any],
    pair_manifest_path: str | Path,
    teacher_head_ids: dict[int, list[str]],
    device: torch.device,
) -> tuple[RiNALMoHeadToHeadStudent, dict[str, Any]]:
    model_cfg = config.get("model", {})
    adapter_dim = int(model_cfg.get("adapter_dim", 64))
    ffn_mult = int(model_cfg.get("adapter_ffn_mult", 2))
    train_backbone = bool(model_cfg.get("train_backbone", True))
    act_ckpt = bool(model_cfg.get("activation_checkpointing", False))

    adapter_specs, student_heads = _build_adapter_specs(
        pair_manifest_path, teacher_head_ids
    )

    backbone = RiNALMoStudentBackbone(
        model_dir=Path(config["student"]["model_dir"]),
        routes=DEFAULT_ROUTES,
        student_heads=student_heads,
    )

    if act_ckpt:
        try:
            backbone.model.gradient_checkpointing_enable()
        except AttributeError:
            pass

    for p in backbone.parameters():
        p.requires_grad_(train_backbone)

    student = RiNALMoHeadToHeadStudent(
        backbone=backbone,
        adapter_specs=adapter_specs,
        adapter_dim=adapter_dim,
        ffn_multiplier=ffn_mult,
    )
    student = student.to(device)
    return student, student.count_parameters()


def _save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    step: int,
    metrics: dict[str, float],
    path: Path,
    config: dict[str, Any] | None = None,
    splits: dict[str, list[int]] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
    }
    if config is not None:
        payload["config"] = config
    if splits is not None:
        payload["splits"] = splits
    torch.save(payload, path)


def _load_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    device: torch.device,
) -> int:
    map_location = {"cuda:0": f"cuda:{device.index}"} if device.type == "cuda" else device.type
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    if scheduler and ckpt.get("scheduler_state_dict"):
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return int(ckpt.get("step", 0))


# ---------------------------------------------------------------------------
# Full-cache helpers
# ---------------------------------------------------------------------------

def _load_first_bucket_route_heads(bucket_manifest_path: str | Path) -> dict[int, list[str]]:
    """Read route head-key lists from the first bucket in the manifest."""
    import json

    from safetensors import safe_open

    bucket_df = pd.read_parquet(bucket_manifest_path).dropna(subset=["file_path"])
    if bucket_df.empty:
        raise FileNotFoundError(f"no bucket files listed in {bucket_manifest_path}")
    first_path = Path(str(bucket_df["file_path"].iloc[0]))
    if not first_path.is_absolute():
        # Manifest may store the path relative to cwd or relative to the cache root.
        candidate_cwd = Path.cwd() / first_path
        candidate_root = Path(bucket_manifest_path).parent.parent / first_path
        first_path = candidate_cwd if candidate_cwd.exists() else candidate_root
    with safe_open(str(first_path), framework="pt") as f:
        metadata = f.metadata() or {}
    return {
        depth: json.loads(metadata.get(f"route_heads_L{depth}", "[]"))
        for depth in ROUTE_DEPTHS
    }


def _make_student_full(
    config: dict[str, Any],
    pair_manifest_path: str | Path,
    route_head_keys: dict[int, list[str]],
    device: torch.device,
) -> tuple[RiNALMoHeadToHeadStudent, dict[str, int]]:
    """Build the student model for the full bucketed slim cache."""
    model_cfg = config.get("model", {})
    student_cfg = config.get("student", {})
    adapter_dim = int(model_cfg.get("adapter_dim", 64))
    ffn_mult = int(model_cfg.get("adapter_ffn_mult", 2))
    train_backbone = bool(model_cfg.get("train_backbone", True))
    act_ckpt = bool(model_cfg.get("activation_checkpointing", False))
    model_dir = Path(student_cfg.get("model_dir", "assets/models/rinalmo-micro"))
    config_overrides = student_cfg.get("config_overrides") or model_cfg.get("config_overrides") or None

    adapter_specs, student_heads = _build_adapter_specs(
        pair_manifest_path, route_head_keys
    )

    backbone = RiNALMoStudentBackbone(
        model_dir=model_dir,
        routes=DEFAULT_ROUTES,
        student_heads=student_heads,
        config_overrides=config_overrides,
    )

    if act_ckpt:
        try:
            backbone.model.gradient_checkpointing_enable()
        except AttributeError:
            pass

    for p in backbone.parameters():
        p.requires_grad_(train_backbone)

    student = RiNALMoHeadToHeadStudent(
        backbone=backbone,
        adapter_specs=adapter_specs,
        adapter_dim=adapter_dim,
        ffn_multiplier=ffn_mult,
    )

    # Optional MLM auxiliary head (small-student S1 arm). Attached as a
    # submodule *before* DDP wrapping so its gradients are synchronised and it
    # is included in checkpoints.
    mlm_weight = float(config.get("training", {}).get("mlm_weight", 0.0))
    if mlm_weight > 0:
        student.mlm_head = RiNALMoMLMHead(backbone)
        # The pooler never contributes to either the alignment or the MLM
        # loss.  Freeze it so that ``find_unused_parameters=False`` can be
        # used under DDP (find_unused=True is incompatible with the two
        # forward passes per iteration that the MLM auxiliary loss requires).
        pooler = getattr(backbone.model, "pooler", None)
        if pooler is not None:
            for p in pooler.parameters():
                p.requires_grad_(False)

    # Optional weight initialisation from an earlier checkpoint (e.g. S0 MLM
    # pretraining). Only matching submodules (``backbone.*`` / ``mlm_head.*``)
    # are loaded.
    init_checkpoint = student_cfg.get("init_checkpoint") or model_cfg.get("init_checkpoint")
    if init_checkpoint:
        ckpt = torch.load(init_checkpoint, map_location="cpu", weights_only=False)
        state = ckpt.get("model_state_dict", ckpt)
        backbone_state = {
            k[len("backbone."):]: v for k, v in state.items() if k.startswith("backbone.")
        }
        if not backbone_state:
            raise RuntimeError(f"no backbone.* keys in init checkpoint {init_checkpoint}")
        missing, unexpected = student.backbone.load_state_dict(backbone_state, strict=False)
        if len(missing) > 0.05 * sum(1 for _ in student.backbone.parameters()):
            raise RuntimeError(
                f"init checkpoint backbone mismatch: {len(missing)} missing keys"
            )
        if hasattr(student, "mlm_head"):
            mlm_state = {
                k[len("mlm_head."):]: v for k, v in state.items() if k.startswith("mlm_head.")
            }
            if mlm_state:
                # decoder.weight is tied to the backbone embedding; drop it.
                mlm_state.pop("decoder.weight", None)
                student.mlm_head.load_state_dict(mlm_state, strict=False)

    # Verify the production RiNALMo-micro assumptions.
    if student.backbone.num_layers != 12:
        raise ValueError(
            f"expected 12 backbone layers, got {student.backbone.num_layers}"
        )
    for route, specs in adapter_specs.items():
        if not 1 <= route <= student.backbone.num_layers:
            raise ValueError(f"route layer {route} out of range")
        for _key, head_idx in specs:
            if not 0 <= head_idx < student.backbone.num_heads:
                raise ValueError(
                    f"student head {head_idx} out of range 0..{student.backbone.num_heads - 1}"
                )

    student = student.to(device)
    return student, student.count_parameters()


def _save_run_metadata(run_dir: Path, config: dict[str, Any]) -> None:
    """Save resolved config, Slurm job ID, git commit and code hash."""
    import hashlib
    import json

    run_dir.mkdir(parents=True, exist_ok=True)
    import yaml

    with open(run_dir / "config_resolved.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh)

    git_commit = "unknown"
    git_dirty = "unknown"
    try:
        git_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parent, text=True
        ).strip()
        git_dirty = subprocess.check_output(
            ["git", "diff", "--quiet"], cwd=Path(__file__).resolve().parent
        )
        git_dirty = "clean"
    except subprocess.CalledProcessError:
        git_dirty = "dirty"
    except FileNotFoundError:
        git_commit = "unknown"
        git_dirty = "unknown"

    # Small code hash covering the modules that define this run.
    code_files = [
        Path(__file__).resolve(),
        Path(__file__).resolve().parent / "full_sequence_dataset.py",
        Path(__file__).resolve().parent / "losses.py",
        Path(__file__).resolve().parent / "gradient_routing.py",
        Path(__file__).resolve().parent / "model.py",
    ]
    h = hashlib.md5()
    for path in code_files:
        if path.exists():
            h.update(path.read_bytes())
    code_hash = h.hexdigest()

    metadata = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID", "unknown"),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "code_hash": code_hash,
        "host": socket.gethostname(),
        "run_dir": str(run_dir),
    }
    with open(run_dir / "metadata.json", "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def _flatten_grad_dict(
    step: int,
    grad_norms: dict[str, float],
    layer_grads: dict[int, float] | None,
    diagnostics: dict[str, float],
) -> dict[str, Any]:
    row: dict[str, Any] = {"step": step}
    row.update(grad_norms)
    if layer_grads:
        for layer_idx, val in layer_grads.items():
            row[f"layer_{layer_idx}"] = val
    row["zero_grad_params"] = diagnostics.get("zero_grad_params", 0.0)
    row["nan_inf_grad_params"] = diagnostics.get("nan_inf_grad_params", 0.0)
    return row


def _lambdas_from_config(loss_lambdas: dict[str, float]) -> tuple[float, float, float]:
    return (
        float(loss_lambdas.get("kl", 1.0)),
        float(loss_lambdas.get("mse", 1.0)),
        float(loss_lambdas.get("cos", 0.1)),
    )


def _average_route_components(
    per_head_list: list[list[dict[str, torch.Tensor]]],
) -> dict[str, float]:
    if not per_head_list:
        return {}
    keys = ["kl", "mse", "cos", "total"]
    out: dict[str, float] = {}
    for key in keys:
        vals = [
            h[key].detach().float().mean().item()
            for heads in per_head_list
            for h in heads
            if key in h
        ]
        out[key] = sum(vals) / max(len(vals), 1)
    return out


def _per_head_metric_dict(
    route: str,
    per_head: list[dict[str, torch.Tensor]],
    adapter_specs: dict[int, list[tuple[str, int]]],
) -> dict[str, float]:
    depth = int(route[1:])
    specs = adapter_specs.get(depth, [])
    out: dict[str, float] = {}
    for t, comps in enumerate(per_head):
        key = specs[t][0] if t < len(specs) else f"H{t}"
        out[f"{route}/{key}/total"] = float(comps["total"].detach().mean().item())
    return out


def _log_train_step(
    writer: Any | None,
    step: int,
    total_loss: float,
    route_losses: dict[str, torch.Tensor],
    route_per_head: dict[str, list[dict[str, torch.Tensor]]],
    grad_norms: dict[str, float],
    layer_grads: dict[int, float] | None,
    lr: float,
    step_time: float,
    adapter_specs: dict[int, list[tuple[str, int]]],
) -> None:
    if writer is None:
        return
    writer.add_scalar("train/loss", total_loss, step)
    writer.add_scalar("train/lr", lr, step)
    writer.add_scalar("train/step_time_sec", step_time, step)
    for route, loss in route_losses.items():
        writer.add_scalar(f"train/{route}/loss", float(loss.detach().mean().item()), step)
    for route, heads in route_per_head.items():
        for metric, val in _per_head_metric_dict(route, heads, adapter_specs).items():
            writer.add_scalar(f"train/{metric}", val, step)
    for name, val in grad_norms.items():
        writer.add_scalar(f"grad/{name}", val, step)
    if layer_grads:
        for layer_idx, val in layer_grads.items():
            tag = "grad/layer/embedding" if layer_idx == -1 else (
                "grad/layer/final" if layer_idx == 12 else f"grad/layer/{layer_idx}"
            )
            writer.add_scalar(tag, val, step)


def _flatten_metrics_for_csv(
    metrics: dict[str, float],
    prefix: str,
    step: int,
) -> dict[str, Any]:
    row: dict[str, Any] = {"step": step, "split": prefix}
    row.update(metrics)
    return row


@torch.no_grad()
def _evaluate_baselines(
    model: RiNALMoHeadToHeadStudent,
    val_loader: DataLoader,
    device: torch.device,
    lambdas: tuple[float, float, float],
    use_amp: bool,
    amp_dtype: torch.dtype,
    baselines: list[str],
) -> dict[str, dict[str, float]]:
    if not baselines:
        return {}

    model.eval()
    route_keys = [f"L{d}" for d in DEFAULT_ROUTES]
    native_running: dict[str, list[float]] = {k: [] for k in route_keys}
    linear_running: dict[str, list[float]] = {k: [] for k in route_keys}
    native_counts: dict[str, int] = {k: 0 for k in route_keys}
    linear_counts: dict[str, int] = {k: 0 for k in route_keys}

    loss_fn = partial(route_alignment_loss, lambdas=lambdas)

    for batch in val_loader:
        batch = _to_device(batch, device)
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        lengths = batch["lengths"]

        if "native_head" in baselines:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                outputs = model.backbone.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_attentions=True,
                    return_dict=True,
                )
            attentions = outputs.attentions  # tuple of [B,H,L_tok,L_tok]
            for route in route_keys:
                depth = int(route[1:])
                attn = attentions[depth - 1]
                heads = model.backbone.student_heads[depth]
                # Strip <cls> and <eos> to align with nucleotide coordinates.
                if attn.shape[-1] > 2:
                    pred = attn[:, heads, 1:-1, 1:-1]
                else:
                    pred = attn[:, heads, :, :]
                if f"teacher_{route}" in batch:
                    loss, _ = loss_fn(pred, batch[f"teacher_{route}"], lengths)
                    native_running[route].append(float(loss.detach().mean().item()))
                    native_counts[route] += 1

        if "linear_projection" in baselines:
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                exits = model.backbone(input_ids, attention_mask)
            for route in route_keys:
                depth = int(route[1:])
                heads = exits[depth]["head_outputs"].float()  # [B,K,L,D]
                teacher = batch.get(f"teacher_{route}")
                if teacher is None:
                    continue
                teacher = teacher.float()
                K = heads.shape[1]
                route_losses = []
                for k in range(K):
                    Hk = heads[:, k, :, :]  # [B,L,D]
                    Tk = teacher[:, k, :, :]  # [B,L,L]
                    b, l, d = Hk.shape
                    mask2d = (
                        torch.arange(l, device=device).unsqueeze(0)
                        < lengths.unsqueeze(1)
                    ) & (
                        torch.arange(l, device=device).unsqueeze(0)
                        < lengths.unsqueeze(1)
                    )
                    row_valid = mask2d.any(dim=-1)  # [B,L]
                    X = Hk[row_valid]  # [N,D]
                    Y = Tk[row_valid]  # [N,L]
                    if X.shape[0] < d:
                        continue
                    try:
                        reg = 1e-3
                        XtX = X.t() @ X
                        W = torch.linalg.solve(
                            XtX + reg * torch.eye(d, device=device, dtype=X.dtype),
                            X.t() @ Y,
                        )  # [D,L]
                        pred_k = Hk @ W  # [B,L,L]
                        loss, _ = loss_fn(
                            pred_k.unsqueeze(1), teacher[:, k : k + 1], lengths
                        )
                        route_losses.append(float(loss.detach().mean().item()))
                    except RuntimeError:
                        continue
                if route_losses:
                    linear_running[route].append(sum(route_losses) / len(route_losses))
                    linear_counts[route] += 1

    out: dict[str, dict[str, float]] = {}
    if "native_head" in baselines:
        out["native_head"] = {
            route: sum(v) / max(len(v), 1) for route, v in native_running.items()
        }
    if "linear_projection" in baselines:
        out["linear_projection"] = {
            route: sum(v) / max(len(v), 1) for route, v in linear_running.items()
        }
    model.train()
    return out


@torch.no_grad()
def run_evaluation(
    model: RiNALMoHeadToHeadStudent,
    val_loader: DataLoader,
    device: torch.device,
    route_weights: dict[str, float],
    lambdas: tuple[float, float, float],
    use_amp: bool,
    amp_dtype: torch.dtype,
    baselines: list[str] | None = None,
) -> dict[str, Any]:
    model.eval()
    loss_fn = partial(route_alignment_loss, lambdas=lambdas)

    route_totals: dict[str, float] = {f"L{d}": 0.0 for d in DEFAULT_ROUTES}
    route_counts: dict[str, int] = {f"L{d}": 0 for d in DEFAULT_ROUTES}
    component_totals: dict[str, float] = {}
    component_counts: dict[str, int] = {}

    for batch in val_loader:
        batch = _to_device(batch, device)
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            pred = model(batch["input_ids"], batch["attention_mask"])
            route_losses = compute_route_losses(pred, batch, loss_fn=loss_fn)

        for route, (loss, per_head) in route_losses.items():
            route_totals[route] += float(loss.detach().mean().item())
            route_counts[route] += 1
            for comps in per_head:
                for key, val in comps.items():
                    tag = f"{route}/{key}"
                    component_totals[tag] = component_totals.get(tag, 0.0) + float(
                        val.detach().mean().item()
                    )
                    component_counts[tag] = component_counts.get(tag, 0) + 1

    n_batches = sum(route_counts.values())
    if n_batches == 0:
        model.train()
        return {}

    metrics: dict[str, float] = {}
    weighted_total = 0.0
    for route in route_totals:
        if route_counts[route]:
            val = route_totals[route] / route_counts[route]
            metrics[f"loss/{route}"] = val
            weighted_total += route_weights.get(route, 1.0) * val
    metrics["loss/total"] = weighted_total
    for tag, total in component_totals.items():
        metrics[f"component/{tag}"] = total / max(component_counts.get(tag, 1), 1)

    if baselines:
        baseline_metrics = _evaluate_baselines(
            model, val_loader, device, lambdas, use_amp, amp_dtype, baselines
        )
        for name, route_vals in baseline_metrics.items():
            for route, val in route_vals.items():
                metrics[f"baseline/{name}/{route}"] = val

    model.train()
    return metrics


def run_training_worker(
    rank: int,
    world_size: int,
    config: dict[str, Any],
    run_dir: Path,
    is_ddp: bool,
) -> None:
    runtime_cfg = config.get("runtime", {})
    training_cfg = config.get("training", {})
    seed = int(runtime_cfg.get("seed", 42))
    torch.manual_seed(seed + rank)

    if is_ddp:
        timeout = int(training_cfg.get("nccl_timeout", 1800))
        setup_ddp(rank, world_size, timeout_sec=timeout)

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    if rank == 0:
        print(
            f"[rank {rank}] worker started on {device}; ddp={is_ddp}; run_dir={run_dir}",
            flush=True,
        )

    slim_cache_root = Path(config["slim_cache_root"])
    pair_manifest_path = Path(config["head_pair_manifest"])

    all_paths = _resolve_shard_paths(
        slim_cache_root,
        config.get("shards"),
        config.get("n_shards"),
        config.get("shard_stride"),
    )

    if config.get("val_shards") is not None:
        val_paths = _resolve_shard_paths(
            slim_cache_root,
            config.get("val_shards"),
            None,
            None,
        )
        val_indices = {int(p.stem.split("_")[1]) for p in val_paths}
        train_paths = [p for p in all_paths if int(p.stem.split("_")[1]) not in val_indices]
    elif config.get("val_fraction") is not None:
        val_fraction = float(config["val_fraction"])
        n_val = max(1, int(len(all_paths) * val_fraction))
        n_val = min(n_val, len(all_paths) - 1)  # keep at least one train shard
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(all_paths), generator=g).tolist()
        val_paths = sorted([all_paths[i] for i in perm[:n_val]])
        train_paths = sorted([all_paths[i] for i in perm[n_val:]])
    else:
        raise ValueError("provide val_shards or val_fraction")

    splits: dict[str, list[int]] = {
        "train": sorted({int(p.stem.split("_")[1]) for p in train_paths}),
        "val": sorted({int(p.stem.split("_")[1]) for p in val_paths}),
    }
    if config.get("test_shards") is not None:
        splits["test"] = sorted({int(p.stem.split("_")[1]) for p in _resolve_shard_paths(slim_cache_root, config["test_shards"], None, None)})

    if not train_paths:
        raise ValueError("no training shards resolved")

    first_shard = _load_first_slim_shard(train_paths)
    teacher_head_ids = {d: first_shard[f"teacher_head_ids_L{d}"] for d in ROUTE_DEPTHS}
    shard_lmax = int(first_shard["teacher_heads_L6"].shape[-1])
    max_len = int(runtime_cfg.get("max_len", shard_lmax))
    max_len = min(max_len, shard_lmax)

    tokenizer = RiNALMoStudentBackbone(model_dir=Path(config["student"]["model_dir"])).tokenizer()

    student, param_counts = _make_student(
        config, pair_manifest_path, teacher_head_ids, device
    )
    adapter_specs = student.adapter_specs

    if rank == 0:
        print(f"Model params: {param_counts}", flush=True)

    if is_ddp:
        student = DDP(student, device_ids=[local_rank] if device.type == "cuda" else None, find_unused_parameters=False)

    unwrapped = student.module if is_ddp else student

    lr = float(training_cfg.get("lr", 1e-4))
    wd = float(training_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=wd)

    max_steps = int(training_cfg.get("max_steps", 1000))
    warmup_steps = int(training_cfg.get("warmup_steps", 0))
    min_lr_ratio = 0.01

    def lr_lambda(step: int) -> float:
        step = min(step, max_steps)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine * (1.0 - min_lr_ratio) + min_lr_ratio

    scheduler = LambdaLR(optimizer, lr_lambda)

    resume = bool(training_cfg.get("resume", False))
    start_step = 0
    latest_path = run_dir / "checkpoints" / "latest.pt"
    if resume and latest_path.exists():
        if rank == 0:
            print(f"Resuming from {latest_path}", flush=True)
        start_step = _load_checkpoint(latest_path, unwrapped, optimizer, scheduler, device)
        if is_ddp:
            dist.barrier()

    train_dataset = SlimShardDataset(
        train_paths,
        tokenizer,
        max_len=max_len,
        cache_size=int(runtime_cfg.get("lru_cache_size", 2)),
    )
    val_dataset = SlimShardDataset(
        val_paths,
        tokenizer,
        max_len=max_len,
        cache_size=int(runtime_cfg.get("lru_cache_size", 2)),
    )

    batch_size = int(training_cfg.get("batch_size", 4))
    num_workers = int(runtime_cfg.get("num_workers", 0))
    length_bucketed = bool(runtime_cfg.get("length_bucketed", True))

    if length_bucketed:
        train_sampler = LengthBucketedBatchSampler(
            train_dataset.sequence_lengths(),
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            seed=seed,
            shuffle=True,
            drop_last=False,
        )
        val_sampler = LengthBucketedBatchSampler(
            val_dataset.sequence_lengths(),
            batch_size=batch_size,
            rank=rank,
            world_size=world_size,
            seed=seed,
            shuffle=False,
            drop_last=False,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_sampler=train_sampler,
            collate_fn=variable_length_collate,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            collate_fn=variable_length_collate,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
    else:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=seed,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            seed=seed,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=train_sampler,
            collate_fn=variable_length_collate,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            sampler=val_sampler,
            collate_fn=variable_length_collate,
            num_workers=num_workers,
            pin_memory=device.type == "cuda",
        )

    if len(train_loader) == 0:
        raise ValueError(f"rank {rank} has zero training batches")

    precision = str(training_cfg.get("precision", "fp32"))
    use_amp = precision in ("bf16", "fp16") and device.type == "cuda"
    if precision == "bf16":
        amp_dtype = torch.bfloat16
    elif precision == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    route_weights = dict(training_cfg.get("route_weights", {"L6": 1.0, "L8": 1.0, "L12": 1.0}))
    lambdas = _lambdas_from_config(training_cfg.get("loss_lambdas", {"kl": 1.0, "mse": 1.0, "cos": 0.1}))
    loss_fn = partial(route_alignment_loss, lambdas=lambdas)

    grad_accum = int(training_cfg.get("grad_accum", 1))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    log_every = int(training_cfg.get("log_every", 10))
    save_every = int(training_cfg.get("save_every", 100))
    eval_every = int(training_cfg.get("eval_every", 100))
    grad_diag_every = int(training_cfg.get("grad_diagnostics_every", 100))
    baselines = list(training_cfg.get("baselines", []))

    writer: Any | None = None
    if rank == 0:
        writer = SummaryWriter(log_dir=run_dir / "tensorboard") if SummaryWriter else None
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    step = start_step
    epoch = 0
    train_iter = iter(train_loader)
    student.train()

    while step < max_steps:
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step_start = time.perf_counter()
        batch = _to_device(batch, device)

        sync_context = (
            student.no_sync()
            if is_ddp and (step + 1) % grad_accum != 0
            else nullcontext()
        )

        with sync_context:
            with torch.amp.autocast(
                device_type=device.type, dtype=amp_dtype, enabled=use_amp
            ):
                pred = student(batch["input_ids"], batch["attention_mask"])
                route_losses_with_per_head = compute_route_losses(pred, batch, loss_fn=loss_fn)
                route_losses = {k: v[0] for k, v in route_losses_with_per_head.items()}
                per_head = {k: v[1] for k, v in route_losses_with_per_head.items()}
                total_loss_tensor = total_route_loss(route_losses, route_weights) / grad_accum

            total_loss_tensor.backward()

        step_time = time.perf_counter() - step_start

        total_loss = float(total_loss_tensor.detach().item()) * grad_accum

        train_metrics: dict[str, float] = {
            "loss/total": total_loss,
        }
        for route, loss in route_losses.items():
            train_metrics[f"loss/{route}"] = float(loss.detach().mean().item())
        for route, heads in per_head.items():
            for comps in heads:
                for key, val in comps.items():
                    tag = f"component/{route}/{key}"
                    train_metrics[tag] = train_metrics.get(tag, 0.0) + float(
                        val.detach().mean().item()
                    )
            for key in ["kl", "mse", "cos", "total"]:
                tag = f"component/{route}/{key}"
                if tag in train_metrics:
                    train_metrics[tag] /= len(heads)

        grad_norms: dict[str, float] = {}
        if (step + 1) % grad_diag_every == 0 or step == 0:
            total_norm = 0.0
            for p in student.parameters():
                if p.grad is not None:
                    total_norm += float(p.grad.norm() ** 2)
            grad_norms["total"] = total_norm**0.5
            grad_norms.update(route_grad_norms(unwrapped))
            layer_grads = layer_grad_norms(unwrapped)
        else:
            layer_grads = None

        train_metrics.update({f"grad/{k}": v for k, v in grad_norms.items()})
        train_metrics = all_reduce_dict(train_metrics, device)

        current_lr = optimizer.param_groups[0]["lr"]
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if rank == 0 and (step % log_every == 0 or step == 0):
            _log_train_step(
                writer,
                step,
                total_loss,
                route_losses,
                per_head,
                grad_norms,
                layer_grads,
                current_lr,
                step_time,
                adapter_specs,
            )
            print(
                f"[step {step}] loss={total_loss:.4f} "
                + " ".join(
                    f"{route}={route_losses[route].detach().mean().item():.4f}"
                    for route in ROUTE_KEYS
                    if route in route_losses
                )
                + f" lr={current_lr:.2e} time={step_time:.2f}s",
                flush=True,
            )

        train_rows.append(_flatten_metrics_for_csv(train_metrics, "train", step))

        if rank == 0 and (step + 1) % save_every == 0:
            _save_checkpoint(
                unwrapped,
                optimizer,
                scheduler,
                step + 1,
                train_metrics,
                run_dir / "checkpoints" / f"step_{step + 1:07d}.pt",
                config=config,
                splits=splits,
            )
            _save_checkpoint(
                unwrapped,
                optimizer,
                scheduler,
                step + 1,
                train_metrics,
                run_dir / "checkpoints" / "latest.pt",
                config=config,
                splits=splits,
            )

        if (step + 1) % eval_every == 0 and len(val_loader) > 0:
            val_metrics = run_evaluation(
                unwrapped,
                val_loader,
                device,
                route_weights,
                lambdas,
                use_amp,
                amp_dtype,
                baselines=baselines,
            )
            val_metrics = all_reduce_dict(val_metrics, device)
            if rank == 0 and val_metrics:
                val_loss = val_metrics.get("loss/total", float("inf"))
                print(
                    f"[step {step + 1}] val loss={val_loss:.4f}",
                    flush=True,
                )
                if writer:
                    for k, v in val_metrics.items():
                        writer.add_scalar(f"val/{k}", v, step + 1)
                val_rows.append(_flatten_metrics_for_csv(val_metrics, "val", step + 1))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(
                        unwrapped,
                        optimizer,
                        scheduler,
                        step + 1,
                        val_metrics,
                        run_dir / "checkpoints" / "best_val.pt",
                        config=config,
                        splits=splits,
                    )

        step += 1

    if rank == 0:
        _save_checkpoint(
            unwrapped,
            optimizer,
            scheduler,
            step,
            train_rows[-1] if train_rows else {},
            run_dir / "checkpoints" / "latest.pt",
            config=config,
            splits=splits,
        )
        if train_rows:
            pd.DataFrame(train_rows).to_csv(run_dir / "train_metrics.csv", index=False)
        if val_rows:
            pd.DataFrame(val_rows).to_csv(run_dir / "val_metrics.csv", index=False)
        if writer:
            writer.close()

    cleanup_ddp()


def run_full_training_worker(
    rank: int,
    world_size: int,
    config: dict[str, Any],
    run_dir: Path,
    is_ddp: bool,
) -> None:
    """DDP training worker for the Stage 4B bucketed full slim cache."""
    runtime_cfg = config.get("runtime", {})
    training_cfg = config.get("training", {})
    optimizer_cfg = config.get("optimizer", {})
    scheduler_cfg = config.get("scheduler", {})
    dynamic_cfg = config.get("dynamic_batch", {})
    seed = int(runtime_cfg.get("seed", 42))
    torch.manual_seed(seed + rank)

    if is_ddp:
        timeout = int(training_cfg.get("nccl_timeout", 1800))
        setup_ddp(rank, world_size, timeout_sec=timeout)

    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = torch.device(
        f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu"
    )

    if rank == 0:
        print(
            f"[rank {rank}] full-cache worker started on {device}; "
            f"ddp={is_ddp}; run_dir={run_dir}",
            flush=True,
        )

    full_cache_root = Path(config["full_slim_cache_root"])
    sequence_manifest_path = Path(config["sequence_manifest"])
    pair_manifest_path = Path(config["head_pair_manifest"])
    bucket_manifest_path = full_cache_root / "manifests" / "slim_cache_manifest.parquet"

    train_index = Path(config["train_split_index"])
    val_index = Path(config["val_split_index"])
    test_index = Path(config["test_split_index"]) if config.get("test_split_index") else None

    tokenizer = RiNALMoStudentBackbone(
        model_dir=Path(config["student"]["model_dir"])
    ).tokenizer()

    route_head_keys = _load_first_bucket_route_heads(bucket_manifest_path)
    student, param_counts = _make_student_full(
        config, pair_manifest_path, route_head_keys, device
    )

    if rank == 0:
        print(f"Model params: {param_counts}", flush=True)

    find_unused = False
    if is_ddp:
        model_cfg = config.get("model", {})
        find_unused = bool(model_cfg.get("find_unused_parameters", False))
        student = DDP(
            student,
            device_ids=[local_rank] if device.type == "cuda" else None,
            find_unused_parameters=find_unused,
        )

    unwrapped = student.module if is_ddp else student
    adapter_specs = unwrapped.adapter_specs

    # DDP with find_unused_parameters=False requires every parameter to
    # receive a gradient each step.  In each alignment adapter the supervised
    # attention map is read from the Q/K path only, so the value path
    # (v_proj / o_proj / ln2 / ffn) never contributes to the loss.  Attach
    # those parameters with a zero-weight term below so their reducer buckets
    # complete; the gradient contribution is exactly 0.
    ddp_zero_params: list[torch.nn.Parameter] = []
    if is_ddp and not find_unused:
        ddp_zero_params = [
            p
            for n, p in unwrapped.named_parameters()
            if p.requires_grad
            and n.startswith("adapters.")
            and any(tag in n for tag in ("v_proj", "o_proj", "ln2", "ffn"))
        ]
        if rank == 0:
            print(
                f"[ddp] zero-weight hook on {len(ddp_zero_params)} adapter "
                "value-path params (find_unused_parameters=False)",
                flush=True,
            )

    # Fused AdamW with separate LR/WD for backbone and adapters.
    base_lr = float(optimizer_cfg.get("lr", training_cfg.get("lr", 1.0e-4)))
    base_wd = float(optimizer_cfg.get("weight_decay", training_cfg.get("weight_decay", 0.01)))
    backbone_lr = float(optimizer_cfg.get("backbone_lr", base_lr))
    adapter_lr = float(optimizer_cfg.get("adapter_lr", base_lr))
    backbone_wd = float(optimizer_cfg.get("backbone_weight_decay", base_wd))
    adapter_wd = float(optimizer_cfg.get("adapter_weight_decay", base_wd))
    fused = bool(optimizer_cfg.get("fused", True)) and device.type == "cuda"

    param_groups = [
        {
            "params": [p for p in unwrapped.backbone.parameters() if p.requires_grad],
            "lr": backbone_lr,
            "weight_decay": backbone_wd,
            "name": "backbone",
        },
        {
            "params": [p for p in unwrapped.adapters.parameters() if p.requires_grad],
            "lr": adapter_lr,
            "weight_decay": adapter_wd,
            "name": "adapters",
        },
    ]
    if hasattr(unwrapped, "mlm_head"):
        # decoder.weight is tied to the backbone embedding — exclude params
        # already owned by the backbone group (identity check).
        backbone_param_ids = {id(p) for p in unwrapped.backbone.parameters()}
        param_groups.append(
            {
                "params": [
                    p for p in unwrapped.mlm_head.parameters()
                    if p.requires_grad and id(p) not in backbone_param_ids
                ],
                "lr": backbone_lr,
                "weight_decay": backbone_wd,
                "name": "mlm",
            }
        )
    param_groups = [g for g in param_groups if len(g["params"]) > 0]

    try:
        optimizer = torch.optim.AdamW(
            param_groups, lr=base_lr, weight_decay=base_wd, fused=fused
        )
    except TypeError:
        # Older PyTorch without fused AdamW.
        optimizer = torch.optim.AdamW(param_groups, lr=base_lr, weight_decay=base_wd)

    max_steps = int(training_cfg.get("max_steps", 1000))
    warmup_steps = int(scheduler_cfg.get("warmup_steps", training_cfg.get("warmup_steps", 0)))
    min_lr_ratio = float(
        scheduler_cfg.get("min_lr_ratio", training_cfg.get("min_lr_ratio", 0.01))
    )

    def lr_lambda(step: int) -> float:
        step = min(step, max_steps)
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, max_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return cosine * (1.0 - min_lr_ratio) + min_lr_ratio

    scheduler = LambdaLR(optimizer, lr_lambda)

    resume = bool(training_cfg.get("resume", True))
    start_step = 0
    latest_path = run_dir / "checkpoints" / "latest.pt"
    if resume and latest_path.exists():
        if rank == 0:
            print(f"Resuming from {latest_path}", flush=True)
        start_step = _load_checkpoint(
            latest_path, unwrapped, optimizer, scheduler, device
        )
        if is_ddp:
            dist.barrier()

    max_len = int(runtime_cfg.get("max_len", 1024))
    train_dataset = FullSlimBucketDataset(
        full_cache_root,
        train_index,
        sequence_manifest_path,
        tokenizer,
        max_len=max_len,
        cache_size=int(runtime_cfg.get("lru_cache_size", 2)),
    )
    val_dataset = FullSlimBucketDataset(
        full_cache_root,
        val_index,
        sequence_manifest_path,
        tokenizer,
        max_len=max_len,
        cache_size=1,
    )

    max_pair_tokens = int(dynamic_cfg.get("max_pair_tokens", 512 * 512 * 2))
    max_sequences = int(
        dynamic_cfg.get(
            "max_sequences_per_batch",
            training_cfg.get("batch_size", 2),
        )
    )
    bucket_size_samples = dynamic_cfg.get("bucket_size_samples")

    train_sampler = LengthBucketedSampler(
        train_dataset.lengths(),
        max_pair_tokens=max_pair_tokens,
        max_sequences_per_batch=max_sequences,
        bucket_size_samples=bucket_size_samples,
        rank=rank,
        world_size=world_size,
        seed=seed,
        shuffle=True,
        drop_last=False,
    )
    val_sampler = LengthBucketedSampler(
        val_dataset.lengths(),
        max_pair_tokens=max_pair_tokens,
        max_sequences_per_batch=max_sequences,
        bucket_size_samples=bucket_size_samples,
        rank=rank,
        world_size=world_size,
        seed=seed,
        shuffle=False,
        drop_last=False,
    )

    num_workers = int(runtime_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=full_variable_length_collate,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=full_variable_length_collate,
        num_workers=0,
        pin_memory=False,
    )

    if len(train_loader) == 0:
        raise ValueError(f"rank {rank} has zero training batches")

    precision = str(training_cfg.get("precision", "fp32"))
    use_amp = precision in ("bf16", "fp16") and device.type == "cuda"
    if precision == "bf16":
        amp_dtype = torch.bfloat16
    elif precision == "fp16":
        amp_dtype = torch.float16
    else:
        amp_dtype = torch.float32

    route_weights = dict(
        training_cfg.get("route_weights", {"L6": 1.0, "L8": 1.0, "L12": 1.0})
    )
    lambdas = _lambdas_from_config(
        training_cfg.get("loss_lambdas", {"kl": 1.0, "mse": 1.0, "cos": 0.1})
    )
    loss_fn = partial(route_alignment_loss, lambdas=lambdas)
    normalize_within_route = bool(
        training_cfg.get("normalize_within_route_by_head_count", True)
    )

    # Optional MLM auxiliary loss (separate masked forward through the
    # backbone; the alignment branch always sees the unmasked input because
    # teacher targets were computed on unmasked sequences).
    mlm_weight = float(training_cfg.get("mlm_weight", 0.0))
    mlm_prob = float(training_cfg.get("mlm_prob", 0.15))
    use_mlm = mlm_weight > 0 and hasattr(unwrapped, "mlm_head")
    mlm_gen = torch.Generator(device=device)
    mlm_gen.manual_seed(seed * 1000 + rank)

    grad_accum = int(training_cfg.get("grad_accum", 1))
    grad_clip = float(training_cfg.get("grad_clip", 1.0))
    log_every = int(training_cfg.get("log_every", 10))
    save_every = int(training_cfg.get("save_every", 100))
    eval_every = int(training_cfg.get("eval_every", 100))
    grad_diag_every = int(training_cfg.get("grad_diagnostics_every", 100))
    baselines = list(training_cfg.get("baselines", []))
    oom_decay = float(training_cfg.get("oom_budget_decay", 0.8))
    oom_min_budget = int(training_cfg.get("oom_min_budget", 64 * 64))

    writer: Any | None = None
    if rank == 0:
        writer = SummaryWriter(log_dir=run_dir / "tensorboard") if SummaryWriter else None
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        _save_run_metadata(run_dir, config)

    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    per_head_rows: list[dict[str, Any]] = []
    grad_rows: list[dict[str, Any]] = []
    throughput_rows: list[dict[str, Any]] = []
    best_val_loss = float("inf")

    step = start_step
    epoch = 0
    train_iter = iter(train_loader)
    student.train()

    current_max_pair_tokens = max_pair_tokens
    current_max_sequences = max_sequences

    while step < max_steps:
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        try:
            batch = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            batch = next(train_iter)

        step_start = time.perf_counter()
        batch = _to_device(batch, device)

        sync_context = (
            student.no_sync()
            if is_ddp and (step + 1) % grad_accum != 0
            else nullcontext()
        )

        oom_happened = False
        with sync_context:
            try:
                with torch.amp.autocast(
                    device_type=device.type, dtype=amp_dtype, enabled=use_amp
                ):
                    pred = student(batch["input_ids"], batch["attention_mask"])
                    route_losses_with_per_head = compute_route_losses(
                        pred,
                        batch,
                        loss_fn=loss_fn,
                        normalize_within_route=normalize_within_route,
                    )
                    route_losses = {k: v[0] for k, v in route_losses_with_per_head.items()}
                    per_head = {k: v[1] for k, v in route_losses_with_per_head.items()}
                    total_loss_tensor = total_route_loss(route_losses, route_weights)
                    mlm_loss_tensor = None
                    if use_mlm:
                        masked_ids, mlm_labels = mask_tokens(
                            batch["input_ids"], batch["attention_mask"],
                            mlm_prob=mlm_prob, generator=mlm_gen,
                        )
                        mlm_out = unwrapped.backbone.model(
                            input_ids=masked_ids,
                            attention_mask=batch["attention_mask"],
                            use_cache=False,
                            return_dict=True,
                        )
                        mlm_logits = unwrapped.mlm_head(mlm_out.last_hidden_state)
                        mlm_loss_tensor = mlm_loss_from_logits(mlm_logits, mlm_labels)
                        total_loss_tensor = total_loss_tensor + mlm_weight * mlm_loss_tensor
                    if ddp_zero_params:
                        zero_term = sum(p.sum() for p in ddp_zero_params)
                        total_loss_tensor = total_loss_tensor + 0.0 * zero_term
                    total_loss_tensor = total_loss_tensor / grad_accum
                total_loss_tensor.backward()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    oom_happened = True
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    if rank == 0:
                        _append_jsonl(
                            run_dir / "oom_events.jsonl",
                            {
                                "step": step,
                                "epoch": epoch,
                                "seq_ids": batch.get("seq_ids", []),
                                "lengths": batch["lengths"].cpu().tolist(),
                                "max_pair_tokens": current_max_pair_tokens,
                                "max_sequences": current_max_sequences,
                                "error": str(exc),
                            },
                        )
                    # Reduce the dynamic budget and rebuild the loader.
                    current_max_pair_tokens = max(
                        oom_min_budget, int(current_max_pair_tokens * oom_decay)
                    )
                    current_max_sequences = max(1, int(current_max_sequences * oom_decay))
                    train_sampler = LengthBucketedSampler(
                        train_dataset.lengths(),
                        max_pair_tokens=current_max_pair_tokens,
                        max_sequences_per_batch=current_max_sequences,
                        bucket_size_samples=bucket_size_samples,
                        rank=rank,
                        world_size=world_size,
                        seed=seed,
                        shuffle=True,
                        drop_last=False,
                    )
                    train_sampler.set_epoch(epoch)
                    train_loader = DataLoader(
                        train_dataset,
                        batch_sampler=train_sampler,
                        collate_fn=full_variable_length_collate,
                        num_workers=num_workers,
                        pin_memory=device.type == "cuda",
                    )
                    train_iter = iter(train_loader)
                else:
                    raise

        if oom_happened:
            continue

        total_loss = float(total_loss_tensor.detach().item()) * grad_accum
        if not math.isfinite(total_loss):
            raise RuntimeError(f"Non-finite training loss at step {step}: {total_loss}")

        step_time = time.perf_counter() - step_start

        train_metrics: dict[str, float] = {"loss/total": total_loss}
        if mlm_loss_tensor is not None:
            train_metrics["loss/mlm"] = float(mlm_loss_tensor.detach().item())
        for route, loss in route_losses.items():
            train_metrics[f"loss/{route}"] = float(loss.detach().mean().item())
        for route, heads in per_head.items():
            for comps in heads:
                for key, val in comps.items():
                    tag = f"component/{route}/{key}"
                    train_metrics[tag] = train_metrics.get(tag, 0.0) + float(
                        val.detach().mean().item()
                    )
            for key in ["kl", "mse", "cos", "total"]:
                tag = f"component/{route}/{key}"
                if tag in train_metrics:
                    train_metrics[tag] /= len(heads)

        grad_norms: dict[str, float] = {}
        layer_grads: dict[int, float] | None = None
        diagnostics: dict[str, float] = {}
        if (step + 1) % grad_diag_every == 0 or step == 0:
            total_norm = 0.0
            for p in student.parameters():
                if p.grad is not None:
                    total_norm += float(p.grad.norm() ** 2)
            grad_norms["total"] = total_norm**0.5
            grad_norms.update(route_grad_norms(unwrapped))
            layer_grads = layer_grad_norms(unwrapped)
            diagnostics = gradient_diagnostics(unwrapped)
            if diagnostics.get("nan_inf_grad_params", 0.0) > 0:
                raise RuntimeError(f"NaN/Inf gradients at step {step}")
            grad_rows.append(
                _flatten_grad_dict(step, grad_norms, layer_grads, diagnostics)
            )

        train_metrics.update({f"grad/{k}": v for k, v in grad_norms.items()})
        train_metrics = all_reduce_dict(train_metrics, device)

        # Fail fast on non-finite per-route grad norms (gradient routing mismatch).
        for name, val in grad_norms.items():
            if name in FULL_ROUTE_KEYS and not math.isfinite(val):
                raise RuntimeError(
                    f"Non-finite gradient norm for route {name} at step {step}"
                )

        current_lr = optimizer.param_groups[0]["lr"]
        if (step + 1) % grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        if rank == 0 and (step % log_every == 0 or step == 0):
            _log_train_step(
                writer,
                step,
                total_loss,
                route_losses,
                per_head,
                grad_norms,
                layer_grads,
                current_lr,
                step_time,
                adapter_specs,
            )
            for route, heads in per_head.items():
                specs = adapter_specs.get(int(route[1:]), [])
                for t, comps in enumerate(heads):
                    head_key = specs[t][0] if t < len(specs) else f"H{t}"
                    per_head_rows.append(
                        {
                            "step": step,
                            "route": route,
                            "head_key": head_key,
                            "kl": float(comps["kl"].detach().mean().item()),
                            "mse": float(comps["mse"].detach().mean().item()),
                            "cos": float(comps["cos"].detach().mean().item()),
                            "total": float(comps["total"].detach().mean().item()),
                        }
                    )
            seqs = int(batch["lengths"].sum().item())
            pair_tokens = int((batch["lengths"].float() ** 2).sum().item())
            mem_mb = (
                torch.cuda.max_memory_allocated(device) / 1024 / 1024
                if device.type == "cuda"
                else 0.0
            )
            throughput_rows.append(
                {
                    "step": step,
                    "batch_size": len(batch["lengths"]),
                    "total_seqs": seqs,
                    "total_pair_tokens": pair_tokens,
                    "step_time_sec": step_time,
                    "throughput_seq_per_sec": seqs / step_time if step_time > 0 else 0.0,
                    "max_memory_mb": mem_mb,
                }
            )
            print(
                f"[step {step}] loss={total_loss:.4f} "
                + " ".join(
                    f"{route}={route_losses[route].detach().mean().item():.4f}"
                    for route in FULL_ROUTE_KEYS
                    if route in route_losses
                )
                + f" lr={current_lr:.2e} time={step_time:.2f}s",
                flush=True,
            )

        train_rows.append(_flatten_metrics_for_csv(train_metrics, "train", step))

        if rank == 0 and (step % log_every == 0 or step == 0):
            pd.DataFrame(train_rows).to_csv(run_dir / "train_metrics.csv", index=False)
            if per_head_rows:
                pd.DataFrame(per_head_rows).to_csv(
                    run_dir / "per_head_metrics.csv", index=False
                )
            if grad_rows:
                pd.DataFrame(grad_rows).to_csv(run_dir / "grad_norms.csv", index=False)
            if throughput_rows:
                pd.DataFrame(throughput_rows).to_csv(
                    run_dir / "throughput_memory.csv", index=False
                )

        if rank == 0 and (step + 1) % save_every == 0:
            _save_checkpoint(
                unwrapped,
                optimizer,
                scheduler,
                step + 1,
                train_metrics,
                run_dir / "checkpoints" / f"step_{step + 1:07d}.pt",
                config=config,
                splits=None,
            )
            _save_checkpoint(
                unwrapped,
                optimizer,
                scheduler,
                step + 1,
                train_metrics,
                run_dir / "checkpoints" / "latest.pt",
                config=config,
                splits=None,
            )

        if (step + 1) % eval_every == 0 and len(val_loader) > 0:
            val_metrics = run_evaluation(
                unwrapped,
                val_loader,
                device,
                route_weights,
                lambdas,
                use_amp,
                amp_dtype,
                baselines=baselines,
            )
            val_metrics = all_reduce_dict(val_metrics, device)
            if rank == 0 and val_metrics:
                val_loss = val_metrics.get("loss/total", float("inf"))
                print(f"[step {step + 1}] val loss={val_loss:.4f}", flush=True)
                if writer:
                    for k, v in val_metrics.items():
                        writer.add_scalar(f"val/{k}", v, step + 1)
                val_rows.append(_flatten_metrics_for_csv(val_metrics, "val", step + 1))
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    _save_checkpoint(
                        unwrapped,
                        optimizer,
                        scheduler,
                        step + 1,
                        val_metrics,
                        run_dir / "checkpoints" / "best_val.pt",
                        config=config,
                        splits=None,
                    )

        step += 1

    if rank == 0:
        _save_checkpoint(
            unwrapped,
            optimizer,
            scheduler,
            step,
            train_rows[-1] if train_rows else {},
            run_dir / "checkpoints" / "latest.pt",
            config=config,
            splits=None,
        )
        if train_rows:
            pd.DataFrame(train_rows).to_csv(run_dir / "train_metrics.csv", index=False)
        if val_rows:
            pd.DataFrame(val_rows).to_csv(run_dir / "val_metrics.csv", index=False)
        if per_head_rows:
            pd.DataFrame(per_head_rows).to_csv(
                run_dir / "per_head_metrics.csv", index=False
            )
        if grad_rows:
            pd.DataFrame(grad_rows).to_csv(run_dir / "grad_norms.csv", index=False)
        if throughput_rows:
            pd.DataFrame(throughput_rows).to_csv(
                run_dir / "throughput_memory.csv", index=False
            )
        if writer:
            writer.close()

    cleanup_ddp()


def launch_training(
    config: dict[str, Any],
    run_dir: Path,
    world_size: int | None = None,
) -> None:
    run_dir = Path(run_dir)
    if world_size is None:
        if "WORLD_SIZE" in os.environ:
            world_size = int(os.environ["WORLD_SIZE"])
        elif config.get("runtime", {}).get("ddp"):
            world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        else:
            world_size = int(config.get("world_size", 1))

    cache_format = str(config.get("cache_format", "slim")).lower()
    use_full = (
        cache_format == "full"
        or config.get("full_slim_cache_root") is not None
    )
    worker_fn = run_full_training_worker if use_full else run_training_worker

    if (
        dist.is_available()
        and "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and world_size > 1
    ):
        rank = int(os.environ["RANK"])
        worker_fn(rank, world_size, config, run_dir, is_ddp=True)
    elif world_size > 1:
        mp.spawn(
            worker_fn,
            args=(world_size, config, run_dir, True),
            nprocs=world_size,
            join=True,
        )
    else:
        worker_fn(0, 1, config, run_dir, is_ddp=False)
