from __future__ import annotations

import gc
import logging
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from rna_head_alignment.data.task_loaders import load_task_samples, load_tasks_config
from rna_head_alignment.io.manifests import write_dataframe, write_json, write_markdown_table
from rna_head_alignment.io.paths import get_device, load_yaml, resolve_path
from rna_head_alignment.models.attention_extractors import extract_attention_bundle
from rna_head_alignment.models.loaders import load_frozen_model
from rna_head_alignment.models.registry import get_model_spec

from .pairwise_screen import (
    add_pairwise_score_parts,
    init_pairwise_score_parts,
    score_pairwise_heads,
    score_pairwise_streamed,
    select_pair_indices,
)
from .rna3d_screen import score_rna3d_heads
from .seqcls_screen import head_feature_stack, score_seqcls_head_features, score_seqcls_heads
from .seqreg_screen import score_seqreg_head_features, score_seqreg_heads
from .token_screen import score_token_head_features, score_token_heads, token_feature_stack


def screen_heads(
    model_name: str,
    task_name: str,
    config_path: str | Path,
    output: str | Path,
    models_yaml: str | Path = "configs/models.yaml",
    tasks_yaml: str | Path = "configs/tasks.yaml",
    limit: int | None = None,
    device: str = "auto",
) -> list[dict[str, Any]]:
    cfg = load_yaml(config_path)
    if limit is None:
        limit = int(cfg.get("limit", 128))
    spec = get_model_spec(models_yaml, model_name)
    task_cfg = load_tasks_config(tasks_yaml)[task_name]
    max_len = int(task_cfg.get("max_len", cfg.get("max_len", 256)))
    primary = str(task_cfg.get("metric", "auprc"))
    def empty_row(note: str, n_samples: int = 0) -> dict[str, Any]:
        return {
            "task": task_name,
            "model": model_name,
            "model_type": spec.model_type,
            "layer": -1,
            "head": -1,
            "metric_primary": float("nan"),
            "auroc": float("nan"),
            "auprc": float("nan"),
            "macro_f1": float("nan"),
            "accuracy": float("nan"),
            "pearson": float("nan"),
            "spearman": float("nan"),
            "mae": float("nan"),
            "mcc": float("nan"),
            "f1": float("nan"),
            "n_samples": n_samples,
            "n_pos": 0,
            "n_neg": 0,
            "split": "none",
            "notes": note,
        }

    out = resolve_path(output)
    if out.exists() and out.stat().st_size > 0:
        return pd.read_csv(out).to_dict("records")
    split_filter = cfg.get("split_filter")
    if split_filter:
        allowed = {split_filter} if isinstance(split_filter, str) else {str(s) for s in split_filter}
        cap = int(cfg.get("post_filter_limit", limit))
        if cap <= 0:
            raise ValueError("post_filter_limit must be positive")
        samples = load_task_samples(task_name, task_cfg, limit=cap, max_len=max_len,
                                    split_filter=allowed)
        samples = [s for s in samples if str(s.split) in allowed]
        samples = samples[:cap]
    else:
        samples = load_task_samples(task_name, task_cfg, limit=limit, max_len=max_len)
    if not samples:
        rows = [empty_row("no_samples_loaded", 0)]
        df = pd.DataFrame(rows)
        write_dataframe(df, out)
        write_json(rows, out.with_suffix(".json"))
        write_markdown_table(rows, out.with_suffix(".md"), f"Head Atlas {task_name}/{model_name}")
        return rows
    resolved_device = get_device(device or cfg.get("device", "auto"))
    try:
        tokenizer, model = load_frozen_model(spec, resolved_device)
    except Exception as e:
        rows = [empty_row(f"model_load_failed: {e!r}", 0)]
        df = pd.DataFrame(rows)
        write_dataframe(df, out)
        write_json(rows, out.with_suffix(".json"))
        write_markdown_table(rows, out.with_suffix(".md"), f"Head Atlas {task_name}/{model_name}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return rows
    task_type = str(task_cfg["task_type"])
    stream_head_features = task_type in {"sequence_classification", "sequence_regression"}
    stream_token_features = task_type == "token_regression"
    stream_pairwise_scores = task_type in {"pairwise", "rna3d_contact"}
    pairwise_cfg = cfg.get("pairwise", {}) or {}
    max_pairs_per_sample = int(pairwise_cfg.get("max_pairs_per_sample", 1024))
    attentions = []
    attention_features = []
    token_features = []
    pairwise_y_parts = []
    pairwise_score_parts = None
    kept_samples = []
    for i, sample in enumerate(tqdm(samples, desc=f"{model_name}/{task_name}", leave=False)):
        try:
            bundle = extract_attention_bundle(
                spec,
                tokenizer,
                model,
                sample.sequence,
                seq_id=sample.seq_id,
                max_len=max_len,
                device=resolved_device,
            )
            if bundle.attentions is None:
                continue
            attn_np = bundle.attentions.detach().cpu().numpy()
            if stream_head_features:
                attention_features.append(head_feature_stack(attn_np))
                kept_samples.append(sample)
            elif stream_token_features:
                token_features.append(token_feature_stack(attn_np))
                kept_samples.append(sample)
            elif stream_pairwise_scores:
                selected = select_pair_indices(
                    sample,
                    int(attn_np.shape[-1]),
                    max_pairs=max_pairs_per_sample,
                    seed=int(cfg.get("seed", 42)) + i,
                )
                if selected is None:
                    continue
                flat_indices, y_selected = selected
                if pairwise_score_parts is None:
                    pairwise_score_parts = init_pairwise_score_parts(int(attn_np.shape[0]), int(attn_np.shape[1]))
                add_pairwise_score_parts(pairwise_score_parts, attn_np, flat_indices)
                pairwise_y_parts.append(y_selected)
                kept_samples.append(sample)
            else:
                attentions.append(attn_np)
            del bundle
            if i and i % 50 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception:
            logging.getLogger(__name__).warning(
                "Screening failed for model=%s task=%s sample=%s",
                model_name, task_name, sample.seq_id, exc_info=True,
            )
            continue
    if stream_head_features and attention_features:
        feature_array = np.stack(attention_features, axis=0)
    else:
        feature_array = None
    has_streamed_pairwise = bool(pairwise_y_parts) and pairwise_score_parts is not None
    if not attentions and feature_array is None and not token_features and not has_streamed_pairwise:
        rows = [empty_row("no_attention_or_forward_failed", len(kept_samples))]
    elif task_type == "sequence_classification":
        if feature_array is not None:
            rows = score_seqcls_head_features(model_name, spec.model_type, task_name, kept_samples, feature_array, primary=primary)
        else:
            rows = score_seqcls_heads(model_name, spec.model_type, task_name, kept_samples, attentions, primary=primary)
    elif task_type == "sequence_regression":
        if feature_array is not None:
            rows = score_seqreg_head_features(model_name, spec.model_type, task_name, kept_samples, feature_array, primary=primary)
        else:
            rows = score_seqreg_heads(model_name, spec.model_type, task_name, kept_samples, attentions, primary=primary)
    elif task_type == "token_regression":
        if token_features:
            rows = score_token_head_features(model_name, spec.model_type, task_name, kept_samples, token_features, primary=primary)
        else:
            rows = score_token_heads(model_name, spec.model_type, task_name, kept_samples, attentions, primary=primary)
    elif task_type == "pairwise":
        if has_streamed_pairwise:
            rows = score_pairwise_streamed(
                model_name,
                spec.model_type,
                task_name,
                pairwise_y_parts,
                pairwise_score_parts,
                len(kept_samples),
                primary=primary,
                note=f"streamed_sampled_pairs_max{max_pairs_per_sample}",
            )
        else:
            rows = score_pairwise_heads(model_name, spec.model_type, task_name, kept_samples, attentions, primary=primary)
    elif task_type == "rna3d_contact":
        if has_streamed_pairwise:
            rows = score_pairwise_streamed(
                model_name,
                spec.model_type,
                task_name,
                pairwise_y_parts,
                pairwise_score_parts,
                len(kept_samples),
                primary=primary,
                note=f"streamed_sampled_pairs_max{max_pairs_per_sample}",
            )
        else:
            rows = score_rna3d_heads(model_name, spec.model_type, task_name, kept_samples, attentions, primary=primary)
    else:
        raise ValueError(f"Unsupported task type: {task_type}")
    if not rows:
        rows = [empty_row("no_scored_heads; usually too few labels or valid targets in sample", len(kept_samples))]
    df = pd.DataFrame(rows)
    write_dataframe(df, out)
    write_json(rows, out.with_suffix(".json"))
    write_markdown_table(rows[:50], out.with_suffix(".md"), f"Head Atlas {task_name}/{model_name}")
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


def make_screening_jobs(
    models_yaml: str | Path,
    tasks_yaml: str | Path,
    model_check: str | Path,
    output: str | Path,
    output_root: str | Path = "outputs/head_screening",
    include_all_models: bool = False,
) -> pd.DataFrame:
    from rna_head_alignment.models.registry import load_model_specs

    specs = load_model_specs(models_yaml)
    tasks = load_tasks_config(tasks_yaml)
    check = pd.read_csv(resolve_path(model_check)) if resolve_path(model_check).exists() else pd.DataFrame()
    usable = set()
    if not check.empty:
        for _, row in check.iterrows():
            if bool(row.get("load_ok")) and bool(row.get("forward_ok")) and bool(row.get("has_attentions")):
                usable.add(str(row["model"]))
    rows = []
    job_id = 0
    root = Path(output_root)
    for model_name, spec in specs.items():
        for task_name in tasks:
            task_enabled = bool(tasks[task_name].get("enabled", True))
            enabled = task_enabled and (
                include_all_models or (spec.attention_teacher and model_name in usable)
            )
            if enabled:
                reason = "ok"
            elif not task_enabled:
                reason = "disabled_task"
            else:
                reason = "no_attention_or_not_teacher"
            output_path = root / task_name / model_name / "head_atlas.csv"
            rows.append(
                {
                    "job_id": job_id,
                    "model": model_name,
                    "task": task_name,
                    "enabled": enabled,
                    "reason": reason,
                    "output_path": output_path.as_posix(),
                }
            )
            job_id += 1
    df = pd.DataFrame(rows)
    write_dataframe(df, output)
    return df


def run_screening_job_table(
    jobs_csv: str | Path,
    config_path: str | Path = "configs/screening.yaml",
    models_yaml: str | Path = "configs/models.yaml",
    tasks_yaml: str | Path = "configs/tasks.yaml",
    report_csv: str | Path = "outputs/head_screening/screening_run_report.csv",
    limit: int | None = None,
    device: str = "auto",
    max_jobs: int | None = None,
    resume: bool = True,
    model_filter: str | None = None,
    task_filter: str | None = None,
    time_budget_hours: float | None = None,
) -> pd.DataFrame:
    """Run enabled model-task rows from a screening job table.

    This is intentionally single-process. Server parallelism is handled by the
    Slurm array script, while this function is useful for local smoke tests and
    for resuming partial job tables without changing shell scripts.
    """
    jobs_path = resolve_path(jobs_csv)
    jobs = pd.read_csv(jobs_path)
    rows: list[dict[str, Any]] = []
    report_path = resolve_path(report_csv)

    def flush_report() -> None:
        df_partial = pd.DataFrame(rows)
        write_dataframe(df_partial, report_path)
        if rows:
            write_json(rows, report_path.with_suffix(".json"))

    ran = 0
    table_start = time.time()
    time_budget_seconds = None if time_budget_hours is None else float(time_budget_hours) * 3600.0
    for _, job in jobs.iterrows():
        model_name = str(job["model"])
        task_name = str(job["task"])
        enabled = str(job.get("enabled", "")).lower() == "true"
        output_path = str(job.get("output_path", f"outputs/head_screening/{task_name}/{model_name}/head_atlas.csv"))
        out = resolve_path(output_path)
        if model_filter and model_name != model_filter:
            continue
        if task_filter and task_name != task_filter:
            continue
        if not enabled:
            rows.append(
                {
                    "model": model_name,
                    "task": task_name,
                    "status": "skipped",
                    "reason": str(job.get("reason", "disabled")),
                    "output_path": output_path,
                    "rows": 0,
                    "seconds": 0.0,
                    "error": "",
                }
            )
            flush_report()
            continue
        if resume and out.exists() and out.stat().st_size > 0:
            try:
                n_rows = len(pd.read_csv(out))
            except Exception:
                n_rows = -1
            rows.append(
                {
                    "model": model_name,
                    "task": task_name,
                    "status": "resume_skipped",
                    "reason": "existing_output",
                    "output_path": output_path,
                    "rows": n_rows,
                    "seconds": 0.0,
                    "error": "",
                }
            )
            flush_report()
            continue
        if max_jobs is not None and ran >= max_jobs:
            break
        estimated_seconds = float(job.get("estimated_seconds", 0) or 0)
        if time_budget_seconds is not None:
            elapsed = time.time() - table_start
            if elapsed + estimated_seconds > time_budget_seconds:
                rows.append(
                    {
                        "model": model_name,
                        "task": task_name,
                        "status": "budget_stopped",
                        "reason": "time_budget_would_be_exceeded",
                        "output_path": output_path,
                        "rows": 0,
                        "seconds": round(elapsed, 3),
                        "error": "",
                    }
                )
                flush_report()
                break
        start = time.time()
        try:
            result = screen_heads(
                model_name,
                task_name,
                config_path,
                output_path,
                models_yaml=models_yaml,
                tasks_yaml=tasks_yaml,
                limit=limit,
                device=device,
            )
            rows.append(
                {
                    "model": model_name,
                    "task": task_name,
                    "status": "ok",
                    "reason": "completed",
                    "output_path": output_path,
                    "rows": len(result),
                    "seconds": round(time.time() - start, 3),
                    "error": "",
                }
            )
        except Exception as e:
            rows.append(
                {
                    "model": model_name,
                    "task": task_name,
                    "status": "failed",
                    "reason": "exception",
                    "output_path": output_path,
                    "rows": 0,
                    "seconds": round(time.time() - start, 3),
                    "error": repr(e),
                    "traceback": traceback.format_exc(limit=12),
                }
            )
        flush_report()
        ran += 1
    df = pd.DataFrame(rows)
    write_dataframe(df, report_path)
    if not df.empty:
        write_json(df.to_dict("records"), report_path.with_suffix(".json"))
    return df
