"""Systems-study benchmark helpers.

Collects historical Slurm costs and runs deterministic microbenchmarks comparing:
  S0: offline cached distillation (student forward + cache read)
  S1: online single-teacher extraction
  S2: online three-teacher extraction
  S3: extrapolated online full-teacher extraction (all 9 models)
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from rna_head_alignment.head2head.eval import _load_model_for_eval
from rna_head_alignment.head2head.full_sequence_dataset import (
    FullSlimBucketDataset,
    LengthBucketedSampler,
    full_variable_length_collate,
)
from rna_head_alignment.io.paths import get_device, load_yaml, resolve_path
from rna_head_alignment.ablation.ablation_runner import get_experiment_config


HISTORICAL_PATTERNS = {
    "teacher_cache": ["cache_teacher_attention", "teacher_attention_array"],
    "slim_cache_build": ["full_slim_cache", "build_slim_cache", "slim_cache"],
    "stage4b_training": ["head2head_train", "stage4b_train", "rinalmo_head2head"],
}


def _elapsed_to_seconds(elapsed: str) -> float:
    """Parse ``sacct`` elapsed strings such as ``12:34:56`` or ``2-12:34:56``."""
    elapsed = str(elapsed).strip()
    if not elapsed or elapsed == "00:00:00":
        return 0.0
    days = 0.0
    if "-" in elapsed:
        day_part, elapsed = elapsed.split("-", 1)
        days = float(day_part)
    parts = elapsed.split(":")
    if len(parts) == 3:
        h, m, s = map(float, parts)
    elif len(parts) == 2:
        h, m = map(float, parts)
        s = 0.0
    else:
        h = m = 0.0
        s = float(parts[0])
    return days * 86400.0 + h * 3600.0 + m * 60.0 + s


def _rss_to_mb(rss: str) -> float:
    """Parse ``sacct`` MaxRSS strings such as ``1.23G`` or ``456K``."""
    rss = str(rss).strip().upper()
    if not rss or rss == "0":
        return 0.0
    match = re.match(r"([0-9.]+)\s*([KMGT]?)B?", rss)
    if not match:
        return 0.0
    value = float(match.group(1))
    unit = match.group(2)
    mult = {"K": 1e-3, "M": 1.0, "G": 1e3, "T": 1e6, "": 1.0}
    return value * mult.get(unit, 1.0)


def collect_sacct_jobs(start_days: int = 30, jobname_patterns: list[str] | None = None) -> pd.DataFrame:
    """Query ``sacct`` for historical job accounting rows.

    Returns an empty DataFrame if ``sacct`` is unavailable or returns no rows.
    """
    if jobname_patterns is None:
        jobname_patterns = sorted({p for v in HISTORICAL_PATTERNS.values() for p in v})
    since = f"now-{start_days}days"
    cmd = [
        "sacct",
        "-n",
        "-P",
        "--delimiter=|",
        "--starttime", since,
        "--format", "JobIDRaw,JobName,State,Elapsed,TotalCPU,MaxRSS,NNodes,ReqGRES",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False, timeout=120)
    except Exception as exc:  # noqa: BLE001
        return pd.DataFrame({"error": [str(exc)]})
    if proc.returncode != 0 or not proc.stdout.strip():
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for line in proc.stdout.strip().splitlines():
        parts = line.split("|")
        if len(parts) < 8:
            continue
        jobid, jobname, state, elapsed, totalcpu, maxrss, nnodes, reqgres = parts[:8]
        if not any(pat.lower() in jobname.lower() for pat in jobname_patterns):
            continue
        rows.append({
            "job_id": jobid,
            "job_name": jobname,
            "state": state,
            "elapsed_sec": _elapsed_to_seconds(elapsed),
            "total_cpu_sec": _elapsed_to_seconds(totalcpu),
            "max_rss_mb": _rss_to_mb(maxrss),
            "n_nodes": int(nnodes) if nnodes.strip() else 1,
            "req_gres": reqgres,
        })
    return pd.DataFrame(rows)


def categorize_historical_jobs(df: pd.DataFrame) -> pd.DataFrame:
    """Add a ``stage`` column inferred from the job name."""
    def _stage(name: str) -> str:
        name_l = name.lower()
        for stage, patterns in HISTORICAL_PATTERNS.items():
            if any(p in name_l for p in patterns):
                return stage
        return "other"
    df = df.copy()
    df["stage"] = df["job_name"].apply(_stage)
    return df


def historical_cost_rows(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Summarize historical Slurm costs by stage."""
    if df.empty:
        return []
    df = categorize_historical_jobs(df)
    rows: list[dict[str, Any]] = []
    for stage, grp in df.groupby("stage"):
        rows.append({
            "scenario": "historical",
            "stage": stage,
            "n_jobs": int(len(grp)),
            "n_completed": int((grp["state"] == "COMPLETED").sum()),
            "total_wall_sec": float(grp["elapsed_sec"].sum()),
            "total_wall_hr": float(grp["elapsed_sec"].sum()) / 3600.0,
            "total_cpu_sec": float(grp["total_cpu_sec"].sum()),
            "max_rss_gb": float(grp["max_rss_mb"].max()) / 1e3,
            "notes": "from sacct",
        })
    return rows


def _sample_seq_ids(
    sequence_manifest_path: str | Path,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """Deterministically sample ``n`` sequences using a hash-based ordering."""
    df = pd.read_parquet(resolve_path(sequence_manifest_path))
    if len(df) <= n:
        return df.copy()
    rng = np.random.default_rng(seed)
    return df.sample(n=n, random_state=rng)


def _write_sample_index(sample_df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_df[["seq_id"]].to_parquet(path, index=False)
    return path


def benchmark_offline_cached(
    config: dict[str, Any],
    sample_df: pd.DataFrame,
    output_dir: Path,
    device: torch.device,
) -> dict[str, Any]:
    """Microbenchmark S0: load cached teacher maps and run the student forward."""
    from rna_head_alignment.ablation.ablation_runner import build_ablation_model

    sample_index = _write_sample_index(sample_df, output_dir / "microbench_sample.parquet")
    full_cache_root = resolve_path(config["full_slim_cache_root"])
    seq_manifest_path = resolve_path(config["sequence_manifest"])

    model = build_ablation_model(config, device)
    model.eval()
    tokenizer = model.backbone.tokenizer()

    dataset = FullSlimBucketDataset(
        full_cache_root,
        sample_index,
        seq_manifest_path,
        tokenizer,
        max_len=int(config.get("runtime", {}).get("max_len", 1024)),
        cache_size=1,
    )
    sampler = LengthBucketedSampler(
        dataset.lengths(),
        max_pair_tokens=int(config.get("dynamic_batch", {}).get("max_pair_tokens", 524288)),
        max_sequences_per_batch=4,
        seed=42,
        shuffle=False,
        drop_last=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=full_variable_length_collate,
        num_workers=0,
    )

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    n_seqs = 0
    start = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            _ = model(batch["input_ids"], batch["attention_mask"])
            n_seqs += int(batch["lengths"].numel())
    elapsed = time.perf_counter() - start

    peak_vram = (
        torch.cuda.max_memory_allocated(device) / 1e9
        if device.type == "cuda"
        else 0.0
    )
    return {
        "scenario": "S0_offline_cached",
        "stage": "microbenchmark",
        "n_sequences": n_seqs,
        "total_sec": elapsed,
        "throughput_seq_per_sec": n_seqs / elapsed if elapsed > 0 else 0.0,
        "peak_vram_gb": peak_vram,
        "notes": "student forward + bucketed cache read",
    }


def benchmark_online_teachers(
    config: dict[str, Any],
    sample_df: pd.DataFrame,
    models: list[str],
    device: torch.device,
) -> dict[str, Any]:
    """Microbenchmark online teacher attention extraction for a list of models."""
    from rna_head_alignment.cache.build_teacher_attention_cache import selected_heads_by_model
    from rna_head_alignment.models.attention_extractors import extract_attention_bundle_batch
    from rna_head_alignment.models.loaders import load_frozen_model
    from rna_head_alignment.models.registry import get_model_spec

    sequences = [
        {"seq_id": row["seq_id"], "sequence": str(row["sequence"])}
        for _, row in sample_df.iterrows()
    ]
    by_model = selected_heads_by_model(config.get("teacher_bank", "configs/teacher_bank_selected.yaml"))
    all_models = sorted(by_model.keys())
    chosen = [m for m in models if m in all_models][: len(models)]
    if len(chosen) < len(models):
        chosen = all_models[: len(models)]

    max_len = int(config.get("runtime", {}).get("max_len", 1024))
    total_sec = 0.0
    peak_vram = 0.0
    n_seqs = len(sequences)

    for model_name in chosen:
        spec = get_model_spec("configs/models.yaml", model_name)
        tokenizer, model = load_frozen_model(spec, str(device))
        heads = by_model.get(model_name, [])
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        _ = extract_attention_bundle_batch(
            spec,
            tokenizer,
            model,
            sequences,
            max_len=max_len,
            device=str(device),
            max_tokens_per_batch=4096,
            output_hidden_states=False,
            output_attentions=True,
            selected_heads=heads,
        )
        elapsed = time.perf_counter() - start
        total_sec += elapsed
        if device.type == "cuda":
            peak_vram = max(peak_vram, torch.cuda.max_memory_allocated(device) / 1e9)
        del model, tokenizer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "scenario": f"S{3 - len(chosen)}_online_{len(chosen)}teacher{'s' if len(chosen) > 1 else ''}",
        "stage": "microbenchmark",
        "n_sequences": n_seqs,
        "n_models": len(chosen),
        "models": chosen,
        "total_sec": total_sec,
        "throughput_seq_per_sec": n_seqs / total_sec if total_sec > 0 else 0.0,
        "peak_vram_gb": peak_vram,
        "notes": f"online extraction of {len(chosen)} teacher model(s)",
    }


def run_microbenchmarks(
    registry_path: str | Path,
    output_dir: str | Path,
    n_sequences: int = 10000,
    seed: int = 42,
    device: str = "auto",
) -> list[dict[str, Any]]:
    """Run S0/S1/S2 microbenchmarks and extrapolate S3."""
    registry = load_yaml(registry_path)
    base = registry.get("base_data", {})
    config = {
        "full_slim_cache_root": base.get("slim_cache_root"),
        "sequence_manifest": base.get("sequence_manifest"),
        "target_head_manifest": base.get("target_head_manifest"),
        "teacher_bank": "configs/teacher_bank_selected.yaml",
        "head_pair_manifest": base.get("head_pair_manifest"),
        "runtime": {"max_len": 1024},
        "dynamic_batch": {"max_pair_tokens": 524288},
        "student": {"model_dir": "assets/models/rinalmo-micro"},
        "model": {"adapter_dim": 96, "adapter_ffn_mult": 2, "adapter_type": "transformer"},
    }
    # Use the A3 experiment architecture for the student model.
    cfg_a3 = get_experiment_config(registry, "A3_head2head_top5_full", seed)
    config.update({k: v for k, v in cfg_a3.items() if k in ("teacher_head_subset", "student_pairing_policy", "teacher_aggregation_policy", "adapter_type", "route_policy", "model", "student", "head_pair_manifest", "target_head_manifest")})
    if "full_slim_cache_root" not in config:
        config["full_slim_cache_root"] = config.get("slim_cache_root")

    resolved_device = torch.device(get_device(device) if device == "auto" else device)
    out_dir = Path(resolve_path(output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_df = _sample_seq_ids(config["sequence_manifest"], n_sequences, seed)
    sample_df.to_parquet(out_dir / "microbench_sample_manifest.parquet", index=False)

    # S0 offline cached.
    s0 = benchmark_offline_cached(config, sample_df, out_dir, resolved_device)

    from rna_head_alignment.cache.build_teacher_attention_cache import selected_heads_by_model
    all_models = sorted(selected_heads_by_model(config["teacher_bank"]).keys())

    # S1 online single teacher.
    s1 = benchmark_online_teachers(config, sample_df, [all_models[0]], resolved_device)
    s1["scenario"] = "S1_online_single_teacher"

    # S2 online three teachers.
    s2 = benchmark_online_teachers(config, sample_df, all_models[:3], resolved_device)
    s2["scenario"] = "S2_online_three_teachers"

    # S3 extrapolated from measured per-model timings in S2.
    per_model_sec = s2["total_sec"] / max(s2.get("n_models", 3), 1)
    s3_total = per_model_sec * len(all_models)
    s3 = {
        "scenario": "S3_online_full_estimate",
        "stage": "extrapolated",
        "n_sequences": n_sequences,
        "n_models": len(all_models),
        "models": all_models,
        "total_sec": s3_total,
        "throughput_seq_per_sec": n_sequences / s3_total if s3_total > 0 else 0.0,
        "peak_vram_gb": s2["peak_vram_gb"],
        "notes": f"extrapolated from S2 per-model time ({len(all_models)} models); NOT measured",
    }

    return [s0, s1, s2, s3]


def make_cost_curve(
    historical_rows: list[dict[str, Any]],
    micro_rows: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write the asynchronous cost curve CSV and PNG."""
    hist_df = pd.DataFrame(historical_rows)
    micro_df = pd.DataFrame(micro_rows)
    curve_rows: list[dict[str, Any]] = []
    for row in historical_rows:
        curve_rows.append({
            "scenario": row["scenario"],
            "stage": row["stage"],
            "metric": "total_wall_hr",
            "value": row["total_wall_hr"],
            "notes": row.get("notes", ""),
        })
    for row in micro_rows:
        curve_rows.append({
            "scenario": row["scenario"],
            "stage": row["stage"],
            "metric": "total_sec",
            "value": row["total_sec"],
            "notes": row.get("notes", ""),
        })
        curve_rows.append({
            "scenario": row["scenario"],
            "stage": row["stage"],
            "metric": "throughput_seq_per_sec",
            "value": row["throughput_seq_per_sec"],
            "notes": row.get("notes", ""),
        })
    curve_df = pd.DataFrame(curve_rows)
    csv_path = output_dir / "asynchronous_cost_curve.csv"
    curve_df.to_csv(csv_path, index=False)

    # Plot.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        micro_plot = micro_df.copy()
        micro_plot["total_hr"] = micro_plot["total_sec"] / 3600.0
        colors = {"S0_offline_cached": "green", "S1_online_single_teacher": "blue", "S2_online_three_teachers": "orange", "S3_online_full_estimate": "red"}
        for _, row in micro_plot.iterrows():
            ax.bar(row["scenario"], row["total_hr"], color=colors.get(row["scenario"], "gray"), label=row["scenario"])
        ax.set_ylabel("Hours for 10k sequences")
        ax.set_title("Asynchronous distillation cost curve (microbenchmark)")
        ax.set_yscale("log")
        plt.xticks(rotation=30, ha="right")
        plt.tight_layout()
        png_path = output_dir / "asynchronous_cost_curve.png"
        fig.savefig(png_path, dpi=150)
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        png_path = output_dir / "asynchronous_cost_curve.png"
        png_path.write_text(f"plot failed: {exc}", encoding="utf-8")
    return csv_path


def write_systems_report(
    historical_rows: list[dict[str, Any]],
    micro_rows: list[dict[str, Any]],
    output_dir: Path,
) -> Path:
    """Write systems_ablation_report.md."""
    hist_df = pd.DataFrame(historical_rows)
    micro_df = pd.DataFrame(micro_rows)
    throughput_df = micro_df[["scenario", "throughput_seq_per_sec", "n_sequences", "notes"]].copy()
    memory_df = micro_df[["scenario", "peak_vram_gb", "notes"]].copy()

    hist_csv = output_dir / "historical_costs.csv"
    hist_df.to_csv(hist_csv, index=False)
    throughput_csv = output_dir / "throughput_comparison.csv"
    throughput_df.to_csv(throughput_csv, index=False)
    memory_csv = output_dir / "memory_comparison.csv"
    memory_df.to_csv(memory_csv, index=False)

    lines = [
        "# Systems Study: Asynchronous Distillation Cost",
        "",
        "## Historical Slurm costs (from sacct)",
        "",
        hist_df.to_string(index=False) if not hist_df.empty else "_No historical sacct rows matched._",
        "",
        "## Microbenchmarks (10,000 deterministic sequences)",
        "",
        micro_df[["scenario", "stage", "n_sequences", "n_models", "total_sec", "throughput_seq_per_sec", "peak_vram_gb", "notes"]].to_string(index=False),
        "",
        "## Throughput comparison",
        "",
        throughput_df.to_string(index=False),
        "",
        "## Memory comparison",
        "",
        memory_df.to_string(index=False),
        "",
        "### Notes",
        "- S3 is an **extrapolation** from S2 per-model timings over all teacher models; it was not measured online.",
        "- Historical costs aggregate matching Slurm job names; zero rows means sacct returned no matches.",
        "- Peak VRAM is the maximum observed GPU memory for the scenario.",
        "",
    ]
    report_path = output_dir / "systems_ablation_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
