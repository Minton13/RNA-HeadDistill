#!/usr/bin/env python
"""Generate all Stage 4B/5 ablation figures using matplotlib only.

Reads summary parquet files produced by the various summarizers and writes
PNG/SVG figures under ``outputs/ablation/reports/figures/``.  If a source file
is missing, a placeholder figure with a text annotation is produced so the
script never fails.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from rna_head_alignment.io.paths import resolve_path

FIGURE_KWARGS = {"dpi": 200, "bbox_inches": "tight"}


def _placeholder(title: str, message: str, ax: plt.Axes | None = None) -> plt.Figure:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    ax.text(0.5, 0.5, message, ha="center", va="center", fontsize=10)
    ax.set_title(title)
    ax.axis("off")
    return fig


def _read_parquet(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def plot_alignment_ablation(df: pd.DataFrame | None, out_dir: Path) -> None:
    """A1-A8 per-route metrics grouped by experiment and route depth."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    if df is None or df.empty:
        _placeholder("Alignment Method Ablation (A1-A8)", "No alignment ablation summary found.", axes[0])
    else:
        # Aggregate per experiment/route_depth/seed.
        metric_cols = [c for c in df.columns if "val_" in c and not c.endswith("_std")]
        if not metric_cols:
            metric_cols = ["val_metric"]
        exp_order = sorted(df["experiment_id"].unique()) if "experiment_id" in df.columns else []
        x = np.arange(len(exp_order))
        width = 0.25
        for idx, route in enumerate([6, 8, 12, "mean"]):
            ax = axes[idx]
            if route == "mean":
                sub = df.copy()
                title = "Mean across routes"
            else:
                sub = df[df.get("route_depth", pd.Series(np.nan, index=df.index)) == route]
                title = f"Route L{route}"
            means: dict[str, list[float]] = {}
            stds: dict[str, list[float]] = {}
            for metric in metric_cols[:3]:
                means[metric] = []
                stds[metric] = []
                for exp in exp_order:
                    exp_sub = sub[sub.get("experiment_id", pd.Series("", index=sub.index)) == exp]
                    vals = exp_sub[metric].dropna() if metric in exp_sub else pd.Series(dtype=float)
                    means[metric].append(float(vals.mean()) if len(vals) else np.nan)
                    stds[metric].append(float(vals.std()) if len(vals) > 1 else 0.0)
            for offset, metric in enumerate(metric_cols[:3]):
                vals = np.array(means[metric])
                err = np.array(stds[metric])
                valid = ~np.isnan(vals)
                ax.bar(
                    x[valid] + (offset - 1) * width,
                    vals[valid],
                    width,
                    yerr=err[valid],
                    label=metric,
                )
            ax.set_xticks(x)
            ax.set_xticklabels(exp_order, rotation=45, ha="right", fontsize=7)
            ax.set_title(title)
            ax.legend(fontsize=7)
            ax.set_ylabel("Metric value")
    fig.suptitle("Alignment Method Ablation (A1-A8)", fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.savefig(out_dir / "alignment_method_ablation.png", **FIGURE_KWARGS)
    fig.savefig(out_dir / "alignment_method_ablation.svg", **FIGURE_KWARGS)
    plt.close(fig)


def plot_data_scaling(df: pd.DataFrame | None, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if df is None or df.empty or "size" not in df.columns:
        _placeholder("Data Scaling", "No data scaling summary found.", ax)
    else:
        for exp in sorted(df["experiment_id"].unique() if "experiment_id" in df.columns else []):
            sub = df[df["experiment_id"] == exp]
            metric_col = [c for c in sub.columns if c.startswith("val_") and not c.endswith("_std")][0]
            agg = sub.groupby("size")[metric_col].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["size"], agg["mean"], yerr=agg["std"], marker="o", label=exp)
        ax.set_xlabel("Training sequences")
        ax.set_ylabel("Validation metric")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.set_title("Alignment Performance vs. Training Data Size")
    fig.savefig(out_dir / "data_scaling.png", **FIGURE_KWARGS)
    fig.savefig(out_dir / "data_scaling.svg", **FIGURE_KWARGS)
    plt.close(fig)


def plot_async_cost_curve(df: pd.DataFrame | None, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    if df is None or df.empty:
        _placeholder("Asynchronous Systems Cost Curve", "No systems benchmark summary found.", ax)
    else:
        # Look for wall_time / gpu_hours / throughput columns.
        cost_col = "gpu_hours" if "gpu_hours" in df.columns else "wall_time_sec"
        metric_col = "val_metric" if "val_metric" in df.columns else None
        if metric_col and cost_col in df.columns and "system" in df.columns:
            for system in sorted(df["system"].unique()):
                sub = df[df["system"] == system]
                ax.scatter(sub[cost_col], sub[metric_col], label=system, s=60)
            ax.set_xlabel("GPU-hours")
            ax.set_ylabel("Validation metric")
            ax.legend(fontsize=8)
            ax.set_title("Asynchronous Distillation Systems: Cost vs. Quality")
        else:
            _placeholder("Asynchronous Systems Cost Curve", "Required columns missing.", ax)
    fig.savefig(out_dir / "async_systems_cost_curve.png", **FIGURE_KWARGS)
    fig.savefig(out_dir / "async_systems_cost_curve.svg", **FIGURE_KWARGS)
    plt.close(fig)


def plot_frozen_probe_p0_p6(df: pd.DataFrame | None, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    if df is None or df.empty or "feature_mode" not in df.columns:
        _placeholder("Frozen Probe P0-P6", "No frozen probe summary found.", ax)
    else:
        tasks = sorted(df["task"].unique())
        modes = [
            "vanilla_hidden",
            "distilled_hidden",
            "distilled_hidden_native",
            "distilled_aligned",
            "distilled_hidden_aligned",
            "distilled_hidden_teacher",
            "distilled_hidden_random_heads",
        ]
        x = np.arange(len(tasks))
        width = 0.11
        for offset, mode in enumerate(modes):
            vals = []
            errs = []
            for task in tasks:
                sub = df[(df["task"] == task) & (df["feature_mode"] == mode)]
                metric_col = [c for c in sub.columns if c.startswith("val_") and not c.endswith("_std")][0] if any(c.startswith("val_") for c in sub.columns) else None
                if metric_col and not sub.empty:
                    vals.append(float(sub[metric_col].mean()))
                    errs.append(float(sub[metric_col].std()) if len(sub) > 1 else 0.0)
                else:
                    vals.append(np.nan)
                    errs.append(0.0)
            vals = np.array(vals)
            valid = ~np.isnan(vals)
            ax.bar(
                x[valid] + (offset - 3) * width,
                vals[valid],
                width,
                yerr=np.array(errs)[valid],
                label=mode.replace("distilled_", ""),
            )
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.set_ylabel("Validation metric")
        ax.set_title("Frozen Probe Feature Modes P0-P6 Across Tasks")
        ax.legend(fontsize=7, ncol=2)
    fig.savefig(out_dir / "frozen_probe_p0_p6.png", **FIGURE_KWARGS)
    fig.savefig(out_dir / "frozen_probe_p0_p6.svg", **FIGURE_KWARGS)
    plt.close(fig)


def plot_finetune_replay_ablation(df: pd.DataFrame | None, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    if df is None or df.empty:
        _placeholder("Fine-Tuning / Replay Ablation", "No route fine-tune summary found.", ax)
    else:
        tasks = sorted(df["task"].unique())
        lambdas = sorted(df["lambda_align"].unique())
        x = np.arange(len(tasks))
        width = 0.18
        for offset, lam in enumerate(lambdas):
            vals = []
            errs = []
            for task in tasks:
                sub = df[(df["task"] == task) & (df["lambda_align"] == lam) & (df["feature_mode"] == "F2")]
                metric_col = [c for c in sub.columns if c.startswith("val_") and not c.endswith("_std")][0] if any(c.startswith("val_") for c in sub.columns) else None
                if metric_col and not sub.empty:
                    vals.append(float(sub[metric_col].mean()))
                    errs.append(float(sub[metric_col].std()) if len(sub) > 1 else 0.0)
                else:
                    vals.append(np.nan)
                    errs.append(0.0)
            vals = np.array(vals)
            valid = ~np.isnan(vals)
            ax.bar(
                x[valid] + (offset - len(lambdas) / 2 + 0.5) * width,
                vals[valid],
                width,
                yerr=np.array(errs)[valid],
                label=f"λ={lam}",
            )
        ax.set_xticks(x)
        ax.set_xticklabels(tasks, rotation=45, ha="right")
        ax.set_ylabel("Validation metric")
        ax.set_title("Fine-Tuning with Alignment Replay: λ_align Ablation")
        ax.legend(fontsize=8)
    fig.savefig(out_dir / "finetune_replay_ablation.png", **FIGURE_KWARGS)
    fig.savefig(out_dir / "finetune_replay_ablation.svg", **FIGURE_KWARGS)
    plt.close(fig)


def plot_head_contribution_heatmap(df: pd.DataFrame | None, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    if df is None or df.empty:
        _placeholder("Head Contribution Heatmap", "No gate-weight data found.", ax)
    else:
        # Expect columns: task, route_depth, head_idx, gate_weight.
        required = {"task", "route_depth", "head_idx", "gate_weight"}
        if required.issubset(df.columns):
            pivot = df.pivot_table(
                index=["task", "route_depth"],
                columns="head_idx",
                values="gate_weight",
                aggfunc="mean",
            )
            im = ax.imshow(pivot.values, aspect="auto", cmap="viridis")
            ax.set_xticks(np.arange(pivot.shape[1]))
            ax.set_xticklabels([f"H{i}" for i in pivot.columns])
            ax.set_yticks(np.arange(pivot.shape[0]))
            ax.set_yticklabels([f"{t} L{r}" for t, r in pivot.index])
            plt.colorbar(im, ax=ax, label="Learned gate weight")
            ax.set_title("Head Contribution Heatmap (Learned Gate Weights)")
        else:
            _placeholder("Head Contribution Heatmap", f"Missing columns {required - set(df.columns)}.", ax)
    fig.savefig(out_dir / "head_contribution_heatmap.png", **FIGURE_KWARGS)
    fig.savefig(out_dir / "head_contribution_heatmap.svg", **FIGURE_KWARGS)
    plt.close(fig)


def plot_length_robustness(df: pd.DataFrame | None, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if df is None or df.empty:
        _placeholder("Length Robustness", "No length-binned data found.", ax)
    else:
        required = {"length_bin", "value", "metric"}
        if required.issubset(df.columns):
            metric = df["metric"].iloc[0]
            agg = df.groupby("length_bin")["value"].agg(["mean", "std"]).reset_index()
            ax.errorbar(agg["length_bin"], agg["mean"], yerr=agg["std"], marker="o")
            ax.set_xlabel("Sequence length bin")
            ax.set_ylabel(metric)
            ax.set_title(f"Length Robustness ({metric})")
            ax.tick_params(axis="x", rotation=45)
        else:
            _placeholder("Length Robustness", f"Missing columns {required - set(df.columns)}.", ax)
    fig.savefig(out_dir / "length_robustness.png", **FIGURE_KWARGS)
    fig.savefig(out_dir / "length_robustness.svg", **FIGURE_KWARGS)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="outputs/ablation/reports/figures")
    args = parser.parse_args()

    out_dir = Path(resolve_path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read all available summary files.
    summaries = {
        "alignment": _read_parquet(Path(resolve_path("outputs/ablation/alignment/core_ablation_summary.parquet"))),
        "data_scaling": _read_parquet(Path(resolve_path("outputs/ablation/alignment/data_scaling_summary.parquet"))),
        "systems": _read_parquet(Path(resolve_path("outputs/systems/asynchronous_distillation/systems_benchmark_summary.parquet"))),
        "frozen_probe": _read_parquet(Path(resolve_path("outputs/probes/frozen/frozen_probe_summary.parquet"))),
        "route_finetune": _read_parquet(Path(resolve_path("outputs/probes/finetune/route_finetune_summary.parquet"))),
        "head_contribution": _read_parquet(Path(resolve_path("outputs/probes/frozen/head_contribution_summary.parquet"))),
        "length_robustness": _read_parquet(Path(resolve_path("outputs/ablation/alignment/length_robustness.parquet"))),
    }

    plot_alignment_ablation(summaries["alignment"], out_dir)
    plot_data_scaling(summaries["data_scaling"], out_dir)
    plot_async_cost_curve(summaries["systems"], out_dir)
    plot_frozen_probe_p0_p6(summaries["frozen_probe"], out_dir)
    plot_finetune_replay_ablation(summaries["route_finetune"], out_dir)
    plot_head_contribution_heatmap(summaries["head_contribution"], out_dir)
    plot_length_robustness(summaries["length_robustness"], out_dir)

    print(f"[figures] wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
