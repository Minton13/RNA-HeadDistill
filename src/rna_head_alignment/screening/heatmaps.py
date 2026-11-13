from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import EmptyDataError

from rna_head_alignment.data.task_loaders import load_tasks_config
from rna_head_alignment.io.manifests import write_dataframe, write_json
from rna_head_alignment.io.paths import resolve_path


CORRELATION_METRICS = {"pearson", "spearman", "mcc"}
BOUNDED_HIGHER_IS_BETTER = {"auroc", "auprc", "macro_f1", "accuracy", "f1", "precision", "recall"}


def _metric_for_task(task: str, task_cfg: dict[str, dict[str, Any]]) -> str:
    return str(task_cfg.get(task, {}).get("metric", "metric_primary")).lower()


def normalize_metric_value(value: Any, metric: str) -> float:
    try:
        score = float(value)
    except Exception:
        return float("nan")
    if not np.isfinite(score):
        return float("nan")
    metric = metric.lower()
    if metric in CORRELATION_METRICS:
        return float(np.clip((score + 1.0) / 2.0, 0.0, 1.0))
    if metric in BOUNDED_HIGHER_IS_BETTER or metric == "metric_primary":
        return float(np.clip(score, 0.0, 1.0))
    if metric in {"mae", "rmse", "mse"}:
        # Loss-like metrics are uncommon for selection here. Convert within the
        # caller when task-level ranking is available; keep NaN for global color.
        return float("nan")
    return float(np.clip(score, 0.0, 1.0))


def _prepare_scores(atlas: pd.DataFrame, tasks_yaml: str | Path) -> pd.DataFrame:
    task_cfg = load_tasks_config(tasks_yaml)
    df = atlas.copy()
    if df.empty:
        return df
    df = df[pd.to_numeric(df.get("layer", -1), errors="coerce").fillna(-1) >= 0].copy()
    df["metric_primary"] = pd.to_numeric(df["metric_primary"], errors="coerce")
    df["task_metric"] = df["task"].map(lambda t: _metric_for_task(str(t), task_cfg))
    df["level"] = df["task"].map(lambda t: str(task_cfg.get(str(t), {}).get("group", "unknown")))
    df["score_norm"] = [
        normalize_metric_value(value, metric)
        for value, metric in zip(df["metric_primary"], df["task_metric"])
    ]
    # If a task uses a loss metric, convert it by within-task percentile so the
    # heatmap remains comparable without pretending the raw scale is comparable.
    for task, group in df.groupby("task", dropna=False):
        loss_mask = group["task_metric"].isin(["mae", "rmse", "mse"])
        if not loss_mask.any():
            continue
        idx = group.index[loss_mask]
        raw = pd.to_numeric(group.loc[idx, "metric_primary"], errors="coerce")
        if raw.notna().sum() >= 2:
            ranks = raw.rank(ascending=False, pct=True)
            df.loc[idx, "score_norm"] = ranks.to_numpy(dtype=float)
    return df


def _score_matrix(df: pd.DataFrame) -> np.ndarray:
    layers = sorted(pd.to_numeric(df["layer"], errors="coerce").dropna().astype(int).unique())
    heads = sorted(pd.to_numeric(df["head"], errors="coerce").dropna().astype(int).unique())
    if not layers or not heads:
        return np.zeros((1, 1), dtype=float) * np.nan
    mat = np.full((max(layers) + 1, max(heads) + 1), np.nan, dtype=float)
    for _, row in df.iterrows():
        layer = int(row["layer"])
        head = int(row["head"])
        mat[layer, head] = float(row["score_norm"])
    return mat


def _plot_panel_grid(
    groups: list[tuple[str, pd.DataFrame]],
    title: str,
    output: Path,
    vmin: float,
    vmax: float,
    cmap: str,
) -> None:
    if not groups:
        return
    max_heads = 1
    for _, group in groups:
        if not group.empty:
            max_heads = max(max_heads, int(pd.to_numeric(group["head"], errors="coerce").max()) + 1)
    fig_h = max(2.5, 1.75 * len(groups))
    fig_w = max(7.0, min(18.0, 2.5 + 0.35 * max_heads))
    fig, axes = plt.subplots(len(groups), 1, figsize=(fig_w, fig_h), squeeze=False)
    last_im = None
    for ax, (label, group) in zip(axes[:, 0], groups):
        mat = _score_matrix(group)
        last_im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(label, fontsize=9, loc="left")
        ax.set_ylabel("layer", fontsize=8)
        ax.set_xlabel("head", fontsize=8)
        ax.tick_params(axis="both", labelsize=7, length=2)
    fig.suptitle(title, fontsize=11)
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), shrink=0.8, pad=0.02)
        cbar.set_label("normalized score, shared 0..1 scale", fontsize=8)
        cbar.ax.tick_params(labelsize=7)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_single(group: pd.DataFrame, title: str, output: Path, vmin: float, vmax: float, cmap: str) -> None:
    if group.empty:
        return
    fig, ax = plt.subplots(figsize=(max(5.0, 0.35 * (int(group["head"].max()) + 1)), 3.6))
    im = ax.imshow(_score_matrix(group), aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("head")
    ax.set_ylabel("layer")
    fig.colorbar(im, ax=ax, label="normalized score, shared 0..1 scale")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_head_heatmaps(
    atlas_path: str | Path,
    tasks_yaml: str | Path = "configs/tasks.yaml",
    output_dir: str | Path = "outputs/head_screening/heatmaps",
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
    top_k: int = 50,
) -> dict[str, Any]:
    atlas_file = resolve_path(atlas_path)
    out_dir = resolve_path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not atlas_file.exists() or atlas_file.stat().st_size == 0:
        raise FileNotFoundError(f"Head atlas not found or empty: {atlas_file}")
    try:
        atlas = pd.read_csv(atlas_file)
    except EmptyDataError:
        atlas = pd.DataFrame()
    scores = _prepare_scores(atlas, tasks_yaml)
    write_dataframe(scores, out_dir / "task_head_scores.csv")

    if scores.empty:
        report = {"input": str(atlas_file), "rows": 0, "task_count": 0, "level_count": 0}
        write_json(report, out_dir / "heatmap_report.json")
        (out_dir / "heatmap_report.md").write_text("# Head Screening Heatmaps\n\nNo usable head rows.\n", encoding="utf-8")
        return report

    task_top = (
        scores.sort_values("score_norm", ascending=False)
        .groupby(["task"], dropna=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    write_dataframe(task_top, out_dir / "task_top_heads.csv")

    level_summary = (
        scores.groupby(["level", "model", "model_type", "layer", "head"], dropna=False)
        .agg(
            score_norm_mean=("score_norm", "mean"),
            score_norm_max=("score_norm", "max"),
            raw_metric_mean=("metric_primary", "mean"),
            n_tasks=("task", "nunique"),
            n_rows=("task", "size"),
        )
        .reset_index()
        .rename(columns={"score_norm_mean": "score_norm"})
    )
    write_dataframe(level_summary, out_dir / "level_head_summary.csv")
    level_top = (
        level_summary.sort_values(["level", "score_norm", "score_norm_max"], ascending=[True, False, False])
        .groupby("level", dropna=False)
        .head(top_k)
        .reset_index(drop=True)
    )
    write_dataframe(level_top, out_dir / "level_top_heads.csv")

    for task, group in scores.groupby("task", dropna=False):
        panels = [(str(model), mg.sort_values(["layer", "head"])) for model, mg in group.groupby("model", dropna=False)]
        _plot_panel_grid(
            panels,
            f"Task head scores: {task}",
            out_dir / "tasks" / f"{task}.png",
            vmin,
            vmax,
            cmap,
        )
        for model, mg in group.groupby("model", dropna=False):
            _plot_single(
                mg.sort_values(["layer", "head"]),
                f"{task} / {model}",
                out_dir / "tasks" / "by_model" / f"{task}__{model}.png",
                vmin,
                vmax,
                cmap,
            )

    for level, group in level_summary.groupby("level", dropna=False):
        panels = [(str(model), mg.sort_values(["layer", "head"])) for model, mg in group.groupby("model", dropna=False)]
        _plot_panel_grid(
            panels,
            f"Level-average head scores: {level}",
            out_dir / "levels" / f"{level}.png",
            vmin,
            vmax,
            cmap,
        )
        for model, mg in group.groupby("model", dropna=False):
            _plot_single(
                mg.sort_values(["layer", "head"]),
                f"{level} average / {model}",
                out_dir / "levels" / "by_model" / f"{level}__{model}.png",
                vmin,
                vmax,
                cmap,
            )

    report = {
        "input": str(atlas_file),
        "output_dir": str(out_dir),
        "rows": int(len(scores)),
        "task_count": int(scores["task"].nunique()),
        "model_count": int(scores["model"].nunique()),
        "level_count": int(scores["level"].nunique()),
        "vmin": vmin,
        "vmax": vmax,
        "task_heatmaps": int(scores["task"].nunique()),
        "level_heatmaps": int(level_summary["level"].nunique()),
    }
    write_json(report, out_dir / "heatmap_report.json")
    lines = [
        "# Head Screening Heatmaps",
        "",
        f"- input atlas: `{atlas_file}`",
        f"- output dir: `{out_dir}`",
        f"- rows: {report['rows']}",
        f"- tasks: {report['task_count']}",
        f"- models: {report['model_count']}",
        f"- levels: {report['level_count']}",
        f"- color scale: `{vmin}` to `{vmax}` on `score_norm`",
        "",
        "## Main Outputs",
        "",
        "- `task_head_scores.csv`: raw and normalized per-head task scores.",
        "- `task_top_heads.csv`: top heads per task.",
        "- `level_head_summary.csv`: per-level average score by model/layer/head.",
        "- `level_top_heads.csv`: top heads per level.",
        "- `tasks/*.png`: one shared-colorbar heatmap grid per task.",
        "- `levels/*.png`: one shared-colorbar heatmap grid per level.",
    ]
    (out_dir / "heatmap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report
