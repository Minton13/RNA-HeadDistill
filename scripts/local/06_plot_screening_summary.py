from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


LEVEL_ORDER = ["L6", "L8", "L12"]
LEVEL_COLORS = {"L6": "#4E79A7", "L8": "#59A14F", "L12": "#E15759"}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _savefig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def _task_order(task_summary: pd.DataFrame) -> list[str]:
    frame = task_summary.copy()
    frame["level_rank"] = frame["level"].map({x: i for i, x in enumerate(LEVEL_ORDER)}).fillna(99)
    return frame.sort_values(["level_rank", "task"])["task"].tolist()


def _model_order(model_summary: pd.DataFrame, task_model_best: pd.DataFrame) -> list[str]:
    strength = task_model_best.groupby("model")["score_norm"].mean().rename("mean_score")
    frame = model_summary.merge(strength, on="model", how="left")
    frame["mean_score"] = frame["mean_score"].fillna(-1.0)
    frame["type_rank"] = frame["type"].map({"RNA": 0, "DNA": 1}).fillna(2)
    frame["unsupported_rank"] = (frame["scored_tasks"].astype(float) <= 0).astype(int)
    return frame.sort_values(
        ["unsupported_rank", "type_rank", "mean_score", "model"],
        ascending=[True, True, False, True],
    )["model"].tolist()


def apply_recheck_status(model_summary: pd.DataFrame, root: Path) -> pd.DataFrame:
    """Update display status with post-repair model capability checks.

    Screening scores remain unchanged; this only prevents server-only models
    with confirmed attentions from being shown as broken/no-attention models.
    """
    out = model_summary.copy()
    recheck_path = root / "outputs" / "model_checks" / "model_attention_recheck_unsupported_after_repairs.csv"
    out["display_status"] = out["status"]
    if not recheck_path.exists():
        return out
    recheck = pd.read_csv(recheck_path)
    attention_ok = set(
        recheck[
            (recheck.get("load_ok", False).astype(bool))
            & (recheck.get("forward_ok", False).astype(bool))
            & (recheck.get("has_attentions", False).astype(bool))
        ]["model"].astype(str)
    )
    mask = out["model"].astype(str).isin(attention_ok) & (out["scored_tasks"].astype(float) <= 0)
    out.loc[mask, "display_status"] = "server_only_attention_ok"
    return out


def plot_task_model_heatmap(
    scores: pd.DataFrame,
    model_summary: pd.DataFrame,
    task_summary: pd.DataFrame,
    out_dir: Path,
    sample_label: str,
) -> pd.DataFrame:
    task_model_best = (
        scores.groupby(["level", "task", "model"], as_index=False)["score_norm"].max()
    )
    task_order = _task_order(task_summary)
    model_order = _model_order(model_summary, task_model_best)
    main_models = model_summary[
        (model_summary["scored_tasks"].astype(float) > 0)
        | (model_summary.get("display_status", "") == "server_only_attention_ok")
    ]["model"].astype(str).tolist()
    model_order = [m for m in model_order if m in set(main_models)]
    matrix = task_model_best.pivot_table(index="task", columns="model", values="score_norm", aggfunc="max")
    matrix = matrix.reindex(index=task_order, columns=model_order)
    matrix.to_csv(out_dir / "task_model_best_score_matrix.csv")

    task_to_level = task_summary.set_index("task")["level"].to_dict()
    row_colors = [LEVEL_COLORS.get(task_to_level.get(task, ""), "#999999") for task in matrix.index]
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("#E6E6E6")

    fig, ax = plt.subplots(figsize=(max(16, 0.42 * len(matrix.columns)), max(9, 0.34 * len(matrix.index))))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        linewidths=0.35,
        linecolor="white",
        cbar_kws={"label": "best normalized score per task-model"},
    )
    ax.set_title(f"Task x Model best head score, {sample_label}")
    ax.set_xlabel("Model")
    ax.set_ylabel("Task")
    ax.tick_params(axis="x", labelrotation=65, labelsize=8)
    ax.tick_params(axis="y", labelsize=9)
    for y, color in enumerate(row_colors):
        ax.add_patch(plt.Rectangle((-0.8, y), 0.22, 1, color=color, clip_on=False, linewidth=0))
    _savefig(out_dir / "task_model_best_score_heatmap.png")

    scored_only = [m for m in model_order if m in set(task_model_best["model"])]
    if scored_only:
        matrix_scored = matrix.reindex(columns=scored_only)
        fig, ax = plt.subplots(figsize=(max(13, 0.45 * len(matrix_scored.columns)), max(9, 0.34 * len(matrix_scored.index))))
        sns.heatmap(
            matrix_scored,
            ax=ax,
            cmap=cmap,
            vmin=0,
            vmax=1,
            linewidths=0.35,
            linecolor="white",
            cbar_kws={"label": "best normalized score per task-model"},
        )
        ax.set_title("Task x Model best head score, scored models only")
        ax.set_xlabel("Model")
        ax.set_ylabel("Task")
        ax.tick_params(axis="x", labelrotation=65, labelsize=8)
        ax.tick_params(axis="y", labelsize=9)
        for y, color in enumerate(row_colors):
            ax.add_patch(plt.Rectangle((-0.8, y), 0.22, 1, color=color, clip_on=False, linewidth=0))
        _savefig(out_dir / "task_model_best_score_heatmap_scored_only.png")
    return task_model_best


def plot_model_level_heatmap(
    task_model_best: pd.DataFrame,
    model_summary: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    model_level = (
        task_model_best.groupby(["model", "level"], as_index=False)["score_norm"].mean()
    )
    matrix = model_level.pivot_table(index="model", columns="level", values="score_norm", aggfunc="mean")
    for level in LEVEL_ORDER:
        if level not in matrix.columns:
            matrix[level] = np.nan
    matrix = matrix[LEVEL_ORDER]
    order = _model_order(model_summary, task_model_best)
    matrix = matrix.reindex(order)
    matrix.to_csv(out_dir / "model_level_mean_score_matrix.csv")

    cmap = plt.cm.magma.copy()
    cmap.set_bad("#E6E6E6")
    fig, ax = plt.subplots(figsize=(7, max(8, 0.28 * len(matrix.index))))
    sns.heatmap(
        matrix,
        ax=ax,
        cmap=cmap,
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.4,
        linecolor="white",
        cbar_kws={"label": "mean normalized score across tasks in level"},
    )
    ax.set_title("Model strength by alignment level")
    ax.set_xlabel("Alignment level")
    ax.set_ylabel("Model")
    _savefig(out_dir / "model_level_strength_heatmap.png")
    return model_level


def plot_task_top_bars(
    scores: pd.DataFrame,
    task_summary: pd.DataFrame,
    out_dir: Path,
) -> pd.DataFrame:
    top3 = (
        scores.sort_values("score_norm", ascending=False)
        .groupby("task", as_index=False)
        .head(3)
        .copy()
    )
    order = _task_order(task_summary)
    top3["rank"] = top3.groupby("task")["score_norm"].rank(method="first", ascending=False).astype(int)
    top3["task"] = pd.Categorical(top3["task"], categories=order, ordered=True)
    top3 = top3.sort_values(["task", "rank"])
    leading_cols = ["task", "rank", "model", "model_type", "layer", "head", "score_norm", "metric_primary", "task_metric", "level"]
    top3 = top3[[c for c in leading_cols if c in top3.columns] + [c for c in top3.columns if c not in leading_cols]]
    top3.to_csv(out_dir / "task_best_head_summary.csv", index=False)
    top3.to_csv(out_dir / "task_top3_head_summary.csv", index=False)

    top = top3[top3["rank"] == 1].copy()

    colors = [LEVEL_COLORS.get(x, "#999999") for x in top["level"]]
    labels = [f"{m} L{int(l)}H{int(h)}" for m, l, h in zip(top["model"], top["layer"], top["head"])]
    fig, ax = plt.subplots(figsize=(12, max(8, 0.42 * len(top))))
    y = np.arange(len(top))
    ax.barh(y, top["score_norm"], color=colors, alpha=0.86)
    ax.set_yticks(y, [str(x) for x in top["task"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("best normalized score")
    ax.set_title("Best head per task")
    for yi, score, label in zip(y, top["score_norm"], labels):
        ax.text(min(float(score) + 0.015, 0.98), yi, label, va="center", fontsize=8)
    _savefig(out_dir / "task_best_head_bar.png")

    rank_colors = {1: "#1F77B4", 2: "#FF7F0E", 3: "#2CA02C"}
    top3_plot = top3.copy()
    top3_plot["task_text"] = top3_plot["task"].astype(str)
    top3_plot["head_label"] = top3_plot.apply(
        lambda r: f"#{int(r['rank'])} {r['model']} L{int(r['layer'])}H{int(r['head'])}", axis=1
    )
    task_order = [str(x) for x in order]
    task_to_base = {task: i * 4 for i, task in enumerate(task_order)}
    offsets = {1: -0.9, 2: 0.0, 3: 0.9}
    top3_plot["y"] = top3_plot.apply(
        lambda r: task_to_base.get(str(r["task_text"]), 0) + offsets.get(int(r["rank"]), 0.0), axis=1
    )

    fig, ax = plt.subplots(figsize=(15, max(10, 0.82 * len(task_order))))
    for rank in [3, 2, 1]:
        sub = top3_plot[top3_plot["rank"] == rank].copy()
        ax.barh(
            sub["y"],
            sub["score_norm"],
            height=0.72,
            color=rank_colors[rank],
            alpha=0.82,
            label=f"rank {rank}",
        )
        for _, row in sub.iterrows():
            ax.text(
                min(float(row["score_norm"]) + 0.012, 0.985),
                float(row["y"]),
                str(row["head_label"]),
                va="center",
                fontsize=7.2,
            )
    ax.set_yticks([task_to_base[t] for t in task_order], task_order)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("normalized score")
    ax.set_title("Top 3 heads per task")
    ax.legend(loc="lower right", frameon=False)
    for i, task in enumerate(task_order):
        color = LEVEL_COLORS.get(task_summary.set_index("task")["level"].to_dict().get(task, ""), "#999999")
        ax.add_patch(plt.Rectangle((-0.045, i * 4 - 1.45), 0.018, 2.9, color=color, clip_on=False, linewidth=0))
    _savefig(out_dir / "task_top3_heads_bar.png")
    return top3


def plot_level_top_heads(level_top: pd.DataFrame, out_dir: Path, top_n: int = 25) -> None:
    rows = []
    for level in LEVEL_ORDER:
        sub = level_top[level_top["level"] == level].head(top_n).copy()
        sub["head_label"] = sub.apply(
            lambda r: f"{r['model']} L{int(r['layer'])}H{int(r['head'])}", axis=1
        )
        rows.append(sub)
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    frame.to_csv(out_dir / "level_top_heads_for_plot.csv", index=False)

    fig, axes = plt.subplots(1, len(LEVEL_ORDER), figsize=(18, max(9, 0.28 * top_n)), sharex=True)
    if len(LEVEL_ORDER) == 1:
        axes = [axes]
    for ax, level in zip(axes, LEVEL_ORDER):
        sub = frame[frame["level"] == level].copy().sort_values("score_norm")
        ax.barh(sub["head_label"], sub["score_norm"], color=LEVEL_COLORS[level], alpha=0.86)
        ax.set_title(level)
        ax.set_xlim(0, 1)
        ax.tick_params(axis="y", labelsize=7)
        ax.set_xlabel("mean normalized score")
    axes[0].set_ylabel("Candidate teacher heads")
    fig.suptitle("Top candidate heads by alignment level", y=1.02)
    _savefig(out_dir / "level_top_heads_bar.png")


def plot_unsupported_reasons(unsupported: pd.DataFrame, out_dir: Path) -> None:
    if unsupported.empty:
        return

    def bucket(reason: str) -> str:
        text = str(reason).lower()
        if "mamba_ssm" in text:
            return "missing mamba_ssm runtime"
        if "no_attention" in text:
            return "no standard attentions"
        if "vocab_size" in text:
            return "checkpoint/config mismatch"
        if "meta" in text:
            return "meta tensor/config issue"
        if "tokenizer" in text or "backend tokenizer" in text:
            return "tokenizer/runtime issue"
        if "offline" in text or "connect" in text:
            return "missing offline files"
        return "other"

    frame = unsupported.copy()
    if "display_status" in frame:
        frame.loc[frame["display_status"] == "server_only_attention_ok", "reason_sample"] = (
            "attention extraction repaired and verified; local 100x16 screening deferred to server"
        )
    frame["reason_bucket"] = frame["reason_sample"].map(bucket)
    if "display_status" in frame:
        frame.loc[frame["display_status"] == "server_only_attention_ok", "reason_bucket"] = "server-only attention OK"
    counts = frame["reason_bucket"].value_counts().sort_values()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh(counts.index, counts.values, color="#9C755F", alpha=0.85)
    ax.set_xlabel("model count")
    ax.set_title("Unsupported / no-attention models in local screening")
    for y, value in enumerate(counts.values):
        ax.text(value + 0.05, y, str(value), va="center")
    _savefig(out_dir / "unsupported_model_reasons.png")
    frame.to_csv(out_dir / "unsupported_model_reason_buckets.csv", index=False)


def make_sankey(
    task_model_best: pd.DataFrame,
    model_summary: pd.DataFrame,
    task_summary: pd.DataFrame,
    out_dir: Path,
) -> bool:
    try:
        import plotly.graph_objects as go
    except Exception:
        return False

    model_type = model_summary.set_index("model")["type"].to_dict()
    task_level = task_summary.set_index("task")["level"].to_dict()
    frame = task_model_best.copy()
    frame["type"] = frame["model"].map(model_type).fillna("Unknown")
    frame["level"] = frame["task"].map(task_level).fillna(frame["level"])
    frame = frame[frame["score_norm"].notna() & (frame["score_norm"] > 0)]

    node_labels: list[str] = []
    node_x: list[float] = []
    node_y: list[float] = []
    node_colors: list[str] = []

    def add_node(label: str, x: float, y: float, color: str) -> int:
        node_labels.append(label)
        node_x.append(x)
        node_y.append(y)
        node_colors.append(color)
        return len(node_labels) - 1

    type_order = ["RNA", "DNA", "Unknown"]
    model_order = _model_order(model_summary, frame)
    task_order = _task_order(task_summary)
    type_nodes = {
        t: add_node(f"type:{t}", 0.02, 0.08 + i * 0.18, "#6B7280")
        for i, t in enumerate(type_order)
        if t in set(frame["type"])
    }
    model_nodes = {
        m: add_node(m, 0.34, (i + 0.5) / max(len(model_order), 1), "#4E79A7" if model_type.get(m) == "RNA" else "#F28E2B")
        for i, m in enumerate(model_order)
        if m in set(frame["model"])
    }
    level_nodes = {
        l: add_node(l, 0.66, 0.18 + i * 0.28, LEVEL_COLORS.get(l, "#999999"))
        for i, l in enumerate(LEVEL_ORDER)
        if l in set(frame["level"])
    }
    task_nodes = {
        t: add_node(t, 0.98, (i + 0.5) / max(len(task_order), 1), LEVEL_COLORS.get(task_level.get(t, ""), "#999999"))
        for i, t in enumerate(task_order)
        if t in set(frame["task"])
    }

    src: list[int] = []
    dst: list[int] = []
    val: list[float] = []
    color: list[str] = []

    def add_links(group_cols: list[str], source_key: str, target_key: str, source_nodes: dict, target_nodes: dict) -> None:
        grouped = frame.groupby(group_cols, as_index=False)["score_norm"].sum()
        for _, row in grouped.iterrows():
            s_key = row[source_key]
            t_key = row[target_key]
            if s_key not in source_nodes or t_key not in target_nodes:
                continue
            src.append(source_nodes[s_key])
            dst.append(target_nodes[t_key])
            val.append(float(row["score_norm"]))
            color.append("rgba(90, 90, 90, 0.22)")

    add_links(["type", "model"], "type", "model", type_nodes, model_nodes)
    add_links(["model", "level"], "model", "level", model_nodes, level_nodes)
    add_links(["level", "task"], "level", "task", level_nodes, task_nodes)

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="fixed",
                node=dict(
                    label=node_labels,
                    x=node_x,
                    y=node_y,
                    pad=14,
                    thickness=15,
                    color=node_colors,
                    line=dict(color="rgba(0,0,0,0.25)", width=0.5),
                ),
                link=dict(source=src, target=dst, value=val, color=color),
            )
        ]
    )
    fig.update_layout(
        title="Head-screening strength flow: model type -> model -> level -> task",
        font=dict(size=11),
        width=1400,
        height=900,
    )
    fig.write_html(out_dir / "screening_strength_sankey.html", include_plotlyjs="cdn")
    return True


def write_report(out_dir: Path, sankey_ok: bool) -> None:
    outputs = [
        ("task_model_best_score_heatmap.png", "Best normalized head score for every task-model pair. Gray cells mean no usable local attention score."),
        ("task_model_best_score_heatmap_scored_only.png", "Same heatmap after removing unsupported/no-attention models."),
        ("model_level_strength_heatmap.png", "Mean normalized model strength across L6/L8/L12 task groups."),
        ("task_best_head_bar.png", "Best single head for each supervised task."),
        ("task_top3_heads_bar.png", "Top three heads for each supervised task, with fixed normalized score bounds."),
        ("level_top_heads_bar.png", "Top candidate teacher heads per alignment level."),
        ("unsupported_model_reasons.png", "Why some staged models did not become attention teachers locally."),
        ("screening_strength_sankey.html", "Interactive flow view; link width is cumulative normalized screening strength."),
    ]
    lines = ["# Screening Summary Plot Guide", ""]
    lines.append("Recommended read order:")
    lines.append("1. Coverage: `task_model_best_score_heatmap.png`")
    lines.append("2. Model choice by level: `model_level_strength_heatmap.png`")
    lines.append("3. Teacher-head candidates: `level_top_heads_bar.png`")
    lines.append("4. Per-task winners: `task_best_head_bar.png`; alternatives: `task_top3_heads_bar.png`")
    lines.append("5. Interactive overview: `screening_strength_sankey.html`")
    lines.append("")
    lines.append("## Files")
    for name, desc in outputs:
        if name.endswith(".html") and not sankey_ok:
            continue
        lines.append(f"- `{name}`: {desc}")
    lines.append("")
    lines.append("All score plots use the existing `score_norm` scale with fixed bounds 0 to 1.")
    lines.append("")
    lines.append("Interpretation note: L6/L8/L12 are task-defined alignment banks, not proof that those exact student layers are optimal. L6 summarizes sequence-level tasks, L8 summarizes token-regression tasks, and L12 summarizes pair/contact tasks.")
    (out_dir / "plot_guide.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--screening-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--top-n", type=int, default=25)
    parser.add_argument("--sample-label", default="local n=100")
    args = parser.parse_args()

    root = args.root.resolve()
    screening_dir = args.screening_dir or root / "outputs" / "head_screening"
    screening_dir = screening_dir.resolve() if screening_dir.is_absolute() else (root / screening_dir).resolve()
    out_dir = args.output_dir or screening_dir / "summary_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    heatmap_dir = screening_dir / "heatmaps"
    scores = _read_csv(heatmap_dir / "task_head_scores.csv")
    level_top = _read_csv(heatmap_dir / "level_top_heads.csv")
    model_summary_path = screening_dir / "all_models_task_completion_summary.csv"
    if not model_summary_path.exists():
        model_summary_path = root / "outputs" / "head_screening" / "all_models_task_completion_summary.csv"
    model_summary = _read_csv(model_summary_path)
    model_summary = apply_recheck_status(model_summary, root)
    task_summary_path = screening_dir / "all_tasks_completion_summary.csv"
    if not task_summary_path.exists():
        task_summary_path = root / "outputs" / "head_screening" / "all_tasks_completion_summary.csv"
    task_summary = _read_csv(task_summary_path)
    unsupported_path = screening_dir / "unsupported_models_local_100.csv"
    if not unsupported_path.exists():
        unsupported_path = root / "outputs" / "head_screening" / "unsupported_models_local_100.csv"
    unsupported = _read_csv(unsupported_path) if unsupported_path.exists() else pd.DataFrame()
    if not unsupported.empty and "display_status" in model_summary:
        unsupported = unsupported.drop(columns=[c for c in ["display_status"] if c in unsupported.columns]).merge(
            model_summary[["model", "display_status"]],
            on="model",
            how="left",
        )

    scores = scores[pd.to_numeric(scores["score_norm"], errors="coerce").notna()].copy()
    scores["score_norm"] = pd.to_numeric(scores["score_norm"], errors="coerce")
    task_model_best = plot_task_model_heatmap(scores, model_summary, task_summary, out_dir, args.sample_label)
    plot_model_level_heatmap(task_model_best, model_summary, out_dir)
    plot_task_top_bars(scores, task_summary, out_dir)
    plot_level_top_heads(level_top, out_dir, top_n=args.top_n)
    plot_unsupported_reasons(unsupported, out_dir)
    sankey_ok = make_sankey(task_model_best, model_summary, task_summary, out_dir)
    write_report(out_dir, sankey_ok)

    manifest = {
        "output_dir": str(out_dir),
        "task_model_rows": int(len(task_model_best)),
        "models": int(model_summary.shape[0]),
        "tasks": int(task_summary.shape[0]),
        "sankey": bool(sankey_ok),
    }
    (out_dir / "plot_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
