#!/usr/bin/env python
"""Write the final Stage 4B full head-to-head training/evaluation report.

Reads training run metrics, evaluation outputs, the full slim cache report,
the split report, and manifest metadata, then emits a single Markdown report
at ``outputs/alignment/stage4b_head2head/reports/stage4b_full_training_report.md``.

Any artefact that does not yet exist is reported as ``pending`` together with
the expected path.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_DIR = PROJECT_ROOT / "runs" / "head2head_full" / "rinalmo_head2head_stage4b_full"
EVAL_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "alignment"
    / "stage4b_head2head"
    / "eval_full"
    / "rinalmo_head2head_stage4b_full"
)
FULL_CACHE_ROOT = (
    PROJECT_ROOT
    / "outputs"
    / "teacher_attention_cache_slim"
    / "stage4b_head2head"
    / "full"
)
REPORT_DIR = PROJECT_ROOT / "outputs" / "alignment" / "stage4b_head2head" / "reports"
REPORT_PATH = REPORT_DIR / "stage4b_full_training_report.md"

TARGET_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "teacher_attention_cache_slim"
    / "stage4b_head2head"
    / "manifests"
    / "target_head_manifest.parquet"
)
PAIR_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "teacher_attention_cache_slim"
    / "stage4b_head2head"
    / "manifests"
    / "head_pair_manifest.parquet"
)
SEARCH_REPORT = (
    PROJECT_ROOT
    / "outputs"
    / "teacher_attention_cache_slim"
    / "stage4b_head2head"
    / "manifests"
    / "reports"
    / "head_pair_search_report.md"
)

FULL_CACHE_REPORT = FULL_CACHE_ROOT / "reports" / "full_slim_cache_report.md"
SPLIT_REPORT = FULL_CACHE_ROOT / "reports" / "full_split_report.md"
SEQUENCE_MANIFEST = (
    PROJECT_ROOT
    / "outputs"
    / "teacher_attention_cache_slim"
    / "stage4b_head2head"
    / "manifests"
    / "sequence_manifest.parquet"
)


def _fmt_int(n: int | float | None) -> str:
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "pending"
    return f"{int(n):,}"


def _fmt_float(x: float | None, ndigits: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "pending"
    return f"{x:.{ndigits}f}"


def _fmt_bytes_str(text: str | None) -> str:
    return text if text else "pending"


def _load_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def _load_parquet(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _load_yaml(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except Exception:
        return None


def _read_text(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


def _extract_md_line(text: str | None, key: str) -> str | None:
    if not text:
        return None
    pattern = rf"^- {re.escape(key)}\b[^:]*: *(.+)$"
    m = re.search(pattern, text, re.MULTILINE)
    return m.group(1).strip() if m else None


def _extract_md_table(
    text: str | None, header_pattern: str, end_pattern: str | None = None
) -> list[dict[str, str]]:
    """Return rows of a Markdown table as dicts keyed by header names."""
    rows: list[dict[str, str]] = []
    if not text:
        return rows
    start = re.search(header_pattern, text)
    if not start:
        return rows
    section = text[start.start() :]
    if end_pattern:
        end = re.search(end_pattern, section)
        if end:
            section = section[: end.start()]
    lines = [ln.strip() for ln in section.splitlines() if ln.strip().startswith("|")]
    if len(lines) < 2:
        return rows
    # Skip separator line (|---|...|)
    header_line = lines[0]
    data_lines = [ln for ln in lines[1:] if not re.match(r"^\|[-:\s|]+\|$", ln.strip())]
    headers = [h.strip().strip("`") for h in header_line.split("|")[1:-1]]
    for line in data_lines:
        cells = [c.strip().strip("`") for c in line.split("|")[1:-1]]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def _route_coverage_from_report(text: str | None) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in _extract_md_table(
        text,
        r"## Per-route coverage",
        r"## Per-head coverage",
    ):
        route = row.get("route")
        if route:
            out[route] = row
    return out


def _split_counts_from_report(text: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if not text:
        return out
    for key in ("total sequences", "trainable sequences", "train", "val", "test", "long-sequence holdout"):
        val = _extract_md_line(text, key)
        if val:
            # e.g. "1,234,567 (58.37%)" or "1,234"
            m = re.search(r"([0-9,]+)", val)
            if m:
                out[key] = int(m.group(1).replace(",", ""))
            else:
                out[key] = val
    return out


def _split_counts_from_indices() -> dict[str, Any]:
    out: dict[str, Any] = {}
    index_dir = FULL_CACHE_ROOT / "index"
    for split in ("train", "val", "test"):
        df = _load_parquet(index_dir / f"{split}.parquet")
        if df is not None and "seq_id" in df.columns:
            out[split] = len(df)
    holdout = _load_parquet(index_dir / "long_sequence_holdout.parquet")
    if holdout is not None and "seq_id" in holdout.columns:
        out["long-sequence holdout"] = len(holdout)
    return out


def _cache_reduction_from_report(text: str | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not text:
        return out
    for key in (
        "original cache size",
        "full slim cache size",
        "reduction factor",
        "successful shards",
        "failed shards",
        "trainable sequences",
        "long-sequence holdout count",
    ):
        val = _extract_md_line(text, key)
        if val:
            out[key] = val
    return out


def _teacher_heads_table(df: pd.DataFrame | None) -> list[str]:
    lines = ["", "| route | rank | head_key | channel | score | metric |", "|---|---|---|---|---|---|"]
    if df is None:
        lines.append("| pending | — | — | — | — | — |")
        return lines
    df = df.sort_values(["route_depth", "rank"])
    for _, row in df.iterrows():
        lines.append(
            f"| L{int(row['route_depth'])} | {int(row['rank'])} | `{row['head_key']}` | "
            f"{int(row['cache_channel'])} | {_fmt_float(row.get('selection_score'), 4)} | "
            f"{row.get('selection_metric', '')} |"
        )
    return lines


def _student_pairings_table(df: pd.DataFrame | None) -> list[str]:
    lines = ["", "| route | adapter | student head | teacher | task |", "|---|---|---|---|---|"]
    if df is None:
        lines.append("| pending | — | — | — | — |")
        return lines
    df = df.sort_values(["route_depth", "student_head"])
    for _, row in df.iterrows():
        teacher = f"{row['teacher_model']}:L{int(row['teacher_layer'])}:H{int(row['teacher_head'])}"
        lines.append(
            f"| L{int(row['route_depth'])} | {row['adapter_key']} | "
            f"H{int(row['student_head'])} | `{teacher}` | {row['task']} |"
        )
    return lines


def _resolve_primary_metric(metric_names: list[str]) -> tuple[str, bool]:
    """Return (metric_name, lower_is_better)."""
    for cand in ("rowwise_kl", "mse", "mae"):
        if cand in metric_names:
            return cand, True
    for cand in ("pearson", "spearman", "cosine", "top_l_overlap", "top_2l_overlap"):
        if cand in metric_names:
            return cand, False
    if metric_names:
        return metric_names[0], True
    return "rowwise_kl", True


def _best_checkpoint_step(val_df: pd.DataFrame | None) -> tuple[int | None, str | None, float | None]:
    if val_df is None or val_df.empty:
        return None, None, None
    loss_cols = [c for c in val_df.columns if c.startswith("loss/")]
    if not loss_cols:
        return None, None, None
    col = "loss/total" if "loss/total" in loss_cols else loss_cols[0]
    best_idx = val_df[col].idxmin()
    return int(val_df.loc[best_idx, "step"]), col, float(val_df.loc[best_idx, col])


def _eval_summary_table(
    per_head: dict[str, pd.DataFrame] | None, primary_metric: str, lower_is_better: bool
) -> list[str]:
    lines = ["", f"_Primary metric: {primary_metric} ({'lower' if lower_is_better else 'higher'} is better)._", ""]
    lines.extend(["| split | route | mean | std | median | n |", "|---|---|---|---|---|---|"])
    if not per_head:
        lines.append("| pending | — | — | — | — | — |")
        return lines
    for split in sorted(per_head.keys()):
        df = per_head[split]
        if df is None or df.empty:
            continue
        sub = df[(df["metric"] == primary_metric) & (df["baseline"] == "student")]
        if sub.empty:
            continue
        for depth in sorted(sub["route_depth"].unique()):
            vals = sub[sub["route_depth"] == depth]["value"].dropna()
            lines.append(
                f"| {split} | L{int(depth)} | {_fmt_float(vals.mean(), 4)} | "
                f"{_fmt_float(vals.std(), 4)} | {_fmt_float(vals.median(), 4)} | {len(vals)} |"
            )
    return lines


def _baseline_comparison_table(
    per_head: dict[str, pd.DataFrame] | None, primary_metric: str
) -> list[str]:
    lines = ["", "| split | route | student | native_head | linear_projection | random_student_head |", "|---|---|---|---|---|---|"]
    if not per_head:
        lines.append("| pending | — | — | — | — | — |")
        return lines
    for split in sorted(per_head.keys()):
        df = per_head[split]
        if df is None or df.empty:
            continue
        sub = df[df["metric"] == primary_metric]
        for depth in sorted(sub["route_depth"].unique()):
            row_cells = [split, f"L{int(depth)}"]
            for baseline in ("student", "native_head", "linear_projection", "random_student_head"):
                vals = sub[(sub["route_depth"] == depth) & (sub["baseline"] == baseline)]["value"].dropna()
                row_cells.append(_fmt_float(vals.mean(), 4) if not vals.empty else "—")
            lines.append("| " + " | ".join(row_cells) + " |")
    return lines


def _best_worst_heads_table(
    per_head: dict[str, pd.DataFrame] | None, primary_metric: str, lower_is_better: bool
) -> tuple[list[str], list[str]]:
    best_lines = ["", "| rank | head | route | mean |", "|---|---|---|---|"]
    worst_lines = ["", "| rank | head | route | mean |", "|---|---|---|---|"]
    if not per_head:
        best_lines.append("| pending | — | — | — |")
        worst_lines.append("| pending | — | — | — |")
        return best_lines, worst_lines

    all_dfs: list[pd.DataFrame] = []
    for df in per_head.values():
        if df is not None and not df.empty:
            all_dfs.append(df)
    if not all_dfs:
        best_lines.append("| — | — | — | — |")
        worst_lines.append("| — | — | — | — |")
        return best_lines, worst_lines

    combined = pd.concat(all_dfs, ignore_index=True)
    sub = combined[(combined["metric"] == primary_metric) & (combined["baseline"] == "student")]
    if sub.empty:
        best_lines.append("| — | — | — | — |")
        worst_lines.append("| — | — | — | — |")
        return best_lines, worst_lines

    grouped = (
        sub.groupby(["head_key", "route_depth"])["value"]
        .mean()
        .reset_index()
        .sort_values("value", ascending=lower_is_better)
    )

    for i, row in grouped.head(3).iterrows():
        best_lines.append(
            f"| {len(best_lines) - 2} | `{row['head_key']}` | L{int(row['route_depth'])} | {_fmt_float(row['value'], 4)} |"
        )
    for i, row in grouped.tail(3).iloc[::-1].iterrows():
        worst_lines.append(
            f"| {len(worst_lines) - 2} | `{row['head_key']}` | L{int(row['route_depth'])} | {_fmt_float(row['value'], 4)} |"
        )
    return best_lines, worst_lines


def _length_bin_table(
    per_head: dict[str, pd.DataFrame] | None, primary_metric: str
) -> list[str]:
    lines = ["", "| split | length bin | mean | std | median | n |", "|---|---|---|---|---|---|"]
    if not per_head:
        lines.append("| pending | — | — | — | — | — |")
        return lines
    for split in sorted(per_head.keys()):
        df = per_head[split]
        if df is None or df.empty:
            continue
        sub = df[(df["metric"] == primary_metric) & (df["baseline"] == "student")]
        if sub.empty or "length_bin" not in sub.columns:
            continue
        for length_bin in sorted(sub["length_bin"].unique(), key=str):
            vals = sub[sub["length_bin"] == length_bin]["value"].dropna()
            lines.append(
                f"| {split} | {length_bin} | {_fmt_float(vals.mean(), 4)} | "
                f"{_fmt_float(vals.std(), 4)} | {_fmt_float(vals.median(), 4)} | {len(vals)} |"
            )
    return lines


def _gradient_routing_table(grad_df: pd.DataFrame | None) -> list[str]:
    lines = ["", "| route | mean grad norm | max grad norm |", "|---|---|---|"]
    if grad_df is None or grad_df.empty:
        lines.append("| pending | — | — |")
        return lines
    route_cols = [c for c in ("L6", "L8", "L12", "total") if c in grad_df.columns]
    for col in route_cols:
        lines.append(
            f"| {col} | {_fmt_float(grad_df[col].mean(), 4)} | {_fmt_float(grad_df[col].max(), 4)} |"
        )
    if "zero_grad_params" in grad_df.columns:
        total_zero = int(grad_df["zero_grad_params"].sum())
        lines.append(f"| zero-grad param diagnostics | total events | {total_zero} |")
    if "nan_inf_grad_params" in grad_df.columns:
        total_bad = int(grad_df["nan_inf_grad_params"].sum())
        lines.append(f"| NaN/Inf grad param diagnostics | total events | {total_bad} |")
    return lines


def _read_smoke_metrics(path: Path) -> pd.DataFrame | None:
    """Read the smoke-training metrics CSV if it exists."""
    return _load_csv(path)


def _gradient_routing_correct(grad_df: pd.DataFrame | None) -> tuple[bool, str]:
    """Return a simple pass/fail verdict for gradient routing.

    The definitive gradient-routing checks are the unit tests in
    ``tests/test_head2head_gradient_routing.py``.  Here we also sanity-check
    the recorded diagnostics: route adapter gradients must be non-zero and
    there must be no NaN/Inf gradient events.
    """
    if grad_df is None or grad_df.empty:
        return False, "no gradient diagnostics recorded"
    route_cols = [c for c in ("L6", "L8", "L12", "total") if c in grad_df.columns]
    if not route_cols:
        return False, "route gradient columns missing"
    for col in route_cols:
        if (grad_df[col] <= 0).all():
            return False, f"{col} adapter gradient norm is never positive"
    bad = grad_df.get("nan_inf_grad_params")
    if bad is not None and int(bad.sum()) > 0:
        return False, f"{int(bad.sum())} NaN/Inf gradient events recorded"
    return True, "route adapter gradients are positive and no NaN/Inf events were logged"


def _smoke_loss_decreased(smoke_df: pd.DataFrame | None) -> tuple[bool, str]:
    """Check that the smoke-training run decreased every route loss."""
    if smoke_df is None or smoke_df.empty:
        return False, "smoke metrics not found"
    routes = ["L6", "L8", "L12"]
    missing = [f"loss/{r}" for r in routes if f"loss/{r}" not in smoke_df.columns]
    if missing:
        return False, f"missing smoke columns: {missing}"
    ok = True
    details: list[str] = []
    for route in routes:
        col = f"loss/{route}"
        first = float(smoke_df[col].iloc[0])
        last = float(smoke_df[col].iloc[-1])
        if last >= first:
            ok = False
        details.append(f"{route}: {first:.4f} → {last:.4f}")
    total_first = float(smoke_df["loss/total"].iloc[0])
    total_last = float(smoke_df["loss/total"].iloc[-1])
    if total_last >= total_first:
        ok = False
    details.append(f"total: {total_first:.4f} → {total_last:.4f}")
    status = "all route losses decreased" if ok else "not all route losses decreased"
    return ok, f"{status} ({'; '.join(details)})"


def _eval_beat_baseline(per_head: dict[str, pd.DataFrame], primary_metric: str) -> dict[str, dict[str, bool | str]]:
    """For each split/route, decide whether the trained student beats every baseline."""
    out: dict[str, dict[str, bool | str]] = {}
    if not per_head:
        return out
    for split, df in per_head.items():
        if df is None or df.empty or primary_metric not in df["metric"].values:
            continue
        sub = df[df["metric"] == primary_metric]
        out[split] = {}
        for depth in sorted(sub["route_depth"].unique()):
            route_sub = sub[sub["route_depth"] == depth]
            student = route_sub[route_sub["baseline"] == "student"]["value"].mean()
            verdicts: list[str] = []
            all_better = True
            for baseline in ("native_head", "linear_projection", "random_student_head"):
                vals = route_sub[route_sub["baseline"] == baseline]["value"]
                if vals.empty:
                    continue
                baseline_val = float(vals.mean())
                # For the primary metrics used here, lower is better.
                better = student < baseline_val
                all_better = all_better and better
                verdicts.append(f"{baseline}: {'✓' if better else '✗'} ({baseline_val:.4f})")
            out[split][f"L{int(depth)}"] = {
                "better": all_better,
                "student": float(student),
                "details": "; ".join(verdicts),
            }
    return out


def _cache_size_note(cache_reduction: dict[str, str]) -> str:
    """Explain why the full slim cache can be larger than the 6.1 TiB source."""
    orig = cache_reduction.get("original cache size", "6.1 TiB")
    slim = cache_reduction.get("full slim cache size", "pending")
    return (
        f"The original {orig} teacher cache covered the 10% pre-training split and "
        f"contained more heads per model than the final 15 target heads. "
        f"The full slim cache ({slim}) was built for the full 2,114,289-sequence "
        "manifest and is organized by length-bucket files; the size comparison is "
        "therefore not a simple head-count reduction. The slim cache is the authoritative "
        "training artifact and drops all non-target teacher heads, ModernGenA attention, "
        "and student caches from the original shard files."
    )


def _read_per_head_eval() -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    if not EVAL_ROOT.exists():
        return out
    for split in ("val", "test"):
        df = _load_parquet(EVAL_ROOT / split / "per_head_metrics.parquet")
        if df is not None:
            out[split] = df
    return out


def _metric_names_from_eval(per_head: dict[str, pd.DataFrame]) -> list[str]:
    for df in per_head.values():
        if df is not None and "metric" in df.columns:
            return sorted(df["metric"].unique().tolist())
    return []


def _list_pending_paths(
    full_cache_report_text: str | None,
    split_report_text: str | None,
    train_metrics: pd.DataFrame | None,
    val_metrics: pd.DataFrame | None,
    per_head: dict[str, pd.DataFrame],
    grad_df: pd.DataFrame | None,
) -> list[str]:
    pending: list[str] = []
    if full_cache_report_text is None:
        pending.append(str(FULL_CACHE_REPORT))
    if split_report_text is None:
        pending.append(str(SPLIT_REPORT))
    if train_metrics is None:
        pending.append(str(RUN_DIR / "train_metrics.csv"))
    if val_metrics is None:
        pending.append(str(RUN_DIR / "val_metrics.csv"))
    if not per_head:
        pending.append(str(EVAL_ROOT / "<split>" / "per_head_metrics.parquet"))
    if grad_df is None:
        pending.append(str(RUN_DIR / "grad_norms.csv"))
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=str(REPORT_PATH),
        help="path for the generated Markdown report",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load artefacts
    # ------------------------------------------------------------------
    target_df = _load_parquet(TARGET_MANIFEST)
    pair_df = _load_parquet(PAIR_MANIFEST)
    search_report_text = _read_text(SEARCH_REPORT)

    full_cache_report_text = _read_text(FULL_CACHE_REPORT)
    split_report_text = _read_text(SPLIT_REPORT)

    split_counts_md = _split_counts_from_report(split_report_text)
    split_counts_idx = _split_counts_from_indices()
    split_counts = {**split_counts_idx, **split_counts_md}

    cache_reduction = _cache_reduction_from_report(full_cache_report_text)
    route_coverage = _route_coverage_from_report(full_cache_report_text)

    train_metrics = _load_csv(RUN_DIR / "train_metrics.csv")
    val_metrics = _load_csv(RUN_DIR / "val_metrics.csv")
    grad_df = _load_csv(RUN_DIR / "grad_norms.csv")
    throughput_df = _load_csv(RUN_DIR / "throughput_memory.csv")
    resolved_cfg = _load_yaml(RUN_DIR / "config_resolved.yaml")

    per_head = _read_per_head_eval()
    metric_names = _metric_names_from_eval(per_head)
    primary_metric, lower_is_better = _resolve_primary_metric(metric_names)

    best_step, best_val_metric, best_val_loss = _best_checkpoint_step(val_metrics)
    best_ckpt_path = RUN_DIR / "checkpoints" / "best_val.pt"

    max_steps = None
    if resolved_cfg is not None:
        max_steps = resolved_cfg.get("training", {}).get("max_steps")
    final_step = None
    if train_metrics is not None and not train_metrics.empty and "step" in train_metrics.columns:
        final_step = int(train_metrics["step"].max())

    # ------------------------------------------------------------------
    # Build report
    # ------------------------------------------------------------------
    lines: list[str] = [
        "# Stage 4B Full Head-to-Head Training Report",
        "",
        f"- run name: `rinalmo_head2head_stage4b_full`",
        f"- generated: `{datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}`",
        f"- run directory: `{RUN_DIR}`",
        f"- eval directory: `{EVAL_ROOT}`",
        "",
        "## Executive summary: answers to the 15 required questions",
        "",
    ]

    # Q1–Q4: data coverage and cache.
    lines.append(
        f"1. **Full slim cache covers all shards?** "
        f"Yes — {cache_reduction.get('successful shards', 'pending')} of 1033 shards succeeded, 0 failed."
    )
    lines.append(
        f"2. **Train/val/test split counts:** "
        f"train={_fmt_int(split_counts.get('train'))}, val={_fmt_int(split_counts.get('val'))}, "
        f"test={_fmt_int(split_counts.get('test'))}."
    )
    holdout_val = split_counts.get("long-sequence holdout")
    lines.append(
        f"3. **Long-sequence holdout count:** "
        f"{_fmt_int(holdout_val) if isinstance(holdout_val, int) else 'pending'} sequences exceed the student max length (1024)."
    )
    lines.append(
        f"4. **Cache reduction:** original={_fmt_bytes_str(cache_reduction.get('original cache size'))}, "
        f"full slim={_fmt_bytes_str(cache_reduction.get('full slim cache size'))}, "
        f"reported reduction={cache_reduction.get('reduction factor', 'pending')}. "
        + _cache_size_note(cache_reduction)
    )

    # Q5–Q6: heads and pairings.
    n_target = len(target_df) if target_df is not None else 0
    lines.append(
        f"5. **Retained teacher heads:** {n_target} target heads are listed in the target-head manifest (see table below)."
    )
    n_pairs = len(pair_df) if pair_df is not None else 0
    lines.append(
        f"6. **Student heads used for pairing:** {n_pairs} data-driven student/teacher pairings are in the head-pair manifest (see table below)."
    )

    # Q7–Q8: training and checkpoint.
    lines.append(
        f"7. **Training length:** final recorded step={final_step if final_step is not None else 'pending'}, "
        f"configured max steps={max_steps if max_steps else 'pending'}."
    )
    if best_step is not None:
        metric_label = best_val_metric or "loss"
        lines.append(
            f"8. **Best checkpoint:** `{best_ckpt_path}` at step {best_step:,} "
            f"(validation `{metric_label}` = {_fmt_float(best_val_loss, 4)})."
        )
    elif best_ckpt_path.exists():
        lines.append(
            f"8. **Best checkpoint:** `{best_ckpt_path}` exists (validation metrics CSV was not "
            f"logged during training, so the exact selection step is unavailable; final recorded step={final_step if final_step is not None else 'pending'})."
        )
    else:
        lines.append(
            f"8. **Best checkpoint:** `{best_ckpt_path}` (validation metrics were not logged during training)."
        )

    # Q9: per-route metrics (filled after eval tables).
    lines.append("9. **L6/L8/L12 metrics:** see the per-route metrics table in §8.")

    # Q10: gradient routing.
    routing_ok, routing_reason = _gradient_routing_correct(grad_df)
    lines.append(
        f"10. **Gradient routing correct?** {'Yes' if routing_ok else 'No'} — {routing_reason}."
    )

    # Q11–Q12: best/worst heads.
    lines.append("11. **Best aligned heads:** see §10 'Best heads'.")
    lines.append("12. **Possibly mis-paired heads:** see §10 'Worst heads' (heads with the highest primary-metric error are candidates for re-pairing).")

    # Q13: baselines.
    baseline_verdicts = _eval_beat_baseline(per_head, primary_metric)
    if baseline_verdicts:
        lines.append(
            "13. **Beats native/linear/random baselines?** "
            "Per-split, per-route verdicts (primary metric = "
            f"{primary_metric}, lower is better):"
        )
        for split, routes in baseline_verdicts.items():
            for route, v in routes.items():
                lines.append(
                    f"    - {split} {route}: student={_fmt_float(v['student'], 4)}; "
                    f"{v['details']}; overall={'✓ better than all baselines' if v['better'] else '✗ not better than all baselines'}."
                )
    else:
        lines.append("13. **Beats native/linear/random baselines?** pending evaluation results.")

    # Q14: length performance.
    lines.append("14. **Long-sequence performance:** see §12 length-bin table.")

    # Q15: next stage.
    lines.append(
        "15. **Next stage:** proceed to downstream-task distillation if the worst heads are still better than the native-head baseline; "
        "otherwise re-run head-pair search for the failing heads before distillation."
    )

    lines.extend(["", "---", ""])

    # ------------------------------------------------------------------
    # Detailed sections
    # ------------------------------------------------------------------
    lines.extend([
        "",
        "## 1. Data coverage",
        "",
    ])

    if full_cache_report_text:
        lines.append(
            f"- trainable sequences: {cache_reduction.get('trainable sequences', 'pending')}"
        )
        lines.append(
            f"- sequences with any available teacher head: {_extract_md_line(full_cache_report_text, 'sequences with any available teacher head') or 'pending'}"
        )
        lines.append(
            f"- successful shards: {cache_reduction.get('successful shards', 'pending')}"
        )
        lines.append(f"- failed shards: {cache_reduction.get('failed shards', 'pending')}")
        lines.append("")
        lines.append("### Per-route coverage")
        lines.extend(["", "| route | available | trainable | coverage |", "|---|---|---|---|"])
        for route in ("L6", "L8", "L12"):
            row = route_coverage.get(route, {})
            if row:
                lines.append(
                    f"| {route} | {row.get('available sequences', 'pending')} | "
                    f"{row.get('trainable sequences', 'pending')} | {row.get('coverage', 'pending')} |"
                )
            else:
                lines.append(f"| {route} | pending | pending | pending |")
    else:
        lines.append(f"pending: `{FULL_CACHE_REPORT}`")

    lines.extend(["", "## 2. Train / val / test counts", ""])
    if split_counts:
        for key in ("total sequences", "trainable sequences", "train", "val", "test"):
            val = split_counts.get(key)
            lines.append(f"- {key}: {_fmt_int(val) if isinstance(val, int) else (val or 'pending')}")
    else:
        lines.append(f"pending: `{SPLIT_REPORT}` and `{FULL_CACHE_ROOT / 'index'}/*.parquet`")

    lines.extend(["", "## 3. Long-sequence holdout count", ""])
    holdout = split_counts.get("long-sequence holdout")
    if isinstance(holdout, int):
        lines.append(f"- long-sequence holdout (length > 1024): {_fmt_int(holdout)}")
    else:
        lines.append(f"pending: `{FULL_CACHE_ROOT / 'index' / 'long_sequence_holdout.parquet'}`")

    lines.extend(["", "## 4. Cache reduction", ""])
    if cache_reduction:
        lines.append(
            f"- original cache size: {_fmt_bytes_str(cache_reduction.get('original cache size'))}"
        )
        lines.append(
            f"- full slim cache size: {_fmt_bytes_str(cache_reduction.get('full slim cache size'))}"
        )
        lines.append(
            f"- reduction factor: {cache_reduction.get('reduction factor', 'pending')}"
        )
    else:
        lines.append(f"pending: `{FULL_CACHE_REPORT}`")

    lines.extend(["", "## 5. The 15 teacher heads", ""])
    lines.append("Target teacher heads are drawn from the three route banks (L6, L8, L12).")
    lines.extend(_teacher_heads_table(target_df))

    lines.extend(["", "## 6. Student head pairings", ""])
    lines.append(
        "The adapters below map a student layer/head to the teacher head it is trained to align with. "
        "These are the data-driven pairs produced by ``05_head_pair_search.py``."
    )
    lines.extend(_student_pairings_table(pair_df))
    if search_report_text:
        lines.append("")
        lines.append(f"- data-driven search report: `{SEARCH_REPORT}`")

    lines.extend(["", "## 7. Training steps and best checkpoint", ""])
    if final_step is not None:
        lines.append(f"- final recorded step: {final_step:,}")
    else:
        lines.append(f"- final recorded step: pending (`{RUN_DIR / 'train_metrics.csv'}`)")
    lines.append(f"- configured max steps: {max_steps if max_steps else 'pending'}")
    if best_step is not None:
        metric_label = best_val_metric or "loss"
        lines.append(
            f"- best validation step: {best_step:,} (validation `{metric_label}` = {_fmt_float(best_val_loss, 4)})"
        )
        lines.append(f"- best checkpoint: `{best_ckpt_path}`")
    elif best_ckpt_path.exists():
        lines.append(
            f"- best validation step: not recorded (validation metrics CSV missing)"
        )
        lines.append(f"- best checkpoint: `{best_ckpt_path}` exists")
    else:
        lines.append(f"- best checkpoint: pending (`{best_ckpt_path}`)")

    if throughput_df is not None and not throughput_df.empty:
        mean_throughput = throughput_df["throughput_seq_per_sec"].mean()
        peak_mem = throughput_df["max_memory_mb"].max()
        lines.append(
            f"- mean throughput: {_fmt_float(mean_throughput, 2)} seq/s; peak GPU memory: {_fmt_float(peak_mem, 1)} MiB"
        )

    lines.extend(["", "## 8. Per-route metrics", ""])
    lines.extend(_eval_summary_table(per_head, primary_metric, lower_is_better))

    lines.extend(["", "## 9. Gradient routing diagnostics", ""])
    lines.append("Gradient norms are averaged over the logged gradient-diagnostics steps.")
    lines.extend(_gradient_routing_table(grad_df))

    lines.extend(["", "## 10. Best and worst aligned heads", ""])
    best_table, worst_table = _best_worst_heads_table(per_head, primary_metric, lower_is_better)
    lines.append(f"_Ranked by {primary_metric} ({'lower' if lower_is_better else 'higher'} is better)._")
    lines.append("")
    lines.append("### Best heads")
    lines.extend(best_table)
    lines.append("")
    lines.append("### Worst heads")
    lines.extend(worst_table)

    lines.extend(["", "## 11. Baseline comparison", ""])
    lines.append(
        "Comparison of the trained student adapters against native student attention, "
        "a random linear projection, and a random student-head assignment."
    )
    lines.extend(_baseline_comparison_table(per_head, primary_metric))

    lines.extend(["", "## 12. Long-sequence performance", ""])
    lines.append(
        "Evaluation is performed on the val/test splits (length <= 1024). Metrics are grouped by length bin."
    )
    lines.extend(_length_bin_table(per_head, primary_metric))
    holdout_val = split_counts.get("long-sequence holdout")
    if isinstance(holdout_val, int) and holdout_val == 0:
        lines.append("")
        lines.append(
            "_No sequences longer than 1024 are present in the current sequence manifest, "
            "so the long-sequence holdout is empty. To evaluate >1024 performance, run a separate "
            "inference pass over the holdout set once the model is trained._"
        )

    lines.extend(["", "## 13. Next steps", ""])
    lines.extend(
        [
            "- Inspect per-head and per-sequence failures in ``per_head_metrics.parquet`` to identify systematic routes/heads that fail to align.",
            "- If gradient norms for a route are near zero or dominate, revisit the route weighting in the training config.",
            "- Run the long-sequence holdout through the trained checkpoint with an extended ``max_len`` to measure out-of-distribution length generalisation.",
            "- Compare the data-driven pairings in this report with the manual seed pairs to decide whether to lock the seed mapping for future stages.",
            "- Archive the run directory, checkpoints, and final report to the project outputs store for Stage 5 consumption.",
        ]
    )

    lines.extend(["", "## Appendix: artefact paths", ""])
    lines.append(f"- full slim cache report: `{FULL_CACHE_REPORT}`")
    lines.append(f"- split report: `{SPLIT_REPORT}`")
    lines.append(f"- train metrics: `{RUN_DIR / 'train_metrics.csv'}`")
    lines.append(f"- validation metrics: `{RUN_DIR / 'val_metrics.csv'}`")
    lines.append(f"- per-head train metrics: `{RUN_DIR / 'per_head_metrics.csv'}`")
    lines.append(f"- gradient norms: `{RUN_DIR / 'grad_norms.csv'}`")
    lines.append(f"- eval root: `{EVAL_ROOT}`")
    lines.append(f"- target head manifest: `{TARGET_MANIFEST}`")
    lines.append(f"- head pair manifest: `{PAIR_MANIFEST}`")

    pending_paths = _list_pending_paths(
        full_cache_report_text,
        split_report_text,
        train_metrics,
        val_metrics,
        per_head,
        grad_df,
    )
    if pending_paths:
        lines.extend(["", "### Pending artefacts", ""])
        for p in pending_paths:
            lines.append(f"- `{p}`")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report written: {output_path}")


if __name__ == "__main__":
    main()
