from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml


L8_TASKS = {
    "chemical_reactivity_token",
    "chemical_reactivity_ribonanza_2a3",
    "chemical_reactivity_ribonanza_dms",
    "chemical_reactivity_rnascope",
    "openvaccine_reactivity_token",
    "openvaccine_deg_mg_ph10_token",
    "openvaccine_deg_mg_50c_token",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _collect_timed_reports(root: Path) -> pd.DataFrame:
    rows = []
    for report in sorted((root / "outputs" / "head_screening").glob("screening_run_report*.csv")):
        df = pd.read_csv(report)
        df["report"] = report.name
        df["report_mtime"] = report.stat().st_mtime
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, ignore_index=True)
    frame["seconds"] = pd.to_numeric(frame.get("seconds", 0), errors="coerce").fillna(0.0)
    frame = frame[(frame.get("status", "") == "ok") & (frame["seconds"] > 0)].copy()
    return frame


def _current_scored_pairs(root: Path) -> pd.DataFrame:
    atlas = pd.read_csv(root / "outputs" / "head_screening" / "all_head_atlas.csv")
    atlas["metric_primary_num"] = pd.to_numeric(atlas["metric_primary"], errors="coerce")
    scored = atlas[atlas["metric_primary_num"].notna()].copy()
    pairs = scored.groupby(["model", "model_type", "task"], as_index=False).agg(
        head_rows=("head", "size"),
        max_metric=("metric_primary_num", "max"),
        n_samples_100=("n_samples", "max"),
    )
    return pairs


def _latest_seconds_by_pair(timed: pd.DataFrame) -> pd.DataFrame:
    if timed.empty:
        return pd.DataFrame(columns=["model", "task", "seconds_100", "report"])
    timed = timed.sort_values(["model", "task", "report_mtime"])
    latest = timed.groupby(["model", "task"], as_index=False).tail(1)
    return latest[["model", "task", "seconds", "report"]].rename(columns={"seconds": "seconds_100"})


def _impute_seconds(plan: pd.DataFrame) -> pd.DataFrame:
    out = plan.copy()
    out["seconds_100_observed"] = out["seconds_100"].notna()
    med_model = out.groupby("model")["seconds_100"].transform("median")
    med_task = out.groupby("task")["seconds_100"].transform("median")
    global_med = float(out["seconds_100"].median()) if out["seconds_100"].notna().any() else 60.0
    out["seconds_100"] = out["seconds_100"].fillna(med_model).fillna(med_task).fillna(global_med)
    out["estimate_note"] = out["seconds_100_observed"].map({True: "observed_100", False: "imputed_median"})
    return out


def build_plan(root: Path, target_limit: int, budget_hours: float, safety: float) -> tuple[pd.DataFrame, dict]:
    tasks_cfg = _load_yaml(root / "configs" / "tasks.yaml").get("tasks", {})
    models_cfg = _load_yaml(root / "configs" / "models.yaml").get("models", {})
    scored_pairs = _current_scored_pairs(root)
    timed = _collect_timed_reports(root)
    seconds = _latest_seconds_by_pair(timed)
    plan = scored_pairs.merge(seconds, on=["model", "task"], how="left")
    plan = _impute_seconds(plan)
    plan["level"] = plan["task"].map({name: cfg.get("group") for name, cfg in tasks_cfg.items()})
    plan["task_type"] = plan["task"].map({name: cfg.get("task_type") for name, cfg in tasks_cfg.items()})
    plan["model_optional"] = plan["model"].map({name: bool(cfg.get("optional", False)) for name, cfg in models_cfg.items()})
    plan["model_large"] = plan["model"].map({name: bool(cfg.get("large", False)) for name, cfg in models_cfg.items()})
    plan["scale_factor"] = float(target_limit) / 100.0
    plan["estimated_seconds"] = plan["seconds_100"] * plan["scale_factor"] * float(safety)

    # Treat locally repaired but unscreened 2.5B as server-only rather than part
    # of the local 40h budget. It has no scored 100-row baseline.
    recheck_path = root / "outputs" / "model_checks" / "model_attention_recheck_unsupported_after_repairs.csv"
    server_only_models: set[str] = set()
    if recheck_path.exists():
        recheck = pd.read_csv(recheck_path)
        fixed = recheck[
            (recheck.get("load_ok", False).astype(bool))
            & (recheck.get("forward_ok", False).astype(bool))
            & (recheck.get("has_attentions", False).astype(bool))
        ]["model"].astype(str)
        all_scored = set(plan["model"].astype(str))
        server_only_models.update([m for m in fixed if m not in all_scored])

    # Conservative local priority:
    # 1. non-large RNA models
    # 2. non-large DNA controls
    # 3. large models that already passed local screening
    def priority(row: pd.Series) -> int:
        model_type = str(row["model_type"]).upper()
        if not bool(row["model_large"]) and model_type == "RNA":
            return 0
        if not bool(row["model_large"]) and model_type == "DNA":
            return 1
        return 2

    plan["priority"] = plan.apply(priority, axis=1)
    plan = plan.sort_values(["priority", "estimated_seconds", "model", "task"]).reset_index(drop=True)
    budget_seconds = budget_hours * 3600.0
    cumsum = plan["estimated_seconds"].cumsum()
    plan["recommended_local"] = cumsum <= budget_seconds
    plan["recommendation"] = plan["recommended_local"].map({True: "local_1000", False: "defer_or_reduce"})

    totals = {
        "target_limit": int(target_limit),
        "budget_hours": float(budget_hours),
        "safety_factor": float(safety),
        "scored_model_task_pairs": int(len(plan)),
        "models": int(plan["model"].nunique()),
        "tasks": int(plan["task"].nunique()),
        "estimated_all_seconds": float(plan["estimated_seconds"].sum()),
        "estimated_all_hours": float(plan["estimated_seconds"].sum() / 3600.0),
        "recommended_local_pairs": int(plan["recommended_local"].sum()),
        "recommended_local_hours": float(plan.loc[plan["recommended_local"], "estimated_seconds"].sum() / 3600.0),
        "deferred_pairs": int((~plan["recommended_local"]).sum()),
        "deferred_hours": float(plan.loc[~plan["recommended_local"], "estimated_seconds"].sum() / 3600.0),
        "server_only_attention_ok_models": sorted(server_only_models),
    }
    return plan, totals


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "None."
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        vals = []
        for c in cols:
            value = row[c]
            if isinstance(value, float):
                vals.append(f"{value:.3f}" if math.isfinite(value) else "")
            else:
                vals.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_outputs(plan: pd.DataFrame, totals: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    plan.to_csv(output_dir / "screening_runtime_plan.csv", index=False)
    (output_dir / "screening_runtime_plan.json").write_text(
        json.dumps({"summary": totals, "jobs": plan.to_dict("records")}, indent=2),
        encoding="utf-8",
    )
    by_model = (
        plan.groupby(["model", "model_type"], as_index=False)
        .agg(
            pairs=("task", "nunique"),
            estimated_hours=("estimated_seconds", lambda s: s.sum() / 3600.0),
            local_pairs=("recommended_local", "sum"),
        )
        .sort_values("estimated_hours", ascending=False)
    )
    by_model.to_csv(output_dir / "runtime_by_model.csv", index=False)
    by_task = (
        plan.groupby(["task", "level", "task_type"], as_index=False)
        .agg(
            models=("model", "nunique"),
            estimated_hours=("estimated_seconds", lambda s: s.sum() / 3600.0),
            local_pairs=("recommended_local", "sum"),
        )
        .sort_values(["level", "estimated_hours"], ascending=[True, False])
    )
    by_task.to_csv(output_dir / "runtime_by_task.csv", index=False)

    local_jobs = plan[plan["recommended_local"]].copy()
    local_jobs[["model", "task"]].to_csv(output_dir / "jobs_local_under_budget.csv", index=False)
    deferred = plan[~plan["recommended_local"]].copy()
    deferred[["model", "task", "estimated_seconds", "recommendation"]].to_csv(
        output_dir / "jobs_deferred_or_reduce.csv", index=False
    )

    target_limit = int(totals["target_limit"])
    run_root = Path(f"outputs/head_screening_limit{target_limit}")
    job_rows = []
    for job_id, (_, row) in enumerate(plan.iterrows()):
        enabled = bool(row["recommended_local"])
        job_rows.append(
            {
                "job_id": job_id,
                "model": row["model"],
                "task": row["task"],
                "enabled": enabled,
                "reason": "under_budget" if enabled else "deferred_to_keep_under_budget",
                "output_path": (run_root / str(row["task"]) / str(row["model"]) / "head_atlas.csv").as_posix(),
                "estimated_seconds": row["estimated_seconds"],
                "estimated_hours": row["estimated_seconds"] / 3600.0,
                "seconds_100": row["seconds_100"],
                "estimate_note": row["estimate_note"],
            }
        )
    job_table = pd.DataFrame(job_rows)
    job_table.to_csv(output_dir / f"screening_jobs_limit{target_limit}_under{int(totals['budget_hours'])}h.csv", index=False)
    job_table[job_table["enabled"]].to_csv(
        output_dir / f"screening_jobs_limit{target_limit}_local_only.csv", index=False
    )

    lines = ["# Screening Runtime Plan", ""]
    lines.append("## Summary")
    for key, value in totals.items():
        if isinstance(value, float):
            lines.append(f"- {key}: `{value:.3f}`")
        else:
            lines.append(f"- {key}: `{value}`")
    lines.append("")
    lines.append("Estimates use observed local 100-sample timings, scaled linearly to the target sample count and multiplied by the safety factor.")
    lines.append("")
    lines.append("## Runtime By Model")
    lines.append(_md_table(by_model.head(30)))
    lines.append("")
    lines.append("## Runtime By Task")
    lines.append(_md_table(by_task))
    lines.append("")
    lines.append("## Recommendation")
    if totals["estimated_all_hours"] <= totals["budget_hours"]:
        lines.append("Run all locally at the requested target limit.")
    else:
        lines.append("Full local run exceeds the budget. Use `jobs_local_under_budget.csv` for the local run, and defer `jobs_deferred_or_reduce.csv` to server or lower sample count.")
    lines.append("")
    lines.append("Large/server-only models should be run on Slurm, especially `nucleotide_transformer_2_5b`.")
    lines.append("")
    lines.append("## Runnable Job Tables")
    lines.append(f"- `screening_jobs_limit{target_limit}_under{int(totals['budget_hours'])}h.csv`: all planned pairs; deferred pairs disabled.")
    lines.append(f"- `screening_jobs_limit{target_limit}_local_only.csv`: only enabled local jobs.")
    lines.append(f"- outputs are written under `{run_root.as_posix()}`.")
    (output_dir / "screening_runtime_plan.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=project_root())
    parser.add_argument("--target-limit", type=int, default=1000)
    parser.add_argument("--budget-hours", type=float, default=40.0)
    parser.add_argument("--safety-factor", type=float, default=1.25)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    output_dir = args.output_dir or root / "outputs" / "head_screening" / "runtime_plans" / f"limit_{args.target_limit}"
    plan, totals = build_plan(root, args.target_limit, args.budget_hours, args.safety_factor)
    write_outputs(plan, totals, output_dir)
    print(json.dumps(totals | {"output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
