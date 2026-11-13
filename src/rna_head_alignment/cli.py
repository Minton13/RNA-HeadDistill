from __future__ import annotations

import argparse
import json
from pathlib import Path

from rna_head_alignment.alignment.train_alignment import train_alignment_debug
from rna_head_alignment.cache.build_teacher_attention_cache import (
    build_teacher_attention_cache,
    collect_teacher_cache_manifests,
    make_teacher_cache_jobs,
)
from rna_head_alignment.cache.validate_attention_cache import validate_attention_cache
from rna_head_alignment.data.rinalmo_corpus import sample_corpus
from rna_head_alignment.data.task_loaders import verify_data
from rna_head_alignment.io.manifests import write_json
from rna_head_alignment.io.paths import check_environment, migrate_assets, resolve_path, set_default_env_from_config, load_yaml
from rna_head_alignment.models.loaders import verify_models
from rna_head_alignment.screening.head_scan import make_screening_jobs, run_screening_job_table, screen_heads
from rna_head_alignment.screening.heatmaps import build_head_heatmaps
from rna_head_alignment.screening.select_teacher_heads import collect_head_screening, propose_teacher_heads


def _print_json(data) -> None:
    print(json.dumps(data, indent=2, ensure_ascii=False))


def cmd_check_env(args: argparse.Namespace) -> None:
    cfg = load_yaml(args.config) if args.config else {}
    set_default_env_from_config(cfg)
    report = check_environment(args.config)
    out = resolve_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, out)
    md = out.with_suffix(".md")
    lines = ["# Environment Check", ""]
    for k, v in report.items():
        lines.append(f"- {k}: `{v}`")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    _print_json(report)


def cmd_migrate_assets(args: argparse.Namespace) -> None:
    report = migrate_assets(args.old_root, args.report_dir)
    _print_json(report)


def cmd_verify_models(args: argparse.Namespace) -> None:
    rows = verify_models(
        args.models,
        args.output,
        device=args.device,
        max_len=args.max_len,
        include_large=args.include_large,
    )
    _print_json({"rows": len(rows), "output": str(resolve_path(args.output))})


def cmd_verify_data(args: argparse.Namespace) -> None:
    rows = verify_data(args.tasks, args.output, limit=args.limit)
    _print_json({"rows": len(rows), "output": str(resolve_path(args.output))})


def cmd_sample_rinalmo(args: argparse.Namespace) -> None:
    n = sample_corpus(
        args.corpus,
        args.output,
        n=args.n,
        fraction=args.fraction,
        min_len=args.min_len,
        max_len=args.max_len,
        deduplicate=args.deduplicate,
        sequence_column=args.sequence_column,
    )
    _print_json({"sampled": n, "output": str(resolve_path(args.output))})


def cmd_screen_heads(args: argparse.Namespace) -> None:
    rows = screen_heads(
        args.model,
        args.task,
        args.config,
        args.output,
        models_yaml=args.models,
        tasks_yaml=args.tasks,
        limit=args.limit,
        device=args.device,
    )
    _print_json({"rows": len(rows), "output": str(resolve_path(args.output))})


def cmd_make_screening_jobs(args: argparse.Namespace) -> None:
    df = make_screening_jobs(
        args.models,
        args.tasks,
        args.model_check,
        args.output,
        output_root=args.output_root,
        include_all_models=args.include_all_models,
    )
    _print_json({"rows": int(len(df)), "output": str(resolve_path(args.output))})


def cmd_run_screening_jobs(args: argparse.Namespace) -> None:
    df = run_screening_job_table(
        args.jobs,
        config_path=args.config,
        models_yaml=args.models,
        tasks_yaml=args.tasks,
        report_csv=args.report,
        limit=args.limit,
        device=args.device,
        max_jobs=args.max_jobs,
        resume=not args.no_resume,
        model_filter=args.model,
        task_filter=args.task,
        time_budget_hours=args.time_budget_hours,
    )
    _print_json({"rows": int(len(df)), "output": str(resolve_path(args.report))})


def cmd_collect(args: argparse.Namespace) -> None:
    df = collect_head_screening(args.input, args.output)
    _print_json({"rows": int(len(df)), "output": str(resolve_path(args.output))})


def cmd_plot_heatmaps(args: argparse.Namespace) -> None:
    report = build_head_heatmaps(
        args.atlas,
        tasks_yaml=args.tasks,
        output_dir=args.output_dir,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap,
        top_k=args.top_k,
    )
    _print_json(report)


def cmd_propose(args: argparse.Namespace) -> None:
    df = propose_teacher_heads(args.atlas, args.config, args.output, args.yaml)
    _print_json({"rows": int(len(df)), "output": str(resolve_path(args.output)), "yaml": str(resolve_path(args.yaml))})


def cmd_cache_teacher(args: argparse.Namespace) -> None:
    rows = build_teacher_attention_cache(
        args.teacher_bank,
        args.sequences,
        args.output,
        models_yaml=args.models,
        max_len=args.max_len,
        shard_size=args.shard_size,
        device=args.device,
        limit=args.limit,
        model_filter=args.model,
        shard_index=args.shard_index,
        shard_start=args.shard_start,
        shard_end=args.shard_end,
        resume=not args.no_resume,
        dtype=args.dtype,
        max_tokens_per_batch=args.max_tokens_per_batch,
        sub_batch_size=args.sub_batch_size,
    )
    _print_json({"rows": len(rows), "output": str(resolve_path(args.output))})


def cmd_make_cache_jobs(args: argparse.Namespace) -> None:
    rows = make_teacher_cache_jobs(
        args.teacher_bank,
        args.sequences,
        args.output,
        shard_size=args.shard_size,
        limit=args.limit,
    )
    _print_json({"rows": len(rows), "output": str(resolve_path(args.output))})


def cmd_collect_cache_manifests(args: argparse.Namespace) -> None:
    rows = collect_teacher_cache_manifests(args.cache, args.output)
    _print_json({"rows": len(rows), "output": str(resolve_path(args.output))})


def cmd_validate_cache(args: argparse.Namespace) -> None:
    report = validate_attention_cache(args.cache, args.output)
    _print_json(report)


def cmd_train_alignment(args: argparse.Namespace) -> None:
    report = train_alignment_debug(args.config, args.teacher_cache, args.output, steps=args.steps)
    _print_json(report)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rna_head_alignment")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check-env")
    s.add_argument("--config", default="configs/paths.yaml")
    s.add_argument("--output", default="outputs/model_checks/env_check.json")
    s.set_defaults(func=cmd_check_env)

    s = sub.add_parser("migrate-assets")
    s.add_argument("--old-root", default="../rna_circuit_distill")
    s.add_argument("--report-dir", default="outputs/model_checks")
    s.set_defaults(func=cmd_migrate_assets)

    s = sub.add_parser("verify-models")
    s.add_argument("--models", default="configs/models.yaml")
    s.add_argument("--output", default="outputs/model_checks/model_attention_check.csv")
    s.add_argument("--max-len", type=int, default=128)
    s.add_argument("--device", default="auto")
    s.add_argument("--include-large", action="store_true")
    s.set_defaults(func=cmd_verify_models)

    s = sub.add_parser("verify-data")
    s.add_argument("--tasks", default="configs/tasks.yaml")
    s.add_argument("--output", default="outputs/model_checks/data_check.csv")
    s.add_argument("--limit", type=int, default=5)
    s.set_defaults(func=cmd_verify_data)

    s = sub.add_parser("sample-rinalmo-corpus")
    s.add_argument("--corpus", required=True)
    s.add_argument("--n", type=int, default=None)
    s.add_argument("--fraction", type=float, default=None)
    s.add_argument("--min-len", type=int, default=32)
    s.add_argument("--max-len", type=int, default=512)
    s.add_argument("--deduplicate", type=lambda x: str(x).lower() != "false", default=True)
    s.add_argument("--sequence-column", default="sequence")
    s.add_argument("--output", required=True)
    s.set_defaults(func=cmd_sample_rinalmo)

    s = sub.add_parser("screen-heads")
    s.add_argument("--model", required=True)
    s.add_argument("--task", required=True)
    s.add_argument("--config", default="configs/screening.yaml")
    s.add_argument("--models", default="configs/models.yaml")
    s.add_argument("--tasks", default="configs/tasks.yaml")
    s.add_argument("--output", required=True)
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--device", default="auto")
    s.set_defaults(func=cmd_screen_heads)

    s = sub.add_parser("make-screening-jobs")
    s.add_argument("--models", default="configs/models.yaml")
    s.add_argument("--tasks", default="configs/tasks.yaml")
    s.add_argument("--model-check", default="outputs/model_checks/model_attention_check.csv")
    s.add_argument("--output", default="outputs/head_screening/screening_jobs.csv")
    s.add_argument("--output-root", default="outputs/head_screening")
    s.add_argument("--include-all-models", action="store_true")
    s.set_defaults(func=cmd_make_screening_jobs)

    s = sub.add_parser("run-screening-jobs")
    s.add_argument("--jobs", default="outputs/head_screening/screening_jobs.csv")
    s.add_argument("--config", default="configs/screening.yaml")
    s.add_argument("--models", default="configs/models.yaml")
    s.add_argument("--tasks", default="configs/tasks.yaml")
    s.add_argument("--report", default="outputs/head_screening/screening_run_report.csv")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--device", default="auto")
    s.add_argument("--max-jobs", type=int, default=None)
    s.add_argument("--model", default=None)
    s.add_argument("--task", default=None)
    s.add_argument("--time-budget-hours", type=float, default=None)
    s.add_argument("--no-resume", action="store_true")
    s.set_defaults(func=cmd_run_screening_jobs)

    s = sub.add_parser("collect-head-screening")
    s.add_argument("--input", required=True)
    s.add_argument("--output", default="outputs/head_screening/all_head_atlas.csv")
    s.set_defaults(func=cmd_collect)

    s = sub.add_parser("plot-head-heatmaps")
    s.add_argument("--atlas", default="outputs/head_screening/all_head_atlas.csv")
    s.add_argument("--tasks", default="configs/tasks.yaml")
    s.add_argument("--output-dir", default="outputs/head_screening/heatmaps")
    s.add_argument("--vmin", type=float, default=0.0)
    s.add_argument("--vmax", type=float, default=1.0)
    s.add_argument("--cmap", default="viridis")
    s.add_argument("--top-k", type=int, default=50)
    s.set_defaults(func=cmd_plot_heatmaps)

    s = sub.add_parser("propose-teacher-heads")
    s.add_argument("--atlas", default="outputs/head_screening/all_head_atlas.csv")
    s.add_argument("--config", default="configs/screening.yaml")
    s.add_argument("--output", default="outputs/teacher_banks/teacher_head_candidates.csv")
    s.add_argument("--yaml", default="outputs/teacher_banks/teacher_bank_template.yaml")
    s.set_defaults(func=cmd_propose)

    s = sub.add_parser("cache-teacher-attention")
    s.add_argument("--teacher-bank", default="configs/teacher_bank_selected.yaml")
    s.add_argument("--sequences", required=True)
    s.add_argument("--output", default="outputs/teacher_attention_cache/debug_1000")
    s.add_argument("--models", default="configs/models.yaml")
    s.add_argument("--max-len", type=int, default=512)
    s.add_argument("--shard-size", type=int, default=64)
    s.add_argument("--device", default="auto")
    s.add_argument("--limit", type=int, default=None)
    s.add_argument("--model", default=None)
    s.add_argument("--shard-index", type=int, default=None)
    s.add_argument("--shard-start", type=int, default=None)
    s.add_argument("--shard-end", type=int, default=None)
    s.add_argument("--dtype", default="fp16", choices=["fp16", "float16", "fp32", "float32"])
    s.add_argument("--max-tokens-per-batch", type=int, default=4096)
    s.add_argument("--sub-batch-size", type=int, default=256)
    s.add_argument("--no-resume", action="store_true")
    s.set_defaults(func=cmd_cache_teacher)

    s = sub.add_parser("make-teacher-cache-jobs")
    s.add_argument("--teacher-bank", default="configs/teacher_bank_selected.yaml")
    s.add_argument("--sequences", required=True)
    s.add_argument("--output", default="outputs/teacher_attention_cache/teacher_cache_jobs.csv")
    s.add_argument("--shard-size", type=int, default=64)
    s.add_argument("--limit", type=int, default=None)
    s.set_defaults(func=cmd_make_cache_jobs)

    s = sub.add_parser("collect-teacher-cache-manifests")
    s.add_argument("--cache", required=True)
    s.add_argument("--output", default=None)
    s.set_defaults(func=cmd_collect_cache_manifests)

    s = sub.add_parser("validate-teacher-cache")
    s.add_argument("--cache", required=True)
    s.add_argument("--output", required=True)
    s.set_defaults(func=cmd_validate_cache)

    s = sub.add_parser("train-alignment")
    s.add_argument("--config", default="configs/student_alignment.yaml")
    s.add_argument("--teacher-cache", required=True)
    s.add_argument("--sequences", default=None)
    s.add_argument("--output", default="outputs/student_alignment/debug_1000")
    s.add_argument("--steps", type=int, default=None)
    s.set_defaults(func=cmd_train_alignment)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
