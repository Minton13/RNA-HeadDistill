#!/usr/bin/env python
"""Stage 4B data-driven head-pair search CLI.

Reads ``configs/head2head/head_pair_search.yaml``, runs the search over the
configured slim shards, and writes the fixed ``head_pair_manifest.parquet``.
The previous manual seed manifest is backed up next to the output.

Example::

    PYTHONPATH=src python scripts/head2head/05_head_pair_search.py \
        --config configs/head2head/head_pair_search.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from rna_head_alignment.head2head.head_manifest import (
    PAIR_MANIFEST_COLUMNS,
    ROUTE_DEPTHS,
    load_manifest,
    save_manifest,
)
from rna_head_alignment.head2head.pair_search import (
    DEFAULT_METRICS,
    build_head_pair_manifest,
)
from rna_head_alignment.io.paths import load_yaml, resolve_path


def _parse_shards(shards: str | list[int] | None) -> list[int]:
    if not shards:
        return []
    if isinstance(shards, (list, tuple)):
        return sorted({int(s) for s in shards})
    return sorted({int(s) for s in str(shards).split(",") if s.strip()})


def _parse_candidate_heads(value: str | int | list[int] | None) -> str | list[int] | None:
    if value is None or value == "all":
        return "all"
    if isinstance(value, int):
        return [value]
    if isinstance(value, list):
        return [int(h) for h in value]
    # YAML may return a comma-separated string.
    return [int(h) for h in str(value).split(",") if h.strip()]


def _build_report(
    manifest: pd.DataFrame,
    seed_manifest: pd.DataFrame,
    config: dict,
    route_depths: tuple[int, ...],
) -> str:
    lines = ["# Stage 4B data-driven head-pair search report", ""]
    lines.append(f"- output rows: {len(manifest)}")
    lines.append(f"- pairing method: {config.get('pairing', 'bipartite')}")
    lines.append(f"- metrics: {config.get('metrics', list(DEFAULT_METRICS))}")
    lines.append(f"- shards: {config.get('shards', [])}")
    lines.append(f"- max_samples_per_shard: {config.get('max_samples_per_shard', 256)}")
    lines.append("")

    seed_lookup = {}
    for _, row in seed_manifest.iterrows():
        teacher_key = (
            f"{row['teacher_model']}:L{row['teacher_layer']}:H{row['teacher_head']}"
            f":L{row['route_depth']}:{row['task']}"
        )
        seed_lookup[(row["route_depth"], teacher_key)] = row["student_head"]

    for depth in route_depths:
        sub = manifest[manifest["route_depth"] == depth].sort_values("student_head")
        lines.append(f"## Route L{depth}")
        lines.append("")
        lines.append("| adapter | student head | teacher | channel | matching_metric | matching_score | vs seed |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, row in sub.iterrows():
            teacher = f"{row['teacher_model']}:L{row['teacher_layer']}:H{row['teacher_head']}"
            teacher_key = (
                f"{row['teacher_model']}:L{row['teacher_layer']}:H{row['teacher_head']}"
                f":L{row['route_depth']}:{row['task']}"
            )
            seed_head = seed_lookup.get((depth, teacher_key))
            vs_seed = "same" if seed_head == row["student_head"] else f"seed=H{seed_head}"
            score = row["matching_score"]
            score_s = f"{score:.4f}" if pd.notna(score) else "NaN"
            lines.append(
                f"| {row['adapter_key']} | H{row['student_head']} | "
                f"`{teacher}` | {row['teacher_cache_channel']} | "
                f"{row['matching_metric']} | {score_s} | {vs_seed} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="head_pair_search.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    cfg.setdefault("student", {})

    route_depths = tuple(cfg.get("route_depths", list(ROUTE_DEPTHS)))
    metrics = cfg.get("metrics", list(DEFAULT_METRICS))
    pairing = cfg.get("pairing", "bipartite")
    shards = _parse_shards(cfg.get("shards"))
    if not shards:
        raise SystemExit("config must specify at least one shard index")

    target_manifest_path = resolve_path(cfg["target_head_manifest"])
    seed_pair_manifest_path = resolve_path(cfg["seed_pair_manifest"])
    out_manifest_path = resolve_path(cfg["out_manifest"])
    cache_root = resolve_path(cfg["slim_cache_root"])
    student_model_dir = resolve_path(cfg["student"]["model_dir"])

    # Backup the seed manifest before overwriting the output path.
    seed_backup_path = out_manifest_path.with_suffix(".seed.parquet")
    save_manifest(load_manifest(seed_pair_manifest_path), seed_backup_path)

    manifest = build_head_pair_manifest(
        target_manifest=target_manifest_path,
        seed_pair_manifest=seed_pair_manifest_path,
        cache_root=cache_root,
        shards=shards,
        student_model_dir=student_model_dir,
        route_depths=route_depths,
        metrics=metrics,
        pairing=pairing,
        max_samples_per_shard=int(cfg.get("max_samples_per_shard", 256)),
        candidate_student_heads=_parse_candidate_heads(
            cfg.get("candidate_student_heads", "all")
        ),
        batch_size=int(cfg.get("batch_size", 8)),
        max_len=int(cfg.get("max_len", 512)),
        precision=cfg.get("precision", "bf16"),
        device=cfg.get("device", "auto"),
        seed=int(cfg.get("seed", 42)),
        spearman_sample_size=int(
            cfg.get("spearman_sample_size", 2048)
        ),
        fallback_to_seed=bool(cfg.get("fallback_to_seed", True)),
        progress=cfg.get("progress", True),
        student_config_overrides=cfg["student"].get("config_overrides"),
        student_init_checkpoint=(
            resolve_path(cfg["student"]["init_checkpoint"])
            if cfg["student"].get("init_checkpoint")
            else None
        ),
    )

    if set(manifest.columns) != set(PAIR_MANIFEST_COLUMNS):
        manifest = manifest[PAIR_MANIFEST_COLUMNS]
    save_manifest(manifest, out_manifest_path)

    # Write per-route score matrices and a Markdown report for inspection.
    report_dir = out_manifest_path.parent / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    seed_manifest = load_manifest(seed_pair_manifest_path)
    report_md = _build_report(manifest, seed_manifest, cfg, route_depths)
    report_path = report_dir / "head_pair_search_report.md"
    report_path.write_text(report_md, encoding="utf-8")

    metadata = {
        "config": cfg,
        "out_manifest": str(out_manifest_path),
        "seed_backup": str(seed_backup_path),
        "n_pairs": len(manifest),
        "route_depths": list(route_depths),
        "metrics": list(metrics),
        "pairing": pairing,
        "shards": shards,
    }
    metadata_path = report_dir / "head_pair_search_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    print(f"[pair_search] wrote {len(manifest)} pairs to {out_manifest_path}")
    print(f"[pair_search] seed manifest backed up to {seed_backup_path}")
    print(f"[pair_search] report: {report_path}")


if __name__ == "__main__":
    main()
