#!/usr/bin/env python
"""Build Stage 4B head manifests (target heads + default head pairs).

Run from the project root with the package importable, e.g.:

    PYTHONPATH=src python scripts/head2head/01_build_manifests.py \
        --bank configs/teacher_bank_selected.yaml \
        --atlas outputs/head_screening/all_head_atlas.csv \
        --out-dir outputs/teacher_attention_cache_slim/stage4b_head2head/manifests

Writes target_head_manifest.parquet and head_pair_manifest.parquet plus a
Markdown summary for each (per-route head list and score backfill status).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rna_head_alignment.head2head.head_manifest import (
    ROUTE_DEPTHS,
    build_default_head_pair_manifest,
    build_target_head_manifest,
    save_manifest,
)


def _target_summary_md(df) -> str:
    lines = ["# Target head manifest (Stage 4B)", ""]
    lines.append(f"Total heads: {len(df)}")
    n_scored = int(df["selection_score"].notna().sum())
    lines.append(
        f"Selection scores backfilled from atlas: {n_scored} / {len(df)} "
        f"(remaining {len(df) - n_scored} stay 'manual')"
    )
    lines.append("")
    for depth in ROUTE_DEPTHS:
        sub = df[df["route_depth"] == depth]
        lines.append(f"## Route L{depth} (student layer {sub['student_route_layer'].iloc[0] if len(sub) else '?'})")
        lines.append("")
        lines.append("| rank | head_key | channel | score | metric |")
        lines.append("|---|---|---|---|---|")
        for _, row in sub.sort_values("rank").iterrows():
            score = row["selection_score"]
            score_s = f"{score:.4f}" if score == score else "NaN"
            lines.append(
                f"| {row['rank']} | `{row['head_key']}` | "
                f"{row['cache_channel']} | {score_s} | {row['selection_metric']} |"
            )
        lines.append("")
    return "\n".join(lines)


def _pair_summary_md(df) -> str:
    lines = ["# Head pair manifest (Stage 4B, manual seed pairs)", ""]
    lines.append(f"Total pairs: {len(df)}")
    lines.append("")
    for depth in ROUTE_DEPTHS:
        sub = df[df["route_depth"] == depth]
        lines.append(f"## Route L{depth}")
        lines.append("")
        lines.append("| adapter | student head | teacher | channel | task |")
        lines.append("|---|---|---|---|---|")
        for _, row in sub.sort_values("student_head").iterrows():
            teacher = f"{row['teacher_model']}:L{row['teacher_layer']}:H{row['teacher_head']}"
            lines.append(
                f"| {row['adapter_key']} | H{row['student_head']} | "
                f"`{teacher}` | {row['teacher_cache_channel']} | {row['task']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", required=True, help="teacher_bank_selected.yaml")
    parser.add_argument(
        "--atlas",
        default=None,
        help="all_head_atlas.csv for selection-score backfill (optional)",
    )
    parser.add_argument("--out-dir", required=True, help="output directory")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = build_target_head_manifest(args.bank, atlas_csv=args.atlas)
    pair = build_default_head_pair_manifest(args.bank)

    target_path = save_manifest(target, out_dir / "target_head_manifest.parquet")
    pair_path = save_manifest(pair, out_dir / "head_pair_manifest.parquet")

    (out_dir / "target_head_manifest.md").write_text(
        _target_summary_md(target), encoding="utf-8"
    )
    (out_dir / "head_pair_manifest.md").write_text(
        _pair_summary_md(pair), encoding="utf-8"
    )

    print(f"target_head_manifest: {len(target)} rows -> {target_path}")
    print(f"head_pair_manifest:   {len(pair)} rows -> {pair_path}")
    scored = int(target["selection_score"].notna().sum())
    print(f"atlas score backfill: {scored}/{len(target)} heads scored")


if __name__ == "__main__":
    main()
