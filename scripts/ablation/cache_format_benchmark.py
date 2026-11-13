#!/usr/bin/env python
"""Run the Stage 4B/5 cache-format benchmark.

Compares C0 (bucketed fp16), C1 (ragged flat fp16), C2 (chunked zarr compressed
fp16), and C3 (row-wise uint8 quantized) on a 10,000-sequence subset.

Example (local smoke):
    PYTHONPATH=src python scripts/ablation/cache_format_benchmark.py \
        --n-sequences 128 --output-dir outputs/systems/cache_format_benchmark_smoke

Slurm example:
    sbatch scripts/server/cache_format_benchmark.sbatch
"""

from __future__ import annotations

import argparse

from rna_head_alignment.io.paths import resolve_path
from rna_head_alignment.systems.cache_formats import run_cache_format_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-root",
        default="outputs/teacher_attention_cache_slim/stage4b_head2head/full",
        help="Root of the existing C0 bucketed full slim cache",
    )
    parser.add_argument(
        "--sequence-manifest",
        default="outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/sequence_manifest.parquet",
    )
    parser.add_argument(
        "--bucket-manifest",
        default="outputs/teacher_attention_cache_slim/stage4b_head2head/full/manifests/slim_cache_manifest.parquet",
    )
    parser.add_argument("--output-dir", default="outputs/systems/cache_format_benchmark")
    parser.add_argument("--n-sequences", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    df = run_cache_format_benchmark(
        cache_root=args.cache_root,
        sequence_manifest_path=args.sequence_manifest,
        bucket_manifest_path=args.bucket_manifest,
        output_dir=args.output_dir,
        n=args.n_sequences,
        seed=args.seed,
    )
    summary_path = resolve_path(args.output_dir) / "cache_format_summary.csv"
    print(f"[cache-formats] wrote {summary_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
