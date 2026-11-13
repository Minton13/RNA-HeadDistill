#!/usr/bin/env python
"""Run the Stage 4B/5 systems study.

Reads historical Slurm accounting (sacct), runs deterministic microbenchmarks on
a 10,000-sequence subset, and writes the asynchronous-distillation cost report.

Example (local smoke, CPU only):
    PYTHONPATH=src python scripts/ablation/systems_benchmark.py \
        --n-sequences 32 --device cpu --output-dir outputs/systems/asynchronous_distillation

Slurm example:
    sbatch scripts/server/systems_benchmark.sbatch
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rna_head_alignment.io.paths import resolve_path
from rna_head_alignment.systems.benchmark import (
    collect_sacct_jobs,
    historical_cost_rows,
    make_cost_curve,
    run_microbenchmarks,
    write_systems_report,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", default="configs/ablation/experiment_registry.yaml")
    parser.add_argument("--output-dir", default="outputs/systems/asynchronous_distillation")
    parser.add_argument("--n-sequences", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto, cpu, or cuda")
    parser.add_argument("--skip-sacct", action="store_true", help="Skip sacct historical query")
    parser.add_argument("--skip-microbench", action="store_true", help="Skip microbenchmarks")
    parser.add_argument("--sacct-days", type=int, default=30)
    args = parser.parse_args()

    out_dir = Path(resolve_path(args.output_dir))
    out_dir.mkdir(parents=True, exist_ok=True)

    historical_rows: list[dict] = []
    if not args.skip_sacct:
        sacct_df = collect_sacct_jobs(start_days=args.sacct_days)
        historical_rows = historical_cost_rows(sacct_df)

    micro_rows: list[dict] = []
    if not args.skip_microbench:
        micro_rows = run_microbenchmarks(
            registry_path=args.registry,
            output_dir=out_dir,
            n_sequences=args.n_sequences,
            seed=args.seed,
            device=args.device,
        )

    make_cost_curve(historical_rows, micro_rows, out_dir)
    report_path = write_systems_report(historical_rows, micro_rows, out_dir)
    print(f"[systems] wrote {report_path}")


if __name__ == "__main__":
    main()
