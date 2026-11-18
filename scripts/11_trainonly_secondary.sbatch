#!/usr/bin/env bash
#SBATCH --job-name=rha_trainonly_sec
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=60G
#SBATCH --time=12:00:00
#SBATCH --output=logs/head_screening/%A_trainonly.out
#SBATCH --error=logs/head_screening/%A_trainonly.err

# Train-only re-screening of secondary_structure_pair (all 37 enabled models,
# sequential in ONE single-GPU job; run_screening_job_table resumes/skips).
# Fixes the mixed-split leak: the original atlas probed each model on the
# first 100 records of the shuffled secondary_full.jsonl (~80/11/9
# train/test/val). This rerun keeps only split=="train" records (n=100/model).

set -euo pipefail
mkdir -p logs/head_screening
export PYTHONPATH="$PWD/src"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

# Override with the python that has the project deps installed, e.g.:
#   RHA_PYTHON=/path/to/env/bin/python sbatch 11_trainonly_secondary.sbatch
PYTHON="${RHA_PYTHON:-python}"

"$PYTHON" -m rna_head_alignment.cli run-screening-jobs \
  --jobs outputs/head_screening_trainonly/screening_jobs_trainonly_secondary.csv \
  --config configs/screening_trainonly_secondary.yaml \
  --report outputs/head_screening_trainonly/screening_run_report_trainonly_secondary.csv \
  --limit 600 \
  --device cuda
