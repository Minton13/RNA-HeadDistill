#!/bin/bash
#SBATCH --job-name=h2h_write_full_report
#SBATCH --output=logs/slurm_head2head/write_full_report_%j.out
#SBATCH --error=logs/slurm_head2head/write_full_report_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --exclude=n1
#
# Generate the final Stage 4B head-to-head training/evaluation report.
# This job is intended to run after head2head_eval_full.sbatch completes.
#
set -euo pipefail

cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "[h2h-write-full-report] job_id=${SLURM_JOB_ID:-unknown} start=$(date -Is)"

python scripts/head2head/07_write_full_report.py

echo "[h2h-write-full-report] done $(date -Is)"
