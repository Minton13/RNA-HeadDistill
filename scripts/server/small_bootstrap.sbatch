#!/bin/bash
#SBATCH --job-name=small_bootstrap
#SBATCH --output=logs/slurm_analysis/small_bootstrap_%j.out
#SBATCH --error=logs/slurm_analysis/small_bootstrap_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=n3
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID:-local}

python scripts/analysis/small_student_bootstrap.py
echo "[small-bootstrap] done $(date -Is)"
