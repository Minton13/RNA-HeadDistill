#!/bin/bash
#SBATCH --job-name=p5_finalize_cache
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --array=0-2
#SBATCH --exclude=n3
#SBATCH --output=logs/slurm_downstream/p5_finalize_%A_%a.out
#SBATCH --error=logs/slurm_downstream/p5_finalize_%A_%a.err

# Link downstream teacher shards into task dirs, write index.json, validate coverage.
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

mkdir -p logs/slurm_downstream

ROUTES=(6 8 12)
ROUTE=${ROUTES[${SLURM_ARRAY_TASK_ID}]}

echo "[p5-finalize] route=L${ROUTE} host=$(hostname) start=$(date -Is)"

python scripts/ablation/finalize_downstream_teacher_cache.py --route-depth "${ROUTE}"

echo "[p5-finalize] route=L${ROUTE} done $(date -Is)"
