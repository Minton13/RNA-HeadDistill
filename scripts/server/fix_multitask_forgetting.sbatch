#!/bin/bash
#SBATCH --job-name=multitask_forgetting
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3
#SBATCH --output=logs/slurm_route_finetune/mt_forgetting_%A.out
#SBATCH --error=logs/slurm_route_finetune/mt_forgetting_%A.err

# Post-hoc pre/post alignment (forgetting) measurement for multitask joint runs.

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p logs/slurm_route_finetune

echo "[mt-forgetting] host=$(hostname) start=$(date -Is)"
python scripts/analysis/measure_multitask_forgetting.py
echo "[mt-forgetting] done $(date -Is)"
