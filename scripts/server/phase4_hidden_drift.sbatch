#!/bin/bash
#SBATCH --job-name=p4_drift
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --exclude=n3
#SBATCH --output=logs/slurm_analysis/p4_drift_%j.out
#SBATCH --error=logs/slurm_analysis/p4_drift_%j.err

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID:-local}

mkdir -p logs/slurm_analysis

python scripts/analysis/phase4_hidden_drift.py --per-task 400 --batch-size 16
echo "[p4-drift] done $(date -Is)"
