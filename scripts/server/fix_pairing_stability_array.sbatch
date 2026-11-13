#!/bin/bash
#SBATCH --job-name=pairing_stab
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3
#SBATCH --array=0-4%5
#SBATCH --output=logs/slurm_ablation/pairing_stab_%A_%a.out
#SBATCH --error=logs/slurm_ablation/pairing_stab_%A_%a.err

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

mkdir -p logs/slurm_ablation

echo "[pairing-stab] bootstrap=${SLURM_ARRAY_TASK_ID} host=$(hostname) start=$(date -Is)"

python scripts/ablation/run_pairing_stability.py \
    --bootstrap "${SLURM_ARRAY_TASK_ID}" \
    --seed 17 \
    --max-samples-per-shard 256 \
    --batch-size 8 \
    --precision bf16

echo "[pairing-stab] bootstrap=${SLURM_ARRAY_TASK_ID} done $(date -Is)"
