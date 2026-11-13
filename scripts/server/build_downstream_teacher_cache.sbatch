#!/bin/bash
#SBATCH --job-name=build_downstream_teacher_cache
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_downstream/build_teacher_cache_%j.out
#SBATCH --error=logs/slurm_downstream/build_teacher_cache_%j.err

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

# GPU cache settings from Stage 4B full training.
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1

mkdir -p logs/slurm_downstream

echo "[build-teacher-cache] start $(date -Is) host=$(hostname) gpu=$CUDA_VISIBLE_DEVICES"

python scripts/ablation/build_downstream_teacher_cache.py \
    --config configs/ablation/downstream_data.yaml \
    --resume

echo "[build-teacher-cache] done $(date -Is)"
