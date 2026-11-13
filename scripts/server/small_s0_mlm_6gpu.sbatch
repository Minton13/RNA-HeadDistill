#!/bin/bash
#SBATCH --job-name=small_s0_mlm
#SBATCH --output=logs/slurm_head2head/small_s0_mlm_%j.out
#SBATCH --error=logs/slurm_head2head/small_s0_mlm_%j.err
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=360G
#SBATCH --exclude=n3
# S0 arm: from-scratch small-student MLM pretraining.
#   S0a: CONFIG=configs/head2head/small_student/mlm_s0a_6gpu.yaml (steps 0-20k)
#   S0b: CONFIG=configs/head2head/small_student/mlm_s0b_6gpu.yaml RESUME=1 (steps 20k-40k)
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

CONFIG=${CONFIG:-configs/head2head/small_student/mlm_s0a_6gpu.yaml}
RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then RESUME_FLAG="--resume"; fi

echo "[small-s0-mlm] job_id=${SLURM_JOB_ID:-unknown} config=$CONFIG resume=${RESUME:-0} start=$(date -Is)"
torchrun --standalone --nproc_per_node=6 scripts/head2head/train_small_mlm.py \
    --config "$CONFIG" \
    $RESUME_FLAG
echo "[small-s0-mlm] done $(date -Is)"
