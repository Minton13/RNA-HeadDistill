#!/bin/bash
#SBATCH --job-name=h2h_train_full_6gpu
#SBATCH --output=logs/slurm_head2head/train_full_6gpu_%j.out
#SBATCH --error=logs/slurm_head2head/train_full_6gpu_%j.err
#SBATCH --time=72:00:00
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=360G
#SBATCH --exclude=n1
# Resume from latest checkpoint: sbatch --export=ALL,RESUME=1 scripts/server/head2head_train_full_6gpu.sbatch
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

# n3 NCCL stability (learned from Stage 4A jobs 11630-11632)
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

CONFIG=${CONFIG:-configs/head2head/train_head2head_full_6gpu.yaml}
RUN_NAME=${RUN_NAME:-rinalmo_head2head_stage4b_full}
RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then RESUME_FLAG="--resume"; fi

echo "[h2h-train-full] job_id=${SLURM_JOB_ID:-unknown} config=$CONFIG run=$RUN_NAME resume=${RESUME:-0} start=$(date -Is)"
torchrun --standalone --nproc_per_node=6 scripts/head2head/train_head2head.py \
    --config "$CONFIG" \
    --run-name "$RUN_NAME" \
    $RESUME_FLAG
echo "[h2h-train-full] done $(date -Is)"
