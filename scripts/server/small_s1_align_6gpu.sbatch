#!/bin/bash
#SBATCH --job-name=small_s1_align
#SBATCH --output=logs/slurm_head2head/small_s1_align_%j.out
#SBATCH --error=logs/slurm_head2head/small_s1_align_%j.err
#SBATCH --time=96:00:00
#SBATCH --gres=gpu:6
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=360G
#SBATCH --exclude=n3
# S1 arm: small student init from S0@20k, alignment + MLM aux, 20k steps.
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO

CKPT=runs/small_student/small_s0_mlm/checkpoints/step_20000.pt
PM=outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/head_pair_manifest_small.parquet
for f in "$CKPT" "$PM"; do
    if [ ! -f "$f" ]; then
        echo "[small-s1] FATAL: missing $f"
        exit 1
    fi
done

CONFIG=${CONFIG:-configs/head2head/small_student/train_s1_6gpu.yaml}
RESUME_FLAG=""
if [ "${RESUME:-0}" = "1" ]; then RESUME_FLAG="--resume"; fi

echo "[small-s1] job_id=${SLURM_JOB_ID:-unknown} config=$CONFIG resume=${RESUME:-0} start=$(date -Is)"
torchrun --standalone --nproc_per_node=6 scripts/head2head/train_head2head.py \
    --config "$CONFIG" \
    --run-name small_s1_align_mlm \
    $RESUME_FLAG
echo "[small-s1] done $(date -Is)"
