#!/bin/bash
#SBATCH --job-name=h2h_eval_full
#SBATCH --output=logs/slurm_head2head/eval_full_%j.out
#SBATCH --error=logs/slurm_head2head/eval_full_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
#SBATCH --exclude=n1
# Usage: sbatch --export=ALL,CHECKPOINT=runs/head2head_full/<run>/checkpoints/best_val.pt scripts/server/head2head_eval_full.sbatch
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# n3 NCCL stability
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1

CONFIG=${CONFIG:-configs/head2head/eval_head2head_full.yaml}
CHECKPOINT=${CHECKPOINT:-runs/head2head_full/rinalmo_head2head_stage4b_full/checkpoints/best_val.pt}
# If the training run never produced a best_val checkpoint (e.g. per-rank val
# loader was empty), fall back to the latest checkpoint so evaluation still runs.
if [ ! -f "$CHECKPOINT" ]; then
    FALLBACK="runs/head2head_full/rinalmo_head2head_stage4b_full/checkpoints/latest.pt"
    echo "[h2h-eval-full] $CHECKPOINT not found, falling back to $FALLBACK"
    CHECKPOINT=$FALLBACK
fi
RUN_NAME=${RUN_NAME:-rinalmo_head2head_stage4b_full}

echo "[h2h-eval-full] job_id=${SLURM_JOB_ID:-unknown} checkpoint=$CHECKPOINT run=$RUN_NAME start=$(date -Is)"
python scripts/head2head/eval_head2head_full.py \
    --config "$CONFIG" \
    --checkpoint "$CHECKPOINT" \
    --run-name "$RUN_NAME" \
    --splits val,test
echo "[h2h-eval-full] done $(date -Is)"
