#!/bin/bash
#SBATCH --job-name=h2h_eval_export
#SBATCH --output=logs/slurm_head2head/eval_export_%j.out
#SBATCH --error=logs/slurm_head2head/eval_export_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=120G
# Usage: sbatch --export=ALL,CHECKPOINT=runs/head2head/<run>/checkpoints/best_val.pt scripts/server/head2head_eval_export.sbatch
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

CHECKPOINT=${CHECKPOINT:-runs/head2head/rinalmo_head2head_stage4b/checkpoints/best_val.pt}
RUN_NAME=${RUN_NAME:-rinalmo_head2head_stage4b}

echo "[h2h-eval] checkpoint=$CHECKPOINT"
python scripts/head2head/eval_head2head.py \
    --config configs/head2head/eval_head2head.yaml \
    --checkpoint "$CHECKPOINT" \
    --run-name "$RUN_NAME"

echo "[h2h-export] exporting predicted teacher heads"
python scripts/head2head/export_predicted_heads.py \
    --config configs/head2head/eval_head2head.yaml \
    --checkpoint "$CHECKPOINT" \
    --run-name "$RUN_NAME" \
    --split val

python scripts/head2head/export_predicted_heads.py \
    --config configs/head2head/eval_head2head.yaml \
    --checkpoint "$CHECKPOINT" \
    --run-name "$RUN_NAME" \
    --split test

echo "[h2h-eval-export] done $(date -Is)"
