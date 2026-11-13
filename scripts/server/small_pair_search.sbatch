#!/bin/bash
#SBATCH --job-name=small_pair_search
#SBATCH --output=logs/slurm_head2head/small_pair_search_%j.out
#SBATCH --error=logs/slurm_head2head/small_pair_search_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=n3
# Head-pair search for the small student using the S0@20k checkpoint.
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

CKPT=runs/small_student/small_s0_mlm/checkpoints/step_20000.pt
if [ ! -f "$CKPT" ]; then
    echo "[small-pair-search] FATAL: missing $CKPT"
    exit 1
fi

echo "[small-pair-search] job_id=${SLURM_JOB_ID:-unknown} start=$(date -Is)"
python scripts/head2head/05_head_pair_search.py \
    --config configs/head2head/small_student/head_pair_search_small.yaml
echo "[small-pair-search] done $(date -Is)"
