#!/bin/bash
#SBATCH --job-name=small_smoke
#SBATCH --output=logs/slurm_head2head/small_smoke_%j.out
#SBATCH --error=logs/slurm_head2head/small_smoke_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --exclude=n3
# Smoke for the small-student arms: 200-step MLM (S0) then 200-step
# alignment + MLM aux (S1, random init). Both must show decreasing losses.
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1

echo "[small-smoke] job_id=${SLURM_JOB_ID:-unknown} start=$(date -Is)"

if [ "${SKIP_MLM:-0}" != "1" ]; then
echo "=== S0 MLM smoke (200 steps) ==="
python scripts/head2head/train_small_mlm.py \
    --config configs/head2head/small_student/mlm_s0_smoke.yaml
fi

echo "=== S1 align+MLM smoke (200 steps, random init) ==="
python scripts/head2head/train_head2head.py \
    --config configs/head2head/small_student/train_s1_smoke.yaml

echo "[small-smoke] done $(date -Is)"
