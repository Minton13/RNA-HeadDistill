#!/bin/bash
#SBATCH --job-name=small_s0_ddp_smoke
#SBATCH --output=logs/slurm_head2head/small_s0_ddp_smoke_%j.out
#SBATCH --error=logs/slurm_head2head/small_s0_ddp_smoke_%j.err
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=64G
#SBATCH --exclude=n3
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1

torchrun --standalone --nproc_per_node=2 scripts/head2head/train_small_mlm.py \
    --config configs/head2head/small_student/mlm_s0_ddp_smoke.yaml
echo "[ddp-smoke] done $(date -Is)"
