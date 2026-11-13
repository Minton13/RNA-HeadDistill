#!/bin/bash
#SBATCH --job-name=multitask_smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_multitask/multitask_smoke_%j.out
#SBATCH --error=logs/slurm_multitask/multitask_smoke_%j.err

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

mkdir -p logs/slurm_multitask

echo "[multitask-smoke] start $(date -Is)"

python scripts/ablation/train_multitask.py \
    --mode joint_no_replay \
    --seed 17 \
    --max-steps 20 \
    --eval-every 20 \
    --save-every 20 \
    --warmup-steps 5 \
    --batch-size 16 \
    --debug-steps 20 \
    --eval-subset-size 1024 \
    --output-root outputs/probes/multitask_smoke

echo "[multitask-smoke] done $(date -Is)"
