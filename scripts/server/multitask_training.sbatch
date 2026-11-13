#!/bin/bash
#SBATCH --job-name=multitask_training
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --array=0-5%2
#SBATCH --exclude=n3
#SBATCH --output=logs/slurm_multitask/multitask_%A_%a.out
#SBATCH --error=logs/slurm_multitask/multitask_%A_%a.err

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

# Two modes x three seeds.
MODES=(joint_no_replay joint_with_replay)
SEEDS=(17 41 89)

N_SEEDS=${#SEEDS[@]}
TASK_ID=${SLURM_ARRAY_TASK_ID}
MODE_IDX=$(( TASK_ID / N_SEEDS ))
SEED_IDX=$(( TASK_ID % N_SEEDS ))

MODE=${MODES[$MODE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "[multitask-array] task=${TASK_ID} mode=${MODE} seed=${SEED} host=$(hostname) start=$(date -Is)"

python scripts/ablation/train_multitask.py \
    --mode "${MODE}" \
    --seed "${SEED}" \
    --warmup-steps 1000 \
    --batch-size 16 \
    --max-steps 5000 \
    --eval-every 1000 \
    --save-every 1000 \
    --eval-subset-size 4096

echo "[multitask-array] task=${TASK_ID} done $(date -Is)"
