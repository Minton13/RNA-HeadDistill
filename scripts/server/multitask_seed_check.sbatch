#!/bin/bash
#SBATCH --job-name=multitask_seed_check
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=n3
#SBATCH --output=logs/slurm_multitask/multitask_seed_check_%j.out
#SBATCH --error=logs/slurm_multitask/multitask_seed_check_%j.err

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

for seed in 17 41; do
    echo "=== seed ${seed} ==="
    python scripts/ablation/train_multitask.py \
        --mode joint_no_replay \
        --seed "${seed}" \
        --max-steps 20 \
        --eval-every 1000 \
        --save-every 1000 \
        --batch-size 8 \
        --output-root outputs/probes/multitask_seed_check \
        --debug-steps 20 \
        --skip-eval
done

echo "[multitask-seed-check] done"
