#!/bin/bash
#SBATCH --job-name=a3_coldstart
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --exclude=n3
#SBATCH --array=0-2%3
#SBATCH --output=logs/slurm_ablation/a3_coldstart_%A_%a.out
#SBATCH --error=logs/slurm_ablation/a3_coldstart_%A_%a.err

# A3 cold-start publication fix: identical config to A3_head2head_top5_full
# but trained from vanilla init (no stage4b warm-start), seeds 17/41/89.

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

mkdir -p logs/slurm_ablation

SEEDS=(17 41 89)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}

echo "[a3-coldstart] task=${SLURM_ARRAY_TASK_ID} seed=${SEED} host=$(hostname) start=$(date -Is)"

python scripts/ablation/run_core_ablation.py \
    --registry configs/ablation/experiment_registry_a3_coldstart.yaml \
    --base-config configs/ablation/core_ablation_base.yaml \
    --experiment-id A3_head2head_top5_full \
    --seed "${SEED}"

echo "[a3-coldstart] task=${SLURM_ARRAY_TASK_ID} done $(date -Is)"
