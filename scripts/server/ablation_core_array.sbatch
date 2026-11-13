#!/bin/bash
#SBATCH --job-name=core_ablation_array
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --array=0-26%6
#SBATCH --output=logs/slurm_ablation/core_ablation_%A_%a.out
#SBATCH --error=logs/slurm_ablation/core_ablation_%A_%a.err

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

# n3 NCCL stability settings learned from earlier Stage 4A/B jobs.
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1

mkdir -p logs/slurm_ablation

# Map task ID to (experiment_id, seed).  Order: all seeds for A0, then A1, ... A8.
readarray -t EXPERIMENTS <<EOF
A0_vanilla
A1_single_teacher_top1
A2_head2head_top3
A3_head2head_top5_full
A4_consensus_top5
A5_random_pairing
A6_linear_adapter
A7_all_from_L12
A8_adapters_only
EOF

SEEDS=(17 41 89)

TASK_ID=${SLURM_ARRAY_TASK_ID}
N_SEEDS=${#SEEDS[@]}
EXP_IDX=$(( TASK_ID / N_SEEDS ))
SEED_IDX=$(( TASK_ID % N_SEEDS ))
EXPERIMENT_ID=${EXPERIMENTS[$EXP_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "[core-ablation-array] task=${TASK_ID} experiment=${EXPERIMENT_ID} seed=${SEED} host=$(hostname) start=$(date -Is)"

python scripts/ablation/run_core_ablation.py \
    --registry configs/ablation/experiment_registry.yaml \
    --base-config configs/ablation/core_ablation_base.yaml \
    --experiment-id "${EXPERIMENT_ID}" \
    --seed "${SEED}"

echo "[core-ablation-array] task=${TASK_ID} done $(date -Is)"
