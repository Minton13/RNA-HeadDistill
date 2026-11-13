#!/bin/bash
#SBATCH --job-name=fix_missing_evals
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=n3
#SBATCH --array=0-8%6
#SBATCH --output=logs/slurm_ablation/fix_evals_%A_%a.out
#SBATCH --error=logs/slurm_ablation/fix_evals_%A_%a.err

# Backfill missing ablation evaluations (eval-only, no training):
#   0-2: A2_head2head_top3 seeds 17/41/89 (val+test)
#   3-4: data_scaling size500000 seeds 17/89 (val+test)
#   5-7: loss_ablation kl_mse_cos/kl_mse/mse_only seed17 (test only; val exists)
#   8:   teacher_count_ablation top5 seed17 (test only; val exists)

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

readarray -t RUN_DIRS <<EOF
outputs/ablation/alignment/A2_head2head_top3/seed_17
outputs/ablation/alignment/A2_head2head_top3/seed_41
outputs/ablation/alignment/A2_head2head_top3/seed_89
outputs/ablation/alignment/data_scaling/size500000_seed17/seed_17
outputs/ablation/alignment/data_scaling/size500000_seed89/seed_89
outputs/ablation/alignment/loss_ablation/kl_mse_cos_seed17/seed_17
outputs/ablation/alignment/loss_ablation/kl_mse_seed17/seed_17
outputs/ablation/alignment/loss_ablation/mse_only_seed17/seed_17
outputs/ablation/alignment/teacher_count_ablation/top5_seed17/seed_17
EOF

readarray -t SPLIT_LIST <<EOF
val,test
val,test
val,test
val,test
val,test
test
test
test
test
EOF

TASK_ID=${SLURM_ARRAY_TASK_ID}
RUN_DIR=${RUN_DIRS[$TASK_ID]}
SPLITS=${SPLIT_LIST[$TASK_ID]}

echo "[fix-evals] task=${TASK_ID} run_dir=${RUN_DIR} splits=${SPLITS} host=$(hostname) start=$(date -Is)"

python scripts/ablation/eval_only_ablation.py \
    --run-dir "${RUN_DIR}" \
    --splits "${SPLITS}"

echo "[fix-evals] task=${TASK_ID} done $(date -Is)"
