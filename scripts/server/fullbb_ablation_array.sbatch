#!/bin/bash
#SBATCH --job-name=fullbb_ablation
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=72:00:00
#SBATCH --exclude=n1,n3
#SBATCH --array=0-20%6
#SBATCH --output=logs/slurm_ablation/fullbb_%A_%a.out
#SBATCH --error=logs/slurm_ablation/fullbb_%A_%a.err

# Full-backbone core ablation re-run (2026-09-11 activation-checkpointing fix).
# The original A1-A7 runs silently trained with frozen backbones (hook-captured
# activations detach under activation checkpointing; backbone got requires_grad
# but zero gradients).  This array re-trains A1-A7 x seeds {17,41,89} from
# vanilla init with activation_checkpointing=false and a fail-loud step-0
# backbone-gradient guard.  A8 (adapters-only) stays as the frozen comparator.
#
# task layout: task = exp_idx * 3 + seed_idx
#   exp order: A1 A2 A3 A4 A5 A6 A7 ; seed order: 17 41 89

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

EXPS=(A1_single_teacher_top1 A2_head2head_top3 A3_head2head_top5_full A4_consensus_top5 A5_random_pairing A6_linear_adapter A7_all_from_L12)
SEEDS=(17 41 89)

TASK_ID=${SLURM_ARRAY_TASK_ID}
EXP=${EXPS[$(( TASK_ID / 3 ))]}
SEED=${SEEDS[$(( TASK_ID % 3 ))]}

echo "[fullbb] task=${TASK_ID} exp=${EXP} seed=${SEED} host=$(hostname) start=$(date -Is)"

# Fail-fast NFS input probe (2026-09-09 n1 DRAIN incident).
probe() {
    timeout 90 "$@" >/dev/null 2>&1 || {
        echo "[fullbb] input probe FAILED: $* $(date -Is)" >&2
        exit 2
    }
}
probe ls outputs/teacher_attention_cache_slim/stage4b_head2head/full/buckets/
probe ls outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/

python scripts/ablation/run_core_ablation.py \
    --registry configs/ablation/experiment_registry_fullbb.yaml \
    --base-config configs/ablation/core_ablation_base.yaml \
    --experiment-id "${EXP}" \
    --seed "${SEED}"

echo "[fullbb] task=${TASK_ID} done $(date -Is)"
