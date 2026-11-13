#!/bin/bash
#SBATCH --job-name=fullbb_smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=n3
#SBATCH --output=logs/slurm_ablation/fullbb_smoke_%A.out
#SBATCH --error=logs/slurm_ablation/fullbb_smoke_%A.err

# Smoke for the full-backbone ablation re-run (2026-09-11 activation-
# checkpointing fix). 300 steps of A3 seed 17 with activation
# checkpointing OFF. Verifies:
#   1. the new fail-loud guard passes (backbone gradients flow),
#   2. memory fits without activation checkpointing at the production
#      dynamic-batch budget,
#   3. step time, for the full-run time-limit estimate.
# Post-check (run after completion): the smoke checkpoint's backbone tensors
# must DIFFER from A8's frozen-vanilla backbone.

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

echo "[fullbb-smoke] host=$(hostname) start=$(date -Is)"

# Fail-fast NFS input probe (2026-09-09 n1 DRAIN incident).
probe() {
    timeout 90 "$@" >/dev/null 2>&1 || {
        echo "[fullbb-smoke] input probe FAILED: $* $(date -Is)" >&2
        exit 2
    }
}
probe ls outputs/teacher_attention_cache_slim/stage4b_head2head/full/buckets/
probe ls outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/

python scripts/ablation/run_core_ablation.py \
    --registry configs/ablation/experiment_registry_fullbb_smoke.yaml \
    --base-config configs/ablation/core_ablation_base.yaml \
    --experiment-id A3_head2head_top5_full \
    --seed 17 \
    --skip-eval

echo "[fullbb-smoke] done $(date -Is)"
