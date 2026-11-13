#!/bin/bash
#SBATCH --job-name=fix_secondary_ft
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --exclude=n3
#SBATCH --array=0-14%6
#SBATCH --output=logs/slurm_route_finetune/fix_secondary_%A_%a.out
#SBATCH --error=logs/slurm_route_finetune/fix_secondary_%A_%a.err

# Backfill secondary_structure_pair route fine-tuning: 5 configs x 3 seeds.
# CONFIG_IDX 0 = F1 (no replay); 1-4 = F2 with lambda in {0.0, 0.01, 0.1, 0.5}.
# Replay subset indices already exist under the F2 cell dirs (deterministic
# per seed); the runner reuses them via _load_or_build_replay_index.

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

mkdir -p logs/slurm_route_finetune

SEEDS=(17 41 89)
LAMBDAS=(0.0 0.01 0.1 0.5)

TASK_ID=${SLURM_ARRAY_TASK_ID}
N_SEEDS=${#SEEDS[@]}
CONFIG_IDX=$(( TASK_ID / N_SEEDS ))
SEED_IDX=$(( TASK_ID % N_SEEDS ))
SEED=${SEEDS[$SEED_IDX]}

if [ "${CONFIG_IDX}" -eq 0 ]; then
    MODE="F1"
    LAMBDA=0.0
else
    MODE="F2"
    LAMBDA=${LAMBDAS[$(( CONFIG_IDX - 1 ))]}
fi

echo "[fix-secondary] task=${TASK_ID} mode=${MODE} seed=${SEED} lambda=${LAMBDA} host=$(hostname) start=$(date -Is)"

# Fail-fast probe of NFS-mounted inputs (2026-09-09 n1 DRAIN incident: job
# 31588_10 blocked in an uninterruptible NFS read, survived scancel, and
# slurmd's failed cleanup drained the node).  If the mount is wedged, die in
# the prolog instead of idling with a GPU held.
probe() {
    timeout 90 "$@" >/dev/null 2>&1 || {
        echo "[fix-secondary] input probe FAILED: $* $(date -Is)" >&2
        exit 2
    }
}
probe ls outputs/teacher_attention_cache_slim/stage4b_head2head/full/buckets/
probe dd if=runs/head2head_full/rinalmo_head2head_stage4b_full/checkpoints/best_val.pt of=/dev/null bs=4M count=4 status=none

python scripts/ablation/run_route_finetune.py \
    --task secondary_structure_pair \
    --feature-mode "${MODE}" \
    --seed "${SEED}" \
    --lambda-align "${LAMBDA}" \
    --mixed-precision

echo "[fix-secondary] task=${TASK_ID} done $(date -Is)"
