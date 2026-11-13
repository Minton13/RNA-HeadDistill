#!/bin/bash
#SBATCH --job-name=h2h_full_slim_cache
#SBATCH --output=logs/slurm_head2head/full_slim_cache_%A_%a.out
#SBATCH --error=logs/slurm_head2head/full_slim_cache_%A_%a.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --array=0-63%16
#
# Build the Stage 4B bucketed full slim cache.  Each array task handles all
# shards where ``shard_idx % 64 == SLURM_ARRAY_TASK_ID``.
#
# Submit with:
#
#     sbatch scripts/server/head2head_build_full_slim_cache.sbatch
#
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
echo "[full-slim-cache] array task $TASK_ID: shards where shard_idx % 64 == $TASK_ID"

for SHARD in $(seq 0 1032); do
    if [ $((SHARD % 64)) -eq $TASK_ID ]; then
        python scripts/head2head/04_build_full_slim_cache.py \
            --shard-idx "$SHARD" \
            --cache-root outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads \
            --target-manifest outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/target_head_manifest.parquet \
            --sequence-manifest outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/sequence_manifest.parquet \
            --out-root outputs/teacher_attention_cache_slim/stage4b_head2head/full \
            --max-position 1024 \
            --resume
    fi
done

echo "[full-slim-cache] array task $TASK_ID done"
