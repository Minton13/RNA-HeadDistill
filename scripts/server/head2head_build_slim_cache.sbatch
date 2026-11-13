#!/bin/bash
#SBATCH --job-name=h2h_slim_cache
#SBATCH --output=logs/slurm_head2head/slim_cache_%A_%a.out
#SBATCH --error=logs/slurm_head2head/slim_cache_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
# Array job: one task per shard chunk. Full build = 1033 shards.
# Smoke first:  sbatch --array=0 scripts/server/head2head_build_slim_cache.sbatch
# Full build:   sbatch --array=0-20 scripts/server/head2head_build_slim_cache.sbatch  (21 tasks x ~50 shards)
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

CHUNK=${SLURM_ARRAY_TASK_ID:-0}
CHUNK_SIZE=${CHUNK_SIZE:-50}
START=$((CHUNK * CHUNK_SIZE))
END=$((START + CHUNK_SIZE))

echo "[slim-cache] chunk=$CHUNK shards [$START, $END)"
python scripts/head2head/02_build_slim_cache.py \
    --cache-root outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads \
    --manifest outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/target_head_manifest.parquet \
    --out-root outputs/teacher_attention_cache_slim/stage4b_head2head \
    --shard-range ${START}:${END} \
    --resume
echo "[slim-cache] chunk=$CHUNK done"
