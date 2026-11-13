#!/usr/bin/env bash
#SBATCH --job-name=rha_tcache_mdl
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=80G
#SBATCH --time=96:00:00
#SBATCH --output=logs/teacher_cache/%x_%j_%A_%a.out
#SBATCH --error=logs/teacher_cache/%x_%j_%A_%a.err

set -euo pipefail
mkdir -p logs/teacher_cache
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
# Limit glibc malloc arenas to reduce long-term RSS growth from many large
# per-shard torch allocations.
export MALLOC_ARENA_MAX=2

MODEL="${MODEL:?MODEL environment variable must be set}"
TEACHER_BANK="${TEACHER_BANK:-configs/teacher_bank_selected.yaml}"
SEQUENCES="${SEQUENCES:-outputs/task_samples/rinalmo_pretrain_10pct_len_sorted.jsonl}"
CACHE_OUTPUT="${CACHE_OUTPUT:-outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads}"
MODELS_YAML="${MODELS_YAML:-configs/models.yaml}"
MAX_LEN="${MAX_LEN:-512}"
SHARD_SIZE="${SHARD_SIZE:-2048}"
DTYPE="${DTYPE:-fp16}"
MAX_TOKENS_PER_BATCH="${MAX_TOKENS_PER_BATCH:-4096}"
SUB_BATCH_SIZE="${SUB_BATCH_SIZE:-64}"

echo "=== Cache teacher attention for model: ${MODEL} (80g) ==="
echo "Start: $(date)"

python -m rna_head_alignment.cli cache-teacher-attention \
  --teacher-bank "${TEACHER_BANK}" \
  --sequences "${SEQUENCES}" \
  --output "${CACHE_OUTPUT}" \
  --models "${MODELS_YAML}" \
  --max-len "${MAX_LEN}" \
  --shard-size "${SHARD_SIZE}" \
  --dtype "${DTYPE}" \
  --max-tokens-per-batch "${MAX_TOKENS_PER_BATCH}" \
  --sub-batch-size "${SUB_BATCH_SIZE}" \
  --model "${MODEL}" \
  --device cuda

echo "End: $(date)"
