#!/usr/bin/env bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=120G
#SBATCH --time=12:00:00
#SBATCH --exclude=n2
#SBATCH --output=logs/teacher_cache/range_%x_%j.out
#SBATCH --error=logs/teacher_cache/range_%x_%j.err

set -euo pipefail
mkdir -p logs/teacher_cache
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:?MODEL is required}"
START_SHARD="${START_SHARD:?START_SHARD is required}"
END_SHARD="${END_SHARD:?END_SHARD is required}"
MAX_TOKENS="${MAX_TOKENS:?MAX_TOKENS is required}"
SUB_BATCH="${SUB_BATCH:?SUB_BATCH is required}"

TEACHER_BANK="configs/teacher_bank_selected.yaml"
SEQUENCES="outputs/task_samples/rinalmo_pretrain_10pct_len_sorted.jsonl"
CACHE_OUTPUT="outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads"
MODELS_YAML="configs/models.yaml"
MAX_LEN=512
SHARD_SIZE=2048
DTYPE="fp16"

if [ "${USE_EXPANDABLE:-0}" = "1" ] || [ "${MODEL}" = "rinalmo_giga" ]; then
    export PYTORCH_ALLOC_CONF=expandable_segments:True
fi

echo "[$(date)] Starting ${MODEL} shards ${START_SHARD}-${END_SHARD} on a single GPU"

python -m rna_head_alignment.cli cache-teacher-attention \
    --teacher-bank "${TEACHER_BANK}" \
    --sequences "${SEQUENCES}" \
    --output "${CACHE_OUTPUT}" \
    --models "${MODELS_YAML}" \
    --max-len "${MAX_LEN}" \
    --shard-size "${SHARD_SIZE}" \
    --dtype "${DTYPE}" \
    --max-tokens-per-batch "${MAX_TOKENS}" \
    --sub-batch-size "${SUB_BATCH}" \
    --model "${MODEL}" \
    --shard-start "${START_SHARD}" \
    --shard-end "${END_SHARD}" \
    --device cuda

echo "[$(date)] Finished ${MODEL} shards ${START_SHARD}-${END_SHARD}"
