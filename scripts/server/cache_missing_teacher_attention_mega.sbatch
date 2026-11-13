#!/usr/bin/env bash
#SBATCH --job-name=rha_tcache_mega
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=120G
#SBATCH --time=12:00:00
#SBATCH --exclude=n2
#SBATCH --output=logs/teacher_cache/mega_missing_%j.out
#SBATCH --error=logs/teacher_cache/mega_missing_%j.err

set -euo pipefail
mkdir -p logs/teacher_cache
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

TEACHER_BANK="configs/teacher_bank_selected.yaml"
SEQUENCES="outputs/task_samples/rinalmo_pretrain_10pct_len_sorted.jsonl"
CACHE_OUTPUT="outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads"
MODELS_YAML="configs/models.yaml"
MAX_LEN=512
SHARD_SIZE=2048
DTYPE="fp16"

echo "[$(date)] Starting rinalmo_mega missing shards on a single GPU"
for shard in 924 925 $(seq 927 1032); do
    python -m rna_head_alignment.cli cache-teacher-attention \
        --teacher-bank "${TEACHER_BANK}" \
        --sequences "${SEQUENCES}" \
        --output "${CACHE_OUTPUT}" \
        --models "${MODELS_YAML}" \
        --max-len "${MAX_LEN}" \
        --shard-size "${SHARD_SIZE}" \
        --dtype "${DTYPE}" \
        --max-tokens-per-batch 2048 \
        --sub-batch-size 32 \
        --model rinalmo_mega \
        --shard-index "${shard}" \
        --device cuda
done
echo "[$(date)] Finished rinalmo_mega missing shards"
