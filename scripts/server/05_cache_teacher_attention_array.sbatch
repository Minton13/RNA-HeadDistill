#!/usr/bin/env bash
#SBATCH --job-name=rha_tcache
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/teacher_cache/%A_%a.out
#SBATCH --error=logs/teacher_cache/%A_%a.err

set -euo pipefail
mkdir -p logs/teacher_cache
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

TEACHER_BANK="${TEACHER_BANK:-configs/teacher_bank_selected.yaml}"
SEQUENCES="${SEQUENCES:-outputs/task_samples/rinalmo_pretrain_10pct_len_sorted.jsonl}"
CACHE_OUTPUT="${CACHE_OUTPUT:-outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads}"
MODELS_YAML="${MODELS_YAML:-configs/models.yaml}"
JOB_CSV="${JOB_CSV:-outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads/teacher_cache_jobs.csv}"
OFFSET="${OFFSET:-0}"
MAX_LEN="${MAX_LEN:-512}"
SHARD_SIZE="${SHARD_SIZE:-2048}"
DTYPE="${DTYPE:-fp16}"
MAX_TOKENS_PER_BATCH="${MAX_TOKENS_PER_BATCH:-4096}"
SUB_BATCH_SIZE="${SUB_BATCH_SIZE:-64}"

read MODEL SHARD ENABLED <<<"$(python - <<'PY'
import csv, os
job = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) + int(os.environ.get("OFFSET", "0"))
path = os.environ.get("JOB_CSV", "outputs/teacher_attention_cache/teacher_cache_jobs.csv")
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if job >= len(rows):
    print("__out_of_range__ 0 false")
    raise SystemExit(0)
r = rows[job]
print(r["model"], r["shard_index"], r["enabled"])
PY
)"
if [[ "${MODEL}" == "__out_of_range__" ]]; then
  echo "Array index ${SLURM_ARRAY_TASK_ID} is outside ${JOB_CSV}; skipping."
  exit 0
fi
if [[ "${ENABLED}" != "True" && "${ENABLED}" != "true" ]]; then
  echo "Skipping disabled cache job ${MODEL}/${SHARD}"
  exit 0
fi
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
  --shard-index "${SHARD}" \
  --device cuda
