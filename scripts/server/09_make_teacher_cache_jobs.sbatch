#!/usr/bin/env bash
#SBATCH --job-name=rha_tcache_jobs
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/teacher_cache/%x_%j.out
#SBATCH --error=logs/teacher_cache/%x_%j.err

set -euo pipefail
mkdir -p logs/teacher_cache outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
export TOKENIZERS_PARALLELISM=false

TEACHER_BANK="${TEACHER_BANK:-configs/teacher_bank_selected.yaml}"
SEQUENCES="${SEQUENCES:-outputs/task_samples/rinalmo_pretrain_10pct_len_sorted.jsonl}"
CACHE_OUTPUT="${CACHE_OUTPUT:-outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads}"
JOB_CSV="${JOB_CSV:-${CACHE_OUTPUT}/teacher_cache_jobs.csv}"
SHARD_SIZE="${SHARD_SIZE:-2048}"

if [[ ! -s "${SEQUENCES}" ]]; then
  echo "Missing sampled sequence file: ${SEQUENCES}" >&2
  echo "Run scripts/server/08_sample_rinalmo_10pct.sbatch first." >&2
  exit 2
fi

python -m rna_head_alignment.cli make-teacher-cache-jobs \
  --teacher-bank "${TEACHER_BANK}" \
  --sequences "${SEQUENCES}" \
  --output "${JOB_CSV}" \
  --shard-size "${SHARD_SIZE}"

N_JOBS="$(python - <<'PY'
import csv, os
path = os.environ["JOB_CSV"]
with open(path, newline="", encoding="utf-8") as f:
    n = sum(1 for _ in csv.DictReader(f))
print(n)
PY
)"
LAST=$((N_JOBS - 1))
echo "Wrote ${N_JOBS} cache jobs to ${JOB_CSV}"
echo "Submit cache inference with:"
echo "  JOB_CSV=${JOB_CSV} SEQUENCES=${SEQUENCES} CACHE_OUTPUT=${CACHE_OUTPUT} SHARD_SIZE=${SHARD_SIZE} sbatch --array=0-${LAST} scripts/server/05_cache_teacher_attention_array.sbatch"
