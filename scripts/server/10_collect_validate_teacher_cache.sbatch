#!/usr/bin/env bash
#SBATCH --job-name=rha_tcache_validate
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/teacher_cache/%x_%j.out
#SBATCH --error=logs/teacher_cache/%x_%j.err

set -euo pipefail
mkdir -p logs/teacher_cache
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
export TOKENIZERS_PARALLELISM=false

CACHE_OUTPUT="${CACHE_OUTPUT:-outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads}"
VALIDATION_MD="${VALIDATION_MD:-${CACHE_OUTPUT}/validation.md}"

python -m rna_head_alignment.cli collect-teacher-cache-manifests \
  --cache "${CACHE_OUTPUT}" \
  --output "${CACHE_OUTPUT}/manifest.csv"

python -m rna_head_alignment.cli validate-teacher-cache \
  --cache "${CACHE_OUTPUT}" \
  --output "${VALIDATION_MD}"

echo "Validation report: ${VALIDATION_MD}"
