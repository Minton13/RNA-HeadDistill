#!/usr/bin/env bash
#SBATCH --job-name=rha_vcache_sel
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/teacher_cache/%x_%j.out
#SBATCH --error=logs/teacher_cache/%x_%j.err

set -euo pipefail
mkdir -p logs/teacher_cache
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
python -m rna_head_alignment.cli validate-teacher-cache \
  --cache outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads \
  --output outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads/validation.md
