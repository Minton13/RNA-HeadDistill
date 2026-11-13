#!/usr/bin/env bash
#SBATCH --job-name=rha_vcache
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/teacher_cache/%x_%j.out
#SBATCH --error=logs/teacher_cache/%x_%j.err

set -euo pipefail
mkdir -p logs/teacher_cache
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
python -m rna_head_alignment.cli validate-teacher-cache --cache outputs/teacher_attention_cache/debug_1000 --output outputs/teacher_attention_cache/debug_1000_validation.md
