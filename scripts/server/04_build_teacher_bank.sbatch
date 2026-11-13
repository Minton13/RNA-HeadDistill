#!/usr/bin/env bash
#SBATCH --job-name=rha_bank
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/teacher_banks/%x_%j.out
#SBATCH --error=logs/teacher_banks/%x_%j.err

set -euo pipefail
mkdir -p logs/teacher_banks
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
python -m rna_head_alignment.cli propose-teacher-heads --atlas outputs/head_screening/all_head_atlas.csv --config configs/screening.yaml --output outputs/teacher_banks/teacher_head_candidates.csv --yaml outputs/teacher_banks/teacher_bank_template.yaml
