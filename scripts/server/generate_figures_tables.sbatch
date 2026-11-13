#!/bin/bash
#SBATCH --job-name=generate_figures_tables
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_pipeline/generate_figures_tables_%j.out
#SBATCH --error=logs/slurm_pipeline/generate_figures_tables_%j.err

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

echo "[figures-tables] start $(date -Is)"
python scripts/ablation/generate_figures.py
python scripts/ablation/generate_tables.py
echo "[figures-tables] done $(date -Is)"
