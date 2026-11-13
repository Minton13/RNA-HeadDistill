#!/usr/bin/env bash
#SBATCH --job-name=rha_models
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=60G
#SBATCH --time=04:00:00
#SBATCH --output=logs/model_checks/%x_%j.out
#SBATCH --error=logs/model_checks/%x_%j.err

set -euo pipefail
mkdir -p logs/model_checks
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python -m rna_head_alignment.cli verify-models --models configs/models.yaml --output outputs/model_checks/model_attention_check.csv --max-len 128 --device cuda --include-large
