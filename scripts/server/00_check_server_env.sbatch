#!/usr/bin/env bash
#SBATCH --job-name=rha_env
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=40G
#SBATCH --time=00:30:00
#SBATCH --output=logs/env/%x_%j.out
#SBATCH --error=logs/env/%x_%j.err

set -euo pipefail
mkdir -p logs/env
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
python -m rna_head_alignment.cli check-env --config configs/paths.yaml
