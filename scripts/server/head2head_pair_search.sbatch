#!/bin/bash
#SBATCH --job-name=h2h_pair_search
#SBATCH --output=logs/slurm_head2head/pair_search_%j.out
#SBATCH --error=logs/slurm_head2head/pair_search_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=60G
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

python scripts/head2head/05_head_pair_search.py \
    --config configs/head2head/head_pair_search.yaml
