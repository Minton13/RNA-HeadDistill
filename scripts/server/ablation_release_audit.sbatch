#!/bin/bash
#SBATCH --job-name=h2h_release_audit
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/slurm_head2head/release_audit_%j.out
#SBATCH --error=logs/slurm_head2head/release_audit_%j.err

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1

python scripts/ablation/00_release_audit.py \
    --config configs/ablation/release_audit.yaml \
    --device cuda
