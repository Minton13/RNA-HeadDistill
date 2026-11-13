#!/bin/bash
#SBATCH --job-name=prep_downstream_datasets
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_downstream/prep_datasets_%j.out
#SBATCH --error=logs/slurm_downstream/prep_datasets_%j.err

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p logs/slurm_downstream

echo "[prep-datasets] start $(date -Is) host=$(hostname)"

python scripts/ablation/prepare_downstream_datasets.py \
    --config configs/ablation/downstream_data.yaml

echo "[prep-datasets] done $(date -Is)"
