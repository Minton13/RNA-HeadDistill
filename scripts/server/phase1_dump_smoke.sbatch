#!/bin/bash
#SBATCH --job-name=p1_dump_smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --exclude=n3
#SBATCH --output=logs/slurm_analysis/p1_dump_smoke_%j.out
#SBATCH --error=logs/slurm_analysis/p1_dump_smoke_%j.err

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1

mkdir -p logs/slurm_analysis

# Smoke: smallest task (ncRNA, test=162), one gated mode + vanilla.
python scripts/analysis/phase1_dump_predictions.py \
    --task ncRNA_family_seqcls \
    --feature-mode distilled_hidden_aligned \
    --seed 17 \
    --splits test

python scripts/analysis/phase1_dump_predictions.py \
    --task ncRNA_family_seqcls \
    --feature-mode vanilla_hidden \
    --seed 17 \
    --splits test

echo "[p1-dump-smoke] done $(date -Is)"
