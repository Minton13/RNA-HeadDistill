#!/bin/bash
#SBATCH --job-name=replay_ablation
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:0
#SBATCH --mem=16G
#SBATCH --time=2:00:00
#SBATCH --output=logs/slurm_pipeline/replay_ablation_%j.out
#SBATCH --error=logs/slurm_pipeline/replay_ablation_%j.err

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

echo "[replay-ablation] start $(date -Is)"
python scripts/ablation/run_alignment_replay_ablation.py
echo "[replay-ablation] done $(date -Is)"
