#!/bin/bash
#SBATCH --job-name=ncrna_reeval
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --exclude=n3
#SBATCH --output=logs/slurm_route_finetune/ncrna_reeval_%A.out
#SBATCH --error=logs/slurm_route_finetune/ncrna_reeval_%A.err

# Re-evaluate the 15 ncRNA_family_seqcls route-finetune cells with fixed
# multiclass metrics (softmax).  Eval-only: loads existing checkpoints.

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p logs/slurm_route_finetune

echo "[ncrna-reeval] host=$(hostname) start=$(date -Is) cells=${CELLS:-all}"
python scripts/analysis/reevaluate_route_finetune_ncrna.py ${CELLS:+--cells $CELLS}
echo "[ncrna-reeval] done $(date -Is)"
