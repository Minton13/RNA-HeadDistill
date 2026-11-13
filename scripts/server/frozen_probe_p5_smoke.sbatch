#!/bin/bash
#SBATCH --job-name=frozen_probe_p5_smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_frozen_probes/frozen_probe_p5_smoke_%A_%a.out
#SBATCH --error=logs/slurm_frozen_probes/frozen_probe_p5_smoke_%A_%a.err

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

mkdir -p logs/slurm_frozen_probes

# Smoke test for P5 distilled_hidden_teacher on a single task/seed.
TASK="${1:-ncRNA_family_seqcls}"
SEED="${2:-17}"
OUTPUT_DIR="outputs/probes/frozen/${TASK}/distilled_hidden_teacher/seed_${SEED}"

echo "[frozen-probe-p5-smoke] task=${TASK} seed=${SEED} host=$(hostname) start=$(date -Is)"

python scripts/ablation/run_frozen_probe.py \
    --task "${TASK}" \
    --feature-mode distilled_hidden_teacher \
    --seed "${SEED}" \
    --max-training-hours 0.5

echo "[frozen-probe-p5-smoke] done $(date -Is)"
