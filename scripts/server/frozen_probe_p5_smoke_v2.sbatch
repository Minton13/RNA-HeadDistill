#!/bin/bash
#SBATCH --job-name=frozen_probe_p5_smoke_v2
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=n3
#SBATCH --output=logs/slurm_frozen_probes/frozen_probe_p5_smoke_v2_%j.out
#SBATCH --error=logs/slurm_frozen_probes/frozen_probe_p5_smoke_v2_%j.err

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
export HF_HOME=$PWD/.hf_cache
export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID:-local}

mkdir -p logs/slurm_frozen_probes

# Smoke test for P5 distilled_hidden_teacher on a single task/seed, full
# training budget (no --max-training-hours cap).
TASK="${1:-splice_donor_seqcls}"
SEED="${2:-17}"

echo "[frozen-probe-p5-smoke-v2] task=${TASK} seed=${SEED} host=$(hostname) start=$(date -Is)"

python scripts/ablation/run_frozen_probe.py \
    --task "${TASK}" \
    --feature-mode distilled_hidden_teacher \
    --seed "${SEED}"

echo "[frozen-probe-p5-smoke-v2] done $(date -Is)"
