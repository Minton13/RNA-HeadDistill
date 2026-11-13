#!/bin/bash
#SBATCH --job-name=frozen_probe_hpv2_smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_frozen_probes/frozen_probe_hpv2_smoke_%j.out
#SBATCH --error=logs/slurm_frozen_probes/frozen_probe_hpv2_smoke_%j.err

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

# Smoke test for the hpv2 expanded optimizer grid: single cell, 2 epochs,
# throwaway output root so nothing production is touched.
echo "[hpv2-smoke] start $(date -Is) host=$(hostname)"

python scripts/ablation/run_frozen_probe.py \
    --task ncRNA_family_seqcls \
    --feature-mode distilled_hidden_aligned \
    --seed 17 \
    --max-epochs 2 \
    --max-training-hours 1.0 \
    --lrs 1e-4,1e-3 \
    --wds 1e-5,1e-4 \
    --output-root outputs/probes/frozen_hpv2_smoke

echo "[hpv2-smoke] done $(date -Is)"
