#!/bin/bash
#SBATCH --job-name=frozen_probe_secondary_smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_frozen_probes/frozen_probe_secondary_smoke_%j.out
#SBATCH --error=logs/slurm_frozen_probes/frozen_probe_secondary_smoke_%j.err

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

OUTPUT_ROOT="outputs/probes/frozen_secondary_smoke"
rm -rf "${OUTPUT_ROOT}/secondary_structure_pair"

echo "[frozen-probe-secondary-smoke] start $(date -Is)"

python scripts/ablation/run_frozen_probe.py \
    --task secondary_structure_pair \
    --feature-mode distilled_hidden_aligned \
    --seed 17 \
    --max-epochs 3 \
    --output-root "${OUTPUT_ROOT}"

echo "[frozen-probe-secondary-smoke] done $(date -Is)"
