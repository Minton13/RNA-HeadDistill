#!/bin/bash
#SBATCH --job-name=systems_benchmark
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_ablation/systems_benchmark_%A.out
#SBATCH --error=logs/slurm_ablation/systems_benchmark_%A.err

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

mkdir -p logs/slurm_ablation

echo "[systems-benchmark] host=$(hostname) start=$(date -Is)"

python scripts/ablation/systems_benchmark.py \
    --registry configs/ablation/experiment_registry.yaml \
    --output-dir outputs/systems/asynchronous_distillation \
    --n-sequences 10000 \
    --seed 42 \
    --device auto

echo "[systems-benchmark] done $(date -Is)"
