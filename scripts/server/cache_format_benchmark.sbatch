#!/bin/bash
#SBATCH --job-name=cache_format_benchmark
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm_ablation/cache_format_benchmark_%A.out
#SBATCH --error=logs/slurm_ablation/cache_format_benchmark_%A.err

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p logs/slurm_ablation

echo "[cache-format-benchmark] host=$(hostname) start=$(date -Is)"

python scripts/ablation/cache_format_benchmark.py \
    --cache-root outputs/teacher_attention_cache_slim/stage4b_head2head/full \
    --sequence-manifest outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/sequence_manifest.parquet \
    --bucket-manifest outputs/teacher_attention_cache_slim/stage4b_head2head/full/manifests/slim_cache_manifest.parquet \
    --output-dir outputs/systems/cache_format_benchmark \
    --n-sequences 10000 \
    --seed 42

echo "[cache-format-benchmark] done $(date -Is)"
