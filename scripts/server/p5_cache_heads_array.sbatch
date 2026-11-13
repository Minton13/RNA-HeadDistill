#!/bin/bash
#SBATCH --job-name=p5_cache_heads
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --array=0-59%6
#SBATCH --exclude=n3
#SBATCH --output=logs/slurm_downstream/p5_cache_heads_%A_%a.out
#SBATCH --error=logs/slurm_downstream/p5_cache_heads_%A_%a.err

# Fill the P5 downstream exact-teacher cache: one array task per
# (route_depth, model, shard group) row of p5_cache_jobs.csv.
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
# Some compute nodes cannot write to the home filesystem; redirect caches.
export HF_HOME=$PWD/.hf_cache
export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID:-local}

mkdir -p logs/slurm_downstream

JOB_CSV=outputs/downstream_teacher_fill/p5_cache_jobs.csv
read ROUTE MODEL SHARD_START SHARD_END <<<"$(python - <<'PY'
import csv, os
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
with open("outputs/downstream_teacher_fill/p5_cache_jobs.csv", newline="") as f:
    rows = list(csv.DictReader(f))
r = rows[idx]
print(r["route_depth"], r["model"], r["shard_start"], r["shard_end"])
PY
)"

echo "[p5-cache-heads] task=${SLURM_ARRAY_TASK_ID} route=L${ROUTE} model=${MODEL} shards=${SHARD_START}-${SHARD_END} host=$(hostname) start=$(date -Is)"

python scripts/ablation/cache_downstream_teacher_heads.py \
    --route-depth "${ROUTE}" \
    --model "${MODEL}" \
    --shard-start "${SHARD_START}" \
    --shard-end "${SHARD_END}" \
    --device cuda

echo "[p5-cache-heads] task=${SLURM_ARRAY_TASK_ID} done $(date -Is)"
