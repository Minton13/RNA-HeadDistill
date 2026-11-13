#!/bin/bash
#SBATCH --job-name=p5_merge_shards
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-12%6
#SBATCH --exclude=n3
#SBATCH --output=logs/slurm_downstream/p5_merge_shards_%A_%a.out
#SBATCH --error=logs/slurm_downstream/p5_merge_shards_%A_%a.err

# Merge per-model partials into final downstream-teacher-cache-v1 shards.
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

mkdir -p logs/slurm_downstream

read ROUTE SHARD_START SHARD_END <<<"$(python - <<'PY'
import csv, os
idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
with open("outputs/downstream_teacher_fill/p5_merge_jobs.csv", newline="") as f:
    rows = list(csv.DictReader(f))
r = rows[idx]
print(r["route_depth"], r["shard_start"], r["shard_end"])
PY
)"

echo "[p5-merge] task=${SLURM_ARRAY_TASK_ID} route=L${ROUTE} shards=${SHARD_START}-${SHARD_END} host=$(hostname) start=$(date -Is)"

python scripts/ablation/merge_downstream_teacher_shards.py \
    --route-depth "${ROUTE}" \
    --shard-start "${SHARD_START}" \
    --shard-end "${SHARD_END}"

echo "[p5-merge] task=${SLURM_ARRAY_TASK_ID} done $(date -Is)"
