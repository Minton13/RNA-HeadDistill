#!/bin/bash
#SBATCH --job-name=h2h_merge_full_slim
#SBATCH --output=logs/slurm_head2head/merge_full_slim_%j.out
#SBATCH --error=logs/slurm_head2head/merge_full_slim_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# Merge Stage 4B full slim cache partial manifests after the array job.
# Optionally depend on the build array job, e.g.:
#
#     sbatch --dependency=afterok:<build_job_id> \
#         scripts/server/head2head_merge_full_slim_cache.sbatch
#
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "[merge-full-slim] merging partial manifests ..."
python scripts/head2head/merge_full_slim_manifest.py \
    --out-root outputs/teacher_attention_cache_slim/stage4b_head2head/full
echo "[merge-full-slim] done"
