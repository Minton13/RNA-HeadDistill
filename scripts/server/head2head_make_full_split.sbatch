#!/bin/bash
#SBATCH --job-name=h2h_make_full_split
#SBATCH --output=logs/slurm_head2head/make_full_split_%j.out
#SBATCH --error=logs/slurm_head2head/make_full_split_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Create the train/val/test split after the merge step.
# Optionally depend on the merge job, e.g.:
#
#     sbatch --dependency=afterok:<merge_job_id> \
#         scripts/server/head2head_make_full_split.sbatch
#
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "[make-full-split] creating train/val/test split ..."
python scripts/head2head/make_full_split.py \
    --sequence-manifest outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/sequence_manifest.parquet \
    --slim-cache-manifest outputs/teacher_attention_cache_slim/stage4b_head2head/full/manifests/slim_cache_manifest.parquet \
    --out-root outputs/teacher_attention_cache_slim/stage4b_head2head/full \
    --max-position 1024 \
    --seed 42
echo "[make-full-split] done"
