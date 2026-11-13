#!/bin/bash
#SBATCH --job-name=h2h_rebuild_seq_manifest
#SBATCH --output=logs/slurm_head2head/rebuild_seq_manifest_%j.out
#SBATCH --error=logs/slurm_head2head/rebuild_seq_manifest_%j.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Rebuild the full Stage 4B sequence manifest from the authoritative source JSONL.
# Also removes the previously-started (wrong-manifest) full slim-cache outputs so
# the pipeline starts fresh.
#
set -euo pipefail

cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

FULL_ROOT="outputs/teacher_attention_cache_slim/stage4b_head2head/full"

# Remove the partial full-slim-cache outputs that were produced from the
# old (4-shard-only) sequence manifest.  The original 6.1 TB teacher cache is
# untouched.
if [ -d "$FULL_ROOT" ]; then
    echo "[rebuild-seq-manifest] removing stale full slim-cache outputs under $FULL_ROOT"
    rm -rf "$FULL_ROOT"
fi
mkdir -p "$FULL_ROOT"

echo "[rebuild-seq-manifest] rebuilding sequence manifest from source JSONL ..."
python scripts/head2head/00b_rebuild_full_sequence_manifest.py \
    --source-jsonl outputs/task_samples/rinalmo_pretrain_10pct_len_sorted.jsonl \
    --target-manifest outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/sequence_manifest.parquet \
    --report outputs/teacher_attention_cache_slim/stage4b_head2head/reports/full_sequence_manifest_report.json

echo "[rebuild-seq-manifest] done"
