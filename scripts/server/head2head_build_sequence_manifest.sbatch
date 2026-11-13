#!/bin/bash
#SBATCH --job-name=h2h_seq_manifest
#SBATCH --output=logs/slurm_head2head/sequence_manifest_%j.out
#SBATCH --error=logs/slurm_head2head/sequence_manifest_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#
# Build the Stage 4B sequence manifest after the slim cache array job has
# completed.  Submit with:
#
#     sbatch scripts/server/head2head_build_sequence_manifest.sbatch
#
# or depend on the slim-cache array job, e.g.:
#
#     sbatch --dependency=afterok:<slim_cache_job_id> \
#         scripts/server/head2head_build_sequence_manifest.sbatch
#
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false

echo "[seq-manifest] building sequence manifest ..."
python scripts/head2head/00_build_sequence_manifest.py \
    --source-jsonl outputs/task_samples/rinalmo_pretrain_10pct_len_sorted.jsonl \
    --slim-cache-root outputs/teacher_attention_cache_slim/stage4b_head2head \
    --target-manifest outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/sequence_manifest.parquet \
    --report outputs/teacher_attention_cache_slim/stage4b_head2head/reports/sequence_manifest_report.json
echo "[seq-manifest] done"
