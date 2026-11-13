#!/bin/bash
#SBATCH --job-name=fix_rebuild_tables
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --exclude=n3
#SBATCH --dependency=afterok:31576:31577:31588:31589
#SBATCH --output=logs/slurm_ablation/fix_rebuild_tables_%A.out
#SBATCH --error=logs/slurm_ablation/fix_rebuild_tables_%A.err

# Publication-fix closing step (CPU-only, no training):
#   1. Rebuild paper/tables/{alignment_ablation.csv/.md, route_finetune.csv,
#      alignment_replay.csv, multitask.csv} from raw artifacts
#      (paper/scripts/rebuild_fixed_tables.py).
#   2. Verify all backfilled cells are present and flip the fix-status rows of
#      the two provenance notes to 已完成 (scripts/analysis/fix_finalize_provenance.py).
#
# Dependencies (afterok):
#   31576 a3_coldstart      (A3 from-scratch retrain x3 + val/test eval)
#   31577 fix_missing_evals (A2 evals; data_scaling/loss/teacher_count evals)
#   31588 fix_secondary_ft  (secondary_structure_pair F1/F2 x15 cells)
#   31589 multitask_forgetting (pre/post alignment forgetting)
# pairing stability (31586+31594) and ncRNA re-eval (31587) already COMPLETED
# before this job was submitted; their on-disk products are inputs here.

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

mkdir -p logs/slurm_ablation

echo "[rebuild-tables] host=$(hostname) job=${SLURM_JOB_ID} start=$(date -Is)"

python "${RHA_ROOT:-$PWD}/../paper/scripts/rebuild_fixed_tables.py"

python scripts/analysis/fix_finalize_provenance.py --job-id "${SLURM_JOB_ID}"

echo "[rebuild-tables] done $(date -Is)"
