#!/bin/bash
#SBATCH --job-name=regenerate_stage5_reports
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm_frozen_probes/regenerate_stage5_reports_%j.out
#SBATCH --error=logs/slurm_frozen_probes/regenerate_stage5_reports_%j.err

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

echo "[regenerate-stage5-reports] start $(date -Is)"

python scripts/ablation/summarize_frozen_probes.py || true
python scripts/ablation/summarize_route_finetune.py || true
python scripts/ablation/summarize_multitask.py || true
python scripts/ablation/generate_tables.py || true
python scripts/ablation/generate_figures.py || true
python scripts/ablation/write_final_reports.py || true

echo "[regenerate-stage5-reports] done $(date -Is)"
