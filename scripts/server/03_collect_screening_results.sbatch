#!/usr/bin/env bash
#SBATCH --job-name=rha_collect
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/head_screening/%x_%j.out
#SBATCH --error=logs/head_screening/%x_%j.err

set -euo pipefail
mkdir -p logs/head_screening
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
python -m rna_head_alignment.cli collect-head-screening --input "outputs/head_screening/*/*/head_atlas.csv" --output outputs/head_screening/all_head_atlas.csv
python -m rna_head_alignment.cli plot-head-heatmaps \
  --atlas outputs/head_screening/all_head_atlas.csv \
  --tasks configs/tasks.yaml \
  --output-dir outputs/head_screening/heatmaps \
  --vmin 0.0 \
  --vmax 1.0
