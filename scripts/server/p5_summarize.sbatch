#!/bin/bash
#SBATCH --job-name=p5_summarize
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --exclude=n3
#SBATCH --output=logs/slurm_frozen_probes/p5_summarize_%j.out
#SBATCH --error=logs/slurm_frozen_probes/p5_summarize_%j.err

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TOKENIZERS_PARALLELISM=false

mkdir -p logs/slurm_frozen_probes

python scripts/analysis/p5_summarize.py
