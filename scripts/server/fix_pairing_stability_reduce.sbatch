#!/bin/bash
#SBATCH --job-name=pairing_stab_reduce
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --exclude=n3
#SBATCH --output=logs/slurm_ablation/pairing_stab_reduce_%A.out
#SBATCH --error=logs/slurm_ablation/pairing_stab_reduce_%A.err

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}

mkdir -p logs/slurm_ablation

echo "[pairing-stab-reduce] host=$(hostname) start=$(date -Is)"
python scripts/ablation/run_pairing_stability.py --reduce --seed 17 --n-bootstrap 5
echo "[pairing-stab-reduce] done $(date -Is)"
