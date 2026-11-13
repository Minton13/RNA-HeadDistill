#!/usr/bin/env bash
#SBATCH --job-name=rha_align
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs/student_alignment/%x_%j.out
#SBATCH --error=logs/student_alignment/%x_%j.err

set -euo pipefail
mkdir -p logs/student_alignment
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
torchrun --nproc_per_node=8 -m rna_head_alignment.cli train-alignment --config configs/student_alignment.yaml --teacher-cache outputs/teacher_attention_cache/debug_1000 --sequences outputs/task_samples/rinalmo_debug_1000.jsonl --output outputs/student_alignment/debug_1000
