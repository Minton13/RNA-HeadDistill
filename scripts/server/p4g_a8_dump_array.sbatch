#!/bin/bash
#SBATCH --job-name=p4g_a8_dump
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --exclude=n3
#SBATCH --array=0-17%6
#SBATCH --output=logs/slurm_analysis/p4g_a8_dump_%A_%a.out
#SBATCH --error=logs/slurm_analysis/p4g_a8_dump_%A_%a.err

set -uo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID:-local}

mkdir -p logs/slurm_analysis

TASKS=(splice_donor_seqcls splice_acceptor_seqcls rbp_binding_seqcls ncRNA_family_seqcls chemical_reactivity_token secondary_structure_pair)
SEEDS=(17 41 89)
MODE=distilled_hidden_aligned_gated

TASK_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))
TASK=${TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

CKPT=outputs/ablation/alignment/A8_adapters_only/seed_17/checkpoints/best_val.pt
CELL_ROOT=outputs/probes/frozen_a8
PRED_ROOT=outputs/probes/predictions_a8

CELL_CKPT="${CELL_ROOT}/${TASK}/${MODE}/seed_${SEED}/best_checkpoint.pt"
DONE="${PRED_ROOT}/${TASK}/${MODE}/seed_${SEED}/dump_check.json"
if [ ! -f "$CELL_CKPT" ]; then
    echo "[p4g-a8-dump][skip] no checkpoint yet: $CELL_CKPT"
    exit 0
fi
if [ -f "$DONE" ]; then
    echo "[p4g-a8-dump][skip] already dumped: $DONE"
    exit 0
fi

SPLITS="val,test"
if [ "$TASK" = "chemical_reactivity_token" ]; then
    SPLITS="test"
fi

echo "[p4g-a8-dump] task=$TASK seed=$SEED splits=$SPLITS"
python scripts/analysis/phase1_dump_predictions.py \
    --task "$TASK" \
    --feature-mode "$MODE" \
    --seed "$SEED" \
    --splits "$SPLITS" \
    --cell-root "$CELL_ROOT" \
    --pred-root "$PRED_ROOT" \
    --checkpoint "$CKPT"
echo "[p4g-a8-dump] done $(date -Is)"
