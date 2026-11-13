#!/bin/bash
#SBATCH --job-name=p1_dump
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=n3
#SBATCH --array=0-17%6
#SBATCH --output=logs/slurm_analysis/p1_dump_%A_%a.out
#SBATCH --error=logs/slurm_analysis/p1_dump_%A_%a.err

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
MODES=(vanilla_hidden distilled_hidden distilled_hidden_native distilled_aligned distilled_hidden_aligned distilled_hidden_random_heads distilled_hidden_teacher)

TASK_IDX=$((SLURM_ARRAY_TASK_ID / 3))
SEED_IDX=$((SLURM_ARRAY_TASK_ID % 3))
TASK=${TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

SPLITS="val,test"
if [ "$TASK" = "chemical_reactivity_token" ]; then
    # Pearson needs no val threshold calibration; val split is large.
    SPLITS="test"
fi

echo "[p1-dump] task=$TASK seed=$SEED splits=$SPLITS array_task=$SLURM_ARRAY_TASK_ID"

FAILED=0
for MODE in "${MODES[@]}"; do
    CKPT="outputs/probes/frozen/${TASK}/${MODE}/seed_${SEED}/best_checkpoint.pt"
    DONE="outputs/probes/predictions/${TASK}/${MODE}/seed_${SEED}/dump_check.json"
    if [ ! -f "$CKPT" ]; then
        echo "[p1-dump][skip] missing checkpoint: $CKPT"
        continue
    fi
    if [ -f "$DONE" ]; then
        echo "[p1-dump][skip] already dumped: $DONE"
        continue
    fi
    echo "[p1-dump] dumping $TASK/$MODE/seed_$SEED"
    python scripts/analysis/phase1_dump_predictions.py \
        --task "$TASK" \
        --feature-mode "$MODE" \
        --seed "$SEED" \
        --splits "$SPLITS" || { echo "[p1-dump][FAIL] $TASK/$MODE/seed_$SEED"; FAILED=1; }
done

if [ "$FAILED" -ne 0 ]; then
    echo "[p1-dump] finished with failures task=$TASK seed=$SEED"
    exit 1
fi
echo "[p1-dump] done task=$TASK seed=$SEED $(date -Is)"
