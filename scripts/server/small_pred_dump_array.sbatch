#!/bin/bash
#SBATCH --job-name=small_pred_dump
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --exclude=n3
#SBATCH --array=0-53%12
#SBATCH --output=logs/slurm_analysis/small_pred_dump_%A_%a.out
#SBATCH --error=logs/slurm_analysis/small_pred_dump_%A_%a.err

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1
export MPLCONFIGDIR=/tmp/mpl_${SLURM_JOB_ID:-local}

mkdir -p logs/slurm_analysis

# Dump per-sequence test/val predictions for every frozen_small cell so the
# bootstrap statistics can be computed without re-running probes.
S0_CKPT=runs/small_student/small_s0_mlm/checkpoints/step_40000.pt
S1_CKPT=runs/small_student/small_s1_align_mlm/checkpoints/step_0020000.pt

CELLS=(s0_hidden s1_hidden s1_aligned)

readarray -t TASKS <<EOF
splice_donor_seqcls
splice_acceptor_seqcls
rbp_binding_seqcls
ncRNA_family_seqcls
chemical_reactivity_token
secondary_structure_pair
EOF

SEEDS=(17 41 89)
N_SEEDS=${#SEEDS[@]}
N_TASKS=${#TASKS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}
CELL_IDX=$(( TASK_ID / (N_TASKS * N_SEEDS) ))
REM=$(( TASK_ID % (N_TASKS * N_SEEDS) ))
TASK_IDX=$(( REM / N_SEEDS ))
SEED_IDX=$(( REM % N_SEEDS ))

CELL=${CELLS[$CELL_IDX]}
TASK=${TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

case "$CELL" in
    s0_hidden)  MODE=distilled_hidden;         CKPT="$S0_CKPT" ;;
    s1_hidden)  MODE=distilled_hidden;         CKPT="$S1_CKPT" ;;
    s1_aligned) MODE=distilled_hidden_aligned; CKPT="$S1_CKPT" ;;
    *) echo "unknown cell $CELL"; exit 1 ;;
esac

CELL_DIR="outputs/probes/frozen_small/${CELL}/${TASK}/${MODE}/seed_${SEED}"
if [ ! -f "${CELL_DIR}/training_summary.json" ]; then
    echo "[small-dump] cell not trained yet: ${CELL_DIR}, exiting nonzero for later rerun"
    exit 1
fi

DONE="outputs/probes/predictions_small/${CELL}/${TASK}/${MODE}/seed_${SEED}"
if ls "${DONE}"/pred_test.* >/dev/null 2>&1; then
    echo "[small-dump] skip: predictions exist for ${CELL}/${TASK}/seed_${SEED}"
    exit 0
fi

echo "[small-dump] task_id=${TASK_ID} cell=${CELL} task=${TASK} seed=${SEED} host=$(hostname) start=$(date -Is)"

python scripts/analysis/phase1_dump_predictions.py \
    --task "${TASK}" \
    --feature-mode "${MODE}" \
    --seed "${SEED}" \
    --splits val,test \
    --cell-root "outputs/probes/frozen_small/${CELL}" \
    --pred-root "outputs/probes/predictions_small/${CELL}" \
    --checkpoint "${CKPT}"

echo "[small-dump] task=${TASK} cell=${CELL} seed=${SEED} done $(date -Is)"
