#!/bin/bash
#SBATCH --job-name=p4g_a8
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=96:00:00
#SBATCH --exclude=n3
#SBATCH --array=0-17%6
#SBATCH --output=logs/slurm_frozen_probes/p4g_a8_%A_%a.out
#SBATCH --error=logs/slurm_frozen_probes/p4g_a8_%A_%a.err

set -euo pipefail
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

mkdir -p logs/slurm_frozen_probes

# P4g probes on the A8 (frozen-backbone adapters_only) student:
# hidden states are vanilla by construction, so any downstream gain must come
# from the aligned branch alone.
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

TASK_ID=${SLURM_ARRAY_TASK_ID}
TASK_IDX=$(( TASK_ID / N_SEEDS ))
SEED_IDX=$(( TASK_ID % N_SEEDS ))

TASK=${TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}
MODE=distilled_hidden_aligned_gated
CKPT=outputs/ablation/alignment/A8_adapters_only/seed_17/checkpoints/best_val.pt
OUTROOT=outputs/probes/frozen_a8

DONE="${OUTROOT}/${TASK}/${MODE}/seed_${SEED}/training_summary.json"
if [ -f "$DONE" ]; then
    echo "[p4g-a8] skip: already trained $TASK seed_$SEED"
    exit 0
fi

echo "[p4g-a8] task_id=${TASK_ID} task=${TASK} seed=${SEED} ckpt=${CKPT} host=$(hostname) start=$(date -Is)"

python scripts/ablation/run_frozen_probe.py \
    --task "${TASK}" \
    --feature-mode "${MODE}" \
    --seed "${SEED}" \
    --checkpoint "${CKPT}" \
    --output-root "${OUTROOT}" \
    --max-training-hours 90

echo "[p4g-a8] task=${TASK} seed=${SEED} done $(date -Is)"
