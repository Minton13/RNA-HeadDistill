#!/bin/bash
#SBATCH --job-name=frozen_probe_array
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-125%16
#SBATCH --output=logs/slurm_frozen_probes/frozen_probe_%A_%a.out
#SBATCH --error=logs/slurm_frozen_probes/frozen_probe_%A_%a.err

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

# n3 NCCL stability settings learned from earlier Stage 4A/B jobs.
export NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1

mkdir -p logs/slurm_frozen_probes

# All downstream tasks x feature modes x seeds.  Order is task-major then
# mode-major then seed-minor so the array can be easily subset by task.
readarray -t TASKS <<EOF
splice_donor_seqcls
splice_acceptor_seqcls
rbp_binding_seqcls
ncRNA_family_seqcls
chemical_reactivity_token
secondary_structure_pair
EOF

readarray -t MODES <<EOF
vanilla_hidden
distilled_hidden
distilled_hidden_native
distilled_aligned
distilled_hidden_aligned
distilled_hidden_teacher
distilled_hidden_random_heads
EOF

SEEDS=(17 41 89)

N_MODES=${#MODES[@]}
N_SEEDS=${#SEEDS[@]}

TASK_ID=${SLURM_ARRAY_TASK_ID}
TASK_IDX=$(( TASK_ID / (N_MODES * N_SEEDS) ))
REM=$(( TASK_ID % (N_MODES * N_SEEDS) ))
MODE_IDX=$(( REM / N_SEEDS ))
SEED_IDX=$(( REM % N_SEEDS ))

TASK=${TASKS[$TASK_IDX]}
MODE=${MODES[$MODE_IDX]}
SEED=${SEEDS[$SEED_IDX]}

echo "[frozen-probe-array] task=${TASK_ID} task=${TASK} mode=${MODE} seed=${SEED} host=$(hostname) start=$(date -Is)"

python scripts/ablation/run_frozen_probe.py \
    --task "${TASK}" \
    --feature-mode "${MODE}" \
    --seed "${SEED}"

echo "[frozen-probe-array] task=${TASK_ID} done $(date -Is)"
