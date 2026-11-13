#!/bin/bash
#SBATCH --job-name=route_finetune_array
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --array=0-89%12
#SBATCH --output=logs/slurm_route_finetune/route_finetune_%A_%a.out
#SBATCH --error=logs/slurm_route_finetune/route_finetune_%A_%a.err

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

mkdir -p logs/slurm_route_finetune

# Tasks x (F1 no-replay + F2 with 4 lambdas) x seeds.
# F0 (frozen) is covered by the frozen probe array, so it is omitted here.
readarray -t TASKS <<EOF
splice_donor_seqcls
splice_acceptor_seqcls
rbp_binding_seqcls
ncRNA_family_seqcls
chemical_reactivity_token
secondary_structure_pair
EOF

SEEDS=(17 41 89)
LAMBDAS=(0.0 0.01 0.1 0.5)

N_SEEDS=${#SEEDS[@]}
N_CONFIGS=$((1 + ${#LAMBDAS[@]}))  # F1 + 4 F2 lambdas
PER_TASK=$((N_CONFIGS * N_SEEDS))

TASK_ID=${SLURM_ARRAY_TASK_ID}
TASK_IDX=$(( TASK_ID / PER_TASK ))
REM=$(( TASK_ID % PER_TASK ))
CONFIG_IDX=$(( REM / N_SEEDS ))
SEED_IDX=$(( REM % N_SEEDS ))

TASK=${TASKS[$TASK_IDX]}
SEED=${SEEDS[$SEED_IDX]}

# CONFIG_IDX 0 = F1 (no replay); 1-4 = F2 with lambdas.
if [ "${CONFIG_IDX}" -eq 0 ]; then
    MODE="F1"
    LAMBDA=0.0
else
    MODE="F2"
    LAMBDA_IDX=$(( CONFIG_IDX - 1 ))
    LAMBDA=${LAMBDAS[$LAMBDA_IDX]}
fi

echo "[route-finetune-array] task=${TASK_ID} task=${TASK} mode=${MODE} seed=${SEED} lambda=${LAMBDA} host=$(hostname) start=$(date -Is)"

python scripts/ablation/run_route_finetune.py \
    --task "${TASK}" \
    --feature-mode "${MODE}" \
    --seed "${SEED}" \
    --lambda-align "${LAMBDA}" \
    --mixed-precision

echo "[route-finetune-array] task=${TASK_ID} done $(date -Is)"
