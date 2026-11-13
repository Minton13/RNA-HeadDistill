#!/bin/bash
#SBATCH --job-name=supplementary_ablation_array
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --array=0-20%6
#SBATCH --output=logs/slurm_ablation/supplementary_%A_%a.out
#SBATCH --error=logs/slurm_ablation/supplementary_%A_%a.err

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

mkdir -p logs/slurm_ablation

# Flattened (ablation_type, variant, seed) combos in the order:
#   data_scaling: 3 sizes x 3 seeds
#   loss_ablation: 4 variants x 1 seed
#   teacher_count_ablation: 3 counts x 1 seed
#   pairing_stability: 5 bootstraps x 1 seed
readarray -t AB_TYPES <<EOF
data_scaling
data_scaling
data_scaling
data_scaling
data_scaling
data_scaling
data_scaling
data_scaling
data_scaling
loss_ablation
loss_ablation
loss_ablation
loss_ablation
teacher_count_ablation
teacher_count_ablation
teacher_count_ablation
pairing_stability
pairing_stability
pairing_stability
pairing_stability
pairing_stability
EOF

readarray -t VARIANTS <<EOF
100000
100000
100000
500000
500000
500000
2114289
2114289
2114289
kl_only
mse_only
kl_mse
kl_mse_cos
1
3
5
0
1
2
3
4
EOF

readarray -t SEEDS <<EOF
17
41
89
17
41
89
17
41
89
17
17
17
17
17
17
17
17
17
17
17
17
EOF

TASK_ID=${SLURM_ARRAY_TASK_ID}
ABLATION_TYPE=${AB_TYPES[$TASK_ID]}
VARIANT=${VARIANTS[$TASK_ID]}
SEED=${SEEDS[$TASK_ID]}

echo "[supplementary-array] task=${TASK_ID} type=${ABLATION_TYPE} variant=${VARIANT} seed=${SEED} host=$(hostname) start=$(date -Is)"

python scripts/ablation/run_supplementary_ablation.py \
    --registry configs/ablation/experiment_registry.yaml \
    --ablation-type "${ABLATION_TYPE}" \
    --variant "${VARIANT}" \
    --seed "${SEED}"

echo "[supplementary-array] task=${TASK_ID} done $(date -Is)"
