#!/bin/bash
#SBATCH --job-name=fix_smoke
#SBATCH --partition=batch
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --exclude=n3
#SBATCH --output=logs/slurm_ablation/fix_smoke_%A.out
#SBATCH --error=logs/slurm_ablation/fix_smoke_%A.err

# Smoke tests for all publication-fix paths (tiny slices / steps only).

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

mkdir -p logs/slurm_ablation
echo "[smoke] host=$(hostname) start=$(date -Is)"

echo "=== SMOKE 1: A3 cold-start training path (20 steps, tmp output) ==="
cat > /tmp/a3_coldstart_smoke_base.yaml <<'YAML'
output_root: outputs/ablation/alignment/_smoke_a3_coldstart
training:
  max_steps: 20
  eval_every: 25
  save_every: 20
  log_every: 5
YAML
python scripts/ablation/run_core_ablation.py \
    --registry configs/ablation/experiment_registry_a3_coldstart.yaml \
    --base-config /tmp/a3_coldstart_smoke_base.yaml \
    --experiment-id A3_head2head_top5_full \
    --seed 17 \
    --skip-eval && echo "SMOKE1 PASS" || echo "SMOKE1 FAIL"

echo "=== SMOKE 2: eval-only ablation path (A2 seed17, 64-row val slice) ==="
python scripts/ablation/smoke_eval_only.py \
    --run-dir outputs/ablation/alignment/A2_head2head_top3/seed_17 \
    --n-rows 64 && echo "SMOKE2 PASS" || echo "SMOKE2 FAIL"

echo "=== SMOKE 3: ncRNA re-eval (single cell) ==="
python scripts/analysis/reevaluate_route_finetune_ncrna.py \
    --cells F1/seed_17/lambda_0.0 && echo "SMOKE3 PASS" || echo "SMOKE3 FAIL"

echo "=== SMOKE 4: pairing stability (bootstrap 0, 8 samples/shard) ==="
python scripts/ablation/run_pairing_stability.py \
    --bootstrap 0 --max-samples-per-shard 8 --batch-size 4 && echo "SMOKE4 PASS" || echo "SMOKE4 FAIL"

echo "=== SMOKE 5: secondary finetune (20 steps, tmp output) ==="
python scripts/ablation/run_route_finetune.py \
    --task secondary_structure_pair \
    --feature-mode F1 \
    --seed 17 \
    --lambda-align 0.0 \
    --max-steps 20 --eval-every 20 --save-every 20 \
    --output-root outputs/probes/finetune_smoke \
    --mixed-precision --skip-eval && echo "SMOKE5 PASS" || echo "SMOKE5 FAIL"

echo "=== SMOKE 6: multitask forgetting (1 mode, 1 seed, 32 seqs) ==="
python scripts/analysis/measure_multitask_forgetting.py \
    --modes joint_no_replay --seeds 17 --max-seqs 32 --replay-size 256 && echo "SMOKE6 PASS" || echo "SMOKE6 FAIL"

echo "[smoke] all done $(date -Is)"
