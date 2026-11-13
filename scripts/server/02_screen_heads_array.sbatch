#!/usr/bin/env bash
#SBATCH --job-name=rha_screen
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem-per-gpu=60G
#SBATCH --time=24:00:00
#SBATCH --array=0-0
#SBATCH --output=logs/head_screening/%A_%a.out
#SBATCH --error=logs/head_screening/%A_%a.err

set -euo pipefail
mkdir -p logs/head_screening
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
JOB_CSV="${JOB_CSV:-outputs/head_screening/screening_jobs.csv}"
SCREENING_CONFIG="${SCREENING_CONFIG:-configs/screening.yaml}"
SCREENING_LIMIT="${SCREENING_LIMIT:-256}"
SCREENING_DEVICE="${SCREENING_DEVICE:-cuda}"
read MODEL TASK ENABLED OUTPUT_PATH <<<"$(python - <<'PY'
import csv, os
job = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0"))
path = os.environ.get("JOB_CSV", "outputs/head_screening/screening_jobs.csv")
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
r = rows[job]
print(r["model"], r["task"], r["enabled"], r["output_path"])
PY
)"
if [[ "${ENABLED}" != "True" && "${ENABLED}" != "true" ]]; then
  echo "Skipping disabled job ${MODEL}/${TASK}"
  exit 0
fi
if [[ -s "${OUTPUT_PATH}" ]]; then
  echo "Resume skip existing ${OUTPUT_PATH}"
  exit 0
fi
python -m rna_head_alignment.cli screen-heads \
  --model "${MODEL}" \
  --task "${TASK}" \
  --config "${SCREENING_CONFIG}" \
  --output "${OUTPUT_PATH}" \
  --limit "${SCREENING_LIMIT}" \
  --device "${SCREENING_DEVICE}"
