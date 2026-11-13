#!/usr/bin/env bash
#SBATCH --job-name=rha_sample10pct
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/rinalmo_sampling/%x_%j.out
#SBATCH --error=logs/rinalmo_sampling/%x_%j.err

set -euo pipefail
mkdir -p logs/rinalmo_sampling outputs/task_samples
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
export TOKENIZERS_PARALLELISM=false
export PYTHONUTF8=1

RINALMO_CORPUS="${RINALMO_CORPUS:-assets/data/rinalmo_pretraining_corpus}"
SEQUENCE_COLUMN="${SEQUENCE_COLUMN:-sequence}"
OUTPUT="${OUTPUT:-outputs/task_samples/rinalmo_pretrain_10pct.jsonl}"
FRACTION="${FRACTION:-0.10}"
MIN_LEN="${MIN_LEN:-32}"
MAX_LEN="${MAX_LEN:-512}"

if [[ ! -e "${RINALMO_CORPUS}" ]]; then
  echo "Missing RiNALMo pretraining corpus: ${RINALMO_CORPUS}" >&2
  echo "Set RINALMO_CORPUS=/path/to/RiNALMo/pretraining/corpus and resubmit." >&2
  exit 2
fi

python -m rna_head_alignment.cli sample-rinalmo-corpus \
  --corpus "${RINALMO_CORPUS}" \
  --fraction "${FRACTION}" \
  --min-len "${MIN_LEN}" \
  --max-len "${MAX_LEN}" \
  --deduplicate true \
  --sequence-column "${SEQUENCE_COLUMN}" \
  --output "${OUTPUT}"

echo "Sample written to ${OUTPUT}"
echo "Manifest written to ${OUTPUT}.manifest.json"
