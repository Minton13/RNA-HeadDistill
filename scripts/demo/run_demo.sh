#!/usr/bin/env bash
# Reviewer demo: toy-data screening -> teacher-head selection -> small attention
# cache -> cache validation, end to end on CPU in a few minutes.
#
# Usage (from the repository root):
#   bash scripts/demo/run_demo.sh
#
# All outputs go to outputs/demo/ (deleted and rebuilt on each run).
# Set DEVICE=cuda to run on a GPU; CPU is the default and suffices.

set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

DEVICE="${DEVICE:-cpu}"
DEMO_OUT=outputs/demo
MODELS=configs/demo/models.yaml
TASKS=configs/demo/tasks.yaml
SCREEN_CFG=configs/demo/screening.yaml
PY=${PYTHON:-python}

echo "=== [1/7] toy data ==="
"$PY" scripts/demo/make_toy_data.py --output-dir data/demo

echo "=== [2/7] check demo model ==="
if [[ ! -f assets/models/rinalmo-micro/model.safetensors ]]; then
  echo "assets/models/rinalmo-micro not found; downloading multimolecule/rinalmo-micro ..."
  "$PY" scripts/demo/fetch_demo_model.py
fi

rm -rf "$DEMO_OUT"
mkdir -p "$DEMO_OUT/head_screening"

echo "=== [3/7] verify model ==="
"$PY" -m rna_head_alignment.cli verify-models \
  --models "$MODELS" \
  --output "$DEMO_OUT/model_checks/model_attention_check.csv" \
  --max-len 96 --device "$DEVICE"

echo "=== [4/7] head screening (2 jobs: 1 model x 2 tasks) ==="
"$PY" -m rna_head_alignment.cli make-screening-jobs \
  --models "$MODELS" --tasks "$TASKS" \
  --model-check "$DEMO_OUT/model_checks/model_attention_check.csv" \
  --output "$DEMO_OUT/head_screening/screening_jobs.csv" \
  --output-root "$DEMO_OUT/head_screening" --include-all-models
"$PY" -m rna_head_alignment.cli run-screening-jobs \
  --jobs "$DEMO_OUT/head_screening/screening_jobs.csv" \
  --config "$SCREEN_CFG" --models "$MODELS" --tasks "$TASKS" \
  --report "$DEMO_OUT/head_screening/screening_run_report.csv" \
  --device "$DEVICE" --no-resume

echo "=== [5/7] collect atlas + heatmaps ==="
"$PY" -m rna_head_alignment.cli collect-head-screening \
  --input "$DEMO_OUT/head_screening/*/*/head_atlas.csv" \
  --output "$DEMO_OUT/head_screening/all_head_atlas.csv"
"$PY" -m rna_head_alignment.cli plot-head-heatmaps \
  --atlas "$DEMO_OUT/head_screening/all_head_atlas.csv" \
  --tasks "$TASKS" --output-dir "$DEMO_OUT/head_screening/heatmaps"

echo "=== [6/7] propose teacher heads (supervised selection) ==="
"$PY" -m rna_head_alignment.cli propose-teacher-heads \
  --atlas "$DEMO_OUT/head_screening/all_head_atlas.csv" \
  --config "$SCREEN_CFG" \
  --output "$DEMO_OUT/teacher_banks/teacher_head_candidates.csv" \
  --yaml "$DEMO_OUT/teacher_banks/teacher_bank.yaml"

echo "=== [7/7] cache selected teacher attention + validate ==="
"$PY" -m rna_head_alignment.cli cache-teacher-attention \
  --teacher-bank "$DEMO_OUT/teacher_banks/teacher_bank.yaml" \
  --models "$MODELS" \
  --sequences data/demo/toy_secondary_structure.jsonl \
  --output "$DEMO_OUT/teacher_attention_cache" \
  --max-len 96 --shard-size 16 --device "$DEVICE"
"$PY" -m rna_head_alignment.cli validate-teacher-cache \
  --cache "$DEMO_OUT/teacher_attention_cache" \
  --output "$DEMO_OUT/teacher_attention_cache_validation.md"

echo
echo "Demo finished. Key artifacts:"
echo "  $DEMO_OUT/head_screening/all_head_atlas.csv            (per-head probe scores)"
echo "  $DEMO_OUT/head_screening/heatmaps/                     (head-score heatmaps)"
echo "  $DEMO_OUT/teacher_banks/teacher_bank.yaml              (selected teacher heads)"
echo "  $DEMO_OUT/teacher_attention_cache/                     (cached attention maps)"
echo "  $DEMO_OUT/teacher_attention_cache_validation.md        (cache report)"
