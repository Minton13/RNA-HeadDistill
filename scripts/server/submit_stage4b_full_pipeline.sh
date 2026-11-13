#!/bin/bash
# Submit the full Stage 4B head-to-head pipeline with Slurm dependencies.
#
# Chain:
#   1. build full slim cache (array)
#   2. merge partial manifests
#   3. create train/val/test split
#   4. train student (6-GPU DDP)
#   5. evaluate on val/test
#   6. write final Stage 4B report
#
# Usage:
#     bash scripts/server/submit_stage4b_full_pipeline.sh
#
set -euo pipefail

PROJECT_ROOT="${RHA_ROOT:-$PWD}"
cd "$PROJECT_ROOT"

REPORT_DIR="outputs/alignment/stage4b_head2head/reports"
LOG_DIR="logs/slurm_head2head"
mkdir -p "$REPORT_DIR" "$LOG_DIR"

SUBMIT_TIME=$(date -Is)

echo "[submit-stage4b-full] submitting pipeline from $PROJECT_ROOT at $SUBMIT_TIME"

REBUILD_ID=$(sbatch --parsable scripts/server/head2head_rebuild_sequence_manifest.sbatch)
echo "[submit-stage4b-full] rebuild manifest job: $REBUILD_ID"

BUILD_ID=$(sbatch --parsable --dependency=afterok:"$REBUILD_ID" scripts/server/head2head_build_full_slim_cache.sbatch)
echo "[submit-stage4b-full] build array job: $BUILD_ID (afterok:$REBUILD_ID)"

MERGE_ID=$(sbatch --parsable --dependency=afterok:"$BUILD_ID" scripts/server/head2head_merge_full_slim_cache.sbatch)
echo "[submit-stage4b-full] merge job: $MERGE_ID (afterok:$BUILD_ID)"

SPLIT_ID=$(sbatch --parsable --dependency=afterok:"$MERGE_ID" scripts/server/head2head_make_full_split.sbatch)
echo "[submit-stage4b-full] split job: $SPLIT_ID (afterok:$MERGE_ID)"

TRAIN_ID=$(sbatch --parsable --dependency=afterok:"$SPLIT_ID" scripts/server/head2head_train_full_6gpu.sbatch)
echo "[submit-stage4b-full] train job: $TRAIN_ID (afterok:$SPLIT_ID)"

EVAL_ID=$(sbatch --parsable --dependency=afterok:"$TRAIN_ID" scripts/server/head2head_eval_full.sbatch)
echo "[submit-stage4b-full] eval job: $EVAL_ID (afterok:$TRAIN_ID)"

REPORT_ID=$(sbatch --parsable --dependency=afterok:"$EVAL_ID" scripts/server/head2head_write_full_report.sbatch)
echo "[submit-stage4b-full] report job: $REPORT_ID (afterok:$EVAL_ID)"

# ---------------------------------------------------------------------------
# Write dependency record (JSON + Markdown)
# ---------------------------------------------------------------------------
JOBS_JSON="$REPORT_DIR/full_pipeline_jobs.json"
JOBS_MD="$REPORT_DIR/full_pipeline_jobs.md"

cat > "$JOBS_JSON" <<EOF
{
  "submitted_at": "$SUBMIT_TIME",
  "project_root": "$PROJECT_ROOT",
  "jobs": {
    "rebuild_manifest": "$REBUILD_ID",
    "build_array": "$BUILD_ID",
    "merge": "$MERGE_ID",
    "split": "$SPLIT_ID",
    "train": "$TRAIN_ID",
    "eval": "$EVAL_ID",
    "report": "$REPORT_ID"
  },
  "dependencies": {
    "build_array": "afterok:$REBUILD_ID",
    "merge": "afterok:$BUILD_ID",
    "split": "afterok:$MERGE_ID",
    "train": "afterok:$SPLIT_ID",
    "eval": "afterok:$TRAIN_ID",
    "report": "afterok:$EVAL_ID"
  }
}
EOF

cat > "$JOBS_MD" <<EOF
# Stage 4B full pipeline job submission

- submitted at: \`$SUBMIT_TIME\`
- project root: \`$PROJECT_ROOT\`

## Job IDs

| step | job id | dependency |
|---|---|---|---
| rebuild manifest | \`$REBUILD_ID\` | — |
| build array | \`$BUILD_ID\` | afterok:$REBUILD_ID |
| merge | \`$MERGE_ID\` | afterok:$BUILD_ID |
| split | \`$SPLIT_ID\` | afterok:$MERGE_ID |
| train | \`$TRAIN_ID\` | afterok:$SPLIT_ID |
| eval | \`$EVAL_ID\` | afterok:$TRAIN_ID |
| report | \`$REPORT_ID\` | afterok:$EVAL_ID |

## Check status

\`\`\`bash
squeue -j $REBUILD_ID,$BUILD_ID,$MERGE_ID,$SPLIT_ID,$TRAIN_ID,$EVAL_ID,$REPORT_ID
\`\`\`
EOF

echo "[submit-stage4b-full] pipeline submitted. job records:"
echo "  JSON: $JOBS_JSON"
echo "  Markdown: $JOBS_MD"
echo "[submit-stage4b-full] job ids: $REBUILD_ID $BUILD_ID $MERGE_ID $SPLIT_ID $TRAIN_ID $EVAL_ID $REPORT_ID"
