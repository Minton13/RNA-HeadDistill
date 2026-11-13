#!/bin/bash
# Submit cleaned frozen-probe and multi-task jobs after the NaN/time-limit fixes.
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

mkdir -p outputs/ablation/reports

FROZEN_JOB=$(sbatch --parsable scripts/server/frozen_probe_array_v2.sbatch)
echo "Submitted frozen probe array v2: ${FROZEN_JOB}"

MULTITASK_JOB=$(sbatch --parsable scripts/server/multitask_training.sbatch)
echo "Submitted multitask training: ${MULTITASK_JOB}"

REPORT_JOB=$(sbatch --parsable \
    --dependency=afterany:${FROZEN_JOB},${MULTITASK_JOB} \
    scripts/server/regenerate_stage5_reports.sbatch)
echo "Submitted report regeneration (afterany frozen/multitask): ${REPORT_JOB}"

cat > outputs/ablation/reports/fixed_pipeline_jobs.json <<EOF
{
  "frozen_probe_array_v2": "${FROZEN_JOB}",
  "multitask_training": "${MULTITASK_JOB}",
  "regenerate_reports": "${REPORT_JOB}"
}
EOF

echo "All jobs submitted. IDs written to outputs/ablation/reports/fixed_pipeline_jobs.json"
