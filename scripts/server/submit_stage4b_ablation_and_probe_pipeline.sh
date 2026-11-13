#!/bin/bash
# Master orchestration script for the full Stage 4B ablation + Stage 5 probe pipeline.
#
# Usage:
#   bash scripts/server/submit_stage4b_ablation_and_probe_pipeline.sh
#
# Environment:
#   RELEASE_AUDIT_JOBID   - if set, pipeline depends on this existing release audit job.
#   SKIP_RELEASE_AUDIT    - if set to "1", do not submit or depend on release audit.
#
# The script estimates total GPU-hours and wall time, writes the plan to
# outputs/ablation/planned_compute.md, submits jobs with dependencies, and
# records job IDs in outputs/ablation/pipeline_jobs.json and pipeline_jobs.md.

set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"

source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"

export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export HF_ENDPOINT=https://hf-mirror.com

mkdir -p outputs/ablation logs/slurm_pipeline

# ---------------------------------------------------------------------------
# 1. Query sacct for the existing Stage 4B full run duration.
# ---------------------------------------------------------------------------
FULL_RUN_JOB_NAME="head2head_train_full"
FULL_RUN_ELAPSED_SEC=0
if command -v sacct &>/dev/null; then
    # Try to find the most recent completed full run.
    readarray -t ELAPSED_LINES < <(sacct -n -P -X --format=JobName,Elapsed --name="${FULL_RUN_JOB_NAME}*" 2>/dev/null | grep -E "^${FULL_RUN_JOB_NAME}" | cut -d'|' -f2 || true)
    if [ ${#ELAPSED_LINES[@]} -gt 0 ]; then
        # Parse the first elapsed time H:M:S or D-H:M:S.
        elapsed="${ELAPSED_LINES[0]}"
        if [[ "$elapsed" =~ ^([0-9]+)-([0-9]+):([0-9]+):([0-9]+)$ ]]; then
            days=${BASH_REMATCH[1]}
            hours=${BASH_REMATCH[2]}
            mins=${BASH_REMATCH[3]}
            secs=${BASH_REMATCH[4]}
            FULL_RUN_ELAPSED_SEC=$(( days*86400 + hours*3600 + mins*60 + secs ))
        elif [[ "$elapsed" =~ ^([0-9]+):([0-9]+):([0-9]+)$ ]]; then
            hours=${BASH_REMATCH[1]}
            mins=${BASH_REMATCH[2]}
            secs=${BASH_REMATCH[3]}
            FULL_RUN_ELAPSED_SEC=$(( hours*3600 + mins*60 + secs ))
        fi
    fi
fi

# ---------------------------------------------------------------------------
# 2. Estimate total GPU-hours and wall time.
# ---------------------------------------------------------------------------
# Rough estimates per job array / step.  These are intentionally conservative.
GPU_HOURS_CORE_ABLATION=216       # 27 array tasks x 1 GPU x 8 h
GPU_HOURS_SUPP_ABLATION=144       # 18 array tasks x 1 GPU x 8 h
GPU_HOURS_SYSTEMS_BENCH=4
GPU_HOURS_CACHE_FORMAT=4
GPU_HOURS_DOWNSTREAM_PREP=0       # CPU job
GPU_HOURS_DOWNSTREAM_CACHE=16     # CPU/IO bound, small GPU
GPU_HOURS_FROZEN_PROBE=48         # 126 array tasks x 1 GPU x ~0.4 h
GPU_HOURS_ROUTE_FINETUNE=180      # 90 array tasks x 1 GPU x 2 h
GPU_HOURS_REPLAY_ABLATION=0       # aggregation, depends on route finetune
GPU_HOURS_MULTITASK=144           # 6 array tasks x 1 GPU x 24 h
GPU_HOURS_FIGURES_TABLES=0        # CPU
GPU_HOURS_FINAL_REPORTS=0         # CPU

TOTAL_GPU_HOURS=$(cat <<EOF | awk '{s+=$1} END {print s}'
${GPU_HOURS_CORE_ABLATION}
${GPU_HOURS_SUPP_ABLATION}
${GPU_HOURS_SYSTEMS_BENCH}
${GPU_HOURS_CACHE_FORMAT}
${GPU_HOURS_DOWNSTREAM_CACHE}
${GPU_HOURS_FROZEN_PROBE}
${GPU_HOURS_ROUTE_FINETUNE}
${GPU_HOURS_MULTITASK}
EOF
)

WALL_DAYS_ESTIMATE=14

{
    echo "# Planned Compute for Stage 4B Ablation + Stage 5 Probe Pipeline"
    echo ""
    echo "- Stage 4B full run elapsed seconds: ${FULL_RUN_ELAPSED_SEC}"
    echo "- Estimated total GPU-hours: ${TOTAL_GPU_HOURS}"
    echo "- Estimated end-to-end wall time: ~${WALL_DAYS_ESTIMATE} days (with % concurrency limits)"
    echo ""
    echo "| step | gpu_hours | notes |"
    echo "|---|---|---|"
    echo "| core_ablation_array | ${GPU_HOURS_CORE_ABLATION} | A0-A8, 27 array tasks |"
    echo "| supplementary_ablation_array | ${GPU_HOURS_SUPP_ABLATION} | data/loss/teacher-count/pairing |"
    echo "| systems_benchmark | ${GPU_HOURS_SYSTEMS_BENCH} | async distillation cost |"
    echo "| cache_format_benchmark | ${GPU_HOURS_CACHE_FORMAT} | I/O format comparison |"
    echo "| downstream_dataset_prep | ${GPU_HOURS_DOWNSTREAM_PREP} | CPU prep |"
    echo "| downstream_teacher_cache | ${GPU_HOURS_DOWNSTREAM_CACHE} | route cache extraction |"
    echo "| frozen_probe_array | ${GPU_HOURS_FROZEN_PROBE} | P0-P6 probes |"
    echo "| route_finetune_array | ${GPU_HOURS_ROUTE_FINETUNE} | F1/F2 per task |"
    echo "| replay_ablation | ${GPU_HOURS_REPLAY_ABLATION} | aggregation |"
    echo "| multitask_training | ${GPU_HOURS_MULTITASK} | joint models |"
    echo "| figures_tables | ${GPU_HOURS_FIGURES_TABLES} | CPU plotting |"
    echo "| final_reports | ${GPU_HOURS_FINAL_REPORTS} | CPU report writing |"
} > outputs/ablation/planned_compute.md

echo "[pipeline] planned compute written to outputs/ablation/planned_compute.md"

# ---------------------------------------------------------------------------
# 3. Submit jobs with dependencies.
# ---------------------------------------------------------------------------
JOB_IDS=()
DEP_STR=""

submit_or_record() {
    local name=$1
    local script=$2
    local deps=$3
    local jid
    if [ -n "${deps}" ]; then
        jid=$(sbatch --dependency="${deps}" "${script}" | awk '{print $4}')
    else
        jid=$(sbatch "${script}" | awk '{print $4}')
    fi
    JOB_IDS+=("${name}:${jid}")
    echo "[pipeline] ${name} -> ${jid}"
}

# 1. Release audit (use existing or submit new).
if [ "${SKIP_RELEASE_AUDIT:-0}" = "1" ]; then
    echo "[pipeline] skipping release audit (SKIP_RELEASE_AUDIT=1)"
    AUDIT_DEP=""
elif [ -n "${RELEASE_AUDIT_JOBID:-}" ]; then
    echo "[pipeline] using existing release audit job ${RELEASE_AUDIT_JOBID}"
    JOB_IDS+=("release_audit:${RELEASE_AUDIT_JOBID}")
    AUDIT_DEP="afterok:${RELEASE_AUDIT_JOBID}"
else
    submit_or_record "release_audit" "scripts/server/ablation_release_audit.sbatch" ""
    AUDIT_DEP="afterok:${JOB_IDS[-1]##*:}"
fi

# Helper to build dependency string from the most recent job name.
dep_after() {
    local target=$1
    for entry in "${JOB_IDS[@]}"; do
        if [[ "$entry" == "${target}:"* ]]; then
            echo "afterok:${entry##*:}"
            return
        fi
    done
    echo ""
}

# 2. Core ablation array.
CORE_DEP=""
if [ -n "${AUDIT_DEP}" ]; then CORE_DEP="${AUDIT_DEP}"; fi
submit_or_record "core_ablation_array" "scripts/server/ablation_core_array.sbatch" "${CORE_DEP}"
CORE_DEP="afterok:${JOB_IDS[-1]##*:}"

# 3. Supplementary ablation array (depends on core ablation A3 checkpoint).
submit_or_record "supplementary_ablation_array" "scripts/server/supplementary_ablation_array.sbatch" "${CORE_DEP}"

# 4. Systems benchmark (independent of ablations but after audit).
SYS_DEP="${AUDIT_DEP}"
[ -z "${SYS_DEP}" ] && SYS_DEP=""
submit_or_record "systems_benchmark" "scripts/server/systems_benchmark.sbatch" "${SYS_DEP}"

# 5. Cache format benchmark (after audit).
submit_or_record "cache_format_benchmark" "scripts/server/cache_format_benchmark.sbatch" "${SYS_DEP}"

# 6. Downstream dataset prep (CPU, after audit).
submit_or_record "downstream_dataset_prep" "scripts/server/prepare_downstream_datasets.sbatch" "${AUDIT_DEP}"
DS_DEP="afterok:${JOB_IDS[-1]##*:}"

# 7. Downstream teacher cache (after dataset prep and audit).
submit_or_record "downstream_teacher_cache" "scripts/server/build_downstream_teacher_cache.sbatch" "${DS_DEP}"
CACHE_DEP="afterok:${JOB_IDS[-1]##*:}"

# 8. Frozen probe array (after downstream datasets and teacher cache).
FROZEN_DEP="${DS_DEP},${CACHE_DEP}"
submit_or_record "frozen_probe_array" "scripts/server/frozen_probe_array.sbatch" "${FROZEN_DEP}"
FROZEN_DEP="afterok:${JOB_IDS[-1]##*:}"

# 9. Route fine-tune array (after core ablation and downstream data).
ROUTE_DEP="${CORE_DEP},${DS_DEP}"
submit_or_record "route_finetune_array" "scripts/server/route_finetune_array.sbatch" "${ROUTE_DEP}"
ROUTE_DEP="afterok:${JOB_IDS[-1]##*:}"

# 10. Replay ablation aggregation (after route fine-tune).
submit_or_record "replay_ablation" "scripts/server/replay_ablation.sbatch" "${ROUTE_DEP}"
REPLAY_DEP="afterok:${JOB_IDS[-1]##*:}"

# 11. Multi-task training (after core ablation and downstream data).
MULTI_DEP="${CORE_DEP},${DS_DEP}"
submit_or_record "multitask_training" "scripts/server/multitask_training.sbatch" "${MULTI_DEP}"
MULTI_DEP="afterok:${JOB_IDS[-1]##*:}"

# 12. Figures/tables (after frozen probes, route finetune, multitask, systems).
FIG_DEP="${FROZEN_DEP},${ROUTE_DEP},${MULTI_DEP}"
# Include systems if it exists.
for entry in "${JOB_IDS[@]}"; do
    if [[ "$entry" == "systems_benchmark:"* ]]; then
        FIG_DEP="${FIG_DEP},afterok:${entry##*:}"
    fi
done
submit_or_record "figures_tables" "scripts/server/generate_figures_tables.sbatch" "${FIG_DEP}"
FIG_DEP="afterok:${JOB_IDS[-1]##*:}"

# 13. Final reports (after figures/tables and replay ablation).
REPORT_DEP="${FIG_DEP},${REPLAY_DEP}"
submit_or_record "final_reports" "scripts/server/write_final_reports.sbatch" "${REPORT_DEP}"

# ---------------------------------------------------------------------------
# 4. Write pipeline_jobs.json and pipeline_jobs.md.
# ---------------------------------------------------------------------------
{
    echo "{"
    echo '  "pipeline": "stage4b_ablation_and_probe_pipeline",'
    echo '  "submitted_at": "'$(date -Is)'",'
    echo '  "full_run_elapsed_sec": '${FULL_RUN_ELAPSED_SEC}','
    echo '  "estimated_gpu_hours": '${TOTAL_GPU_HOURS}','
    echo '  "jobs": ['
    n=${#JOB_IDS[@]}
    for i in "${!JOB_IDS[@]}"; do
        name=${JOB_IDS[$i]%%:*}
        jid=${JOB_IDS[$i]##*:}
        comma=$([ $i -lt $((n-1)) ] && echo "," || echo "")
        echo '    {"name": "'${name}'", "job_id": "'${jid}'"}'"${comma}"
    done
    echo '  ]'
    echo "}"
} > outputs/ablation/pipeline_jobs.json

{
    echo "# Stage 4B Ablation + Stage 5 Probe Pipeline Jobs"
    echo ""
    echo "Submitted: $(date -Is)"
    echo ""
    echo "| step | job_id |"
    echo "|---|---|"
    for entry in "${JOB_IDS[@]}"; do
        name=${entry%%:*}
        jid=${entry##*:}
        echo "| ${name} | ${jid} |"
    done
} > outputs/ablation/pipeline_jobs.md

echo "[pipeline] wrote outputs/ablation/pipeline_jobs.json"
echo "[pipeline] wrote outputs/ablation/pipeline_jobs.md"
echo "[pipeline] submit complete. Use 'squeue -u $USER' to monitor."
