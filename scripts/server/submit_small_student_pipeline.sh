#!/bin/bash
# Submit the small-student (from-scratch) experiment pipeline with Slurm deps.
#
#   S0a (MLM 0-20k, 6 GPU)
#     ├── pair search (1 GPU, afterok S0a)
#     │     └── S1 (align+MLM 20k, 6 GPU, afterok pair search)
#     └── S0b (MLM 20k-40k resume, 6 GPU, afterok S0a)
#
# Run only after the smoke job has passed:
#   bash scripts/server/submit_small_student_pipeline.sh
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
mkdir -p outputs/small_student

S0A=$(sbatch --parsable scripts/server/small_s0_mlm_6gpu.sbatch)
echo "S0A=$S0A"

PAIR=$(sbatch --parsable --dependency=afterok:${S0A} scripts/server/small_pair_search.sbatch)
echo "PAIR=$PAIR"

S0B=$(sbatch --parsable --dependency=afterok:${S0A} \
    --export=ALL,CONFIG=configs/head2head/small_student/mlm_s0b_6gpu.yaml,RESUME=1 \
    scripts/server/small_s0_mlm_6gpu.sbatch)
echo "S0B=$S0B"

S1=$(sbatch --parsable --dependency=afterok:${PAIR} scripts/server/small_s1_align_6gpu.sbatch)
echo "S1=$S1"

python3 - "$S0A" "$PAIR" "$S0B" "$S1" <<'EOF'
import json, sys, datetime
s0a, pair, s0b, s1 = sys.argv[1:5]
doc = {
    "submitted_at": datetime.datetime.now().isoformat(timespec="seconds"),
    "jobs": {
        "s0a_mlm_0_20k": {"id": s0a, "dep": None},
        "pair_search": {"id": pair, "dep": f"afterok:{s0a}"},
        "s0b_mlm_20k_40k": {"id": s0b, "dep": f"afterok:{s0a}"},
        "s1_align_mlm_20k": {"id": s1, "dep": f"afterok:{pair}"},
    },
}
for ext, text in [("json", json.dumps(doc, indent=2)),
                  ("md", "\n".join(
                      ["# Small-student pipeline jobs", "",
                       f"- S0a MLM 0-20k: `{s0a}`",
                       f"- pair search: `{pair}` (afterok:{s0a})",
                       f"- S0b MLM 20k-40k: `{s0b}` (afterok:{s0a})",
                       f"- S1 align+MLM 20k: `{s1}` (afterok:{pair})"]))]:
    with open(f"outputs/small_student/pipeline_jobs.{ext}", "w") as fh:
        fh.write(text + "\n")
print("wrote outputs/small_student/pipeline_jobs.{json,md}")
EOF
