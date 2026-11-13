#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."

MODEL=rinalmo_mega
START=937
END=1032
NJOBS=16
MAX_TOKENS=4096
SUB_BATCH=64
USE_EXPANDABLE=1

RANGE=$((END - START + 1))
CHUNK=$(( (RANGE + NJOBS - 1) / NJOBS ))

for i in $(seq 0 $((NJOBS - 1))); do
    s=$((START + i * CHUNK))
    e=$((s + CHUNK - 1))
    if [ "$e" -gt "$END" ]; then
        e=$END
    fi
    if [ "$s" -gt "$END" ]; then
        break
    fi
    echo "Submitting ${MODEL} shards ${s}-${e}"
    MODEL="${MODEL}" START_SHARD="${s}" END_SHARD="${e}" \
        MAX_TOKENS="${MAX_TOKENS}" SUB_BATCH="${SUB_BATCH}" USE_EXPANDABLE="${USE_EXPANDABLE}" \
        sbatch scripts/server/cache_teacher_attention_range.sbatch
done
