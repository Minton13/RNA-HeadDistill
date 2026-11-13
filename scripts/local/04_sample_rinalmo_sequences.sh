#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/src"
CORPUS="${1:?Usage: 04_sample_rinalmo_sequences.sh /path/to/rinalmo_corpus}"
python -m rna_head_alignment.cli sample-rinalmo-corpus --corpus "${CORPUS}" --n 1000 --min-len 32 --max-len 512 --deduplicate true --output outputs/task_samples/rinalmo_debug_1000.jsonl
