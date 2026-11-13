#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/src"
OLD_ROOT="${1:-../rna_circuit_distill}"
python -m rna_head_alignment.cli migrate-assets --old-root "${OLD_ROOT}"
