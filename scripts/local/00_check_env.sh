#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/src"
export TOKENIZERS_PARALLELISM=false
python -m rna_head_alignment.cli check-env --config configs/paths.yaml
