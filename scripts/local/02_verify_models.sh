#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/src"
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
python -m rna_head_alignment.cli verify-models --models configs/models.yaml --output outputs/model_checks/model_attention_check.csv --max-len 128 --device auto
