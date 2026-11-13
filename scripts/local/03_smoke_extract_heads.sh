#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${PWD}/src"
python -m rna_head_alignment.cli screen-heads --model rinalmo_micro --task secondary_structure_pair --config configs/screening.yaml --output outputs/head_screening/secondary_structure_pair/rinalmo_micro/head_atlas.csv --limit 8 --device auto
