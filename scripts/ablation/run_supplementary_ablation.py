#!/usr/bin/env python
"""CLI wrapper for supplementary ablations.

Example (single run):
    PYTHONPATH=src python scripts/ablation/run_supplementary_ablation.py \
        --ablation-type data_scaling --variant 100000 --seed 17

Array use (inside Slurm):
    python scripts/ablation/run_supplementary_ablation.py \
        --ablation-type "$ABLATION_TYPE" --variant "$VARIANT" --seed "$SEED"
"""

from __future__ import annotations

from rna_head_alignment.ablation.supplementary import main

if __name__ == "__main__":
    main()
