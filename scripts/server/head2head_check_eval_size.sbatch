#!/bin/bash
#SBATCH --job-name=h2h_check_eval_size
#SBATCH --output=logs/slurm_head2head/check_eval_size_%j.out
#SBATCH --error=logs/slurm_head2head/check_eval_size_%j.err
#SBATCH --time=00:10:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --exclude=n1
set -euo pipefail
cd "${RHA_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH=$PWD/src:${PYTHONPATH:-}
python - <<'PY'
import pandas as pd
from rna_head_alignment.head2head.full_sequence_dataset import LengthBucketedSampler

cfg = {
    "max_pair_tokens": 524288,
    "max_sequences_per_batch": 8,
    "bucket_size_samples": 1000,
}
for split in ["val", "test"]:
    df = pd.read_parquet(f"outputs/teacher_attention_cache_slim/stage4b_head2head/full/index/{split}.parquet")
    lengths = df["length"].tolist()
    sampler = LengthBucketedSampler(
        lengths,
        max_pair_tokens=cfg["max_pair_tokens"],
        max_sequences_per_batch=cfg["max_sequences_per_batch"],
        bucket_size_samples=cfg["bucket_size_samples"],
        rank=0,
        world_size=1,
        seed=42,
        shuffle=False,
        drop_last=False,
    )
    batches = list(sampler)
    n_batches = len(batches)
    n_rows = sum(len(b) for b in batches) * 15 * 8 * 4  # seqs * heads * metrics * baselines
    print(f"{split}: sequences={len(lengths)} batches={n_batches} approx_rows={n_rows}")
PY
