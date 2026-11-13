#!/usr/bin/env bash
#SBATCH --job-name=rha_tcache_val
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/teacher_cache/validate_%x_%j.out
#SBATCH --error=logs/teacher_cache/validate_%x_%j.err

set -euo pipefail
mkdir -p logs/teacher_cache
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
export TOKENIZERS_PARALLELISM=false

MODEL="${MODEL:?MODEL is required}"
CACHE_OUTPUT="${CACHE_OUTPUT:-outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads}"
VALIDATION_MD="${VALIDATION_MD:-${CACHE_OUTPUT}/validation_${MODEL}.md}"
export MODEL CACHE_OUTPUT VALIDATION_MD

python - <<'PY'
import os
from pathlib import Path
import pandas as pd
import torch
from rna_head_alignment.io.paths import ensure_parent, resolve_path

cache = Path(os.environ["CACHE_OUTPUT"])
model = os.environ["MODEL"]
validation_md = os.environ["VALIDATION_MD"]
manifest_path = cache / "manifest.csv"
manifest = pd.read_csv(manifest_path)
manifest = manifest[manifest["model"] == model]
report = {
    "cache": str(cache),
    "model": model,
    "manifest_exists": manifest_path.exists(),
    "shards_checked": 0,
    "bad_shards": 0,
    "n_sequences": 0,
    "n_maps": 0,
    "errors": [],
}
for _, row in manifest.iterrows():
    path = Path(str(row.get("path", "")))
    if not path.is_absolute():
        path = resolve_path(path)
    if not path.exists():
        report["bad_shards"] += 1
        report["errors"].append(f"missing shard: {path}")
        continue
    payload = torch.load(path, map_location="cpu")
    report["shards_checked"] += 1
    attn = payload.get("attention")
    mask = payload.get("valid_mask")
    if not isinstance(attn, torch.Tensor) or attn.ndim != 4:
        report["bad_shards"] += 1
        report["errors"].append(f"{path}: attention not [B,K,L,L]")
        continue
    if not torch.isfinite(attn).all():
        report["bad_shards"] += 1
        report["errors"].append(f"{path}: NaN/Inf in attention")
    if not isinstance(mask, torch.Tensor) or mask.shape[0] != attn.shape[0] or mask.shape[1] != attn.shape[-1]:
        report["bad_shards"] += 1
        report["errors"].append(f"{path}: valid_mask shape mismatch")
    row_sums = attn.sum(dim=-1)
    valid_rows = mask[:, None, :].expand(-1, attn.shape[1], -1) if isinstance(mask, torch.Tensor) else torch.ones_like(row_sums, dtype=torch.bool)
    if valid_rows.any():
        deviation = (row_sums[valid_rows] - 1.0).abs().mean().item()
        if deviation > 0.2:
            report["errors"].append(f"{path}: row normalization mean deviation {deviation:.4f}")
    report["n_sequences"] += int(attn.shape[0])
    report["n_maps"] += int(attn.shape[0] * attn.shape[1])

ok = report["manifest_exists"] and report["bad_shards"] == 0 and report["shards_checked"] > 0
report["ok"] = bool(ok)
out = ensure_parent(Path(validation_md))
lines = [
    f"# Teacher Attention Cache Validation ({model})",
    "",
    f"- cache: `{cache}`",
    f"- model: `{model}`",
    f"- ok: `{report['ok']}`",
    f"- shards checked: {report['shards_checked']}",
    f"- bad shards: {report['bad_shards']}",
    f"- sequences: {report['n_sequences']}",
    f"- attention maps: {report['n_maps']}",
    "",
    "## Errors",
    "",
]
lines.extend(f"- {e}" for e in report["errors"]) if report["errors"] else lines.append("None.")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report)
PY

echo "Validation report: ${VALIDATION_MD}"
