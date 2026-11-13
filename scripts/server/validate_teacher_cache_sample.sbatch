#!/usr/bin/env bash
#SBATCH --job-name=rha_tcache_validate
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/teacher_cache/%x_%j.out
#SBATCH --error=logs/teacher_cache/%x_%j.err

set -euo pipefail
mkdir -p logs/teacher_cache
source "${CONDA_PROFILE:?set CONDA_PROFILE to your conda.sh, e.g. $HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${RHA_CONDA_ENV:-rnallm}"
export PYTHONPATH="$PWD/src"
export TOKENIZERS_PARALLELISM=false

CACHE_OUTPUT="${CACHE_OUTPUT:-outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads}"
VALIDATION_MD="${VALIDATION_MD:-${CACHE_OUTPUT}/validation.md}"
export CACHE_OUTPUT VALIDATION_MD

python - <<'PY'
import os
from pathlib import Path
import pandas as pd
import torch
from rna_head_alignment.io.paths import ensure_parent, resolve_path

cache = Path(os.environ["CACHE_OUTPUT"])
validation_md = os.environ["VALIDATION_MD"]
manifest_path = cache / "manifest.csv"
manifest = pd.read_csv(manifest_path)

# Sample shards: every model gets early/middle/late; plus any shard with recorded errors.
manifest["shard"] = manifest["shard"].astype(int)
sample_rows = []
for model, group in manifest.groupby("model"):
    group = group.sort_values("shard")
    n = len(group)
    idxs = [0, n // 2, n - 1]
    sample_rows.append(group.iloc[idxs])

sample_df = pd.concat(sample_rows).drop_duplicates()
sample_paths = [Path(p) if Path(p).is_absolute() else resolve_path(p) for p in sample_df["path"].tolist()]

report = {
    "cache": str(cache),
    "manifest_exists": manifest_path.exists(),
    "shards_checked": 0,
    "bad_shards": 0,
    "n_sequences": 0,
    "n_maps": 0,
    "errors": [],
    "sample_note": "full file existence checked; tensor checks performed on a representative sample of shards",
}

# Full existence check
missing = sum(1 for p in [Path(p) if Path(p).is_absolute() else resolve_path(p) for p in manifest["path"].tolist()] if not p.exists())
if missing:
    report["bad_shards"] += missing
    report["errors"].append(f"{missing} shard files missing")

# Sample tensor check
def check_one(path: Path) -> dict:
    payload = torch.load(path, map_location="cpu")
    attn = payload.get("attention")
    mask = payload.get("valid_mask")
    if not isinstance(attn, torch.Tensor) or attn.ndim != 4:
        return {"bad": True, "error": f"{path}: attention not [B,K,L,L]", "n_seq": 0, "n_maps": 0}
    if not torch.isfinite(attn).all():
        return {"bad": True, "error": f"{path}: NaN/Inf in attention", "n_seq": 0, "n_maps": 0}
    if not isinstance(mask, torch.Tensor) or mask.shape[0] != attn.shape[0] or mask.shape[1] != attn.shape[-1]:
        return {"bad": True, "error": f"{path}: valid_mask shape mismatch", "n_seq": 0, "n_maps": 0}
    row_sums = attn.sum(dim=-1)
    valid_rows = mask[:, None, :].expand(-1, attn.shape[1], -1)
    if valid_rows.any():
        deviation = (row_sums[valid_rows] - 1.0).abs().mean().item()
        if deviation > 0.2:
            return {"bad": False, "error": f"{path}: row normalization mean deviation {deviation:.4f}", "n_seq": int(attn.shape[0]), "n_maps": int(attn.shape[0] * attn.shape[1])}
    return {"bad": False, "error": None, "n_seq": int(attn.shape[0]), "n_maps": int(attn.shape[0] * attn.shape[1])}

for path in sample_paths:
    res = check_one(path)
    report["shards_checked"] += 1
    if res["bad"]:
        report["bad_shards"] += 1
    if res["error"]:
        report["errors"].append(res["error"])
    report["n_sequences"] += res["n_seq"]
    report["n_maps"] += res["n_maps"]

ok = report["manifest_exists"] and report["bad_shards"] == 0 and report["shards_checked"] > 0
report["ok"] = bool(ok)
out = ensure_parent(Path(validation_md))
lines = [
    "# Teacher Attention Cache Validation",
    "",
    f"- cache: `{cache}`",
    f"- ok: `{report['ok']}`",
    f"- shards checked (tensor): {report['shards_checked']} (sample)",
    f"- total shard files: {len(manifest)}",
    f"- missing files: {missing}",
    f"- bad shards: {report['bad_shards']}",
    f"- sequences (sample): {report['n_sequences']}",
    f"- attention maps (sample): {report['n_maps']}",
    f"- note: {report['sample_note']}",
    "",
    "## Errors",
    "",
]
lines.extend(f"- {e}" for e in report["errors"]) if report["errors"] else lines.append("None.")
out.write_text("\n".join(lines) + "\n", encoding="utf-8")
print(report)
PY

echo "Validation report: ${VALIDATION_MD}"
