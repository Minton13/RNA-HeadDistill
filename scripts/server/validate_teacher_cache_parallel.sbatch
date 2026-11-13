#!/usr/bin/env bash
#SBATCH --job-name=rha_tcache_validate
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
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
N_WORKERS="${N_WORKERS:-8}"
export CACHE_OUTPUT VALIDATION_MD N_WORKERS

python - <<'PY'
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import torch
from rna_head_alignment.io.paths import ensure_parent, resolve_path

cache = Path(os.environ["CACHE_OUTPUT"])
validation_md = os.environ["VALIDATION_MD"]
n_workers = int(os.environ["N_WORKERS"])
manifest_path = cache / "manifest.csv"
manifest = pd.read_csv(manifest_path)

paths = []
for _, row in manifest.iterrows():
    p = Path(str(row.get("path", "")))
    if not p.is_absolute():
        p = resolve_path(p)
    paths.append(p)

def check_one(path: Path) -> dict:
    if not path.exists():
        return {"bad": True, "error": f"missing shard: {path}", "n_seq": 0, "n_maps": 0}
    try:
        payload = torch.load(path, map_location="cpu")
    except Exception as e:
        return {"bad": True, "error": f"{path}: load error {repr(e)}", "n_seq": 0, "n_maps": 0}
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

report = {
    "cache": str(cache),
    "manifest_exists": manifest_path.exists(),
    "shards_checked": 0,
    "bad_shards": 0,
    "n_sequences": 0,
    "n_maps": 0,
    "errors": [],
}

with ThreadPoolExecutor(max_workers=n_workers) as exe:
    futures = {exe.submit(check_one, p): p for p in paths}
    for fut in as_completed(futures):
        res = fut.result()
        report["shards_checked"] += 1
        if res["bad"]:
            report["bad_shards"] += 1
        if res["error"]:
            report["errors"].append(res["error"])
        report["n_sequences"] += res["n_seq"]
        report["n_maps"] += res["n_maps"]
        if report["shards_checked"] % 500 == 0:
            print(f"checked {report['shards_checked']}/{len(paths)} shards", flush=True)

ok = report["manifest_exists"] and report["bad_shards"] == 0 and report["shards_checked"] > 0
report["ok"] = bool(ok)
out = ensure_parent(Path(validation_md))
lines = [
    "# Teacher Attention Cache Validation",
    "",
    f"- cache: `{cache}`",
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
