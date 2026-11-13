from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch

from rna_head_alignment.io.paths import ensure_parent, resolve_path
from rna_head_alignment.cache.build_teacher_attention_cache import collect_teacher_cache_manifests


def validate_attention_cache(cache: str | Path, output: str | Path) -> dict[str, Any]:
    root = resolve_path(cache)
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists() and (root / "manifest_parts").exists():
        collect_teacher_cache_manifests(root, manifest_path)
    report: dict[str, Any] = {
        "cache": str(root),
        "manifest_exists": manifest_path.exists(),
        "shards_checked": 0,
        "bad_shards": 0,
        "n_sequences": 0,
        "n_maps": 0,
        "errors": [],
    }
    if not manifest_path.exists():
        report["errors"].append("manifest.csv missing")
    else:
        manifest = pd.read_csv(manifest_path) if manifest_path.stat().st_size else pd.DataFrame()
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
    out = ensure_parent(output)
    lines = [
        "# Teacher Attention Cache Validation",
        "",
        f"- cache: `{root}`",
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
    return report
