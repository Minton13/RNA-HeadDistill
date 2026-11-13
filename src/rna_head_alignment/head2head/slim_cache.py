"""Stage 4B slim teacher attention cache.

Re-packs the 9-model teacher attention cache
(``outputs/teacher_attention_cache/rinalmo_pretrain_10pct_selected_heads/``)
into one slim shard per shard index that keeps *all* attention data for the
15 preserved teacher heads but stores metadata (seq_ids / sequences /
lengths) exactly once instead of 9 times.

Schema of each slim shard ``shard_{idx:05d}.pt`` (``SLIM_SCHEMA_VERSION``):

- ``schema_version``: str
- ``shard_id``: str, e.g. ``'00000'``
- ``seq_ids``: list[str], ``sequences``: list[str], ``lengths``: int32 [B]
- ``teacher_heads_L6`` / ``_L8`` / ``_L12``: fp16 [B, T_route, Lmax, Lmax]
- ``teacher_head_ids_L6`` / ``_L8`` / ``_L12``: list[str] of head_keys,
  aligned with dim 1 of the matching tensor
- ``metadata``: dict with source_shards, channel_provenance,
  dropped_seq_ids, n_dropped, original_dtypes, Lmax, built_at

No ``valid_mask`` is stored. Channel lookup is always validated against each
source shard's ``head_keys``; nothing is assumed from position alone.

I/O hygiene: exactly one shard index is processed per call; source shards
are loaded, sliced, and released immediately. Nothing large is cached at
module level.
"""

from __future__ import annotations

import gc
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

from .head_manifest import ROUTE_DEPTHS, load_manifest

SLIM_SCHEMA_VERSION = "stage4b-v1"

REFERENCE_MODEL = "rnafm"


def _load_target_manifest(target_manifest) -> pd.DataFrame:
    if isinstance(target_manifest, (str, Path)):
        return load_manifest(target_manifest)
    return target_manifest


def _discover_models(cache_root: str | Path) -> list[str]:
    cache_root = Path(cache_root)
    return sorted(
        p.name.split("=", 1)[1]
        for p in cache_root.glob("model=*")
        if p.is_dir()
    )


def _shard_path(cache_root: Path, model: str, shard_idx: int) -> Path:
    return cache_root / f"model={model}" / f"shard_{shard_idx:05d}.pt"


def _load_shard(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def _route_tensor_key(depth: int) -> str:
    return f"teacher_heads_L{depth}"


def _route_ids_key(depth: int) -> str:
    return f"teacher_head_ids_L{depth}"


def build_slim_shard(
    shard_idx: int,
    cache_root: str | Path,
    out_root: str | Path,
    target_manifest,
    models: list[str] | None = None,
    strict: bool = True,
) -> dict:
    """Build one slim shard from the 9 model shards of the same index.

    Returns a per-shard report dict (never raises on data problems when
    ``strict=False``; missing head_keys / missing model shards are then
    skipped and recorded in the report). Raises on any violation when
    ``strict=True``. An Lmax mismatch across models is always an error.
    """
    cache_root = Path(cache_root)
    out_root = Path(out_root)
    manifest = _load_target_manifest(target_manifest)
    if models is None:
        models = _discover_models(cache_root)

    report: dict = {
        "shard_id": f"{shard_idx:05d}",
        "ok": False,
        "error": None,
        "out_path": None,
        "out_bytes": 0,
        "source_bytes": 0,
        "models_used": [],
        "n_seqs": 0,
        "n_dropped": 0,
        "dropped_seq_ids": [],
        "skipped_heads": [],
        "warnings": [],
        "route_heads": {},
    }
    try:
        # ---- Pass 1: light metadata per model (attention dropped at once). --
        metas: dict[str, dict] = {}
        for model in models:
            path = _shard_path(cache_root, model, shard_idx)
            if not path.exists():
                msg = f"missing shard file: {path}"
                if strict:
                    raise FileNotFoundError(msg)
                report["warnings"].append(msg)
                continue
            shard = _load_shard(path)
            report["source_bytes"] += path.stat().st_size
            head_keys = list(shard["head_keys"])
            metas[model] = {
                "path": path,
                "seq_ids": list(shard["seq_ids"]),
                "head_key_to_channel": {k: i for i, k in enumerate(head_keys)},
                "Lmax": int(shard["attention"].shape[-1]),
                "dtype": shard["attention"].dtype,
            }
            del shard
            gc.collect()

        if not metas:
            raise RuntimeError(f"no model shards found for shard {shard_idx}")

        # ---- Cross-model seq_id alignment (reference order preserved). ----
        ref_model = REFERENCE_MODEL if REFERENCE_MODEL in metas else next(iter(metas))
        ref_ids = metas[ref_model]["seq_ids"]
        id_sets = {m: set(meta["seq_ids"]) for m, meta in metas.items()}
        kept_ids = [s for s in ref_ids if all(s in id_sets[m] for m in metas)]
        dropped = [s for s in ref_ids if s not in set(kept_ids)]
        report["dropped_seq_ids"] = dropped
        report["n_dropped"] = len(dropped)
        if dropped:
            report["warnings"].append(
                f"dropped {len(dropped)} seq_ids not present in every model "
                f"(reference={ref_model}); see dropped_seq_ids"
            )

        # ---- Lmax consistency across models is a hard requirement. ----
        lmax_ref = metas[ref_model]["Lmax"]
        for model, meta in metas.items():
            if meta["Lmax"] != lmax_ref:
                raise ValueError(
                    f"Lmax mismatch in shard {shard_idx}: {ref_model} has "
                    f"{lmax_ref}, {model} has {meta['Lmax']}"
                )

        # ---- Pass 2: slice target channels per model, release immediately. --
        # route -> list of (head_key, [B, Lmax, Lmax] fp16 slice)
        route_slices: dict[int, list[tuple[str, torch.Tensor]]] = {
            d: [] for d in ROUTE_DEPTHS
        }
        channel_provenance: list[dict] = []
        manifest_by_model = {
            m: grp for m, grp in manifest.groupby("teacher_model", sort=False)
        }
        for model in metas:
            meta = metas[model]
            rows = manifest_by_model.get(model)
            if rows is None or len(rows) == 0:
                continue
            shard = _load_shard(meta["path"])
            id2row = {s: i for i, s in enumerate(shard["seq_ids"])}
            row_idx = torch.tensor(
                [id2row[s] for s in kept_ids], dtype=torch.long
            )
            attn = shard["attention"]
            if attn.shape[0] != len(shard["seq_ids"]):
                raise ValueError(
                    f"{meta['path']}: attention batch {attn.shape[0]} != "
                    f"seq_ids {len(shard['seq_ids'])}"
                )
            for _, row in rows.iterrows():
                head_key = row["head_key"]
                channel = meta["head_key_to_channel"].get(head_key)
                if channel is None:
                    msg = (
                        f"head_key {head_key!r} not in {meta['path']} "
                        f"head_keys {list(meta['head_key_to_channel'])}"
                    )
                    if strict:
                        raise KeyError(msg)
                    report["warnings"].append(msg)
                    report["skipped_heads"].append(head_key)
                    continue
                depth = int(row["route_depth"])
                # positional cache_channel must agree with the head_key lookup
                if int(row["cache_channel"]) != channel:
                    msg = (
                        f"cache_channel mismatch for {head_key}: manifest "
                        f"{int(row['cache_channel'])} vs shard {channel}"
                    )
                    if strict:
                        raise ValueError(msg)
                    report["warnings"].append(msg)
                route_slices[depth].append(
                    (head_key, attn[row_idx, channel].to(torch.float16))
                )
                channel_provenance.append(
                    {
                        "head_key": head_key,
                        "route_depth": depth,
                        "model": model,
                        "channel": channel,
                        "source_shard": str(meta["path"]),
                    }
                )
            report["models_used"].append(model)
            del shard, attn
            gc.collect()

        # ---- Reference metadata, stored exactly once. ----
        ref_shard = _load_shard(metas[ref_model]["path"])
        ref_id2row = {s: i for i, s in enumerate(ref_shard["seq_ids"])}
        ref_rows = [ref_id2row[s] for s in kept_ids]
        sequences = [ref_shard["sequences"][i] for i in ref_rows]
        lengths = ref_shard["lengths"][torch.tensor(ref_rows, dtype=torch.long)].to(
            torch.int32
        )
        del ref_shard
        gc.collect()

        slim: dict = {
            "schema_version": SLIM_SCHEMA_VERSION,
            "shard_id": f"{shard_idx:05d}",
            "seq_ids": kept_ids,
            "sequences": sequences,
            "lengths": lengths,
        }
        for depth in ROUTE_DEPTHS:
            slices = route_slices[depth]
            if slices:
                tensor = torch.stack([t for _, t in slices], dim=1)
            else:
                tensor = torch.zeros(
                    (len(kept_ids), 0, lmax_ref, lmax_ref), dtype=torch.float16
                )
            slim[_route_tensor_key(depth)] = tensor
            slim[_route_ids_key(depth)] = [k for k, _ in slices]
            report["route_heads"][f"L{depth}"] = len(slices)
            if tensor.numel() > 0:
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    report["warnings"].append(
                        f"NaN/Inf present in teacher_heads_L{depth} "
                        f"(kept as-is from fp16 source)"
                    )

        slim["metadata"] = {
            "source_shards": {m: str(metas[m]["path"]) for m in metas},
            "channel_provenance": channel_provenance,
            "dropped_seq_ids": dropped,
            "n_dropped": len(dropped),
            "original_dtypes": {m: str(metas[m]["dtype"]) for m in metas},
            "Lmax": lmax_ref,
            "reference_model": ref_model,
            "built_at": datetime.now(timezone.utc).isoformat(),
        }

        out_root.mkdir(parents=True, exist_ok=True)
        out_path = out_root / f"shard_{shard_idx:05d}.pt"
        torch.save(slim, out_path)

        report["ok"] = True
        report["out_path"] = str(out_path)
        report["out_bytes"] = out_path.stat().st_size
        report["n_seqs"] = len(kept_ids)
    except Exception as exc:  # noqa: BLE001 - report, don't swallow
        report["error"] = f"{type(exc).__name__}: {exc}"
        if strict:
            raise
    finally:
        gc.collect()
    return report


def verify_slim_equivalence(
    shard_idx: int,
    cache_root: str | Path,
    slim_path: str | Path,
    target_manifest,
) -> list[dict]:
    """Bitwise (fp16) check that each slim channel equals its source channel.

    Reloads the original model shards for ``shard_idx`` and compares each
    target head's slim slice against the original attention channel with
    ``torch.equal``. Returns one result dict per manifest head.
    """
    cache_root = Path(cache_root)
    manifest = _load_target_manifest(target_manifest)
    slim = _load_shard(Path(slim_path))
    slim_ids = list(slim["seq_ids"])

    results: list[dict] = []
    for model, rows in manifest.groupby("teacher_model", sort=False):
        src_path = _shard_path(cache_root, model, shard_idx)
        shard = _load_shard(src_path)
        head_key_to_channel = {k: i for i, k in enumerate(shard["head_keys"])}
        id2row = {s: i for i, s in enumerate(shard["seq_ids"])}
        missing_in_src = [s for s in slim_ids if s not in id2row]
        row_idx = torch.tensor(
            [id2row[s] for s in slim_ids if s in id2row], dtype=torch.long
        )
        for _, row in rows.iterrows():
            head_key = row["head_key"]
            depth = int(row["route_depth"])
            res = {
                "shard_id": slim["shard_id"],
                "head_key": head_key,
                "route_depth": depth,
                "model": model,
                "channel_in_orig": head_key_to_channel.get(head_key),
                "channel_in_slim": None,
                "equal": False,
                "note": "",
            }
            slim_ids_list = slim[_route_ids_key(depth)]
            if res["channel_in_orig"] is None:
                res["note"] = "head_key missing in source shard"
            elif head_key not in slim_ids_list:
                res["note"] = "head_key missing in slim shard"
            elif missing_in_src:
                res["note"] = (
                    f"{len(missing_in_src)} slim seq_ids missing in source shard"
                )
            else:
                j = slim_ids_list.index(head_key)
                res["channel_in_slim"] = j
                orig = shard["attention"][row_idx, res["channel_in_orig"]].to(
                    torch.float16
                )
                res["equal"] = bool(
                    torch.equal(slim[_route_tensor_key(depth)][:, j], orig)
                )
                if not res["equal"]:
                    res["note"] = "tensor mismatch"
            results.append(res)
        del shard
        gc.collect()
    del slim
    gc.collect()
    return results


def iter_shard_indices(manifest_csv_or_dir: str | Path) -> list[int]:
    """List available shard indices.

    - If given a directory (cache root): union of ``shard_*.pt`` indices
      found under ``model=*`` subdirectories.
    - If given a CSV: unique values of the ``shard_idx`` / ``shard_id`` /
      ``shard`` column (digits extracted from strings like ``'00012'``).
    """
    p = Path(manifest_csv_or_dir)
    if p.is_dir():
        idxs: set[int] = set()
        for mdir in p.glob("model=*"):
            for f in mdir.glob("shard_*.pt"):
                idxs.add(int(f.stem.split("_", 1)[1]))
        return sorted(idxs)
    df = pd.read_csv(p)
    for col in ("shard_idx", "shard_id", "shard"):
        if col in df.columns:
            digits = df[col].astype(str).str.extract(r"(\d+)")[0].dropna()
            return sorted(set(digits.astype(int)))
    raise ValueError(f"no shard column found in {p}")
