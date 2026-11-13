"""Cache-format benchmark helpers.

Compares four formats on a deterministic 10,000-sequence subset:
  C0: existing bucketed fp16 safetensors
  C1: ragged flat fp16 chunks
  C2: chunked zarr compressed fp16 (falls back to npz if zarr missing)
  C3: row-wise uint8 quantized with per-row scale
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file

from rna_head_alignment.head2head.head_manifest import ROUTE_DEPTHS
from rna_head_alignment.head2head.losses import route_alignment_loss
from rna_head_alignment.io.paths import resolve_path

ROUTE_KEYS = tuple(f"L{d}" for d in ROUTE_DEPTHS)


def _sample_manifest_rows(
    sequence_manifest_path: str | Path,
    n: int,
    seed: int,
) -> pd.DataFrame:
    df = pd.read_parquet(resolve_path(sequence_manifest_path)).reset_index().rename(columns={"index": "manifest_idx"})
    if len(df) <= n:
        return df.copy()
    rng = np.random.default_rng(seed)
    return df.sample(n=n, random_state=rng).copy()


def _build_manifest_index_to_bucket(
    cache_root: Path,
    bucket_manifest: pd.DataFrame,
) -> dict[int, tuple[Path, int]]:
    """Map sequence_manifest row index -> (bucket_path, position_in_bucket)."""
    mapping: dict[int, tuple[Path, int]] = {}
    cache_root = Path(cache_root)
    for _, row in bucket_manifest.iterrows():
        file_path = row["file_path"]
        bucket_path = cache_root / file_path
        if not bucket_path.exists():
            # Try alternative relative to cwd.
            alt = Path.cwd() / file_path
            if alt.exists():
                bucket_path = alt
            else:
                continue
        try:
            with safe_open(str(bucket_path), framework="pt") as f:
                indices = f.get_tensor("sequence_manifest_indices").to(torch.int64).numpy()
        except Exception:
            continue
        for pos, idx in enumerate(indices):
            if idx >= 0:
                mapping[int(idx)] = (bucket_path, pos)
    return mapping


def _read_route_heads(bucket_path: Path) -> dict[int, list[str]]:
    with safe_open(str(bucket_path), framework="pt") as f:
        metadata = f.metadata() or {}
    return {
        depth: json.loads(metadata.get(f"route_heads_L{depth}", "[]"))
        for depth in ROUTE_DEPTHS
    }


def _pad_maps_to_batch(maps: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Pad a list of [T, L, L] tensors to [B, T, Lmax, Lmax] and return lengths."""
    lengths = torch.tensor([m.shape[-1] for m in maps], dtype=torch.long)
    max_l = int(lengths.max())
    batch = torch.zeros(len(maps), maps[0].shape[0], max_l, max_l, dtype=maps[0].dtype)
    for b, m in enumerate(maps):
        l = m.shape[-1]
        batch[b, :, :l, :l] = m
    return batch, lengths


def iter_sample_records(
    cache_root: str | Path,
    sequence_manifest_path: str | Path,
    bucket_manifest_path: str | Path,
    n: int = 10000,
    seed: int = 42,
    batch_size: int = 1024,
) -> Iterator[list[dict[str, Any]]]:
    """Yield chunks of sampled records from the full bucketed slim cache.

    Each record is ``{"seq_id": str, "length": int, "routes": {depth: [T,L,L]}}``.
    """
    cache_root = resolve_path(cache_root)
    sample_df = _sample_manifest_rows(sequence_manifest_path, n, seed)
    bucket_manifest = pd.read_parquet(resolve_path(bucket_manifest_path))
    mapping = _build_manifest_index_to_bucket(cache_root, bucket_manifest)

    records: list[dict[str, Any]] = []
    for _, row in sample_df.iterrows():
        manifest_idx = int(row["manifest_idx"])
        if manifest_idx not in mapping:
            continue
        bucket_path, position = mapping[manifest_idx]
        records.append({
            "seq_id": str(row["seq_id"]),
            "length": int(row["length"]),
            "bucket_path": bucket_path,
            "position": position,
        })
        if len(records) >= batch_size:
            yield _load_records(records)
            records = []
    if records:
        yield _load_records(records)


def _load_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Load route attention tensors for a chunk of records grouped by bucket."""
    by_bucket: dict[Path, list[tuple[int, dict[str, Any]]]] = {}
    for i, rec in enumerate(records):
        by_bucket.setdefault(rec["bucket_path"], []).append((i, rec))
    loaded: list[dict[str, Any] | None] = [None] * len(records)
    for bucket_path, items in by_bucket.items():
        tensors = load_file(str(bucket_path), device="cpu")
        route_heads = _read_route_heads(bucket_path)
        for orig_idx, rec in items:
            pos = rec["position"]
            length = rec["length"]
            routes: dict[int, torch.Tensor] = {}
            for depth in ROUTE_DEPTHS:
                key = f"teacher_heads_L{depth}"
                if key not in tensors:
                    continue
                t = tensors[key][pos, :, :length, :length].to(torch.float16)
                routes[depth] = t
            loaded[orig_idx] = {
                "seq_id": rec["seq_id"],
                "length": length,
                "routes": routes,
                "route_heads": route_heads,
            }
    return [r for r in loaded if r is not None]


def _random_prediction(teacher: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Generate a random row-normalized prediction map for loss simulation."""
    B, T, L, _ = teacher.shape
    pred = torch.randn_like(teacher)
    mask = (torch.arange(L, device=pred.device).view(1, 1, 1, L) < lengths.view(B, 1, 1, 1)).float()
    pred = pred * mask + (1 - mask) * -1e9
    pred = F.softmax(pred, dim=-1)
    return pred


def _simulate_training_steps(
    records_iter: Iterator[list[dict[str, Any]]],
    n_steps: int = 500,
    batch_size: int = 4,
    device: torch.device = torch.device("cpu"),
) -> dict[str, float]:
    """Run a lightweight 500-step simulation using the format's read path."""
    buffer: list[dict[str, Any]] = []
    kl_total = mse_total = cos_total = 0.0
    steps = 0
    for chunk in records_iter:
        buffer.extend(chunk)
        while len(buffer) >= batch_size and steps < n_steps:
            batch_records = buffer[:batch_size]
            buffer = buffer[batch_size:]
            maps_by_depth: dict[int, list[torch.Tensor]] = {d: [] for d in ROUTE_DEPTHS}
            lengths_list: list[int] = []
            for rec in batch_records:
                lengths_list.append(rec["length"])
                for d, t in rec["routes"].items():
                    maps_by_depth[d].append(t)
            if not maps_by_depth:
                continue
            step_kl = step_mse = step_cos = 0.0
            n_routes = 0
            for d, maps in maps_by_depth.items():
                if not maps:
                    continue
                teacher, lengths = _pad_maps_to_batch(maps)
                teacher = teacher.to(device)
                lengths = lengths.to(device)
                pred = _random_prediction(teacher, lengths)
                loss, per_head = route_alignment_loss(pred, teacher, lengths)
                step_kl += sum(float(h["kl"].item()) for h in per_head) / max(len(per_head), 1)
                step_mse += sum(float(h["mse"].item()) for h in per_head) / max(len(per_head), 1)
                step_cos += sum(float(h["cos"].item()) for h in per_head) / max(len(per_head), 1)
                n_routes += 1
            if n_routes:
                kl_total += step_kl / n_routes
                mse_total += step_mse / n_routes
                cos_total += step_cos / n_routes
            steps += 1
            if steps >= n_steps:
                break
        if steps >= n_steps:
            break
    denom = max(steps, 1)
    return {
        "train_500_kl": kl_total / denom,
        "train_500_mse": mse_total / denom,
        "train_500_cos": cos_total / denom,
    }


def _compute_reconstruction_error(
    original_records: list[dict[str, Any]],
    reconstructed_records: list[dict[str, Any]],
    sample_count: int = 50,
) -> dict[str, float]:
    """Compare ``reconstructed_records`` to ``original_records``."""
    if not original_records or not reconstructed_records:
        return {"reconstruction_max_abs_error": 0.0, "reconstruction_mean_abs_error": 0.0, "reconstruction_cosine": 1.0}
    rng = np.random.default_rng(42)
    indices = rng.choice(len(original_records), size=min(sample_count, len(original_records)), replace=False)
    max_err = 0.0
    sum_err = 0.0
    n = 0
    cos_sum = 0.0
    for idx in indices:
        orig = original_records[idx]
        recon = reconstructed_records[idx]
        for d in orig["routes"]:
            if d not in recon["routes"]:
                continue
            o = orig["routes"][d].float()
            r = recon["routes"][d].float()
            L = min(o.shape[-1], r.shape[-1])
            o = o[:, :L, :L]
            r = r[:, :L, :L]
            diff = (o - r).abs()
            max_err = max(max_err, float(diff.max()))
            sum_err += float(diff.sum())
            n += int(diff.numel())
            flat_o = o.reshape(-1)
            flat_r = r.reshape(-1)
            norm = flat_o.norm(2) * flat_r.norm(2)
            if norm > 0:
                cos_sum += float(F.cosine_similarity(flat_o.unsqueeze(0), flat_r.unsqueeze(0)).item())
            else:
                cos_sum += 1.0
    return {
        "reconstruction_max_abs_error": max_err,
        "reconstruction_mean_abs_error": sum_err / max(n, 1),
        "reconstruction_cosine": cos_sum / max(len(indices), 1),
    }


# ---------------------------------------------------------------------------
# Format writers / readers
# ---------------------------------------------------------------------------


class C0BucketedFormat:
    """Reference: existing bucketed fp16 safetensors."""

    name = "C0_bucketed_fp16"

    def __init__(
        self,
        cache_root: Path,
        sequence_manifest_path: Path,
        bucket_manifest_path: Path,
        n: int,
        seed: int,
    ) -> None:
        self.cache_root = cache_root
        self.sequence_manifest_path = sequence_manifest_path
        self.bucket_manifest_path = bucket_manifest_path
        self.n = n
        self.seed = seed

    def write(self, out_dir: Path) -> tuple[Path, float]:
        """C0 already exists; only record the touched bucket files."""
        sample_df = _sample_manifest_rows(self.sequence_manifest_path, self.n, self.seed)
        bucket_manifest = pd.read_parquet(self.bucket_manifest_path)
        mapping = _build_manifest_index_to_bucket(self.cache_root, bucket_manifest)
        touched = set()
        for _, row in sample_df.iterrows():
            m = mapping.get(int(row["manifest_idx"]))
            if m:
                touched.add(m[0])
        out_dir.mkdir(parents=True, exist_ok=True)
        meta = {"touched_bucket_files": [str(p) for p in sorted(touched)], "n_sequences": len(sample_df)}
        meta_path = out_dir / "c0_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        disk_bytes = sum(p.stat().st_size for p in touched if p.exists())
        return out_dir, disk_bytes

    def read_iter(self) -> Iterator[list[dict[str, Any]]]:
        yield from iter_sample_records(
            self.cache_root,
            self.sequence_manifest_path,
            self.bucket_manifest_path,
            self.n,
            self.seed,
            batch_size=1024,
        )


class C1RaggedFlatFormat:
    """Ragged flat fp16 chunks."""

    name = "C1_ragged_flat_fp16"

    def __init__(self, source_iter: Iterator[list[dict[str, Any]]]) -> None:
        self.source_iter = source_iter
        self.chunk_size = 1024

    def write(self, out_dir: Path) -> tuple[Path, float]:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_start = time.perf_counter()
        chunk_idx = 0
        disk_bytes = 0.0
        current: list[dict[str, Any]] = []
        for records in self.source_iter:
            current.extend(records)
            while len(current) >= self.chunk_size:
                chunk = current[: self.chunk_size]
                current = current[self.chunk_size :]
                path = out_dir / f"chunk_{chunk_idx:04d}.pt"
                self._write_chunk(chunk, path)
                disk_bytes += path.stat().st_size
                chunk_idx += 1
        if current:
            path = out_dir / f"chunk_{chunk_idx:04d}.pt"
            self._write_chunk(current, path)
            disk_bytes += path.stat().st_size
        write_sec = time.perf_counter() - write_start
        (out_dir / "write_time.json").write_text(json.dumps({"write_sec": write_sec}), encoding="utf-8")
        return out_dir, disk_bytes

    def _write_chunk(self, records: list[dict[str, Any]], path: Path) -> None:
        routes: dict[int, dict[str, Any]] = {d: {"data": [], "offsets": [0], "lengths": []} for d in ROUTE_DEPTHS}
        route_heads: dict[int, list[str]] | None = None
        for rec in records:
            if route_heads is None:
                route_heads = rec.get("route_heads", {d: [] for d in ROUTE_DEPTHS})
            for d, t in rec["routes"].items():
                T, L, _ = t.shape
                routes[d]["lengths"].append(L)
                for h in range(T):
                    routes[d]["data"].append(t[h].reshape(-1))
                    routes[d]["offsets"].append(routes[d]["offsets"][-1] + L * L)
        payload: dict[str, Any] = {"route_heads": route_heads}
        for d in ROUTE_DEPTHS:
            if routes[d]["data"]:
                payload[f"data_L{d}"] = torch.cat(routes[d]["data"]).to(torch.float16)
                payload[f"offsets_L{d}"] = torch.tensor(routes[d]["offsets"], dtype=torch.int64)
                payload[f"lengths_L{d}"] = torch.tensor(routes[d]["lengths"], dtype=torch.int32)
        torch.save(payload, path)

    def read_iter(self, out_dir: Path) -> Iterator[list[dict[str, Any]]]:
        for path in sorted(out_dir.glob("chunk_*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            route_heads = payload.get("route_heads", {d: [] for d in ROUTE_DEPTHS})
            # Determine number of sequences from lengths arrays.
            n_seqs = len(next(iter(v for v in payload.values() if isinstance(v, torch.Tensor) and v.dim() == 1 and v.dtype == torch.int32)))
            for i in range(n_seqs):
                rec: dict[str, Any] = {"seq_id": f"{path.stem}_seq{i}", "routes": {}}
                for d in ROUTE_DEPTHS:
                    lengths = payload[f"lengths_L{d}"]
                    L = int(lengths[i].item())
                    offsets = payload[f"offsets_L{d}"]
                    data = payload[f"data_L{d}"]
                    T = len(route_heads.get(d, []))
                    if T == 0:
                        continue
                    start = int(offsets[i * T].item())
                    end = int(offsets[(i + 1) * T].item())
                    flat = data[start:end].to(torch.float16)
                    maps = []
                    for h in range(T):
                        lh = int(flat[h * L * L : (h + 1) * L * L].shape[0]) ** 0.5
                        lh = int(lh)
                        maps.append(flat[h * L * L : (h + 1) * L * L].reshape(lh, lh))
                    rec["routes"][d] = torch.stack(maps, dim=0)
                    rec["length"] = L
                rec["route_heads"] = route_heads
                yield [rec]


class C2ZarrFormat:
    """Chunked zarr compressed fp16 (npz fallback if zarr missing)."""

    name = "C2_chunked_zarr_fp16"

    def __init__(self, source_iter: Iterator[list[dict[str, Any]]]) -> None:
        self.source_iter = source_iter

    def write(self, out_dir: Path) -> tuple[Path, float]:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_start = time.perf_counter()
        try:
            import zarr
            store = zarr.DirectoryStore(str(out_dir))
            root_group = zarr.group(store=store, overwrite=True)
            for d in ROUTE_DEPTHS:
                root_group.create(f"data_L{d}", shape=(0,), chunks=(65536,), dtype="float16")
                root_group.create(f"offsets_L{d}", shape=(0,), chunks=(1024,), dtype="int64")
                root_group.create(f"lengths_L{d}", shape=(0,), chunks=(1024,), dtype="int32")
                root_group.attrs[f"head_keys_L{d}"] = []

            route_heads: dict[int, list[str]] | None = None
            counts: dict[int, int] = {d: 0 for d in ROUTE_DEPTHS}
            for records in self.source_iter:
                if route_heads is None and records:
                    route_heads = records[0].get("route_heads", {d: [] for d in ROUTE_DEPTHS})
                    for d in ROUTE_DEPTHS:
                        root_group.attrs[f"head_keys_L{d}"] = route_heads.get(d, [])
                self._append_records_zarr(root_group, records, counts)
            for d in ROUTE_DEPTHS:
                root_group.attrs[f"n_seqs_L{d}"] = counts[d]
            disk_bytes = sum(p.stat().st_size for p in out_dir.rglob("*") if p.is_file())
            backend = "zarr"
        except Exception as exc:  # noqa: BLE001
            # Fallback to compressed npz chunks.
            disk_bytes = self._write_npz_fallback(out_dir)
            backend = f"npz_fallback_{exc}"
        write_sec = time.perf_counter() - write_start
        (out_dir / "write_time.json").write_text(
            json.dumps({"write_sec": write_sec, "backend": backend}), encoding="utf-8"
        )
        return out_dir, disk_bytes

    def _append_records_zarr(
        self,
        zgroup: Any,
        records: list[dict[str, Any]],
        counts: dict[int, int],
    ) -> None:
        for d in ROUTE_DEPTHS:
            data_list: list[torch.Tensor] = []
            local_offsets: list[int] = [0]
            lengths: list[int] = []
            for rec in records:
                t = rec["routes"].get(d)
                if t is None:
                    continue
                T, L, _ = t.shape
                counts[d] += 1
                lengths.append(L)
                for h in range(T):
                    data_list.append(t[h].reshape(-1))
                    local_offsets.append(local_offsets[-1] + L * L)
            if data_list:
                prev_total = int(zgroup[f"offsets_L{d}"].shape[0] and zgroup[f"offsets_L{d}"][-1])
                full_offsets = [o + prev_total for o in local_offsets]
                # Avoid duplicating the previous final offset across chunks.
                append_offsets = full_offsets if prev_total == 0 else full_offsets[1:]
                data_arr = torch.cat(data_list).numpy().astype(np.float16)
                zgroup[f"data_L{d}"].append(data_arr)
                zgroup[f"offsets_L{d}"].append(np.array(append_offsets, dtype=np.int64))
                zgroup[f"lengths_L{d}"].append(np.array(lengths, dtype=np.int32))

    def _write_npz_fallback(self, out_dir: Path) -> float:
        """Write compressed .npz files when zarr is unavailable."""
        chunk_idx = 0
        disk_bytes = 0.0
        current: list[dict[str, Any]] = []
        chunk_size = 1024
        for records in self.source_iter:
            current.extend(records)
            while len(current) >= chunk_size:
                chunk = current[:chunk_size]
                current = current[chunk_size:]
                path = out_dir / f"chunk_{chunk_idx:04d}.npz"
                arrays = self._records_to_arrays(chunk)
                np.savez_compressed(path, **arrays)
                disk_bytes += path.stat().st_size
                chunk_idx += 1
        if current:
            path = out_dir / f"chunk_{chunk_idx:04d}.npz"
            arrays = self._records_to_arrays(current)
            np.savez_compressed(path, **arrays)
            disk_bytes += path.stat().st_size
        return disk_bytes

    def _records_to_arrays(self, records: list[dict[str, Any]]) -> dict[str, np.ndarray]:
        arrays: dict[str, np.ndarray] = {}
        route_heads = records[0].get("route_heads", {d: [] for d in ROUTE_DEPTHS})
        for d in ROUTE_DEPTHS:
            data_list: list[torch.Tensor] = []
            offsets = [0]
            lengths: list[int] = []
            for rec in records:
                t = rec["routes"].get(d)
                if t is None:
                    continue
                T, L, _ = t.shape
                lengths.append(L)
                for h in range(T):
                    data_list.append(t[h].reshape(-1))
                    offsets.append(offsets[-1] + L * L)
            if data_list:
                arrays[f"data_L{d}"] = torch.cat(data_list).numpy().astype(np.float16)
                arrays[f"offsets_L{d}"] = np.array(offsets, dtype=np.int64)
                arrays[f"lengths_L{d}"] = np.array(lengths, dtype=np.int32)
        arrays["route_heads"] = np.array(json.dumps(route_heads), dtype=object)
        return arrays

    def read_iter(self, out_dir: Path) -> Iterator[list[dict[str, Any]]]:
        # Unified reader handles zarr or npz fallback.
        zarr_files = list(out_dir.glob("*.zarr")) if (out_dir / ".zgroup").exists() else []
        if zarr_files or (out_dir / ".zgroup").exists():
            yield from self._read_zarr(out_dir)
        else:
            yield from self._read_npz(out_dir)

    def _read_zarr(self, out_dir: Path) -> Iterator[list[dict[str, Any]]]:
        import zarr
        zgroup = zarr.open(str(out_dir), mode="r")
        route_heads = {d: list(zgroup.attrs.get(f"head_keys_L{d}", [])) for d in ROUTE_DEPTHS}
        n_seqs = min(int(zgroup.attrs.get(f"n_seqs_L{d}", 0)) for d in ROUTE_DEPTHS if route_heads[d]) or 0
        for i in range(n_seqs):
            rec: dict[str, Any] = {"seq_id": f"zarr_seq{i}", "routes": {}}
            for d in ROUTE_DEPTHS:
                if not route_heads[d]:
                    continue
                lengths = zgroup[f"lengths_L{d}"]
                L = int(lengths[i])
                offsets = zgroup[f"offsets_L{d}"]
                data = zgroup[f"data_L{d}"]
                T = len(route_heads[d])
                start = int(offsets[i * T])
                end = int(offsets[(i + 1) * T])
                flat = torch.from_numpy(np.array(data[start:end], dtype=np.float16))
                maps = [flat[h * L * L : (h + 1) * L * L].reshape(L, L) for h in range(T)]
                rec["routes"][d] = torch.stack(maps, dim=0)
                rec["length"] = L
            rec["route_heads"] = route_heads
            yield [rec]

    def _read_npz(self, out_dir: Path) -> Iterator[list[dict[str, Any]]]:
        for path in sorted(out_dir.glob("chunk_*.npz")):
            arrs = np.load(path, allow_pickle=True)
            route_heads = json.loads(str(arrs["route_heads"]))
            n_seqs = len(next(v for k, v in arrs.items() if k.startswith("lengths_")))
            for i in range(n_seqs):
                rec: dict[str, Any] = {"seq_id": f"{path.stem}_seq{i}", "routes": {}}
                for d in ROUTE_DEPTHS:
                    if not route_heads.get(str(d), []):
                        continue
                    L = int(arrs[f"lengths_L{d}"][i])
                    offsets = arrs[f"offsets_L{d}"]
                    data = arrs[f"data_L{d}"]
                    T = len(route_heads[str(d)])
                    start = int(offsets[i * T])
                    end = int(offsets[(i + 1) * T])
                    flat = torch.from_numpy(data[start:end].astype(np.float16))
                    maps = [flat[h * L * L : (h + 1) * L * L].reshape(L, L) for h in range(T)]
                    rec["routes"][d] = torch.stack(maps, dim=0)
                    rec["length"] = L
                rec["route_heads"] = {int(k): v for k, v in route_heads.items()}
                yield [rec]


class C3QuantizedFormat:
    """Row-wise uint8 quantized with per-row scale."""

    name = "C3_rowwise_uint8_quantized"

    def __init__(self, source_iter: Iterator[list[dict[str, Any]]]) -> None:
        self.source_iter = source_iter
        self.chunk_size = 1024

    def write(self, out_dir: Path) -> tuple[Path, float]:
        out_dir.mkdir(parents=True, exist_ok=True)
        write_start = time.perf_counter()
        chunk_idx = 0
        disk_bytes = 0.0
        current: list[dict[str, Any]] = []
        for records in self.source_iter:
            current.extend(records)
            while len(current) >= self.chunk_size:
                chunk = current[: self.chunk_size]
                current = current[self.chunk_size :]
                path = out_dir / f"chunk_{chunk_idx:04d}.pt"
                self._write_chunk(chunk, path)
                disk_bytes += path.stat().st_size
                chunk_idx += 1
        if current:
            path = out_dir / f"chunk_{chunk_idx:04d}.pt"
            self._write_chunk(current, path)
            disk_bytes += path.stat().st_size
        write_sec = time.perf_counter() - write_start
        (out_dir / "write_time.json").write_text(json.dumps({"write_sec": write_sec}), encoding="utf-8")
        return out_dir, disk_bytes

    def _write_chunk(self, records: list[dict[str, Any]], path: Path) -> None:
        routes: dict[int, dict[str, Any]] = {
            d: {"data": [], "scales": [], "offsets": [0], "lengths": []} for d in ROUTE_DEPTHS
        }
        route_heads: dict[int, list[str]] | None = None
        for rec in records:
            if route_heads is None:
                route_heads = rec.get("route_heads", {d: [] for d in ROUTE_DEPTHS})
            for d, t in rec["routes"].items():
                T, L, _ = t.shape
                routes[d]["lengths"].append(L)
                for h in range(T):
                    attn = t[h].float()
                    row_scales = attn.abs().max(dim=-1).values.clamp_min(1e-8)
                    quantized = ((attn / row_scales.unsqueeze(-1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8)
                    routes[d]["data"].append(quantized.reshape(-1))
                    routes[d]["scales"].append(row_scales.to(torch.float16))
                    routes[d]["offsets"].append(routes[d]["offsets"][-1] + L)
        payload: dict[str, Any] = {"route_heads": route_heads}
        for d in ROUTE_DEPTHS:
            if routes[d]["data"]:
                payload[f"data_L{d}"] = torch.cat(routes[d]["data"])
                payload[f"scales_L{d}"] = torch.cat(routes[d]["scales"])
                payload[f"offsets_L{d}"] = torch.tensor(routes[d]["offsets"], dtype=torch.int64)
                payload[f"lengths_L{d}"] = torch.tensor(routes[d]["lengths"], dtype=torch.int32)
        torch.save(payload, path)

    def read_iter(self, out_dir: Path) -> Iterator[list[dict[str, Any]]]:
        for path in sorted(out_dir.glob("chunk_*.pt")):
            payload = torch.load(path, map_location="cpu", weights_only=False)
            route_heads = payload.get("route_heads", {d: [] for d in ROUTE_DEPTHS})
            n_seqs = len(next(iter(v for v in payload.values() if isinstance(v, torch.Tensor) and v.dim() == 1 and v.dtype == torch.int32)))
            for i in range(n_seqs):
                rec: dict[str, Any] = {"seq_id": f"{path.stem}_seq{i}", "routes": {}}
                for d in ROUTE_DEPTHS:
                    lengths = payload[f"lengths_L{d}"]
                    L = int(lengths[i].item())
                    offsets = payload[f"offsets_L{d}"]
                    data = payload[f"data_L{d}"]
                    scales = payload[f"scales_L{d}"]
                    T = len(route_heads.get(d, []))
                    if T == 0:
                        continue
                    seq_start = int(offsets[i * T * L].item())
                    seq_end = int(offsets[(i + 1) * T * L].item())
                    q_flat = data[seq_start:seq_end].float()
                    s_flat = scales[seq_start:seq_end].float()
                    maps = []
                    for h in range(T):
                        q = q_flat[h * L : (h + 1) * L].reshape(L, L)
                        s = s_flat[h * L : (h + 1) * L]
                        dequant = (q / 127.5 - 1.0) * s.unsqueeze(-1)
                        maps.append(dequant.to(torch.float16))
                    rec["routes"][d] = torch.stack(maps, dim=0)
                    rec["length"] = L
                rec["route_heads"] = route_heads
                yield [rec]


def _uncompressed_bytes(records: list[dict[str, Any]]) -> int:
    total = 0
    for rec in records:
        for t in rec["routes"].values():
            total += t.numel() * 2  # fp16
    return total


def run_cache_format_benchmark(
    cache_root: str | Path,
    sequence_manifest_path: str | Path,
    bucket_manifest_path: str | Path,
    output_dir: str | Path,
    n: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """Run the full cache-format benchmark and return a summary DataFrame."""
    out_root = Path(resolve_path(output_dir))
    out_root.mkdir(parents=True, exist_ok=True)

    # Pre-sample records once so original is available for error comparisons.
    original_chunks: list[list[dict[str, Any]]] = []
    for chunk in iter_sample_records(
        cache_root,
        sequence_manifest_path,
        bucket_manifest_path,
        n=n,
        seed=seed,
        batch_size=1024,
    ):
        original_chunks.append([{k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in rec.items()} for rec in chunk])

    uncompressed_total = sum(_uncompressed_bytes(chunk) for chunk in original_chunks)

    results: list[dict[str, Any]] = []

    # C0
    c0 = C0BucketedFormat(cache_root, resolve_path(sequence_manifest_path), resolve_path(bucket_manifest_path), n, seed)
    c0_dir, c0_bytes = c0.write(out_root / "C0_bucketed_fp16")
    read_start = time.perf_counter()
    for _ in c0.read_iter():
        pass
    c0_read_sec = time.perf_counter() - read_start
    results.append({
        "format": c0.name,
        "disk_size_gb": c0_bytes / 1e9,
        "compression_ratio": uncompressed_total / max(c0_bytes, 1),
        "write_time_sec": 0.0,
        "read_time_sec": c0_read_sec,
        "read_throughput_seq_per_sec": n / max(c0_read_sec, 1e-9),
        **_simulate_training_steps(c0.read_iter(), n_steps=500),
        "reconstruction_max_abs_error": 0.0,
        "reconstruction_mean_abs_error": 0.0,
        "reconstruction_cosine": 1.0,
    })

    def _original_iter() -> Iterator[list[dict[str, Any]]]:
        for chunk in original_chunks:
            yield chunk

    # C1
    c1 = C1RaggedFlatFormat(_original_iter())
    c1_dir, c1_bytes = c1.write(out_root / "C1_ragged_flat_fp16")
    read_start = time.perf_counter()
    c1_recon: list[dict[str, Any]] = []
    for recs in c1.read_iter(c1_dir):
        c1_recon.extend(recs)
    c1_read_sec = time.perf_counter() - read_start
    c1_original = [rec for chunk in original_chunks for rec in chunk][: len(c1_recon)]
    c1_err = _compute_reconstruction_error(c1_original, c1_recon)
    results.append({
        "format": c1.name,
        "disk_size_gb": c1_bytes / 1e9,
        "compression_ratio": uncompressed_total / max(c1_bytes, 1),
        "write_time_sec": json.loads((c1_dir / "write_time.json").read_text())["write_sec"],
        "read_time_sec": c1_read_sec,
        "read_throughput_seq_per_sec": n / max(c1_read_sec, 1e-9),
        **_simulate_training_steps(c1.read_iter(c1_dir), n_steps=500),
        **c1_err,
    })

    # C2
    def _original_iter2() -> Iterator[list[dict[str, Any]]]:
        for chunk in original_chunks:
            yield chunk

    c2 = C2ZarrFormat(_original_iter2())
    c2_dir, c2_bytes = c2.write(out_root / "C2_chunked_zarr_fp16")
    read_start = time.perf_counter()
    c2_recon: list[dict[str, Any]] = []
    for recs in c2.read_iter(c2_dir):
        c2_recon.extend(recs)
    c2_read_sec = time.perf_counter() - read_start
    c2_original = [rec for chunk in original_chunks for rec in chunk][: len(c2_recon)]
    c2_err = _compute_reconstruction_error(c2_original, c2_recon)
    results.append({
        "format": c2.name,
        "disk_size_gb": c2_bytes / 1e9,
        "compression_ratio": uncompressed_total / max(c2_bytes, 1),
        "write_time_sec": json.loads((c2_dir / "write_time.json").read_text())["write_sec"],
        "read_time_sec": c2_read_sec,
        "read_throughput_seq_per_sec": n / max(c2_read_sec, 1e-9),
        **_simulate_training_steps(c2.read_iter(c2_dir), n_steps=500),
        **c2_err,
    })

    # C3
    def _original_iter3() -> Iterator[list[dict[str, Any]]]:
        for chunk in original_chunks:
            yield chunk

    c3 = C3QuantizedFormat(_original_iter3())
    c3_dir, c3_bytes = c3.write(out_root / "C3_rowwise_uint8_quantized")
    read_start = time.perf_counter()
    c3_recon: list[dict[str, Any]] = []
    for recs in c3.read_iter(c3_dir):
        c3_recon.extend(recs)
    c3_read_sec = time.perf_counter() - read_start
    c3_original = [rec for chunk in original_chunks for rec in chunk][: len(c3_recon)]
    c3_err = _compute_reconstruction_error(c3_original, c3_recon)
    results.append({
        "format": c3.name,
        "disk_size_gb": c3_bytes / 1e9,
        "compression_ratio": uncompressed_total / max(c3_bytes, 1),
        "write_time_sec": json.loads((c3_dir / "write_time.json").read_text())["write_sec"],
        "read_time_sec": c3_read_sec,
        "read_throughput_seq_per_sec": n / max(c3_read_sec, 1e-9),
        **_simulate_training_steps(c3.read_iter(c3_dir), n_steps=500),
        **c3_err,
    })

    df = pd.DataFrame(results)
    df.to_csv(out_root / "cache_format_summary.csv", index=False)
    md_path = out_root / "cache_format_summary.md"
    md_path.write_text(
        "# Cache Format Benchmark Summary\n\n" + df.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    return df
