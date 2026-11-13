from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rna_head_alignment.io.paths import resolve_path


def _node_sequence(data: dict[str, Any]) -> tuple[str, list[Any]]:
    nodes = data.get("nodes")
    if isinstance(nodes, dict):
        items = sorted(nodes.items(), key=lambda kv: str(kv[0]))
        residues = []
        ids = []
        for node_id, attrs in items:
            ids.append(node_id)
            if isinstance(attrs, dict):
                residues.append(str(attrs.get("nt") or attrs.get("residue") or attrs.get("base") or "N")[:1])
            else:
                residues.append("N")
        return "".join(residues).upper().replace("T", "U"), ids
    if isinstance(nodes, list):
        residues = []
        ids = []
        for i, node in enumerate(nodes):
            ids.append(node.get("id", i) if isinstance(node, dict) else i)
            if isinstance(node, dict):
                residues.append(str(node.get("nt") or node.get("residue") or node.get("base") or "N")[:1])
            else:
                residues.append("N")
        return "".join(residues).upper().replace("T", "U"), ids
    seq = str(data.get("sequence") or data.get("seq") or "").upper().replace("T", "U")
    return seq, list(range(len(seq)))


def _edges(data: dict[str, Any]) -> list[tuple[Any, Any]]:
    edges = data.get("edges") or data.get("links") or []
    out = []
    if isinstance(edges, dict):
        for key, attrs in edges.items():
            if isinstance(key, str) and "," in key:
                a, b = key.split(",", 1)
                out.append((a.strip(" ()'\""), b.strip(" ()'\"")))
            elif isinstance(attrs, dict):
                out.append((attrs.get("source"), attrs.get("target")))
    elif isinstance(edges, list):
        for edge in edges:
            if isinstance(edge, dict):
                out.append((edge.get("source") or edge.get("u") or edge.get("from"), edge.get("target") or edge.get("v") or edge.get("to")))
            elif isinstance(edge, (list, tuple)) and len(edge) >= 2:
                out.append((edge[0], edge[1]))
    return [(a, b) for a, b in out if a is not None and b is not None]


def load_rnaglib_json(path: str | Path) -> dict[str, Any]:
    p = resolve_path(path)
    data = json.loads(p.read_text(encoding="utf-8"))
    sequence, node_ids = _node_sequence(data)
    index = {str(node_id): i for i, node_id in enumerate(node_ids)}
    L = len(sequence)
    pair = np.zeros((L, L), dtype=np.float32)
    for a, b in _edges(data):
        ia = index.get(str(a))
        ib = index.get(str(b))
        if ia is None or ib is None or ia == ib:
            continue
        pair[ia, ib] = 1.0
        pair[ib, ia] = 1.0
    return {
        "seq_id": p.stem,
        "sequence": sequence,
        "pair_matrix": pair,
        "split": "train",
        "source": "rnaglib",
    }


def iter_rnaglib_dir(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    p = resolve_path(path)
    files = sorted(p.glob("*.json"))
    rows = []
    for file in files[: limit or None]:
        try:
            item = load_rnaglib_json(file)
            if item["sequence"]:
                rows.append(item)
        except Exception:
            continue
    return rows


def iter_contact_lmdb_dir(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    try:
        import lmdb  # type: ignore
    except Exception as e:
        raise ImportError("RNAscope contact map data is LMDB; install optional dependency `lmdb`.") from e
    p = resolve_path(path)
    lmdb_dirs = sorted([x for x in p.glob("*.lmdb") if x.is_dir()])
    rows: list[dict[str, Any]] = []
    for lmdb_dir in lmdb_dirs:
        split = "train"
        name = lmdb_dir.name.lower()
        if "val" in name:
            split = "val"
        elif "test" in name:
            split = "test"
        env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False)
        with env.begin() as txn:
            cursor = txn.cursor()
            for key, value in cursor:
                if limit and len(rows) >= limit:
                    break
                item = _decode_lmdb_value(value)
                seq = str(item.get("sequence") or item.get("seq") or "").upper().replace("T", "U")
                seq = "".join(ch for ch in seq if ch in "ACGUN")
                if not seq:
                    continue
                L = len(seq)
                pair = np.zeros((L, L), dtype=np.float32)
                pairs = item.get("pairs") or item.get("contacts") or item.get("contact_map")
                if isinstance(pairs, np.ndarray) and pairs.ndim == 2 and pairs.shape == (L, L):
                    pair = pairs.astype(np.float32)
                elif isinstance(pairs, list):
                    for edge in pairs:
                        if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                            i, j = int(edge[0]), int(edge[1])
                            if 0 <= i < L and 0 <= j < L:
                                pair[i, j] = pair[j, i] = 1.0
                elif isinstance(item.get("tertiary"), np.ndarray):
                    coords = item["tertiary"].astype(np.float32)
                    if coords.ndim == 2 and coords.shape[1] >= 3:
                        n = min(L, coords.shape[0])
                        coords = coords[:n, :3]
                        finite = np.isfinite(coords).all(axis=1)
                        dist = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
                        contact = (dist <= 8.0) & finite[:, None] & finite[None, :]
                        np.fill_diagonal(contact, False)
                        pair[:n, :n] = contact.astype(np.float32)
                rows.append(
                    {
                        "seq_id": key.decode("utf-8", errors="ignore"),
                        "sequence": seq,
                        "pair_matrix": pair,
                        "split": split,
                        "source": "rnascope_contact_lmdb",
                    }
                )
        env.close()
        if limit and len(rows) >= limit:
            break
    return rows


def _decode_lmdb_value(value: bytes) -> dict[str, Any]:
    import io
    import pickle

    try:
        obj = json.loads(value.decode("utf-8"))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    try:
        obj = pickle.loads(value)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        pass
    try:
        npz = np.load(io.BytesIO(value), allow_pickle=True)
        return {k: npz[k] for k in npz.files}
    except Exception:
        return {}
