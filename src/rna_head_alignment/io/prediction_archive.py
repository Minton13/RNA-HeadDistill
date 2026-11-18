"""Verify prediction-file inventory, including configuration path collisions."""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path


def verify_prediction_archive(index: str | Path, root: str | Path) -> dict:
    root = Path(root).resolve()
    rows = list(csv.DictReader(Path(index).open(encoding="utf-8")))
    checked, errors, cache, paths = [], [], {}, {}
    for row in rows:
        status = row["status"]
        identity = {key: row[key] for key in ("task", "config", "seed", "split")}
        if status in {"missing_on_disk", "present_index_only"}:
            checked.append({**identity, "result": status})
            continue
        if status != "present":
            errors.append({**identity, "error": "unknown_status"})
            continue
        relative = row["packaged_as"]
        path = (root / relative).resolve()
        if not relative or not path.is_relative_to(root):
            errors.append({**identity, "error": "unsafe_path"})
            continue
        expected = (row["sha256"], int(row["bytes"]))
        if relative in paths and paths[relative] != expected:
            errors.append({**identity, "error": "configuration_path_collision", "path": relative})
        paths[relative] = expected
        if not path.is_file():
            errors.append({**identity, "error": "missing_packaged_file", "path": relative})
            continue
        if relative not in cache:
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            cache[relative] = (digest.hexdigest(), path.stat().st_size)
        if cache[relative] != expected:
            errors.append({**identity, "error": "sha256_or_size_mismatch", "path": relative})
        else:
            checked.append({**identity, "result": "verified"})
    return {"index_rows": len(rows), "unique_files_checked": len(cache),
            "verified_entries": sum(row["result"] == "verified" for row in checked),
            "error_count": len(errors), "entries": checked, "errors": errors}
