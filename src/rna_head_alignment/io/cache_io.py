from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .paths import ensure_parent, resolve_path


def save_pt(obj: dict[str, Any], path: str | Path) -> Path:
    p = ensure_parent(path)
    torch.save(obj, p)
    return p


def load_pt(path: str | Path) -> dict[str, Any]:
    return torch.load(resolve_path(path), map_location="cpu")


def file_nonempty(path: str | Path) -> bool:
    p = resolve_path(path)
    return p.exists() and p.is_file() and p.stat().st_size > 0
