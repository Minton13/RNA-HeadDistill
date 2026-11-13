from __future__ import annotations

import hashlib


def stable_split(seq_id: str, train: float = 0.8, val: float = 0.1) -> str:
    x = int(hashlib.md5(seq_id.encode("utf-8")).hexdigest()[:8], 16) / 0xFFFFFFFF
    if x < train:
        return "train"
    if x < train + val:
        return "val"
    return "test"
