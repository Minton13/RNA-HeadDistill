from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .paths import ensure_parent, resolve_path


def write_json(data: Any, path: str | Path) -> Path:
    p = ensure_parent(path)
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return p


def read_json(path: str | Path) -> Any:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def write_csv(rows: Iterable[dict[str, Any]], path: str | Path) -> Path:
    p = ensure_parent(path)
    rows = list(rows)
    if not rows:
        p.write_text("", encoding="utf-8")
        return p
    fieldnames = sorted({k for row in rows for k in row})
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return p


def write_dataframe(df: pd.DataFrame, path: str | Path) -> Path:
    p = ensure_parent(path)
    df.to_csv(p, index=False)
    return p


def write_markdown_table(rows: list[dict[str, Any]], path: str | Path, title: str) -> Path:
    p = ensure_parent(path)
    lines = [f"# {title}", ""]
    if rows:
        cols = list(rows[0].keys())
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join("---" for _ in cols) + "|")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
    else:
        lines.append("No rows.")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p
