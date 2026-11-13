from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sort sampled sequence JSONL by length to reduce attention-cache padding.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--descending", action="store_true")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    with src.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            rows.append((int(obj.get("length") or len(obj.get("sequence", ""))), line))
    rows.sort(key=lambda item: item[0], reverse=args.descending)

    with dst.open("w", encoding="utf-8") as f:
        for _, line in rows:
            f.write(line)

    manifest = {
        "input": str(src.resolve()),
        "output": str(dst.resolve()),
        "records": len(rows),
        "sort_key": "length",
        "descending": bool(args.descending),
        "min_len": rows[0][0] if rows else None,
        "max_len": rows[-1][0] if rows else None,
        "notes": "Use this sorted file for teacher attention cache jobs to reduce within-shard Lmax padding.",
    }
    dst.with_suffix(dst.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
