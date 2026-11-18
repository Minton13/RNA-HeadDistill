"""Check the prediction index before replay; missing/colliding files fail."""
import argparse
import json
from pathlib import Path
from rna_head_alignment.io.prediction_archive import verify_prediction_archive

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True, type=Path)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = verify_prediction_archive(args.index, args.root)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"entries", "errors"}}, indent=2))
    raise SystemExit(2 if result["error_count"] else 0)
