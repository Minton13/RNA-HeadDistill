#!/usr/bin/env python
"""Download the demo model (RiNALMo-micro, ~120 MB) into assets/models/.

The pipeline loads teachers from local directories with
``local_files_only=True``, so the weights must be staged before the demo runs.
Requires network access to Hugging Face (set HF_ENDPOINT to a mirror such as
https://hf-mirror.com if huggingface.co is unreachable).
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

REPO_ID = "multimolecule/rinalmo-micro"
KEEP_SUFFIXES = {".json", ".txt", ".safetensors", ".md"}
KEEP_FILES = {"config.json", "tokenizer_config.json", "vocab.txt", "model.safetensors"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo-id", default=REPO_ID)
    ap.add_argument("--output", default="assets/models/rinalmo-micro")
    args = ap.parse_args()

    out = Path(args.output)
    if (out / "model.safetensors").exists() and (out / "config.json").exists():
        print(f"[demo] model already staged at {out}")
        return

    from huggingface_hub import snapshot_download

    with tempfile.TemporaryDirectory() as tmp:
        snap = snapshot_download(args.repo_id, cache_dir=tmp)
        snap_path = Path(snap)
        out.mkdir(parents=True, exist_ok=True)
        copied = []
        for item in sorted(snap_path.iterdir()):
            if item.is_file() and (item.name in KEEP_FILES or item.suffix in KEEP_SUFFIXES):
                shutil.copy2(item, out / item.name)
                copied.append(item.name)
    print(f"[demo] staged {args.repo_id} -> {out} ({len(copied)} files: {', '.join(copied)})")


if __name__ == "__main__":
    main()
