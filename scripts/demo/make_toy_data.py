#!/usr/bin/env python
"""Generate a tiny deterministic toy dataset for the reviewer demo.

Two synthetic tasks are written under ``data/demo/`` (paths relative to the
project root):

- ``toy_secondary_structure.jsonl`` — pairwise task. Each sequence contains one
  or two GC/AU Watson-Crick hairpin stems with a short loop; labels are given
  as a dot-bracket string, which the loader converts to an L x L pair matrix.
- ``toy_family_seqcls.csv`` — 3-class sequence-classification task. Each
  sequence embeds one family-specific motif (GNRA tetraloop, a YCGY duplex
  signature, or no motif), so attention-head probes have a real signal to find.

Both files share the JSONL/CSV schemas used by the real tasks
(``sequence`` plus ``dot_bracket`` or ``label``), so the demo exercises the
same code path as the full screening pipeline. Everything is deterministic
given ``--seed``.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

COMPLEMENT = {"A": "U", "U": "A", "G": "C", "C": "G"}
BASES = "ACGU"


def _rand_seq(rng: np.random.Generator, length: int, gc_bias: float = 0.5) -> list[str]:
    probs = np.array([1 - gc_bias, gc_bias, gc_bias, 1 - gc_bias]) / 2.0  # A, C, G, U
    return list(rng.choice(list(BASES), size=length, p=probs))


def _add_hairpin(seq: list[str], dot: list[str], rng: np.random.Generator, start: int, stem: int, loop: int) -> None:
    for k in range(stem):
        b = str(rng.choice(list(BASES)))
        seq[start + k] = b
        seq[start + stem + loop + (stem - 1 - k)] = COMPLEMENT[b]
        dot[start + k] = "("
        dot[start + stem + loop + (stem - 1 - k)] = ")"
    for k in range(loop):
        dot[start + stem + k] = "."


def make_hairpin_record(rng: np.random.Generator, idx: int) -> dict:
    length = int(rng.integers(44, 73))
    seq = _rand_seq(rng, length)
    dot = ["."] * length
    n_hairpins = int(rng.integers(1, 3))
    cursor = int(rng.integers(0, 6))
    for _ in range(n_hairpins):
        stem = int(rng.integers(4, 8))
        loop = int(rng.integers(4, 7))
        span = 2 * stem + loop
        if cursor + span > length - 2:
            break
        _add_hairpin(seq, dot, rng, cursor, stem, loop)
        cursor += span + int(rng.integers(2, 8))
    return {
        "seq_id": f"toy_ss_{idx:04d}",
        "sequence": "".join(seq),
        "dot_bracket": "".join(dot),
        "split": ("train" if idx % 10 < 7 else "validation") if idx % 10 < 9 else "test",
    }


def make_family_record(rng: np.random.Generator, idx: int) -> dict:
    families = {"tetraloop": "GAAA", "duplex": "CGCG", "none": None}
    names = sorted(families)
    label = names[idx % len(names)]
    length = int(rng.integers(40, 65))
    seq = _rand_seq(rng, length)
    motif = families[label]
    if motif is not None:
        pos = int(rng.integers(4, length - 8))
        seq[pos : pos + len(motif)] = motif
    return {
        "seq_id": f"toy_fam_{idx:04d}",
        "sequence": "".join(seq),
        "label": label,
        "split": ("train" if idx % 10 < 7 else "validation") if idx % 10 < 9 else "test",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--output-dir", default="data/demo")
    ap.add_argument("--n-pairwise", type=int, default=48)
    ap.add_argument("--n-seqcls", type=int, default=60)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    ss_path = out_dir / "toy_secondary_structure.jsonl"
    with ss_path.open("w", encoding="utf-8") as f:
        for i in range(args.n_pairwise):
            f.write(json.dumps(make_hairpin_record(rng, i)) + "\n")

    fam_path = out_dir / "toy_family_seqcls.csv"
    with fam_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seq_id", "sequence", "label", "split"])
        writer.writeheader()
        for i in range(args.n_seqcls):
            writer.writerow(make_family_record(rng, i))

    print(f"[demo] wrote {args.n_pairwise} pairwise records -> {ss_path}")
    print(f"[demo] wrote {args.n_seqcls} seqcls records -> {fam_path}")


if __name__ == "__main__":
    main()
