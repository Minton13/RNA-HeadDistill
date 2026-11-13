from __future__ import annotations

from pathlib import Path
from typing import Iterable

from rna_head_alignment.data.rinalmo_corpus import sample_corpus


def sample_sequences(
    corpus: str | Path,
    output: str | Path,
    n: int | None = None,
    fraction: float | None = None,
    min_len: int = 32,
    max_len: int = 512,
    deduplicate: bool = True,
    sequence_column: str = "sequence",
) -> int:
    return sample_corpus(
        corpus=corpus,
        output=output,
        n=n,
        fraction=fraction,
        min_len=min_len,
        max_len=max_len,
        deduplicate=deduplicate,
        sequence_column=sequence_column,
    )
