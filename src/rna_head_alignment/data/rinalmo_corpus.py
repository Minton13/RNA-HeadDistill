from __future__ import annotations

import gzip
import json
import random
import time
from pathlib import Path
from typing import Iterator

import pandas as pd
from Bio import SeqIO

from rna_head_alignment.io.paths import ensure_parent, resolve_path


def _normalize(seq: str) -> str:
    return "".join(ch for ch in str(seq).upper().replace("T", "U") if ch in "ACGUN")


def _iter_jsonl(path: Path, sequence_column: str) -> Iterator[dict[str, str]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            obj = json.loads(line)
            seq = _normalize(obj.get(sequence_column) or obj.get("sequence") or obj.get("seq") or "")
            if seq:
                yield {"seq_id": str(obj.get("seq_id") or obj.get("id") or f"{path.stem}_{i}"), "sequence": seq}


def _iter_fasta(path: Path) -> Iterator[dict[str, str]]:
    with (gzip.open(path, "rt") if path.suffix == ".gz" else path.open("r", encoding="utf-8")) as f:
        for record in SeqIO.parse(f, "fasta"):
            seq = _normalize(str(record.seq))
            if seq:
                yield {"seq_id": str(record.id), "sequence": seq}


def _iter_table(path: Path, sequence_column: str) -> Iterator[dict[str, str]]:
    if path.suffix.lower() == ".parquet":
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(path)
            schema_names = set(pf.schema_arrow.names)
            seq_col = sequence_column if sequence_column in schema_names else None
            if seq_col is None:
                for candidate in ("sequence", "seq", "Sequence"):
                    if candidate in schema_names:
                        seq_col = candidate
                        break
            if seq_col is None:
                raise ValueError(f"{path} has no sequence column; columns={pf.schema_arrow.names}")
            id_col = None
            for candidate in ("seq_id", "id", "name", "rnacentral_id", "urs"):
                if candidate in schema_names:
                    id_col = candidate
                    break
            columns = [seq_col] + ([id_col] if id_col else [])
            offset = 0
            for batch in pf.iter_batches(columns=columns, batch_size=100_000):
                data = batch.to_pydict()
                seqs = data[seq_col]
                ids = data.get(id_col) if id_col else None
                for j, raw in enumerate(seqs):
                    seq = _normalize(raw)
                    if seq:
                        seq_id = str(ids[j]) if ids is not None else f"{path.stem}_{offset + j}"
                        yield {"seq_id": seq_id, "sequence": seq}
                offset += len(seqs)
            return
        except ImportError:
            df = pd.read_parquet(path, columns=[sequence_column])
    else:
        sep = "\t" if path.suffix.lower() == ".tsv" else ","
        df = pd.read_csv(path, sep=sep)
    for i, row in df.iterrows():
        seq = _normalize(row.get(sequence_column) or row.get("sequence") or row.get("seq") or "")
        if seq:
            yield {"seq_id": str(row.get("seq_id", row.get("id", f"{path.stem}_{i}"))), "sequence": seq}


def iter_corpus(path: str | Path, sequence_column: str = "sequence") -> Iterator[dict[str, str]]:
    p = resolve_path(path)
    files = sorted(p.rglob("*")) if p.is_dir() else [p]
    for file in files:
        name = file.name.lower()
        if not file.is_file():
            continue
        if name.endswith((".fa", ".fasta", ".fna", ".fa.gz", ".fasta.gz", ".fna.gz")):
            yield from _iter_fasta(file)
        elif name.endswith((".jsonl", ".jsonl.gz")):
            yield from _iter_jsonl(file, sequence_column)
        elif name.endswith((".csv", ".tsv", ".parquet")):
            yield from _iter_table(file, sequence_column)


def sample_corpus(
    corpus: str | Path,
    output: str | Path,
    n: int | None = None,
    fraction: float | None = None,
    min_len: int = 32,
    max_len: int = 512,
    deduplicate: bool = True,
    sequence_column: str = "sequence",
    seed: int = 42,
) -> int:
    if n is None and fraction is None:
        raise ValueError("Provide either n or fraction")
    rng = random.Random(seed)
    seen: set[str] = set()
    pool = []
    processed = 0
    length_skipped = 0
    duplicate_skipped = 0
    sampled_out = 0
    out = ensure_parent(output)

    # Fractional sampling can be very large for the RiNALMo pretraining corpus.
    # Stream directly to disk when there is no explicit cap instead of holding
    # the selected 10% in memory.
    stream_fraction = fraction is not None and n is None
    writer = out.open("w", encoding="utf-8") if stream_fraction else None
    last_log = time.time()
    for item in iter_corpus(corpus, sequence_column=sequence_column):
        processed += 1
        seq = item["sequence"]
        if not (min_len <= len(seq) <= max_len):
            length_skipped += 1
            continue
        if deduplicate and seq in seen:
            duplicate_skipped += 1
            continue
        seen.add(seq)
        item = {
            "seq_id": item["seq_id"],
            "sequence": seq,
            "length": len(seq),
            "source": "rinalmo_pretraining_corpus",
        }
        if stream_fraction:
            if rng.random() <= fraction:
                assert writer is not None
                writer.write(json.dumps(item, ensure_ascii=False) + "\n")
                sampled_out += 1
        elif fraction is not None:
            if rng.random() <= fraction:
                pool.append(item)
        else:
            pool.append(item)
            if len(pool) >= int(n):
                break
        now = time.time()
        if now - last_log > 30:
            print(
                "sample_corpus: "
                f"processed={processed:,} sampled={sampled_out:,} "
                f"length_skipped={length_skipped:,} duplicate_skipped={duplicate_skipped:,}",
                flush=True,
            )
            last_log = now

    if writer is not None:
        writer.close()
    if fraction is not None and n is not None and len(pool) > n:
        pool = rng.sample(pool, n)
    if not stream_fraction:
        with out.open("w", encoding="utf-8") as f:
            for item in pool:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        sampled_out = len(pool)

    manifest = {
        "corpus": str(resolve_path(corpus)),
        "output": str(out),
        "n": n,
        "fraction": fraction,
        "min_len": min_len,
        "max_len": max_len,
        "deduplicate": deduplicate,
        "sequence_column": sequence_column,
        "seed": seed,
        "processed_records": processed,
        "length_skipped": length_skipped,
        "duplicate_skipped": duplicate_skipped,
        "sampled_records": sampled_out,
    }
    out.with_suffix(out.suffix + ".manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return sampled_out
