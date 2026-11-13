from __future__ import annotations

import ctypes
import gc
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from tqdm import tqdm

from rna_head_alignment.io.manifests import write_csv, write_json
from rna_head_alignment.io.paths import get_device, load_yaml, resolve_path
from rna_head_alignment.models.attention_extractors import extract_attention_bundle, extract_attention_bundle_batch
from rna_head_alignment.models.loaders import load_frozen_model
from rna_head_alignment.models.registry import get_model_spec


def read_sequence_jsonl(path: str | Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows = []
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if limit and len(rows) >= limit:
                break
            if not line.strip():
                continue
            obj = json.loads(line)
            seq = str(obj.get("sequence", "")).upper().replace("T", "U")
            if not seq:
                continue
            rows.append(
                {
                    "seq_id": str(obj.get("seq_id") or f"seq_{len(rows)}"),
                    "sequence": seq,
                    "length": len(seq),
                }
            )
    return rows


def count_sequence_jsonl(path: str | Path, limit: int | None = None) -> int:
    count = 0
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if limit and count >= limit:
                break
            if line.strip():
                count += 1
    return count


def read_sequence_jsonl_shard(
    path: str | Path,
    shard_size: int,
    shard_index: int,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    start = shard_index * shard_size
    end = start + shard_size
    rows: list[dict[str, Any]] = []
    seen = 0
    with resolve_path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if limit and seen >= limit:
                break
            if not line.strip():
                continue
            if seen < start:
                seen += 1
                continue
            if seen >= end:
                break
            obj = json.loads(line)
            seq = str(obj.get("sequence", "")).upper().replace("T", "U")
            if seq:
                rows.append(
                    {
                        "seq_id": str(obj.get("seq_id") or f"seq_{seen}"),
                        "sequence": seq,
                        "length": len(seq),
                    }
                )
            seen += 1
    return rows


def selected_heads_by_model(teacher_bank: str | Path) -> dict[str, list[dict[str, Any]]]:
    bank = load_yaml(teacher_bank)
    by_model: dict[str, list[dict[str, Any]]] = {}
    for bank_name, bank_cfg in (bank.get("banks") or {}).items():
        for task, task_cfg in (bank_cfg.get("tasks") or {}).items():
            for head in task_cfg.get("teacher_heads") or []:
                item = dict(head)
                item.setdefault("task", task)
                item.setdefault("bank", bank_name)
                by_model.setdefault(str(item["model"]), []).append(item)
    for bank_name, mappings in (bank.get("student_teacher_mapping") or {}).items():
        for mapping in mappings or []:
            for head in mapping.get("teacher_group") or []:
                item = dict(head)
                item.setdefault("task", item.get("task", "manual"))
                item.setdefault("bank", bank_name)
                by_model.setdefault(str(item["model"]), []).append(item)
    for model, heads in by_model.items():
        seen = set()
        unique = []
        for h in heads:
            key = (int(h["layer"]), int(h["head"]), str(h.get("task", "")))
            if key not in seen:
                unique.append(h)
                seen.add(key)
        by_model[model] = unique
    return by_model


def _head_key(model_name: str, head: dict[str, Any]) -> str:
    return (
        f"{model_name}:L{int(head['layer'])}:H{int(head['head'])}:"
        f"{head.get('bank', '')}:{head.get('task', '')}"
    )


def _pad_stack(items: list[torch.Tensor], fill: float = 0.0) -> torch.Tensor:
    max_len = max(int(x.shape[-1]) for x in items)
    out = []
    for x in items:
        k, l, _ = x.shape
        padded = torch.full((k, max_len, max_len), fill, dtype=x.dtype)
        padded[:, :l, :l] = x
        out.append(padded)
    return torch.stack(out, dim=0)


def build_teacher_attention_cache(
    teacher_bank: str | Path,
    sequences: str | Path,
    output: str | Path,
    models_yaml: str | Path = "configs/models.yaml",
    max_len: int = 512,
    shard_size: int = 64,
    device: str = "auto",
    limit: int | None = None,
    model_filter: str | None = None,
    shard_index: int | None = None,
    shard_start: int | None = None,
    shard_end: int | None = None,
    resume: bool = True,
    dtype: str = "fp16",
    max_tokens_per_batch: int = 4096,
    sub_batch_size: int = 256,
) -> list[dict[str, Any]]:
    out_root = resolve_path(output)
    out_root.mkdir(parents=True, exist_ok=True)
    if shard_index is not None:
        seqs = read_sequence_jsonl_shard(sequences, shard_size=shard_size, shard_index=shard_index, limit=limit)
        shard_batches = [(shard_index, seqs, shard_index * shard_size)]
    elif shard_start is not None or shard_end is not None:
        start = shard_start or 0
        end = shard_end
        shard_batches = []
        for shard_no in range(start, end + 1):
            seqs = read_sequence_jsonl_shard(sequences, shard_size=shard_size, shard_index=shard_no, limit=limit)
            shard_batches.append((shard_no, seqs, shard_no * shard_size))
    else:
        seqs = read_sequence_jsonl(sequences, limit=limit)
        shard_batches = [
            (shard_no, seqs[start : start + shard_size], start)
            for shard_no, start in enumerate(range(0, len(seqs), shard_size))
        ]
    by_model = selected_heads_by_model(teacher_bank)
    if model_filter:
        by_model = {model_filter: by_model.get(model_filter, [])}
    manifest = []
    if not by_model:
        write_json({"warning": "no selected teacher heads", "teacher_bank": str(teacher_bank)}, out_root / "manifest.json")
        write_csv([], out_root / "manifest.csv")
        return []
    resolved_device = get_device(device)
    for model_name, heads in by_model.items():
        if not heads:
            continue
        spec = get_model_spec(models_yaml, model_name)
        tokenizer, model = load_frozen_model(spec, resolved_device)
        model_dir = out_root / f"model={model_name}"
        model_dir.mkdir(parents=True, exist_ok=True)
        for shard_no, batch, global_start in shard_batches:
            shard_path = model_dir / f"shard_{shard_no:05d}.pt"
            if resume and shard_path.exists() and shard_path.stat().st_size > 0:
                manifest.append(
                    {
                        "model": model_name,
                        "shard": shard_no,
                        "path": str(shard_path),
                        "status": "exists",
                        "sequence_start": global_start,
                        "sequence_end": global_start + len(batch),
                        "n_sequences": len(batch),
                        "n_heads": len(heads),
                    }
                )
                continue
            maps = []
            valid_masks = []
            seq_ids = []
            kept_sequences = []
            lengths = []
            errors = []
            sub_pbar = tqdm(total=len(batch), desc=f"cache {model_name} shard {shard_no}", leave=False)
            original_sub_batch_size = sub_batch_size
            for sub_start in range(0, len(batch), sub_batch_size):
                sub_batch = batch[sub_start : sub_start + sub_batch_size]
                remaining = sub_batch
                current_size = min(original_sub_batch_size, len(remaining))
                while remaining:
                    chunk = remaining[:current_size]
                    try:
                        sub_iter = extract_attention_bundle_batch(
                            spec,
                            tokenizer,
                            model,
                            chunk,
                            max_len=max_len,
                            device=resolved_device,
                            max_tokens_per_batch=max_tokens_per_batch,
                            output_hidden_states=False,
                            output_attentions=True,
                            selected_heads=heads,
                        )
                        for item, bundle in zip(chunk, sub_iter):
                            try:
                                if bundle.attentions is None:
                                    raise RuntimeError("model returned no attentions")
                                out_dtype = torch.float16 if dtype.lower() in {"fp16", "float16", "half"} else torch.float32
                                # Selected heads are already extracted and mapped to nucleotide
                                # positions by the extractor; the result is [K, L, L].
                                stacked = bundle.attentions.to(dtype=out_dtype).detach().cpu()
                                maps.append(stacked)
                                valid_masks.append(torch.ones(stacked.shape[-1], dtype=torch.bool))
                                seq_ids.append(item["seq_id"])
                                kept_sequences.append(bundle.sequence)
                                lengths.append(stacked.shape[-1])
                            except Exception as e:
                                import traceback
                                traceback.print_exc()
                                print(f"SEQ ERROR {model_name} shard {shard_no} seq {item['seq_id']}: {repr(e)}", flush=True)
                                errors.append({"seq_id": item["seq_id"], "error": repr(e)})
                        sub_pbar.update(len(chunk))
                        remaining = remaining[current_size:]
                        current_size = min(original_sub_batch_size, len(remaining))
                    except torch.cuda.OutOfMemoryError as e:
                        # Release cached memory and retry with a smaller chunk.
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        if current_size == 1:
                            print(f"OOM single-seq {model_name} shard {shard_no} seq {remaining[0]['seq_id']}: {repr(e)}", flush=True)
                            errors.append({"seq_id": remaining[0]["seq_id"], "error": repr(e)})
                            sub_pbar.update(1)
                            remaining = remaining[1:]
                            current_size = min(original_sub_batch_size, len(remaining))
                        else:
                            new_size = max(1, current_size // 2)
                            print(f"OOM retry {model_name} shard {shard_no} reducing sub-batch {current_size} -> {new_size}", flush=True)
                            current_size = new_size
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"SUBBATCH ERROR {model_name} shard {shard_no}: {repr(e)}", flush=True)
                        for item in remaining[:current_size]:
                            errors.append({"seq_id": item["seq_id"], "error": repr(e)})
                        sub_pbar.update(current_size)
                        remaining = remaining[current_size:]
                        current_size = min(original_sub_batch_size, len(remaining))
            sub_pbar.close()
            if maps:
                attn = _pad_stack(maps)
                lmax = attn.shape[-1]
                mask = torch.zeros((len(valid_masks), lmax), dtype=torch.bool)
                for i, m in enumerate(valid_masks):
                    mask[i, : m.numel()] = m
                payload = {
                    "model": model_name,
                    "task_or_bank": sorted({str(h.get("bank", "")) for h in heads}),
                    "selected_heads": heads,
                    "head_keys": [_head_key(model_name, h) for h in heads],
                    "seq_ids": seq_ids,
                    "sequences": kept_sequences,
                    "lengths": torch.tensor(lengths, dtype=torch.long),
                    "attention": attn,
                    "valid_mask": mask,
                    "attention_dtype": str(attn.dtype),
                    "errors": errors,
                    "attention_layout": "attention[b, k, i, j] where k follows selected_heads/head_keys",
                    "source_sequences": str(resolve_path(sequences)),
                    "sequence_start": global_start,
                    "sequence_end": global_start + len(batch),
                }
                torch.save(payload, shard_path)
                status = "ok"
                n_seqs = len(maps)
                n_errs = len(errors)
                # Explicit cleanup to avoid memory accumulation across shards
                del attn, mask, payload, maps, valid_masks, seq_ids, kept_sequences, lengths, errors
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                # Force glibc to return freed memory to the OS; without this the
                # process RSS can grow monotonically and trigger Slurm OOM.
                try:
                    ctypes.CDLL("libc.so.6").malloc_trim(0)
                except Exception:
                    pass
            else:
                status = "empty"
                n_seqs = 0
                n_errs = len(errors)
            manifest.append(
                {
                    "model": model_name,
                    "shard": shard_no,
                    "path": str(shard_path),
                    "status": status,
                    "sequence_start": global_start,
                    "sequence_end": global_start + len(batch),
                    "n_sequences": n_seqs,
                    "n_heads": len(heads),
                    "errors": n_errs,
                }
            )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
    part_dir = out_root / "manifest_parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    part_stem = "manifest"
    if model_filter is not None:
        part_stem += f"_{model_filter}"
    if shard_index is not None:
        part_stem += f"_shard_{shard_index:05d}"
    write_csv(manifest, part_dir / f"{part_stem}.csv")
    write_json(manifest, part_dir / f"{part_stem}.json")
    if model_filter is None and shard_index is None:
        write_csv(manifest, out_root / "manifest.csv")
        write_json(manifest, out_root / "manifest.json")
    return manifest


def make_teacher_cache_jobs(
    teacher_bank: str | Path,
    sequences: str | Path,
    output: str | Path,
    shard_size: int = 64,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    n_sequences = count_sequence_jsonl(sequences, limit=limit)
    by_model = selected_heads_by_model(teacher_bank)
    rows: list[dict[str, Any]] = []
    job_id = 0
    n_shards = (n_sequences + shard_size - 1) // shard_size if n_sequences else 0
    for model, heads in by_model.items():
        for shard in range(n_shards):
            rows.append(
                {
                    "job_id": job_id,
                    "model": model,
                    "shard_index": shard,
                    "n_heads": len(heads),
                    "shard_size": shard_size,
                    "n_sequences_total": n_sequences,
                    "enabled": bool(heads),
                }
            )
            job_id += 1
    write_csv(rows, output)
    return rows


def collect_teacher_cache_manifests(cache: str | Path, output: str | Path | None = None) -> list[dict[str, Any]]:
    root = resolve_path(cache)
    part_dir = root / "manifest_parts"
    rows: list[dict[str, Any]] = []
    if part_dir.exists():
        for path in sorted(part_dir.glob("*.csv")):
            if path.stat().st_size == 0:
                continue
            rows.extend(pd.read_csv(path).to_dict("records"))
    if not rows and (root / "manifest.csv").exists():
        rows.extend(pd.read_csv(root / "manifest.csv").to_dict("records"))
    out = resolve_path(output) if output else root / "manifest.csv"
    write_csv(rows, out)
    write_json(rows, Path(out).with_suffix(".json"))
    return rows
