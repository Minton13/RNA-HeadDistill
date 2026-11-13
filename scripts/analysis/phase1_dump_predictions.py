#!/usr/bin/env python3
"""Phase 1.2: dump per-sequence predictions from a trained frozen-probe cell.

For each (task, mode, seed) cell this script rebuilds the exact model used by
``run_frozen_probe.py``, loads ``best_checkpoint.pt``, runs inference on
val/test, and stores per-sequence predictions for downstream bootstrap
statistics.  It also recomputes the published metrics from the dumped
predictions and asserts they match ``eval_{split}.json`` (tolerance 1e-3).

Outputs (under ``outputs/probes/predictions/{task}/{mode}/seed_{seed}/``):

* seqcls: ``pred_{split}.parquet`` with seq_id/label/logits
* token regression: ``pred_{split}.pt`` list of per-seq dicts
* pairwise: ``pred_{split}.pt`` list of per-seq dicts (fp16 prob maps)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT / "scripts" / "ablation") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "ablation"))

from run_frozen_probe import (  # noqa: E402
    _build_probe,
    _load_stage4b_student,
    _load_vanilla_backbone,
    _resolve_checkpoint,
    _set_seed,
)
from rna_head_alignment.io.paths import get_device  # noqa: E402
from rna_head_alignment.probes.datasets import get_downstream_dataloaders  # noqa: E402
from rna_head_alignment.probes.feature_extractor import ProbeFeatureExtractor  # noqa: E402
from rna_head_alignment.probes.metrics import (  # noqa: E402
    _top_k_overlap_per_sample,
    pairwise_metrics_streaming,
    sequence_classification_metrics,
    token_regression_metrics,
)
from rna_head_alignment.probes.task_config import (  # noqa: E402
    TASK_TO_ROUTE_DEPTH,
    TASK_TO_TYPE,
    default_teacher_cache_dir,
    downstream_data_dir,
)

PRED_ROOT = Path("outputs/probes/predictions")
CELL_ROOT = Path("outputs/probes/frozen")


def _load_cell(task: str, mode: str, seed: int, device: torch.device,
               cell_root: Path = CELL_ROOT, checkpoint: str | None = None):
    """Rebuild extractor + probe and load the trained cell weights."""
    cell_dir = cell_root / task / mode / f"seed_{seed}"
    ckpt = torch.load(cell_dir / "best_checkpoint.pt", map_location=device, weights_only=False)

    if mode == "vanilla_hidden":
        model = _load_vanilla_backbone(device)
    else:
        model = _load_stage4b_student(_resolve_checkpoint(checkpoint), device)
    for p in model.parameters():
        p.requires_grad_(False)

    route_depth = TASK_TO_ROUTE_DEPTH[task]
    extractor = ProbeFeatureExtractor(
        model=model,
        feature_mode=mode,
        task=task,
        route_depth=route_depth,
        teacher_cache_dir=default_teacher_cache_dir(task),
    )
    extractor.to(device)
    # Lazily create gates/projection so state-dict keys exist.
    if mode == "distilled_hidden_teacher":
        extractor._ensure_components(5)
    else:
        dummy_ids = torch.zeros((1, 10), dtype=torch.long, device=device)
        dummy_mask = torch.ones((1, 10), dtype=torch.long, device=device)
        with torch.no_grad():
            _ = extractor(dummy_ids, dummy_mask, seq_ids=["dummy"])

    missing, unexpected = extractor.load_state_dict(
        ckpt["feature_extractor_state_dict"], strict=False
    )
    crit_missing = [
        k for k in missing
        if k in ("gates", "residual_gate")
        or k.startswith("projection.")
        or k.startswith("attended_proj.")
    ]
    if crit_missing:
        raise RuntimeError(f"Critical extractor params missing on load: {crit_missing}")

    label_map = json.loads((cell_dir / "label_map.json").read_text()) if (cell_dir / "label_map.json").exists() else None
    num_classes = len(label_map) if label_map else None
    probe = _build_probe(TASK_TO_TYPE[task], extractor.output_dim(), num_classes)
    probe.to(device)
    probe.load_state_dict(ckpt["probe_state_dict"], strict=True)

    extractor.eval()
    probe.eval()
    return extractor, probe, num_classes, ckpt.get("summary", {})


@torch.no_grad()
def _infer_seqcls(extractor, probe, loader, device):
    rows = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        seq_ids = batch.get("seq_ids", [""] * input_ids.shape[0])
        features = extractor(input_ids, attention_mask, seq_ids=list(seq_ids))
        logits = probe(features, mask=attention_mask)
        logits_np = logits.detach().float().cpu().numpy()
        labels_np = batch["labels"].numpy()
        for i, sid in enumerate(seq_ids):
            rows.append({
                "seq_id": sid,
                "label": labels_np[i].item() if hasattr(labels_np[i], "item") else labels_np[i],
                "logits": logits_np[i].tolist(),
            })
    return rows


@torch.no_grad()
def _infer_token(extractor, probe, loader, device):
    seqs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        seq_ids = batch.get("seq_ids", [])
        features = extractor(input_ids, attention_mask, seq_ids=list(seq_ids))
        preds = probe(features)
        mask = batch["label_mask"].cpu().numpy().astype(bool)
        labels = batch["labels"].numpy()
        preds_np = preds.detach().float().cpu().numpy()
        for i, sid in enumerate(seq_ids):
            m = mask[i]
            seqs.append({
                "seq_id": sid,
                "preds": preds_np[i][m].astype(np.float32),
                "labels": labels[i][m].astype(np.float32),
            })
    return seqs


@torch.no_grad()
def _infer_pairwise(extractor, probe, loader, device):
    seqs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        seq_ids = batch.get("seq_ids", [])
        features = extractor(input_ids, attention_mask, seq_ids=list(seq_ids))
        logits = probe(features)
        probs = torch.sigmoid(logits).detach().cpu()
        mask = batch["label_mask"].cpu()
        labels = batch["labels"].cpu()
        lengths = batch["lengths"].cpu()
        for i, sid in enumerate(seq_ids):
            L = int(lengths[i].item())
            seqs.append({
                "seq_id": sid,
                "length": L,
                "probs": probs[i, :L, :L].half(),
                "labels": labels[i, :L, :L].float(),
                "mask": mask[i, :L, :L].bool(),
            })
    return seqs


def _metrics_from_dump(task: str, num_classes: int | None, dumped) -> dict[str, float]:
    task_type = TASK_TO_TYPE[task]
    if task_type == "sequence_classification":
        labels = np.array([r["label"] for r in dumped])
        logits = np.array([r["logits"] for r in dumped])
        if logits.ndim > 1 and logits.shape[1] == 1:
            logits = logits[:, 0]
        return sequence_classification_metrics(labels, logits, num_classes or 2)
    if task_type == "token_regression":
        preds = np.concatenate([s["preds"] for s in dumped])
        labels = np.concatenate([s["labels"] for s in dumped])
        return token_regression_metrics(labels, preds, np.ones_like(labels, dtype=bool))
    if task_type == "pairwise":
        flat_true, flat_probs, top_l, top_2l = [], [], [], []
        for s in dumped:
            m = s["mask"].numpy().astype(bool)
            flat_true.append(s["labels"].numpy()[m])
            flat_probs.append(s["probs"].float().numpy()[m])
            L = s["length"]
            if L > 1:
                top_l.append(_top_k_overlap_per_sample(s["probs"].float(), s["labels"], L, k=L))
                top_2l.append(_top_k_overlap_per_sample(s["probs"].float(), s["labels"], L, k=2 * L))
        return pairwise_metrics_streaming(
            np.concatenate(flat_true), np.concatenate(flat_probs), top_l, top_2l
        )
    raise ValueError(task_type)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--feature-mode", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-len", type=int, default=512)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cell-root", default=str(CELL_ROOT),
                        help="Root containing trained probe cells (default outputs/probes/frozen)")
    parser.add_argument("--pred-root", default=str(PRED_ROOT),
                        help="Root for dumped predictions (default outputs/probes/predictions)")
    parser.add_argument("--checkpoint", default=None,
                        help="Override student checkpoint (e.g. A8 adapters-only best_val.pt)")
    args = parser.parse_args()

    task, mode, seed = args.task, args.feature_mode, args.seed
    _set_seed(seed)
    device = torch.device(get_device(args.device))
    cell_root = Path(args.cell_root)

    out_dir = Path(args.pred_root) / task / mode / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    extractor, probe, num_classes, summary = _load_cell(
        task, mode, seed, device, cell_root=cell_root, checkpoint=args.checkpoint
    )
    print(f"[dump] loaded cell {task}/{mode}/seed_{seed} best_val={summary.get('best_val_metric')}", flush=True)

    tokenizer = (
        extractor.backbone.tokenizer()
        if hasattr(extractor.backbone, "tokenizer")
        else extractor.model.backbone.tokenizer()
    )
    train_loader, val_loader, test_loader, _ = get_downstream_dataloaders(
        task=task,
        data_dir=downstream_data_dir(task),
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_len=args.max_len,
        num_workers=0,
    )
    split_loaders = {"val": val_loader, "test": test_loader}

    task_type = TASK_TO_TYPE[task]
    infer_fn = {
        "sequence_classification": _infer_seqcls,
        "token_regression": _infer_token,
        "pairwise": _infer_pairwise,
    }[task_type]

    check = {}
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        loader = split_loaders[split]
        dumped = infer_fn(extractor, probe, loader, device)
        if task_type == "sequence_classification":
            pd.DataFrame(dumped).to_parquet(out_dir / f"pred_{split}.parquet", index=False)
        else:
            torch.save(dumped, out_dir / f"pred_{split}.pt")
        # Consistency check against published metrics.
        metrics = _metrics_from_dump(task, num_classes, dumped)
        ref_path = cell_root / task / mode / f"seed_{seed}" / f"eval_{split}.json"
        ref = json.loads(ref_path.read_text()) if ref_path.exists() else {}
        diffs = {
            k: abs(metrics.get(k, float("nan")) - ref.get(k, float("nan")))
            for k in ref
            if isinstance(ref.get(k), float) and ref.get(k) == ref.get(k)
        }
        max_diff = max(diffs.values()) if diffs else 0.0
        check[split] = {"metrics": metrics, "max_abs_diff_vs_eval_json": max_diff}
        print(f"[dump] {task}/{mode}/seed_{seed} {split}: n={len(dumped)} max_diff={max_diff:.6f}", flush=True)
        if max_diff > 1e-3:
            print(f"[dump][WARN] metric mismatch on {split}: recomputed={metrics} ref={ref}", flush=True)

    with open(out_dir / "dump_check.json", "w", encoding="utf-8") as fh:
        json.dump(check, fh, indent=2, default=str)
    print(f"[dump] done {task}/{mode}/seed_{seed}", flush=True)


if __name__ == "__main__":
    main()
