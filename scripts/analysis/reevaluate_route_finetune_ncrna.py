#!/usr/bin/env python3
"""Re-evaluate route-finetune ncRNA_family_seqcls cells with fixed metrics.

Background: the original route fine-tune run evaluated multiclass logits
without softmax, so AUROC/AUPRC/F1/MCC were all NaN and the best checkpoint
was never selected (``best_val_metric=-inf``; eval fell on the final state).
The metrics code in ``src/rna_head_alignment/probes/metrics.py`` has since
been fixed (softmax + per-class one-vs-rest).  This script re-evaluates the
15 existing cells (F1 + F2 x {0, 0.01, 0.1, 0.5}) x seeds {17, 41, 89} from
their saved checkpoints.  **No training is performed.**

Per cell:
  1. Evaluate every ``checkpoint_step_*.pt`` on the val split (fixed metrics).
  2. Select the checkpoint with the best val AUROC (the training-primary
     metric for sequence classification).
  3. Evaluate the selected checkpoint on the test split.
  4. Write ``eval_val_fixed.json`` / ``eval_test_fixed.json``,
     ``predictions_{val,test}_fixed.npz`` and ``checkpoint_selection.json``
     into the cell directory.  Original (buggy) files are left untouched.

Usage:
    PYTHONPATH=src python scripts/analysis/reevaluate_route_finetune_ncrna.py \
        [--cells F1/seed_17/lambda_0.0 ...]  [--limit-cells N]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/ablation"))

from run_route_finetune import _build_probe, _load_stage4b_student, _resolve_checkpoint
from rna_head_alignment.probes.datasets import get_downstream_dataloaders
from rna_head_alignment.probes.eval import evaluate_probe
from rna_head_alignment.probes.feature_extractor import ProbeFeatureExtractor
from rna_head_alignment.probes.task_config import TASK_TO_ROUTE_DEPTH, downstream_data_dir

TASK = "ncRNA_family_seqcls"
SEEDS = [17, 41, 89]
LAMBDAS = ["0.0", "0.01", "0.1", "0.5"]
FINETUNE_ROOT = ROOT / "outputs/probes/finetune"


def _load_cell_checkpoint(student, probe, ckpt_path: Path, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    student_state = ckpt.get("student_state_dict")
    probe_state = ckpt.get("probe_state_dict")
    if student_state is None or probe_state is None:
        raise RuntimeError(f"{ckpt_path} missing student/probe state dict")
    student.load_state_dict(student_state, strict=True)
    probe.load_state_dict(probe_state, strict=True)
    student.to(device).eval()
    probe.to(device).eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cells", nargs="*", default=None,
                        help="Limit to cells like F1/seed_17/lambda_0.0 (default: all 15)")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if (args.device == "auto" and torch.cuda.is_available()) else ("cpu" if args.device == "auto" else args.device)
    )
    print("device:", device, flush=True)

    student = _load_stage4b_student(_resolve_checkpoint(None), device)
    tokenizer = student.backbone.tokenizer()

    data_dir = downstream_data_dir(TASK)
    train_loader, val_loader, test_loader, label_map = get_downstream_dataloaders(
        task=TASK, data_dir=data_dir, tokenizer=tokenizer,
        batch_size=16, max_len=512, num_workers=0,
    )
    num_classes = len(label_map) if label_map else None
    print("num_classes:", num_classes, flush=True)

    route_depth = TASK_TO_ROUTE_DEPTH[TASK]
    extractor = ProbeFeatureExtractor(
        model=student, feature_mode="distilled_hidden", task=TASK, route_depth=route_depth,
    )
    extractor.to(device)

    cells: list[tuple[str, str, str]] = []
    for seed in SEEDS:
        cells.append(("F1", f"seed_{seed}", "lambda_0.0"))
        for lam in LAMBDAS:
            cells.append(("F2", f"seed_{seed}", f"lambda_{lam}"))
    if args.cells:
        wanted = set(args.cells)
        cells = [c for c in cells if f"{c[0]}/{c[1]}/{c[2]}" in wanted]

    for mode, seed_dir, lam_dir in cells:
        cell_dir = FINETUNE_ROOT / TASK / mode / seed_dir / lam_dir
        if not cell_dir.is_dir():
            print(f"[skip] missing cell dir {cell_dir}", flush=True)
            continue
        ckpts = sorted(cell_dir.glob("checkpoint_step_*.pt"),
                       key=lambda p: int(p.stem.rsplit("_", 1)[1]))
        if not ckpts:
            print(f"[skip] no step checkpoints in {cell_dir}", flush=True)
            continue

        probe = _build_probe("sequence_classification", student.backbone.hidden_size, num_classes)
        probe.to(device)

        val_results: list[dict] = []
        for ckpt_path in ckpts:
            _load_cell_checkpoint(student, probe, ckpt_path, device)
            res = evaluate_probe(probe, extractor, val_loader, TASK, device,
                                 num_classes=num_classes, mixed_precision=False)
            auroc = res["metrics"].get("auroc", float("nan"))
            val_results.append({"checkpoint": ckpt_path.name,
                                "step": int(ckpt_path.stem.rsplit("_", 1)[1]),
                                "val_metrics": res["metrics"]})
            print(f"[{mode}/{seed_dir}/{lam_dir}] {ckpt_path.name} val_auroc={auroc}", flush=True)

        best = max(val_results, key=lambda r: (not np.isnan(r["val_metrics"].get("auroc", float("nan"))),
                                               r["val_metrics"].get("auroc", float("-inf"))))
        best_ckpt = cell_dir / best["checkpoint"]

        _load_cell_checkpoint(student, probe, best_ckpt, device)
        for split, loader in [("val", val_loader), ("test", test_loader)]:
            res = evaluate_probe(probe, extractor, loader, TASK, device,
                                 num_classes=num_classes, mixed_precision=False,
                                 return_predictions=True)
            metrics = {k: v for k, v in res["metrics"].items()}
            (cell_dir / f"eval_{split}_fixed.json").write_text(
                json.dumps(metrics, indent=2, default=str), encoding="utf-8")
            np.savez(cell_dir / f"predictions_{split}_fixed.npz",
                     predictions=res["predictions"], labels=res["labels"])
            print(f"[{mode}/{seed_dir}/{lam_dir}] best={best['checkpoint']} {split}: {metrics}", flush=True)

        selection = {
            "selected_checkpoint": best["checkpoint"],
            "selected_step": best["step"],
            "selection_metric": "val_auroc",
            "note": "re-evaluated with fixed multiclass metrics (softmax); no retraining",
            "per_checkpoint_val": val_results,
        }
        (cell_dir / "checkpoint_selection.json").write_text(
            json.dumps(selection, indent=2, default=str), encoding="utf-8")

    print("done", flush=True)


if __name__ == "__main__":
    main()
