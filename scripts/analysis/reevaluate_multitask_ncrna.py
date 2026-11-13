#!/usr/bin/env python3
"""Re-evaluate multitask ncRNA results with fixed metrics code."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # repository root
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/ablation"))

import torch

from train_multitask import _load_stage4b_student, _resolve_checkpoint, _build_probe
from rna_head_alignment.probes.multitask import MultiTaskHead2HeadStudent
from rna_head_alignment.probes.datasets import get_downstream_dataloaders
from rna_head_alignment.probes.task_config import downstream_data_dir, TASK_TO_ROUTE_DEPTH, TASK_TO_TYPE
from rna_head_alignment.probes.feature_extractor import ProbeFeatureExtractor
from rna_head_alignment.probes.eval import evaluate_probe


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    checkpoint_path = _resolve_checkpoint(None)
    student = _load_stage4b_student(checkpoint_path, device)

    task = "ncRNA_family_seqcls"
    data_dir = downstream_data_dir(task)
    train_loader, val_loader, test_loader, label_map = get_downstream_dataloaders(
        task=task, data_dir=data_dir, tokenizer=student.backbone.tokenizer(),
        batch_size=16, max_len=512, num_workers=0,
    )
    num_classes = len(label_map) if label_map else None
    print("num_classes:", num_classes)

    extractor = ProbeFeatureExtractor(
        model=student, feature_mode="distilled_hidden", task=task,
        route_depth=TASK_TO_ROUTE_DEPTH[task],
    )
    extractor.to(device)

    for mode in ["joint_no_replay", "joint_with_replay"]:
        for seed in [17, 41, 89]:
            ckpt_path = ROOT / f"outputs/probes/multitask/{mode}/seed_{seed}/best_checkpoint.pt"
            if not ckpt_path.exists():
                print(f"Missing: {ckpt_path}")
                continue

            probe = _build_probe(TASK_TO_TYPE[task], student.backbone.hidden_size, num_classes)
            multitask_model = MultiTaskHead2HeadStudent(student=student, probes={task: probe})
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            multitask_model.load_state_dict(ckpt["model_state_dict"], strict=False)
            multitask_model.to(device)
            multitask_model.eval()

            for split, loader in [("val", val_loader), ("test", test_loader)]:
                result = evaluate_probe(
                    multitask_model.probes[task], extractor, loader, task, device,
                    num_classes=num_classes, mixed_precision=False,
                )
                out_path = ROOT / f"outputs/probes/multitask/{mode}/seed_{seed}/{task}/eval_{split}_fixed.json"
                with open(out_path, "w", encoding="utf-8") as fh:
                    json.dump(result["metrics"], fh, indent=2)
                print(f"{mode}/seed_{seed}/{split}: {result['metrics']}")


if __name__ == "__main__":
    main()
