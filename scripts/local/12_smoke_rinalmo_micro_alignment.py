from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.optim import AdamW

from rna_head_alignment.alignment.aligners import LayerHeadAlignerBank
from rna_head_alignment.alignment.attention_losses import mask_normalized_mse
from rna_head_alignment.alignment.student_model import RiNALMoMicroStudent, load_student_head_config


def read_sequences(path: Path, n: int, max_len: int) -> list[dict[str, str]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            seq = str(obj["sequence"]).upper().replace("T", "U")[:max_len]
            if seq:
                rows.append({"seq_id": str(obj.get("seq_id", f"seq_{len(rows)}")), "sequence": seq})
            if len(rows) >= n:
                break
    if not rows:
        raise ValueError(f"No sequences read from {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Local RiNALMo-micro alignment smoke test.")
    parser.add_argument("--sequences", default="outputs/task_samples/rinalmo_pretrain_10pct_len_sorted.jsonl")
    parser.add_argument("--config", default="configs/student_alignment.yaml")
    parser.add_argument("--models", default="configs/models.yaml")
    parser.add_argument("--output", default="outputs/student_alignment/local_rinalmo_micro_smoke")
    parser.add_argument("--n-sequences", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--max-len", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=1e-6)
    args = parser.parse_args()

    out = Path(args.output)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    seqs = read_sequences(Path(args.sequences), args.n_sequences, args.max_len)
    selected = load_student_head_config(args.config)
    mapping = {bank: cfg.heads for bank, cfg in selected.items()}

    student = RiNALMoMicroStudent(models_yaml=args.models, device=args.device)
    aligners = LayerHeadAlignerBank(mapping).to(student.device)
    optimizer = AdamW(list(student.model.parameters()) + list(aligners.parameters()), lr=args.lr)

    teacher_targets: list[dict[str, torch.Tensor]] = []
    with torch.no_grad():
        for item in seqs:
            targets = student.extract_selected_attentions(item["sequence"], selected, max_len=args.max_len)
            teacher_targets.append({k: v.detach().cpu() for k, v in targets.items()})

    log_path = out / "train_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["step", "seq_id", "loss_total", "loss_L6", "loss_L8", "loss_L12", "seq_len"],
        )
        writer.writeheader()
        for step in range(args.steps):
            index = step % len(seqs)
            item = seqs[index]
            current = student.extract_selected_attentions(item["sequence"], selected, max_len=args.max_len)
            losses: dict[str, torch.Tensor] = {}
            valid = torch.ones((1, len(item["sequence"])), dtype=torch.bool, device=student.device)
            for bank, attn_stack in current.items():
                cfg = selected[bank]
                target_stack = teacher_targets[index][bank].to(student.device)
                per_head_losses = []
                for local_i, student_head in enumerate(cfg.heads):
                    pred = aligners.forward_one(bank, student_head, attn_stack[local_i].unsqueeze(0))
                    teacher = target_stack[local_i].unsqueeze(0)
                    per_head_losses.append(mask_normalized_mse(pred, teacher, valid))
                losses[bank] = torch.stack(per_head_losses).mean()
            loss_total = torch.stack(list(losses.values())).mean()
            optimizer.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(list(student.model.parameters()) + list(aligners.parameters()), 1.0)
            optimizer.step()
            writer.writerow(
                {
                    "step": step,
                    "seq_id": item["seq_id"],
                    "loss_total": float(loss_total.detach().cpu()),
                    "loss_L6": float(losses.get("L6", torch.tensor(float("nan"))).detach().cpu()),
                    "loss_L8": float(losses.get("L8", torch.tensor(float("nan"))).detach().cpu()),
                    "loss_L12": float(losses.get("L12", torch.tensor(float("nan"))).detach().cpu()),
                    "seq_len": len(item["sequence"]),
                }
            )

    ckpt = out / "checkpoints" / "local_smoke_last.pt"
    torch.save(
        {
            "student_state_dict": student.model.state_dict(),
            "aligners": aligners.state_dict(),
            "selected": {k: {"layer_index": v.layer, "heads": v.heads, "display_layer": v.display_layer} for k, v in selected.items()},
            "sequences": seqs,
        },
        ckpt,
    )
    report = {
        "ok": True,
        "output": str(out.resolve()),
        "log": str(log_path.resolve()),
        "checkpoint": str(ckpt.resolve()),
        "n_sequences": len(seqs),
        "steps": args.steps,
        "max_len": args.max_len,
        "device": args.device,
        "selected": {k: {"layer_index": v.layer, "display_layer": v.display_layer, "heads": v.heads} for k, v in selected.items()},
        "note": "Local smoke uses RiNALMo-micro initial attentions as temporary teacher targets; server run uses selected external teacher-head cache.",
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
