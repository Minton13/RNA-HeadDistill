#!/usr/bin/env python3
"""Phase 4 diagnostic: hidden-state drift between vanilla and distilled student.

Quantifies the "distillation damage" directly: for the same sequences, compare
the vanilla RiNALMo-micro hidden states with the Stage 4B release student
hidden states at the three route depths (6/8/12).

Metrics per route depth:
* tokenwise cosine similarity (masked mean)
* relative MSE  ||H_a - H_v||^2 / ||H_v||^2
* linear CKA (Gram-accumulated, memory-safe)

Sequences: up to --per-task sequences drawn from each downstream task's test
split (task-relevant, length-diverse).

Output: outputs/probes/analysis_v2/hidden_drift/hidden_drift.csv + .png
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "src"))
if str(_PROJECT_ROOT / "scripts" / "ablation") not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT / "scripts" / "ablation"))

from run_frozen_probe import (  # noqa: E402
    _load_stage4b_student,
    _load_vanilla_backbone,
    _resolve_checkpoint,
    _set_seed,
)
from rna_head_alignment.head2head.sequence_dataset import align_tokens_to_sequence  # noqa: E402
from rna_head_alignment.probes.task_config import TASK_TO_ROUTE_DEPTH, downstream_data_dir  # noqa: E402

ROUTE_DEPTHS = (6, 8, 12)
OUT_DIR = Path("outputs/probes/analysis_v2/hidden_drift")


class _GramAcc:
    """Accumulate centered Gram statistics for linear CKA."""

    def __init__(self) -> None:
        self.n = 0
        self.sum_x = None
        self.sum_y = None
        self.gxx = None
        self.gyy = None
        self.gxy = None

    def update(self, x: torch.Tensor, y: torch.Tensor) -> None:
        # x, y: [N, D] float64 on CPU
        n = x.shape[0]
        sx = x.sum(0)
        sy = y.sum(0)
        if self.n == 0:
            self.sum_x, self.sum_y = sx, sy
            self.gxx = x.T @ x
            self.gyy = y.T @ y
            self.gxy = x.T @ y
            self.n = n
        else:
            self.sum_x += sx
            self.sum_y += sy
            self.gxx += x.T @ x
            self.gyy += y.T @ y
            self.gxy += x.T @ y
            self.n += n

    def cka(self) -> float:
        n = self.n
        cxx = self.gxx - torch.outer(self.sum_x, self.sum_x) / n
        cyy = self.gyy - torch.outer(self.sum_y, self.sum_y) / n
        cxy = self.gxy - torch.outer(self.sum_x, self.sum_y) / n
        hsic = (cxy * cxy).sum()
        norm = torch.sqrt((cxx * cxx).sum() * (cyy * cyy).sum())
        return float(hsic / norm.clamp_min(1e-12))


@torch.no_grad()
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-task", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-len", type=int, default=512)
    args = parser.parse_args()

    _set_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vanilla = _load_vanilla_backbone(device).eval()
    student = _load_stage4b_student(_resolve_checkpoint(None), device).eval()
    distilled = student.backbone if hasattr(student, "backbone") else student
    tokenizer = vanilla.tokenizer()

    # ------------------------------------------------------------- sequences
    seqs: list[tuple[str, str]] = []  # (seq_id, sequence)
    for task in TASK_TO_ROUTE_DEPTH:
        path = Path(downstream_data_dir(task)) / "test.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path, columns=["seq_id", "sequence"])
        df = df.sample(n=min(args.per_task, len(df)), random_state=0)
        seqs.extend(zip(df["seq_id"], df["sequence"]))
    # Sort by length for efficient batching.
    seqs.sort(key=lambda t: len(t[1]))
    print(f"[drift] {len(seqs)} sequences", flush=True)

    stats = {d: {"cos": [], "mse_num": 0.0, "mse_den": 0.0, "cka": _GramAcc()} for d in ROUTE_DEPTHS}

    for start in range(0, len(seqs), args.batch_size):
        batch = seqs[start : start + args.batch_size]
        ids_list, masks_list = [], []
        for _sid, seq in batch:
            ids, seq_len = align_tokens_to_sequence(tokenizer, seq, args.max_len)
            ids_list.append(ids)
            masks_list.append(seq_len)
        Lmax = max(len(t) for t in ids_list)
        input_ids = torch.zeros(len(batch), Lmax, dtype=torch.long)
        attn_mask = torch.zeros(len(batch), Lmax, dtype=torch.long)
        for i, (ids, sl) in enumerate(zip(ids_list, masks_list)):
            input_ids[i, : len(ids)] = ids
            attn_mask[i, :sl] = 1
        input_ids = input_ids.to(device)
        attn_mask = attn_mask.to(device)

        ev = vanilla(input_ids, attn_mask)
        ea = distilled(input_ids, attn_mask)
        m = attn_mask.bool()
        for d in ROUTE_DEPTHS:
            hv = ev[d]["hidden"][m].double().cpu()   # [Ntok, D]
            ha = ea[d]["hidden"][m].double().cpu()
            cos = torch.nn.functional.cosine_similarity(hv, ha, dim=-1)
            stats[d]["cos"].append(cos)
            stats[d]["mse_num"] += float(((hv - ha) ** 2).sum())
            stats[d]["mse_den"] += float((hv**2).sum())
            stats[d]["cka"].update(hv, ha)
        if start % (args.batch_size * 20) == 0:
            print(f"[drift] {start}/{len(seqs)}", flush=True)

    rows = []
    for d in ROUTE_DEPTHS:
        cos = torch.cat(stats[d]["cos"])
        rows.append({
            "route_depth": d,
            "n_tokens": int(cos.numel()),
            "cosine_mean": float(cos.mean()),
            "cosine_p05": float(cos.quantile(0.05)),
            "rel_mse": stats[d]["mse_num"] / max(stats[d]["mse_den"], 1e-12),
            "linear_cka": stats[d]["cka"].cka(),
        })
    df = pd.DataFrame(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_DIR / "hidden_drift.csv", index=False)
    print(df.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax2 = ax.twinx()
    x = np.arange(len(df))
    ax.bar(x - 0.15, df["cosine_mean"], width=0.3, label="tokenwise cosine")
    ax.bar(x + 0.15, df["linear_cka"], width=0.3, label="linear CKA")
    ax2.plot(x, df["rel_mse"], "o--", color="crimson", label="relative MSE")
    ax.set_xticks(x)
    ax.set_xticklabels([f"L{d}" for d in df["route_depth"]])
    ax.set_ylabel("similarity")
    ax2.set_ylabel("relative MSE")
    ax.set_ylim(0, 1.05)
    ax.set_title("Hidden-state drift: distilled student vs vanilla RiNALMo-micro")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "hidden_drift.png", dpi=200)
    print(f"[drift] wrote {OUT_DIR}")


if __name__ == "__main__":
    main()
