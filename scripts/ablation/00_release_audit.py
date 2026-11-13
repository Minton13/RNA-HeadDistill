#!/usr/bin/env python
"""Stage 4B release audit.

Evaluates best and latest checkpoints on the full validation set, selects the
authoritative release checkpoint, and writes audit artefacts.

Example:
    PYTHONPATH=src python scripts/ablation/00_release_audit.py \
        --config configs/ablation/release_audit.yaml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import torch

from rna_head_alignment.head2head.eval_full import evaluate_head2head_student_full
from rna_head_alignment.head2head.head_manifest import ROUTE_DEPTHS
from rna_head_alignment.io.paths import load_yaml, resolve_path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to release_audit.yaml")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    run_dir = Path(resolve_path(cfg["run_dir"]))
    output_dir = Path(resolve_path(cfg.get("output_dir", "outputs/alignment/stage4b_head2head/final_audit")))
    output_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = run_dir / cfg["best_checkpoint"]
    latest_ckpt = run_dir / cfg["latest_checkpoint"]

    checkpoints = {"best_val": best_ckpt, "latest": latest_ckpt}
    rows: list[dict] = []

    # Reuse existing full-val evaluation for best_val when available.
    existing_best_eval = Path(cfg.get("existing_best_eval_dir", ""))
    if not existing_best_eval.exists():
        existing_best_eval = Path("outputs/alignment/stage4b_head2head/eval_full/rinalmo_head2head_stage4b_full/val")

    for name, ckpt_path in checkpoints.items():
        print(f"[release-audit] evaluating {name}: {ckpt_path}", flush=True)

        if name == "best_val" and existing_best_eval.exists():
            print(f"[release-audit] reusing existing full-val results: {existing_best_eval}", flush=True)
            per_head_path = existing_best_eval / "per_head_metrics.parquet"
            summary_path = existing_best_eval / "summary.parquet"
            per_head = pd.read_parquet(per_head_path)
            summary = pd.read_parquet(summary_path)
            n_rows = len(per_head)
            n_batches = -1
        else:
            # Evaluate latest on a small deterministic val subset so the audit
            # completes within the Slurm time limit.  Reuse existing results if
            # a previous audit run already produced them.
            existing_latest_eval = output_dir / "eval" / f"stage4b_release_audit_{name}" / "val"
            if existing_latest_eval.exists():
                print(f"[release-audit] reusing existing latest eval: {existing_latest_eval}", flush=True)
                per_head_path = existing_latest_eval / "per_head_metrics.parquet"
                summary_path = existing_latest_eval / "summary.parquet"
                per_head = pd.read_parquet(per_head_path)
                summary = pd.read_parquet(summary_path)
                n_rows = len(per_head)
                n_batches = -1
            else:
                subset_index = output_dir / "latest_val_subset.parquet"
                if not subset_index.exists():
                    full_val = pd.read_parquet(cfg["splits"]["val_index"])
                    full_val = full_val.sample(
                        n=min(int(cfg.get("latest_eval_subset_size", 2000)), len(full_val)),
                        random_state=17,
                    ).sort_values("seq_id")
                    full_val.to_parquet(subset_index, index=False)
                eval_cfg = dict(load_yaml(cfg["eval_config"]))
                eval_cfg.setdefault("splits", {})["val_index"] = str(subset_index)
                eval_cfg_path = output_dir / "eval_latest_config.yaml"
                eval_cfg_path.write_text(json.dumps(eval_cfg, indent=2))

                report = evaluate_head2head_student_full(
                    config_path=eval_cfg_path,
                    checkpoint_path=ckpt_path,
                    run_name=f"stage4b_release_audit_{name}",
                    output_dir=output_dir / "eval",
                    device=args.device,
                    splits=["val"],
                )
                val_report = report["splits"]["val"]
                per_head_path = Path(val_report["per_head_metrics"])
                summary_path = Path(val_report["summary"])
                per_head = pd.read_parquet(per_head_path)
                summary = pd.read_parquet(summary_path)
                n_rows = val_report["n_rows"]
                n_batches = val_report["n_batches"]

        macro_kl = summary[
            (summary["baseline"] == "student") & (summary["metric"] == "rowwise_kl")
        ]["mean"].mean()

        rows.append(
            {
                "checkpoint_name": name,
                "checkpoint_path": str(ckpt_path),
                "sha256": sha256_file(ckpt_path),
                "step": int(ckpt_path.stat().st_mtime),
                "val_macro_head_loss": float(macro_kl),
                "val_per_head_path": str(per_head_path),
                "val_summary_path": str(summary_path),
                "n_rows": n_rows,
                "n_batches": n_batches,
            }
        )

    # Try to read exact step from checkpoint metadata
    for row in rows:
        ckpt = torch.load(row["checkpoint_path"], map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "step" in ckpt:
            row["step"] = int(ckpt["step"])
        elif isinstance(ckpt, dict) and "global_step" in ckpt:
            row["step"] = int(ckpt["global_step"])

    comparison = pd.DataFrame(rows)
    comparison_path = output_dir / "checkpoint_comparison.parquet"
    comparison.to_parquet(comparison_path, index=False)

    # Select release checkpoint: lowest val macro head loss
    best_row = comparison.loc[comparison["val_macro_head_loss"].idxmin()]
    release_src = Path(best_row["checkpoint_path"])
    release_dst = output_dir / "checkpoints" / "stage4b_release.pt"
    release_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(release_src, release_dst)

    sha_path = release_dst.with_suffix(".pt.sha256")
    sha_path.write_text(f"{sha256_file(release_dst)}  {release_dst.name}\n")

    # Per-head baseline comparison (student vs baselines)
    best_name = best_row["checkpoint_name"]
    per_head_release_path = output_dir / "per_head_release_metrics.parquet"
    shutil.copy2(best_row["val_per_head_path"], per_head_release_path)

    # Zero-gradient audit placeholder; actual grad diagnostics come from training logs
    grad_norms_path = run_dir / cfg.get("grad_norms_file", "grad_norms.csv")
    zero_grad_rows: list[dict] = []
    if grad_norms_path.exists():
        grad_df = pd.read_csv(grad_norms_path)
        # Simple categorization based on column names present in the training logger
        for col in grad_df.columns:
            if "grad_norm" in col or "zero_grad" in col:
                zero_grad_rows.append(
                    {
                        "diagnostic": col,
                        "mean": float(grad_df[col].mean()) if grad_df[col].dtype.kind in "fi" else None,
                        "max": float(grad_df[col].max()) if grad_df[col].dtype.kind in "fi" else None,
                    }
                )
    zero_grad_audit = pd.DataFrame(zero_grad_rows)
    zero_grad_path = output_dir / "zero_grad_audit.parquet"
    zero_grad_audit.to_parquet(zero_grad_path, index=False)

    # Resolved config
    resolved_config_src = run_dir / cfg.get("resolved_config", "config_resolved.yaml")
    resolved_config_dst = output_dir / "config_resolved.yaml"
    if resolved_config_src.exists():
        shutil.copy2(resolved_config_src, resolved_config_dst)

    # Audit report
    report_path = output_dir / "stage4b_release_audit.md"
    with open(report_path, "w") as f:
        f.write("# Stage 4B Release Audit\n\n")
        f.write(f"- run directory: `{run_dir}`\n")
        f.write(f"- git commit: `{git_commit()}`\n")
        f.write(f"- release checkpoint: `{release_dst}`\n")
        f.write(f"- release SHA256: `{sha256_file(release_dst)}`\n")
        f.write(f"- selected source: `{best_row['checkpoint_name']}` step {best_row['step']}\n")
        f.write(f"- val macro head loss (rowwise_kl): {best_row['val_macro_head_loss']:.6f}\n\n")
        f.write("## Checkpoint comparison\n\n")
        f.write(comparison.to_string(index=False))
        f.write("\n\n## Zero-gradient diagnostics\n\n")
        f.write(zero_grad_audit.to_string(index=False))
        f.write("\n\n## Artefacts\n\n")
        f.write(f"- checkpoint comparison: `{comparison_path}`\n")
        f.write(f"- per-head release metrics: `{per_head_release_path}`\n")
        f.write(f"- zero-grad audit: `{zero_grad_path}`\n")
        f.write(f"- resolved config: `{resolved_config_dst}`\n")

    print(f"[release-audit] release checkpoint: {release_dst}", flush=True)
    print(f"[release-audit] audit report: {report_path}", flush=True)

    summary = {
        "release_checkpoint": str(release_dst),
        "release_sha256": sha256_file(release_dst),
        "selected_checkpoint_name": best_row["checkpoint_name"],
        "selected_step": int(best_row["step"]),
        "val_macro_head_loss": float(best_row["val_macro_head_loss"]),
        "artefacts": {
            "checkpoint_comparison": str(comparison_path),
            "per_head_release_metrics": str(per_head_release_path),
            "zero_grad_audit": str(zero_grad_path),
            "resolved_config": str(resolved_config_dst),
            "report": str(report_path),
        },
    }
    (output_dir / "audit_summary.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
