#!/usr/bin/env python
"""Verify rebuilt publication-fix tables and close out the provenance notes.

Run AFTER paper/scripts/rebuild_fixed_tables.py (same Slurm job). Steps:

1. Verify the rebuilt tables actually contain the backfilled cells
   (A2 evals, A3 cold-start, secondary F1/F2, ncRNA fixed evals,
   multitask alignment-forgetting). Any miss -> exit 1, notes untouched.
2. Flip "**进行中**" -> "**已完成**" on the fix-status rows of
   paper/notes/ablation_systems_provenance.md (jobs 31576/31577) and
   paper/notes/finetune_multitask_provenance.md (jobs 31588/31589).
3. Append a closure row (with headline numbers from the rebuilt CSVs) to
   each fix-status table.

Usage:
    python scripts/analysis/fix_finalize_provenance.py --job-id $SLURM_JOB_ID
    python scripts/analysis/fix_finalize_provenance.py --dry-run --skip-verify \
        --tables-dir /tmp/rebuild_tables_dryrun
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]  # workspace root containing paper/ and rna_head_alignment/
TAB = ROOT / "paper/tables"
NOTES = ROOT / "paper/notes"

ABL_NOTE = NOTES / "ablation_systems_provenance.md"
FT_NOTE = NOTES / "finetune_multitask_provenance.md"


def fail(msg: str) -> None:
    print(f"[finalize][FATAL] {msg}", file=sys.stderr)
    sys.exit(1)


def verify(tab: Path) -> dict[str, float]:
    """Assert all backfilled cells are present; return headline numbers."""
    aa = pd.read_csv(tab / "alignment_ablation.csv")
    rf = pd.read_csv(tab / "route_finetune.csv")
    mt = pd.read_csv(tab / "multitask.csv")

    def row(df: pd.DataFrame, col: str, key: str) -> pd.Series:
        hit = df[df[col] == key]
        if hit.empty:
            fail(f"{col}=={key} not found in table")
        return hit.iloc[0]

    a2 = row(aa, "experiment", "A2_head2head_top3")
    a3c = row(aa, "experiment", "A3_head2head_top5_coldstart")
    if int(a2["n_seeds_metrics"]) != 3:
        fail(f"A2 n_seeds_metrics={a2['n_seeds_metrics']} (expect 3) — 31577 evals missing?")
    if int(a3c["n_seeds_metrics"]) != 3:
        fail(f"A3 coldstart n_seeds_metrics={a3c['n_seeds_metrics']} (expect 3) — 31576 incomplete?")

    sec = row(rf, "task", "secondary_structure_pair")
    ncr = row(rf, "task", "ncRNA_family_seqcls")
    for r, name in ((sec, "secondary"), (ncr, "ncRNA")):
        if int(r["F1_n"]) != 3 or int(r["F2_replay_n"]) != 3:
            fail(f"{name}: F1_n={r['F1_n']} F2_replay_n={r['F2_replay_n']} (expect 3/3) — 31588/31587 incomplete?")

    for task in ("ncRNA_family_seqcls", "secondary_structure_pair"):
        r = row(mt, "task", task)
        if int(r["independent_F1_n"]) != 3:
            fail(f"multitask {task}: independent_F1_n={r['independent_F1_n']} (expect 3)")
    for mode in ("joint_no_replay", "joint_with_replay"):
        for col in (f"{mode}_forget_cos_delta_mean", f"{mode}_forget_kl_delta_mean"):
            if mt[col].isna().any():
                fail(f"multitask column {col} has NaN — 31589 forgetting jsons missing?")

    out = {
        "a2_cos": float(a2["Cosine_mean"]), "a2_cos_std": float(a2["Cosine_std"]),
        "a2_kl": float(a2["KL_mean"]),
        "a3c_cos": float(a3c["Cosine_mean"]), "a3c_cos_std": float(a3c["Cosine_std"]),
        "a3c_kl": float(a3c["KL_mean"]), "a3c_kl_std": float(a3c["KL_std"]),
        "sec_f1": float(sec["F1_mean"]),
        "sec_f2": float(sec["F2_replay_mean"]), "sec_f2_std": float(sec["F2_replay_std"]),
        "fg_norep_cos": float(mt[f"joint_no_replay_forget_cos_delta_mean"].iloc[0]),
        "fg_norep_kl": float(mt[f"joint_no_replay_forget_kl_delta_mean"].iloc[0]),
        "fg_rep_cos": float(mt[f"joint_with_replay_forget_cos_delta_mean"].iloc[0]),
        "fg_rep_kl": float(mt[f"joint_with_replay_forget_kl_delta_mean"].iloc[0]),
    }
    print("[finalize] verification passed")
    return out


def flip_statuses(text: str, job_prefixes: dict[str, int], note_name: str, dry_run: bool) -> str:
    lines = text.splitlines(keepends=True)
    counts = {k: 0 for k in job_prefixes}
    for i, line in enumerate(lines):
        if "**进行中**" not in line:
            continue
        for jp in job_prefixes:
            if f"job {jp}" in line:
                lines[i] = line.replace("**进行中**", "**已完成**")
                counts[jp] += 1
                if dry_run:
                    print(f"[dry-run] {note_name}: flip line {i + 1}: {line.strip()[:110]}")
                break
    for jp, expect in job_prefixes.items():
        if counts[jp] != expect:
            fail(f"{note_name}: expected {expect} '进行中' row(s) for job {jp}, found {counts[jp]}")
    return "".join(lines)


def append_row(text: str, row: str, note_name: str) -> str:
    stripped = text.rstrip("\n")
    last = stripped.splitlines()[-1].strip()
    if not last.startswith("|"):
        fail(f"{note_name}: last non-empty line is not a table row; refusing to append")
    return stripped + "\n" + row + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-id", default="local")
    ap.add_argument("--tables-dir", default=str(TAB))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-verify", action="store_true")
    args = ap.parse_args()

    tab = Path(args.tables_dir)
    nums: dict[str, float] = {}
    if args.skip_verify:
        aa = pd.read_csv(tab / "alignment_ablation.csv")
        rf = pd.read_csv(tab / "route_finetune.csv")
        mt = pd.read_csv(tab / "multitask.csv")
        sec = rf[rf.task == "secondary_structure_pair"].iloc[0]
        nums = {
            "a2_cos": np.nan, "a2_cos_std": np.nan, "a2_kl": np.nan,
            "a3c_cos": np.nan, "a3c_cos_std": np.nan, "a3c_kl": np.nan, "a3c_kl_std": np.nan,
            "sec_f1": float(sec["F1_mean"]), "sec_f2": float(sec["F2_replay_mean"]),
            "sec_f2_std": float(sec["F2_replay_std"]),
            "fg_norep_cos": np.nan, "fg_norep_kl": np.nan, "fg_rep_cos": np.nan, "fg_rep_kl": np.nan,
        }
        for mode, pre in (("joint_no_replay", "fg_norep"), ("joint_with_replay", "fg_rep")):
            c = mt[f"{mode}_forget_cos_delta_mean"]
            k = mt[f"{mode}_forget_kl_delta_mean"]
            if c.notna().any():
                nums[f"{pre}_cos"] = float(c.dropna().iloc[0])
                nums[f"{pre}_kl"] = float(k.dropna().iloc[0])
    else:
        nums = verify(tab)

    def f4(x: float) -> str:
        return "n/a" if np.isnan(x) else f"{x:.4f}"

    # --- ablation_systems_provenance.md (jobs 31576, 31577) ---
    abl_row = (
        f"| 表格重建 + 收尾 | **已完成**，job {args.job_id} | "
        f"rebuild_fixed_tables.py 重建 alignment_ablation/route_finetune/alignment_replay/multitask 四表。"
        f"A3 冷启动（从零 20k 步，n=3）test cosine {f4(nums['a3c_cos'])}±{f4(nums['a3c_cos_std'])}、"
        f"KL {f4(nums['a3c_kl'])}±{f4(nums['a3c_kl_std'])}，与 A1/A4–A8 同口径可比；"
        f"A2 补齐后 test cosine {f4(nums['a2_cos'])}±{f4(nums['a2_cos_std'])}（n=3，早停 ≤10k 步）。"
        f"两份 provenance 修复状态段同步收尾 |"
    )
    abl_text = ABL_NOTE.read_text(encoding="utf-8")
    abl_new = flip_statuses(abl_text, {"31576": 1, "31577": 4}, ABL_NOTE.name, args.dry_run)
    abl_new = append_row(abl_new, abl_row, ABL_NOTE.name)

    # --- finetune_multitask_provenance.md (jobs 31588, 31589) ---
    ft_row = (
        f"| 表格重建 + 收尾 | **已完成**，job {args.job_id} | "
        f"secondary F1 test top_l={f4(nums['sec_f1'])}（三 seed 结构性相同），"
        f"F2(λ=0.1) top_l={f4(nums['sec_f2'])}±{f4(nums['sec_f2_std'])}（n=3）；"
        f"multitask 遗忘量已并入 multitask.csv：no_replay Δcos={nums['fg_norep_cos']:+.4f} "
        f"ΔKL={nums['fg_norep_kl']:+.4f}，with_replay Δcos={nums['fg_rep_cos']:+.4f} "
        f"ΔKL={nums['fg_rep_kl']:+.4f}（macro over L6/8/12） |"
    )
    ft_text = FT_NOTE.read_text(encoding="utf-8")
    ft_new = flip_statuses(ft_text, {"31588": 1, "31589": 1}, FT_NOTE.name, args.dry_run)
    ft_new = append_row(ft_new, ft_row, FT_NOTE.name)

    if args.dry_run:
        print("[dry-run] would append to", ABL_NOTE.name, "->", abl_row[:160], "...")
        print("[dry-run] would append to", FT_NOTE.name, "->", ft_row[:160], "...")
        print("[dry-run] no files written")
        return

    ABL_NOTE.write_text(abl_new, encoding="utf-8")
    FT_NOTE.write_text(ft_new, encoding="utf-8")
    print(f"[finalize] updated {ABL_NOTE}")
    print(f"[finalize] updated {FT_NOTE}")


if __name__ == "__main__":
    main()
