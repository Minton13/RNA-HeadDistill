from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import pandas as pd

from rna_head_alignment.io.manifests import write_dataframe, write_json, write_markdown_table
from rna_head_alignment.io.paths import load_yaml, resolve_path, save_yaml


def collect_head_screening(input_pattern: str, output: str | Path) -> pd.DataFrame:
    paths = [Path(p) for p in glob.glob(str(resolve_path(input_pattern)))]
    frames = []
    for p in paths:
        if p.exists() and p.stat().st_size > 0:
            try:
                frames.append(pd.read_csv(p))
            except Exception:
                pass
    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    write_dataframe(df, output)
    if not df.empty:
        write_json(df.to_dict("records"), resolve_path(output).with_suffix(".json"))
    return df


def propose_teacher_heads(
    atlas_path: str | Path,
    config_path: str | Path,
    output: str | Path,
    yaml_output: str | Path,
) -> pd.DataFrame:
    cfg = load_yaml(config_path)
    selection = cfg.get("selection", {})
    per_model_k = int(selection.get("per_model_per_task_top_k", 5))
    global_task_k = int(selection.get("global_per_task_top_k", 30))
    atlas = pd.read_csv(resolve_path(atlas_path))
    if atlas.empty:
        candidates = atlas
    else:
        atlas = atlas.copy()
        atlas["metric_primary"] = pd.to_numeric(atlas["metric_primary"], errors="coerce")
        atlas = atlas[atlas["layer"] >= 0].sort_values("metric_primary", ascending=False)
        parts = []
        for (_, _), group in atlas.groupby(["task", "model"], dropna=False):
            parts.append(group.head(per_model_k))
        candidates = pd.concat(parts, ignore_index=True) if parts else atlas.head(0)
        candidates = (
            candidates.sort_values("metric_primary", ascending=False)
            .groupby("task", dropna=False)
            .head(global_task_k)
            .reset_index(drop=True)
        )
    write_dataframe(candidates, output)
    write_json(candidates.to_dict("records"), resolve_path(output).with_suffix(".json"))
    write_markdown_table(candidates.head(100).to_dict("records"), resolve_path(output).with_suffix(".md"), "Teacher Head Candidates")
    bank = {"banks": {}, "student_teacher_mapping": {"L6": [], "L8": [], "L12": []}}
    for bank_name, bank_cfg in (cfg.get("layer_banks") or {}).items():
        bank["banks"][bank_name] = {"student_layer": bank_cfg["student_layer"], "tasks": {}}
        if bank_name == "L6":
            bank["banks"][bank_name]["dna_student_heads"] = {
                "fraction": bank_cfg.get("dna_student_head_fraction", 0.333),
                "mode": "dedicated_student_heads",
            }
        for task in bank_cfg.get("tasks", []):
            rows = candidates[candidates["task"] == task].head(global_task_k)
            bank["banks"][bank_name]["tasks"][task] = {
                "teacher_heads": [
                    {
                        "model": str(r["model"]),
                        "model_type": str(r.get("model_type", "")),
                        "layer": int(r["layer"]),
                        "head": int(r["head"]),
                        "metric": float(r["metric_primary"]) if pd.notna(r["metric_primary"]) else None,
                        "task": task,
                        "dna_informed": str(r.get("model_type", "")).upper() == "DNA",
                    }
                    for _, r in rows.iterrows()
                ]
            }
    save_yaml(bank, yaml_output)
    return candidates
