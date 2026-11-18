#!/usr/bin/env python
"""Build Nature Communications Source Data for the RNA head-distillation paper.

Outputs (written only under release/source_data/):
  SourceData.xlsx        one sheet per figure panel / table
  csv/<SheetName>.csv    the same data, one CSV per sheet

Every sheet reproduces exactly the numbers plotted in the paper figures or
printed in the paper tables.  Where a figure script recomputes values from a
raw source (Fig2a/Fig2b Jaccard, Fig3b cost curves, Fig4a/4c per-seed bars,
TableS3 concentration), the same logic is re-implemented here from the same
source files (see README.md for the exact code provenance per sheet).

Inputs are read-only:
  rna_head_alignment/outputs/...  (atlas, bootstrap, systems CSVs)
  paper/tables/...                (published table CSVs)
  paper/assets/fig1_real_data/... (packaged npz slices for Fig. 1)

Run on the cluster (python is forbidden on the login node):
  srun --nodelist=n3 --partition=batch --cpus-per-task=4 --mem=16G \
      --time=00:30:00 release/.venv/bin/python \
      release/source_data/build_source_data.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("<REPO_ROOT>")
RHA = ROOT / "rna_head_alignment"
PAPER = ROOT / "paper"
OUT = ROOT / "release" / "source_data"
CSV_DIR = OUT / "csv"

ATLAS_CSV = RHA / "outputs/head_screening/all_head_atlas.csv"
HEADFUNC_CSV = PAPER / "tables/head_functionality_stats.csv"
SEL_HEADS_CSV = PAPER / "tables/selected_teacher_heads.csv"
SYS_DIR = RHA / "outputs/systems/asynchronous_distillation"
ABL_CSV = PAPER / "tables/alignment_ablation.csv"
ABL_DELTA_CSV = PAPER / "tables/alignment_ablation_delta.csv"
AV2 = RHA / "outputs/probes/analysis_v2"
PE_CSV = AV2 / "bootstrap/point_estimates.csv"
A8_PE_CSV = AV2 / "bootstrap_a8/a8_point_estimates.csv"
S4_CSV = AV2 / "bootstrap/p4_vs_baselines_summary.csv"
S8_CSV = AV2 / "bootstrap_a8/a8_vs_baselines_summary.csv"
BS_CSV = PAPER / "tables/downstream_primary_bootstrap.csv"
FULL_METRICS_CSV = PAPER / "tables/downstream_full_metrics.csv"
FIG1_DIR = PAPER / "assets/fig1_real_data"
SANKEY_DIR = RHA / "outputs/asset_summaries"
PILOT_DIR = PAPER / "tables/pilot"

# shared constants, mirrored from paper/scripts/_paper_style.py
TASK_ORDER = [
    "splice_donor_seqcls", "splice_acceptor_seqcls", "rbp_binding_seqcls",
    "ncRNA_family_seqcls", "chemical_reactivity_token",
    "secondary_structure_pair",
]
TASK_METRIC = {
    "splice_donor_seqcls": ("auroc", "AUROC"),
    "splice_acceptor_seqcls": ("auroc", "AUROC"),
    "rbp_binding_seqcls": ("auroc", "AUROC"),
    "ncRNA_family_seqcls": ("auroc", "AUROC"),
    "chemical_reactivity_token": ("pearson", "Pearson r"),
    "secondary_structure_pair": ("top_l_precision", "Top-L precision"),
}

REPORT: dict = {"sheets": {}, "checks": []}


def note(msg: str) -> None:
    print(msg, flush=True)
    REPORT["checks"].append(msg)


# ---------------------------------------------------------------- atlas cleaning
def load_valid_atlas() -> pd.DataFrame:
    """Identical cleaning to paper/scripts/plot_fig2_atlas.py and
    paper/notes/_analysis_head_function.py."""
    atlas = pd.read_csv(ATLAS_CSV, low_memory=False)
    atlas = atlas[~atlas["notes"].fillna("").str.contains("model_load_failed")]
    atlas = atlas[atlas["metric_primary"].notna()]
    per_head = atlas[atlas["head"] >= 0]
    combo_max = per_head.groupby(["task", "model"])["metric_primary"].max()
    deg = set(combo_max[combo_max <= 1e-9].index.tolist())
    keys = list(zip(per_head["task"], per_head["model"]))
    keep = pd.Series([k not in deg for k in keys], index=per_head.index)
    valid = per_head[keep].copy()
    note(f"atlas: {len(valid)} valid per-head rows, "
         f"{valid['task'].nunique()} tasks, {valid['model'].nunique()} models, "
         f"{len(deg)} degenerate combos removed")
    return valid


def heatmap_wide(g: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """layer x head matrix of metric_primary, exactly as plot_fig2_atlas.py."""
    n_layers = int(g["layer"].max()) + 1
    n_heads = int(g["head"].max()) + 1
    mat = np.full((n_layers, n_heads), np.nan)
    for _, r in g.iterrows():
        mat[int(r["layer"]), int(r["head"])] = r["metric_primary"]
    cols = [f"head_{j}" for j in range(n_heads)]
    wide = pd.DataFrame(mat, columns=cols)
    wide.insert(0, "layer", range(n_layers))
    return wide, mat


# ---------------------------------------------------------------- Fig 1
def fig1a() -> pd.DataFrame:
    npz = np.load(FIG1_DIR / "panel_a_probe_atlas.npz", allow_pickle=False)
    combos = [("secondary_structure_pair", "rinalmo_giga"),
              ("splice_site_seqcls", "splicebert")]
    rows = []
    for k, (task, model) in enumerate(combos):
        mat = np.asarray(npz[f"mat_raw_{k}"], dtype=np.float64)
        label = str(npz[f"label_{k}"].item())
        n_nan = int(np.isnan(mat).sum())
        note(f"Fig1a {label}: shape={mat.shape} n_nan={n_nan}")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                rows.append({"heatmap": label, "task": task, "model": model,
                             "layer": i, "head": j, "probe_score": v})
    return pd.DataFrame(rows)


def fig1c() -> pd.DataFrame:
    npz = np.load(FIG1_DIR / "panel_c_attention.npz", allow_pickle=False)
    seq = str(npz["sequence"].item())
    note(f"Fig1c sequence (len {len(seq)}): {seq}")
    rows = []
    for depth in (6, 8, 12):
        head_id = str(npz[f"head_id_L{depth}"].item())
        for map_type in ("teacher", "student"):
            mat = np.asarray(npz[f"{map_type}_L{depth}"], dtype=np.float64)
            for i in range(mat.shape[0]):
                for j in range(mat.shape[1]):
                    rows.append({"map": f"{map_type}_L{depth}",
                                 "map_type": map_type, "route_depth": depth,
                                 "head_id": head_id, "row": i, "col": j,
                                 "attention_value": mat[i, j]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Fig 2
def fig2a(valid: pd.DataFrame) -> dict[str, pd.DataFrame]:
    # six combos in the order used by plot_fig2_atlas.py (`picks` list)
    picks = [
        ("secondary_structure_pair", "rinalmo_giga"),
        ("rbp_binding_seqcls", "carbon_500m"),
        ("rbp_binding_seqcls", "mrnafm"),
        ("secondary_structure_pair", "rnaernie"),
        ("chemical_reactivity_token", "rnafm"),
        ("splice_site_seqcls", "splicebert"),
    ]
    out = {}
    for k, (task, model) in enumerate(picks, start=1):
        g = valid[(valid["task"] == task) & (valid["model"] == model)]
        wide, mat = heatmap_wide(g)
        med = float(np.nanmedian(mat))
        t5 = float(np.sort(mat[~np.isnan(mat)].ravel())[-5:].mean())
        note(f"Fig2a_{k} {task} x {model}: {mat.shape[0]}L x {mat.shape[1]}H, "
             f"median={med:.4f}, top5={t5:.4f}, n_nan={int(np.isnan(mat).sum())}")
        out[f"Fig2a_{k}"] = wide
    return out


def fig2b(valid: pd.DataFrame) -> pd.DataFrame:
    """Mean Jaccard of top-10 head sets, replicated from plot_fig2_atlas.py."""
    tasks = sorted(valid["task"].unique())
    K = 10
    jac_sum = pd.DataFrame(0.0, index=tasks, columns=tasks)
    jac_cnt = pd.DataFrame(0, index=tasks, columns=tasks)
    null_jac = []
    for model, gm in valid.groupby("model"):
        tops = {}
        for task, g in gm.groupby("task"):
            top = g.nlargest(K, "metric_primary")
            tops[task] = set(zip(top["layer"], top["head"]))
        n_heads_model = len(gm.groupby(["layer", "head"]).size())
        null_jac.append(K / (2 * n_heads_model - K))
        for i, t1 in enumerate(tasks):
            if t1 not in tops:
                continue
            for t2 in tasks[i:]:
                if t2 not in tops:
                    continue
                a, b = tops[t1], tops[t2]
                j = len(a & b) / len(a | b)
                jac_sum.loc[t1, t2] += j
                jac_sum.loc[t2, t1] += j
                jac_cnt.loc[t1, t2] += 1
                jac_cnt.loc[t2, t1] += 1
    jac = jac_sum / jac_cnt.replace(0, np.nan)
    null_mean = float(np.mean(null_jac))
    off = jac.values[~np.eye(len(tasks), dtype=bool)]
    off = off[~np.isnan(off)]
    note(f"Fig2b: {len(tasks)} tasks, null_mean={null_mean:.4f} "
         f"(paper caption ~0.032), off-diag mean={off.mean():.4f} "
         f"max={off.max():.4f}")
    out = jac.copy()
    out.insert(0, "task", out.index)
    return out.reset_index(drop=True)


def fig2c() -> pd.DataFrame:
    sel = pd.read_csv(SEL_HEADS_CSV)
    assert len(sel) == 15, f"expected 15 selected heads, got {len(sel)}"
    return sel


# ---------------------------------------------------------------- Fig 3
FULL_CORPUS = 2_072_010
SCEN_ORDER = ["S0_offline_cached", "S1_online_single_teacher",
              "S2_online_three_teachers", "S3_online_full_estimate"]


def fig3a() -> pd.DataFrame:
    thr = pd.read_csv(SYS_DIR / "throughput_comparison.csv").set_index("scenario")
    mem = pd.read_csv(SYS_DIR / "memory_comparison.csv").set_index("scenario")
    rows = []
    for s in SCEN_ORDER:
        rows.append({
            "scenario": s,
            "measurement": ("measured (offline cache)" if s == "S0_offline_cached"
                            else "extrapolated from S2 per-model time"
                            if s == "S3_online_full_estimate"
                            else "measured"),
            "throughput_seq_per_sec": float(thr.loc[s, "throughput_seq_per_sec"]),
            "n_sequences": int(thr.loc[s, "n_sequences"]),
            "peak_vram_gb": float(mem.loc[s, "peak_vram_gb"]),
            "notes": str(thr.loc[s, "notes"]),
        })
    return pd.DataFrame(rows)


def fig3b() -> pd.DataFrame:
    thr = pd.read_csv(SYS_DIR / "throughput_comparison.csv") \
        .set_index("scenario")["throughput_seq_per_sec"]
    t = {k: FULL_CORPUS / thr[k] / 3600.0 for k in thr.index}  # GPU-h per pass
    one_time = t["S3_online_full_estimate"]
    curves = {
        "offline cache (1x 9-teacher extraction + Nx cached reads)":
            ("S0_offline_cached", True),
        "online, 1 teacher": ("S1_online_single_teacher", False),
        "online, 3 teachers": ("S2_online_three_teachers", False),
        "online, 9 teachers (extrapolated)": ("S3_online_full_estimate", False),
    }
    rows = []
    for label, (skey, amortized) in curves.items():
        for n in range(1, 51):
            cost = (one_time + n * t[skey]) if amortized else n * t[skey]
            rows.append({"n_experiments": float(n), "curve": label,
                         "cumulative_gpu_hours": cost})
    for label, skey in [("online, 1 teacher", "S1_online_single_teacher"),
                        ("online, 3 teachers", "S2_online_three_teachers"),
                        ("online, 9 teachers (extrapolated)",
                         "S3_online_full_estimate")]:
        k = one_time / (t[skey] - t["S0_offline_cached"])
        yk = one_time + k * t["S0_offline_cached"]
        note(f"Fig3b break-even offline vs {label}: N={k:.2f}, {yk:.1f} GPU-h")
        rows.append({"n_experiments": k,
                     "curve": f"break-even: offline cache vs {label}",
                     "cumulative_gpu_hours": yk})
    return pd.DataFrame(rows)


def fig3c() -> pd.DataFrame:
    abl = pd.read_csv(ABL_CSV).set_index("experiment")
    exps = ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8"]
    full = dict(zip(exps, ["single_teacher_top1", "head2head_top3",
                           "head2head_top5_full", "consensus_top5",
                           "random_pairing", "linear_adapter", "all_from_L12",
                           "adapters_only"]))
    rows = []
    for e in exps:
        r = abl.loc[f"{e}_{full[e]}"]
        rows.append({
            "experiment_id": e,
            "experiment": f"{e}_{full[e]}",
            "n_seeds_metrics": r["n_seeds_metrics"],
            "KL_mean": r["KL_mean"], "KL_std": r["KL_std"],
            "Cosine_mean": r["Cosine_mean"], "Cosine_std": r["Cosine_std"],
            "KL_matched9_mean": r["KL_matched9_mean"],
            "KL_matched9_std": r["KL_matched9_std"],
            "Cosine_matched9_mean": r["Cosine_matched9_mean"],
            "Cosine_matched9_std": r["Cosine_matched9_std"],
            "n_seeds_matched9": r["n_seeds_matched9"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Fig 4
def _pe_series(pe: pd.DataFrame, a8: pd.DataFrame, task: str, mode: str,
               metric: str) -> pd.DataFrame:
    if mode == "A8_P4g":
        sub = a8[(a8.task == task) & (a8.line == "A8_P4g")]
    else:
        sub = pe[(pe.task == task) & (pe["mode"] == mode)]
    sub = sub[["seed", "n_test", metric]].dropna(subset=[metric])
    return sub.rename(columns={metric: "value"})


def _star(p: float) -> str:
    return "***" if (p >= 0.999 or p <= 0.001) else "ns"


def fig4a(pe: pd.DataFrame, a8: pd.DataFrame, s4: pd.DataFrame,
          s8: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASK_ORDER:
        metric, _ = TASK_METRIC[task]
        for mode in ("P0", "P4", "P4g", "A8_P4g"):
            sub = _pe_series(pe, a8, task, mode, metric)
            mu = float(sub["value"].mean())
            sd = float(sub["value"].std(ddof=1))
            if mode == "P0":
                comp = None
            elif mode == "A8_P4g":
                comp = s8[(s8.task == task) & (s8.comparison == "A8_P4g-P0")]
            else:
                comp = s4[(s4.task == task) & (s4.comparison == f"{mode}-P0")]
            dm = dl = dh = pp = np.nan
            st = ""
            if comp is not None and len(comp):
                dm, dl, dh, pp = comp[["mean", "ci_lo", "ci_hi", "p_pos"]] \
                    .iloc[0].to_numpy()
                st = _star(pp)
            for _, r in sub.iterrows():
                rows.append({
                    "task": task, "metric": metric, "method": mode,
                    "seed": int(r["seed"]), "n_test": int(r["n_test"]),
                    "value": float(r["value"]),
                    "method_mean": mu, "method_sd": sd,
                    "delta_vs_P0_mean": dm, "delta_ci_lo": dl,
                    "delta_ci_hi": dh, "p_pos": pp,
                    "significance_vs_P0": st,
                })
    df = pd.DataFrame(rows)
    note(f"Fig4a: {len(df)} per-seed rows "
         f"({df.groupby(['task', 'method']).ngroups} task x method cells)")
    return df


def fig4c(pe: pd.DataFrame, a8: pd.DataFrame) -> pd.DataFrame:
    task, metric = "splice_acceptor_seqcls", "auroc"
    rows = []
    for mode in ("P0", "P1", "P4", "P4g", "A8_P4g"):
        sub = _pe_series(pe, a8, task, mode, metric)
        mu = float(sub["value"].mean())
        sd = float(sub["value"].std(ddof=1))
        for _, r in sub.iterrows():
            rows.append({"task": task, "metric": metric, "method": mode,
                         "seed": int(r["seed"]), "n_test": int(r["n_test"]),
                         "value": float(r["value"]),
                         "method_mean": mu, "method_sd": sd})
    return pd.DataFrame(rows)


def fig4b(s4: pd.DataFrame, s8: pd.DataFrame, bs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task in TASK_ORDER:
        metric, _ = TASK_METRIC[task]
        for m in ("P4", "P4g", "A8_P4g", "P5"):
            if m == "A8_P4g":
                r = s8[(s8.task == task) & (s8.comparison == "A8_P4g-P0")]
                dm, dl, dh, pp = r[["mean", "ci_lo", "ci_hi", "p_pos"]] \
                    .iloc[0].to_numpy()
            elif m == "P5":
                r = bs[(bs.task == task) & (bs.mode_tag == "P5")
                       & (bs.baseline_tag == "P0")]
                dm, dl, dh, pp = r[["delta_mean", "ci_lo", "ci_hi", "p_pos"]] \
                    .iloc[0].to_numpy()
            else:
                r = s4[(s4.task == task) & (s4.comparison == f"{m}-P0")]
                dm, dl, dh, pp = r[["mean", "ci_lo", "ci_hi", "p_pos"]] \
                    .iloc[0].to_numpy()
            rows.append({"task": task, "metric": metric, "method": m,
                         "comparison": f"{m}-P0",
                         "delta_mean": float(dm), "ci_lo": float(dl),
                         "ci_hi": float(dh), "p_pos": float(pp),
                         "significance": _star(float(pp))})
    df = pd.DataFrame(rows)
    note(f"Fig4b: {len(df)} forest rows (expect 24)")
    assert len(df) == 24
    return df


# ---------------------------------------------------------------- tables
def table2(pe: pd.DataFrame, a8: pd.DataFrame) -> pd.DataFrame:
    """Manuscript Table 2: primary metric per task for P0/P1/P4/P4g/A8-P4g/
    P5/P6 (mean +/- sd, 3 seeds).  P0/P1/P4/P5/P6 come from
    downstream_full_metrics.csv; P4g and A8-P4g are absent there and are
    computed from the per-seed bootstrap point estimates (documented in
    README)."""
    fm = pd.read_csv(FULL_METRICS_CSV)
    rows = []
    for task in TASK_ORDER:
        metric, _ = TASK_METRIC[task]
        for tag in ("P0", "P1", "P4", "P5", "P6"):
            r = fm[(fm.task == task) & (fm.mode_tag == tag)
                   & (fm.metric == metric)]
            assert len(r) == 1, f"Table2: no row {task}/{tag}/{metric}"
            r = r.iloc[0]
            rows.append({"task": task, "mode": r["mode"], "mode_tag": tag,
                         "metric": metric, "mean": r["mean"], "std": r["std"],
                         "n_seeds": int(r["n_seeds"]),
                         "source_file": "paper/tables/downstream_full_metrics.csv"})
        for tag, src in (("P4g", "pe"), ("A8_P4g", "a8")):
            sub = _pe_series(pe, a8, task, tag, metric)
            rows.append({
                "task": task,
                "mode": tag, "mode_tag": tag, "metric": metric,
                "mean": float(sub["value"].mean()),
                "std": float(sub["value"].std(ddof=1)),
                "n_seeds": int(len(sub)),
                "source_file": ("rna_head_alignment/outputs/probes/analysis_v2/"
                                "bootstrap/point_estimates.csv" if src == "pe"
                                else "rna_head_alignment/outputs/probes/"
                                "analysis_v2/bootstrap_a8/a8_point_estimates.csv"),
            })
    df = pd.DataFrame(rows)
    note(f"Table2: {len(df)} rows (expect 42 = 6 tasks x 7 modes)")
    assert len(df) == 42
    # spot-check against the manuscript numbers
    exp = {("splice_donor_seqcls", "P0"): (0.768, 0.003),
           ("splice_donor_seqcls", "P4g"): (0.827, 0.001),
           ("splice_acceptor_seqcls", "A8_P4g"): (0.770, 0.001),
           ("secondary_structure_pair", "P4"): (0.523, 0.028),
           ("secondary_structure_pair", "P5"): (0.335, 0.000),
           ("rbp_binding_seqcls", "P6"): (0.875, 0.000)}
    for (task, tag), (m, s) in exp.items():
        r = df[(df.task == task) & (df.mode_tag == tag)].iloc[0]
        ok = abs(r["mean"] - m) < 5e-4 and abs(r["std"] - s) < 5e-4
        note(f"Table2 check {task}/{tag}: got {r['mean']:.4f}+-{r['std']:.4f} "
             f"vs manuscript {m}+-{s} -> {'OK' if ok else 'MISMATCH'}")
        assert ok
    return df


MODEL_DISPLAY = {
    "rinalmo_giga": "RiNALMo-giga", "rinalmo_mega": "RiNALMo-mega",
    "rinalmo_micro": "RiNALMo-micro", "rnaernie": "RNAErnie",
    "rnafm": "RNA-FM", "plantrna_fm": "PlantRNA-FM", "rnabert": "RNABERT",
    "utrlm_mrl": "UTR-LM (MRL)", "moderngena_base": "ModernGENA-base",
    "mrnafm": "mRNA-FM", "utrbert_5mer": "UTRBERT-5mer",
    "utrlm_te_el": "UTR-LM (TE-EL)", "utrbert_3mer": "UTRBERT-3mer",
    "utrbert_4mer": "UTRBERT-4mer", "utrbert_6mer": "UTRBERT-6mer",
    "splicebert": "SpliceBERT", "carbon_500m": "CARBON-500M",
    "gena_lm_bigbird_base_sparse": "GENA-LM bigbird sparse",
    "nucleotide_transformer_v2_500m": "NT-v2-500M",
    "nucleotide_transformer_v2_50m_3mer": "NT-v2-50M-3mer",
    "nucleotide_transformer_v2_50m": "NT-v2-50M",
    "gena_lm_bert_base_t2t_multi": "GENA-LM t2t-multi",
    "gena_lm_bert_base": "GENA-LM base",
    "gena_lm_bert_base_lastln_t2t": "GENA-LM lastln-t2t",
    "gena_lm_bert_base_t2t": "GENA-LM t2t", "dnabert_6": "DNABERT-6",
}


def tableS3(valid: pd.DataFrame) -> pd.DataFrame:
    """Per-model concentration on secondary_structure_pair, recomputed from
    the atlas exactly as paper/notes/_analysis_head_function.py
    (paper/tables/head_functionality_stats.csv) and supplementary.tex
    tab:s1_concentration."""
    g = valid[valid["task"] == "secondary_structure_pair"]
    rows = []
    for model, gm in g.groupby("model"):
        s = gm["metric_primary"]
        med = float(s.median())
        t5 = float(s.nlargest(min(5, len(s))).mean())
        rows.append({
            "model_display": MODEL_DISPLAY.get(model, model),
            "model": model,
            "model_type": gm["model_type"].iloc[0],
            "n_heads": int(len(gm)),
            "median": med,
            "top5_mean": t5,
            "ratio_top5_over_median": t5 / med if med > 1e-6 else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("ratio_top5_over_median",
                                        ascending=False).reset_index(drop=True)
    note(f"TableS3: {len(df)} models (manuscript lists 26)")
    # cross-check against the published stats CSV (that file rounds to 4
    # decimals, so compare after rounding both sides)
    ref = pd.read_csv(HEADFUNC_CSV)
    ref = ref[ref["task"] == "secondary_structure_pair"].set_index("model")
    n_mismatch = 0
    for _, r in df.iterrows():
        rr = ref.loc[r["model"]]
        if not (abs(round(rr["median"], 4) - round(r["median"], 4)) < 1e-12
                and abs(round(rr["top5_mean"], 4)
                        - round(r["top5_mean"], 4)) < 1e-12):
            n_mismatch += 1
            note(f"TableS3 MISMATCH vs head_functionality_stats.csv: "
                 f"{r['model']}")
    note(f"TableS3 cross-check vs head_functionality_stats.csv: "
         f"{n_mismatch} mismatches")
    # spot-check the manuscript's first/last rows
    top = df.iloc[0]
    note(f"TableS3 top row: {top['model']} heads={top['n_heads']} "
         f"median={top['median']:.4f} top5={top['top5_mean']:.4f} "
         f"ratio={top['ratio_top5_over_median']:.1f} "
         f"(manuscript: RiNALMo-giga 660 0.0045 0.5356 120.1)")
    return df


def figS1b() -> pd.DataFrame:
    """Fixed study facts transcribed from paper/scripts/plot_si_dataflow.py
    (alignment-cache split + six downstream task native splits)."""
    cache = {"train": 2_072_010, "val": 21_143, "test": 21_136}
    cache_use = {"train": "student distillation training",
                 "val": "head pairing + early stopping",
                 "test": "alignment eval (row KL / cosine)"}
    tasks = {
        "Splice donor": (36_437, 4_670, 4_512),
        "Splice acceptor": (36_475, 4_709, 4_435),
        "RBP binding": (962_667, 120_152, 142_760),
        "ncRNA family": (1_403, 169, 162),
        "Chemical reactivity": (126_909, 26_463, 14_436),
        "Secondary structure": (18_407, 2_273, 2_312),
    }
    task_use = {"train": "probe/fine-tune training",
                "val": "checkpoint selection",
                "test": "test bootstrap metrics"}
    rows = []
    for split, n in cache.items():
        rows.append({"dataset": "teacher-attention cache",
                     "dataset_type": "alignment_cache", "split": split,
                     "n_sequences": n, "use": cache_use[split]})
    for name, (tr, va, te) in tasks.items():
        for split, n in zip(("train", "val", "test"), (tr, va, te)):
            rows.append({"dataset": name, "dataset_type": "downstream_task",
                         "split": split, "n_sequences": n,
                         "use": task_use[split]})
    df = pd.DataFrame(rows)
    note(f"FigS1b: cache total = {sum(cache.values()):,} "
         f"(expect 2,114,289); task train total = "
         f"{sum(v[0] for v in tasks.values()):,}")
    return df


def pilot_heatmap(model: str) -> pd.DataFrame:
    df = pd.read_csv(PILOT_DIR / f"pilot_mcc_heatmap_{model}.csv")
    piv = df.pivot(index="layer", columns="head", values="mcc_mean")
    wide = piv.copy()
    wide.columns = [f"head_{int(c)}" for c in piv.columns]
    wide.insert(0, "layer", piv.index.astype(int))
    note(f"FigS2 pilot {model}: {piv.shape[0]}L x {piv.shape[1]}H, "
         f"n_samples per head = {sorted(df['n_samples'].unique())}")
    return wide.reset_index(drop=True)


# ---------------------------------------------------------------- main
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    valid = load_valid_atlas()
    pe = pd.read_csv(PE_CSV)
    a8 = pd.read_csv(A8_PE_CSV)
    s4 = pd.read_csv(S4_CSV)
    s8 = pd.read_csv(S8_CSV)
    bs = pd.read_csv(BS_CSV)

    sheets: dict[str, pd.DataFrame] = {}
    sheets["Fig1a"] = fig1a()
    sheets["Fig1c"] = fig1c()
    sheets.update(fig2a(valid))
    sheets["Fig2b"] = fig2b(valid)
    sheets["Fig2c"] = fig2c()
    sheets["Fig3a"] = fig3a()
    sheets["Fig3b"] = fig3b()
    sheets["Fig3c"] = fig3c()
    sheets["Fig4a"] = fig4a(pe, a8, s4, s8)
    sheets["Fig4b"] = fig4b(s4, s8, bs)
    sheets["Fig4c"] = fig4c(pe, a8)
    sheets["Table1"] = pd.read_csv(SEL_HEADS_CSV)
    sheets["Table2"] = table2(pe, a8)
    sheets["FigS1a_nodes"] = pd.read_csv(
        SANKEY_DIR / "pretrain_benchmark_datasource_level_sankey_nodes_counts_v2.csv",
        encoding="utf-8-sig")
    sheets["FigS1a_links"] = pd.read_csv(
        SANKEY_DIR / "pretrain_benchmark_datasource_level_sankey_links_counts_v2.csv",
        encoding="utf-8-sig")
    sheets["FigS1b"] = figS1b()
    sheets["FigS2a"] = pilot_heatmap("micro")
    sheets["FigS2b"] = pilot_heatmap("mega")
    sheets["FigS2c"] = pilot_heatmap("giga")
    sheets["FigS3"] = pd.read_csv(PILOT_DIR / "pilot_decoder_curves.csv")
    sheets["TableS1"] = pd.read_csv(PILOT_DIR / "pilot_grid_summary.csv")
    sheets["TableS2"] = pd.read_csv(PILOT_DIR / "pilot_rna3db_val_curve.csv")
    sheets["TableS3"] = tableS3(valid)
    sheets["TableS4a"] = pd.read_csv(PAPER / "tables/route_finetune.csv")
    sheets["TableS4b"] = pd.read_csv(PAPER / "tables/multitask.csv")
    sheets["TableS4c"] = pd.read_csv(PAPER / "tables/alignment_replay.csv")
    sheets["TableS5a"] = pd.read_csv(ABL_CSV)
    sheets["TableS5b"] = pd.read_csv(ABL_DELTA_CSV)

    xlsx = OUT / "SourceData.xlsx"
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xw:
        for name, df in sheets.items():
            df.to_excel(xw, sheet_name=name, index=False)
            df.to_csv(CSV_DIR / f"{name}.csv", index=False)
            REPORT["sheets"][name] = {"rows": int(len(df)),
                                      "cols": int(df.shape[1])}
    note(f"wrote {xlsx} with {len(sheets)} sheets")

    # ---------------------------- validation: reload the workbook ----------
    from openpyxl import load_workbook
    wb = load_workbook(xlsx, read_only=True)
    print("\n=== validation: SourceData.xlsx ===")
    total_rows = 0
    empty = []
    for name in wb.sheetnames:
        ws = wb[name]
        n_rows, n_cols = ws.max_row - 1, ws.max_column  # minus header
        total_rows += max(n_rows, 0)
        flag = ""
        if n_rows < 1:
            empty.append(name)
            flag = "  <-- EMPTY"
        print(f"  {name:16s} {n_rows:>6} data rows x {n_cols:>3} cols{flag}")
    wb.close()
    print(f"total: {len(wb.sheetnames)} sheets, {total_rows} data rows")
    assert not empty, f"empty sheets: {empty}"
    assert len(wb.sheetnames) == len(sheets)


if __name__ == "__main__":
    sys.exit(main())
