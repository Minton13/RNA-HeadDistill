# Metric Definition Audit — v1.0.1

Purpose: pin down, for every quantitative surface of the paper, *which* metric
implementation produced the number and whether two numbers may be compared
directly. Classification of values: **MEASURED** (direct pipeline output),
**DERIVED** (recomputed from measured values), **EXTRAPOLATED**,
**HYPOTHETICAL**.

## 1. Atlas / head-screening metrics (Fig. 2, Supp. Fig. 2, Supp. Table 1)

Source code: `src/rna_head_alignment/screening/metrics.py` (`binary_metrics`,
`regression_metrics`), task→metric mapping in `configs/tasks.yaml`.

| Task | `metric_primary` definition | Code |
|---|---|---|
| secondary_structure_pair | AUPRC over sampled valid pairs (max 1024 pairs/sequence, seed=42+i) | `binary_metrics`, `pairwise_screen.score_pairwise_streamed` |
| rna3d_rnaglib, rnascope_contact_map | AUPRC (same path) | same |
| splice_site_seqcls, rbp_binding_seqcls | AUROC (pooled per-head logistic probe scores) | `binary_metrics` via `seqcls_screen` |
| ncRNA_family_seqcls | macro-F1 | `seqcls_screen` |
| mrna_*/chem/openvaccine (regression) | Pearson | `seqreg_screen` / `token_screen` |

All values MEASURED. Cleaning for Fig. 2 (drop load-failed, NaN, degenerate
(task,model), head≥0) replicated in `paper/audit/heads/audit_atlas_compute.py`
and asserted in the figure scripts.

## 2. Table 1 "Score" and percentile

- `Score` = the selected head's `metric_primary` **on the task it was selected
  from** (`selection_metric = metric_primary`; source
  `outputs/teacher_banks/teacher_head_candidates.csv`).
- `percentile` = rank of that score within the head's own (task, model) pool,
  ×100. Two computation variants are on record:
  `paper/audit/heads/table1_15heads_audit.csv` columns `pctl_csv` (as shipped
  in the selection CSV; 5/15 rank-1) vs `pctl_recleaned` (recomputed after
  audit cleaning; 3/15 rank-1; median 93.8 vs 94.2). **UNRESOLVED_METRIC**:
  the two variants differ for 2 heads; the manuscript table must state which
  one it prints. Both are DERIVED from the same MEASURED atlas.

## 3. Train-only secondary re-screening (job 65397)

- Same AUPRC implementation (`binary_metrics`), same n=100 sequences/model.
- **Only the sample pool differs**: stored atlas used the first 100 records of
  the shuffled `secondary_full.jsonl` (measured composition 77 train / 14 val /
  9 test); the rerun uses `split_filter=["train"]`, `post_filter_limit=100`.
- Comparability: same metric *definition* → rank-level comparisons valid
  (used to confirm L31H13/L31H1 keep ranks 1/2). Point estimates across pools
  are **not** same-distribution estimates and must not be tabulated side by
  side as such. Status: definition-comparable YES, point-estimate-comparable NO.

## 4. Downstream evaluation (Fig. 4, Table 2)

- Per-seed metrics computed by the probe evaluation (`outputs/probes/frozen/...`
  `eval_*` columns; MEASURED) and cross-checked against the dumped per-sequence
  predictions (`dumpcheck_*` columns, max|diff| = 0.0 for AUROC cells checked;
  DERIVED, identical).
- Bootstrap CIs: `paper/audit/downstream/p0g_bootstrap_recompute.py`
  (DERIVED from per-sequence predictions).
- fp32 vs fp16: `paper/audit/downstream/fp32_reforward_summary.md`
  (P0 fp32≈fp16 ≤1.1e-5; P4 ±0.018; bootstrap P4−P0=+0.2318 [+0.2240,+0.2400]).
  Manuscript text keeps +0.227 [+0.220,+0.235] (fp16 dumps vs eval, secondary
  per-cell V10 audit) — these are **different quantities**, do not merge.

## 5. Systems numbers (Fig. 3)

- 76.7/44.1/12.4/4.13 seq/s, 0.32/4.86/10.5 GB: MEASURED (single A100,
  forward-only; S3* marked extrapolated in figure).
- 7.51 GPU-h full-corpus forward-only: DERIVED (rescaled microbenchmark).
- 33.58 GPU-h student training: MEASURED (6×A100, 5,000 updates).
- ≥235 GPU-h teacher cache build: MEASURED for 7/9 teachers, lower bound
  (giga/mega logs unrecoverable). See `paper/figures/systems/fig3_v2_measured_costs.csv`.
