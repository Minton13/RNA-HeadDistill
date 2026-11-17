# Paper reproduction

Every main-text and SI figure/table of the paper can be regenerated from the
published derived-data package (the 8.8 TiB raw attention cache and the raw
per-seed training artifacts are **not** redistributed; the package ships the
aggregated CSVs/NPZs those figures and tables were built from).

## Prerequisites

1. Install the package with analysis extras (Python 3.10–3.11):

   ```bash
   pip install -e ".[analysis]"
   ```

2. Download the derived-data package from the repository's **GitHub Release**
   ([GitHub Releases](https://github.com/Minton13/RNA-HeadDistill/releases)), verify and unpack:

   ```bash
   # asset: dataset-rna-head-distill_v1.0.0.tar.gz (~2.1 MB)
   wget https://github.com/Minton13/RNA-HeadDistill/releases/download/v1.0.0/dataset-rna-head-distill_v1.0.0.tar.gz
   tar -xzf dataset-rna-head-distill_v1.0.0.tar.gz
   cd dataset-rna-head-distill && sha256sum -c SHA256SUMS && cd ..
   export RNA_HEAD_DISTILL_DATA=$PWD/dataset-rna-head-distill
   ```

   All scripts also accept an explicit `--data-root <dir>`; `--outdir`
   defaults to `./reproduced_figures`.

The trained student checkpoint is published as the separate release asset
`student-rinalmo-micro-head2head_v1.0.0.tar.gz` (~844 MB, AGPL-3.0 — see
[MODEL_WEIGHTS.md](MODEL_WEIGHTS.md)); unpack it the same way. It is only
needed for evaluating or fine-tuning the student, not for regenerating
figures/tables.

## Figures

| Figure | Command | Reads from `--data-root` |
|---|---|---|
| Fig. 1 | `python scripts/reproduction/plot_fig1_schematic.py --outdir figs/` | `fig1_real_data/*.npz` |
| Fig. 2 | `python scripts/reproduction/plot_fig2_atlas.py --outdir figs/` | `atlas/all_head_atlas.csv`, `selection/selected_teacher_heads.csv` |
| Fig. 3 | `python scripts/reproduction/plot_fig3_async.py --outdir figs/` | `systems/*.csv`, `metrics/ablation/alignment_ablation.csv` |
| Fig. 4 | `python scripts/reproduction/plot_fig4_downstream.py --outdir figs/` | `metrics/bootstrap/*.csv`, `metrics/downstream/downstream_primary_bootstrap.csv` |
| Fig. S1 | `python scripts/reproduction/plot_si_dataflow.py --outdir figs/` | `dataflow/*_counts_v2.csv` |
| Fig. S2 + S3 | `python scripts/reproduction/plot_si_pilot.py --outdir figs/` | `pilot/pilot_mcc_heatmap_*.csv`, `pilot/pilot_decoder_curves.csv` |

Fig. 1 falls back to synthetic schematic thumbnails when the
`fig1_real_data/*.npz` files are absent (the panel geometry is data-independent);
with the data package present, the probe-score heatmaps (panel a) and the
teacher/student attention maps (panel c) are the real ones.

## Tables

One command rebuilds every paper table into `<outdir>/tables/` and writes
`tables_manifest.json` recording, per table, whether it was **rebuilt** from
derived data or **copied canonical** from the package:

```bash
python scripts/reproduction/reproduce_tables.py --outdir repro/
```

| Table | Strategy | Source |
|---|---|---|
| Table 1 (selected heads) | canonical | `selection/selected_teacher_heads.csv` |
| Table 2 (downstream) | canonical | `metrics/downstream/downstream_full_metrics.csv` |
| Table S1 (pilot grid) | canonical | `pilot/pilot_grid_summary.csv` |
| Table S2 (pilot RNA3DB) | canonical | `pilot/pilot_rna3db_val_curve.csv` |
| Table S3 (head-score concentration) | rebuilt | `atlas/all_head_atlas.csv` |
| Table S4 (fine-tuning alternatives) | canonical | `metrics/ablation/route_finetune.csv`, `multitask.csv` |
| Table S5 (alignment ablation) | canonical | `metrics/ablation/alignment_ablation.csv` |
| supporting: delta / replay / pairing / loss / data-scaling / teacher-count | canonical or rebuilt | `metrics/ablation/` (see manifest) |

`scripts/reproduction/rebuild_fixed_tables.py` and
`scripts/reproduction/rebuild_pairing_stability.py` are the original
raw-artifact table pipelines, ported for provenance: they re-aggregate the
per-seed evaluation parquets/JSONs, which are not in the data package, so
they print a `[skip]` notice unless the raw artifacts are staged under
`<data-root>/metrics/ablation/raw/…`. The pilot canonical CSVs
(`pilot/*.csv`) ship with the data package because the legacy pilot project's
raw per-run logs (tens of GB, separate repository) are not redistributable.

## Verification

1. `python scripts/reproduction/reproduce_tables.py --outdir repro/` — every
   table row in `repro/tables/` must be numerically identical to the
   `figures_source/` CSVs (Source Data) shipped in the same package, e.g.
   `diff <(sort repro/tables/head_functionality_stats.csv) <(sort "$RNA_HEAD_DISTILL_DATA/figures_source/head_functionality_stats.csv")`
   where a same-named Source Data CSV exists.
2. Each figure script prints the path it wrote plus `real`/`fallback` data
   status (Fig. 1); expect `real` for all thumbnails with the full package.
3. `pip install -e . && pytest` — the bundled test suite (75 tests, CPU-only,
   no network) validates the analysis/training code itself.
