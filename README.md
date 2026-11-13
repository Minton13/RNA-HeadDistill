# Functional Attention Heads Are Transferable Units of RNA Foundation Models

Code and analysis pipeline for the RECOMB submission *"Functional attention heads
are transferable units of RNA foundation models"*.

The repository implements the full workflow of the paper:

1. **Head screening (atlas).** Every attention head of 26 RNA/DNA foundation
   models is probed on 15 supervised downstream tasks (84,066 per-head probe
   records), producing a per-model, per-layer, per-head functionality atlas.
2. **Supervised teacher-head selection.** Top-scoring heads are selected into
   three route banks (student layers L6/L8/L12) with full
   model–layer–head identity preserved; the final 15 distillation targets are
   curated in `configs/teacher_bank_selected.yaml`.
3. **One-time teacher cache.** The nine teacher models run once over a 2.1M
   sequence sample of the RiNALMo pretraining corpus; the selected full
   `L x L` attention maps are stored as a slim fp16 cache (8.8 TiB). Teachers
   never run again.
4. **Head-to-head asynchronous distillation.** Each teacher head is paired with
   one native RiNALMo-micro student head; the student head output passes through
   a single-layer Transformer alignment adapter and is trained with a direct
   per-head loss (row-wise KL + MSE + cosine) against the cached map, with
   depth-routed gradients (route exits at layers 6/8/12). Trained with 6-GPU
   DDP (`torchrun`).
5. **Downstream evaluation.** Frozen probes (and fine-tuning controls) on six
   downstream tasks, three seeds, with paired bootstrap statistics.
6. **Ablations (A1–A8) and systems benchmark.** Alignment-method ablations,
   data/loss/teacher-count scaling, pairing stability, route fine-tuning and
   multitask controls, plus throughput/memory/cost measurements of asynchronous
   vs. online distillation.

```
 26 teacher models                15 supervised tasks
        |                                |
        v                                v
  extract per-head attentions  ->  linear/MLR probes per head   [screening]
        |                                |
        +------------>  head atlas (84,066 records)  <---------+
                                 |
                                 v
                   supervised selection: 15 target heads
                   (route banks L6 / L8 / L12)                 [selection]
                                 |
   RiNALMo pretraining corpus    v
   10% sample (2.1M seqs)  ->  cache selected teacher attention maps
                                 (8.8 TiB fp16, one-time)       [cache]
                                 |
                                 v
             head-to-head alignment training of RiNALMo-micro
             (per-head adapters, depth-routed gradients, DDP)   [distill]
                                 |
                                 v
        downstream frozen probes / fine-tune controls (3 seeds) [evaluate]
                                 |
                                 v
              ablations A1-A8 + systems benchmark               [analysis]
```

## Repository layout

```
rna_head_alignment/
├── pyproject.toml              # package metadata + dependencies
├── README.md
├── src/rna_head_alignment/     # installable package (all pipeline logic)
│   ├── cli.py                  # single CLI entry point (see below)
│   ├── models/                 # backbone loading, attention extraction, registry
│   ├── screening/              # per-head probe screening, heatmaps, head selection
│   ├── cache/                  # teacher attention cache build/validate
│   ├── head2head/              # head-pairing, slim cache, DDP training, eval
│   ├── probes/                 # downstream frozen probes / fine-tuning / multitask
│   ├── ablation/               # ablation runners
│   ├── systems/                # throughput/memory/cost benchmark, cache formats
│   ├── data/                   # task loaders (15 tasks), corpus sampler, splits
│   └── io/                     # paths, manifests, env checks
├── configs/                    # model/task registries + pipeline configs (YAML)
│   └── demo/                   # tiny configs for the reviewer demo
├── scripts/
│   ├── demo/                   # one-command toy-data demo (Quickstart)
│   ├── local/                  # local helper scripts
│   ├── head2head/              # cache/manifest/training drivers
│   ├── ablation/               # ablation + table/report drivers
│   ├── analysis/               # bootstrap + figure-source aggregation
│   └── server/                 # Slurm templates used for the paper runs
├── tests/                      # pytest suite (CPU-only, no network)
├── docs/internal/              # internal handoff notes (not part of the paper)
└── assets/manifests/           # data/model provenance inventory
```

`outputs/`, `runs/`, `cache/`, `logs/`, `assets/models/`, `assets/data/` are
generated or staged locally and are excluded from version control (see
`.gitignore`).

## Installation

Requires Python 3.10+; PyTorch with CUDA for real runs (CPU suffices for the
demo and the test suite).

```bash
# conda (recommended)
conda create -n rna-head-alignment python=3.10 -y
conda activate rna-head-alignment
pip install torch --index-url https://download.pytorch.org/whl/cu128  # or your CUDA build

cd rna_head_alignment
pip install -e .
# optional extras: Slurm-side logging and analysis/plot helpers
pip install -e ".[server,analysis]"
```

Notable dependencies (see `pyproject.toml` for the full list): `torch`,
`transformers`, `multimolecule` (provides the RiNALMo student and several RNA
teacher architectures), `scikit-learn`, `pandas`, `pyarrow`, `safetensors`,
`lmdb` (only for the RNAscope contact-map LMDB loader; imported lazily),
`biopython`. The `analysis` extra adds `plotly`/`seaborn`/`zarr` used by some
helper scripts and the zarr cache-format benchmark backend (an `npz` fallback
is used when `zarr` is absent).

Model weights are loaded from local directories under `assets/models/` with
`local_files_only=True`. Every model used in the paper is public on Hugging
Face; the registry (`configs/models.yaml`) maps each model name to its staged
directory. For the demo you only need RiNALMo-micro (~120 MB):

```bash
python scripts/demo/fetch_demo_model.py   # downloads multimolecule/rinalmo-micro
# if huggingface.co is unreachable: HF_ENDPOINT=https://hf-mirror.com python scripts/demo/fetch_demo_model.py
```

## Quickstart: 10-minute demo on toy data

One command runs the core pipeline — per-head screening, teacher-head
selection, and a small teacher-attention cache with validation — on two
synthetic tasks (toy hairpin secondary structure + motif family
classification) generated deterministically by `scripts/demo/make_toy_data.py`:

```bash
bash scripts/demo/run_demo.sh
```

Requirements: `pip install -e .` plus the staged RiNALMo-micro weights (the
script downloads them automatically if missing). Runs on CPU; measured ~1.5
minutes on a 2024 server CPU (set `DEVICE=cuda` to use a GPU). Outputs land in
`outputs/demo/` (rebuilt on each run):

| Artifact | Content |
|---|---|
| `outputs/demo/head_screening/all_head_atlas.csv` | 480 per-head probe scores (2 tasks x 240 heads) |
| `outputs/demo/head_screening/heatmaps/` | per-task head-score heatmaps (same style as Fig. 2a) |
| `outputs/demo/teacher_banks/teacher_bank.yaml` | selected teacher heads (route banks L6/L12) |
| `outputs/demo/teacher_attention_cache/` | cached fp16 attention maps for the selected heads |
| `outputs/demo/teacher_attention_cache_validation.md` | cache validation report |

The demo exercises the same CLI commands and code paths as the full pipeline
below; only the data and model count differ.

## Full pipeline (paper-scale)

All functionality is exposed through one CLI, `python -m rna_head_alignment.cli
<command>`. Paths in configs are relative to this directory. The commands below
assume the model weights and downstream datasets are staged under `assets/`
(see *Data availability*).

```bash
# 0. environment / asset checks
python -m rna_head_alignment.cli check-env    --config configs/paths.yaml
python -m rna_head_alignment.cli verify-models --models configs/models.yaml \
    --output outputs/model_checks/model_attention_check.csv --max-len 128
python -m rna_head_alignment.cli verify-data   --tasks configs/tasks.yaml \
    --output outputs/model_checks/data_check.csv --limit 5

# 1. head screening (atlas) — one job per model x task, Slurm-array friendly
python -m rna_head_alignment.cli make-screening-jobs --models configs/models.yaml \
    --tasks configs/tasks.yaml \
    --model-check outputs/model_checks/model_attention_check.csv \
    --output outputs/head_screening/screening_jobs.csv
python -m rna_head_alignment.cli run-screening-jobs \
    --jobs outputs/head_screening/screening_jobs.csv --config configs/screening.yaml \
    --report outputs/head_screening/screening_run_report.csv --device cuda
python -m rna_head_alignment.cli collect-head-screening \
    --input "outputs/head_screening/*/*/head_atlas.csv" \
    --output outputs/head_screening/all_head_atlas.csv
python -m rna_head_alignment.cli plot-head-heatmaps \
    --atlas outputs/head_screening/all_head_atlas.csv --tasks configs/tasks.yaml \
    --output-dir outputs/head_screening/heatmaps

# 2. supervised teacher-head selection
python -m rna_head_alignment.cli propose-teacher-heads \
    --atlas outputs/head_screening/all_head_atlas.csv --config configs/screening.yaml \
    --output outputs/teacher_banks/teacher_head_candidates.csv \
    --yaml  outputs/teacher_banks/teacher_bank_template.yaml
# the paper's 15 targets are the curated configs/teacher_bank_selected.yaml

# 3. distillation corpus sample (10% of the RiNALMo pretraining corpus)
python -m rna_head_alignment.cli sample-rinalmo-corpus \
    --corpus assets/data/rinalmo_pretraining_corpus --fraction 0.10 \
    --output outputs/task_samples/rinalmo_pretrain_10pct.jsonl

# 4. teacher attention cache (8.8 TiB at paper scale; --limit for small runs)
python -m rna_head_alignment.cli cache-teacher-attention \
    --teacher-bank configs/teacher_bank_selected.yaml \
    --sequences outputs/task_samples/rinalmo_pretrain_10pct.jsonl \
    --output outputs/teacher_attention_cache/run0 --device cuda
python -m rna_head_alignment.cli validate-teacher-cache \
    --cache outputs/teacher_attention_cache/run0 \
    --output outputs/teacher_attention_cache/run0_validation.md
```

Stages 4–6 at paper scale (slim-cache build over 2,114,289 sequences,
head-pair search, 6-GPU DDP training, downstream probes, ablations, systems
benchmark) are driven by the Slurm templates in `scripts/server/`, documented
in *Reproduction map* below. Key entry points:

| Stage | Driver | Config |
|---|---|---|
| Slim cache manifests/build | `scripts/server/head2head_build_sequence_manifest.sbatch`, `head2head_build_full_slim_cache.sbatch` | `configs/head2head/slim_cache_smoke.yaml` (schema example) |
| Head-pair search | `scripts/server/head2head_pair_search.sbatch` | `configs/head2head/head_pair_search.yaml` |
| Head-to-head training (6-GPU DDP) | `scripts/server/head2head_train_full_6gpu.sbatch` | `configs/head2head/train_head2head_full_6gpu.yaml` |
| Downstream data prep | `scripts/server/prepare_downstream_datasets.sbatch` | `configs/ablation/downstream_data.yaml` |
| Frozen probes (126-job array) | `scripts/server/frozen_probe_array.sbatch` | `configs/ablation/frozen_probe_base.yaml` |
| Route fine-tune / multitask | `scripts/server/route_finetune_array.sbatch`, `multitask_training.sbatch` | `configs/ablation/experiment_registry.yaml` |
| Ablations A1–A8 | `scripts/server/ablation_core_array.sbatch`, `fullbb_ablation_array.sbatch`, `supplementary_ablation_array.sbatch`, `replay_ablation.sbatch` | `configs/ablation/` |
| Systems benchmark | `scripts/server/systems_benchmark.sbatch` | — |

> **Note on `scripts/server/`:** these are the Slurm templates used for the
> paper runs on our cluster. They are site templates: before submitting, set
> `RHA_ROOT` (repository root; defaults to the Slurm submit directory) and
> `RHA_CONDA_ENV` (conda env name or path; default `rnallm`), and adapt the
> `#SBATCH` resource lines to your cluster.

## Reproduction map

The manuscript lives in the sibling directory `../paper/` (LaTeX in
`../paper/manuscript/`, figures in `../paper/figures/`, aggregated tables in
`../paper/tables/`). Two tiers of reproduction:

- **Tier 1 — figures/tables from aggregated artifacts (minutes, CPU).** Uses
  the small CSVs/parquets already aggregated under `outputs/` and
  `../paper/tables/`. This is what reviewers can realistically re-run.
- **Tier 2 — regenerate the raw artifacts (GPU cluster).** Requires the staged
  models/datasets and, for student re-training, the 8.8 TiB teacher cache (see
  *Data availability*).

| Paper item | Tier-1 script | Inputs (all small, no 8.8 TiB cache) |
|---|---|---|
| Fig. 1 (schematic) | `../paper/scripts/plot_fig1_schematic.py` | `outputs/head_screening/all_head_atlas.csv`; one sample slim-cache bucket + trained checkpoint for real thumbnails (graceful glyph fallback without them) |
| Fig. 2 (head atlas) | `../paper/scripts/plot_fig2_atlas.py` | `outputs/head_screening/all_head_atlas.csv`, `outputs/teacher_attention_cache_slim/stage4b_head2head/manifests/target_head_manifest.parquet`, `../paper/tables/selected_teacher_heads.csv` |
| Fig. 3 (async economics + A1–A8) | `../paper/scripts/plot_fig3_async.py` | `outputs/systems/asynchronous_distillation/{throughput,memory}_comparison.csv`, `asynchronous_cost_curve.csv`, `../paper/tables/alignment_ablation.csv` |
| Fig. 4 (downstream) | `../paper/scripts/plot_fig4_downstream.py` | `outputs/probes/analysis_v2/bootstrap/*.csv`, `bootstrap_a8/*.csv`, `../paper/tables/downstream_primary_bootstrap.csv` |
| Fig. S1 (data flow) | `../paper/scripts/plot_si_dataflow.py` | cache manifest split counts, `outputs/asset_summaries/*sankey*_v2.csv` |
| Table 1 (15 teacher heads) | `../paper/notes/_analysis_head_function.py` | `all_head_atlas.csv` + target-head manifest → `../paper/tables/selected_teacher_heads.csv` |
| Table 2 (frozen probes) | `scripts/analysis/paper_bootstrap_fast.py`, `scripts/analysis/paper_full_metrics.py` | `outputs/probes/analysis_v2/` → `../paper/tables/downstream_primary_bootstrap.csv` |
| Table S1 (per-model concentration) | `../paper/notes/_analysis_head_function.py` | `all_head_atlas.csv` → `../paper/tables/head_functionality_stats.csv` |
| Table S2 (fine-tune/multitask) | `../paper/scripts/rebuild_fixed_tables.py` | `outputs/probes/finetune/`, `outputs/probes/multitask/` → `route_finetune.csv`, `multitask.csv` |
| Table S3 (core ablation) | `../paper/scripts/rebuild_fixed_tables.py` | `outputs/ablation/alignment_fullbb/` → `alignment_ablation.csv` (+ `alignment_ablation_delta.csv`, `alignment_replay.csv`) |

Tier-2 regeneration of the upstream artifacts:

| Artifact | Entry point | Scale / expected runtime |
|---|---|---|
| `all_head_atlas.csv` (84,066 records) | `make-screening-jobs` + `scripts/server/02_screen_heads_array.sbatch` (`SCREENING_LIMIT=256`) + `03_collect_screening_results.sbatch` | 26 models x 15 tasks, one A100 per array job; minutes per job, embarrassingly parallel |
| 15-head teacher bank | `propose-teacher-heads` + manual curation | minutes |
| 2.1M-sequence corpus sample | `sample-rinalmo-corpus` / `scripts/server/08_sample_rinalmo_10pct.sbatch` | CPU, hours (corpus scan) |
| Slim teacher cache (8.8 TiB) | `scripts/server/head2head_build_full_slim_cache.sbatch` chain (1033 shards) | multi-day Slurm array over 9 teacher models, fp16 |
| Trained student (`runs/head2head_full/`) | `scripts/server/head2head_train_full_6gpu.sbatch` | 20,000 steps, 6 x A100 DDP, 72 h wall-time budget |
| Frozen-probe results | `scripts/server/frozen_probe_array.sbatch` (126 jobs) + `scripts/ablation/summarize_frozen_probes.py` | one GPU per job |
| Ablations A1–A8 | `scripts/server/ablation_core_array.sbatch`, `fullbb_ablation_array.sbatch`, `supplementary_ablation_array.sbatch` | one GPU per variant x seed |
| Systems numbers | `scripts/server/systems_benchmark.sbatch` | single GPU, hours |

## Data availability

- **Teacher attention cache.** The slim cache (8.8 TiB fp16, 2,114,289
  sequences x 15 teacher heads, length-bucketed `safetensors` shards with
  parquet manifests) cannot be redistributed publicly. It can be rebuilt
  exactly with the pipeline above (teacher weights are public; the corpus
  sample and selected-head bank are specified in
  `configs/teacher_bank_selected.yaml` and `configs/paths.yaml`). The cache is
  also **available from the authors upon request**.
- **Head atlas and aggregated results.** The per-head probe atlas
  (`all_head_atlas.csv`), selected-head tables, downstream bootstrap summaries,
  and systems measurements are small CSVs/parquets under `outputs/` and
  `../paper/tables/` and are included with the submission.
- **Downstream benchmark datasets** are public: dnagpt splice sites (donor/
  acceptor), rbp24 (RBP binding), Rfam (ncRNA family), Ribonanza (chemical
  reactivity, 2A3/DMS), bpRNA90/PDB (secondary structure); screening-only tasks
  additionally use OpenVaccine, RNAscope (Kaggle), CDS-BART/GleghornLab mRNA
  stability, and RNAGlib/RNAscope contact maps. Provenance, source URLs, and
  per-file checksums are catalogued in `assets/manifests/`. Dataset preparation
  for downstream evaluation is `configs/ablation/downstream_data.yaml` +
  `scripts/ablation/prepare_downstream_datasets.py`.
- **Model weights**: all 26 screened models are public on Hugging Face
  (`configs/models.yaml` maps names to staged directories; several need
  `trust_remote_code`, handled automatically). RiNALMo-micro is both a screened
  teacher and the student backbone.

## Hardware and software environment

Paper runs were executed on a Slurm cluster with NVIDIA A100-PCIE-40GB GPUs:
head screening and probes as single-GPU array jobs; head-to-head training on 6
GPUs with `torchrun` DDP (`NCCL_IB_DISABLE=1 NCCL_P2P_DISABLE=1` on our
interconnect). Reference software environment: Python 3.10, PyTorch
2.9.1+cu128, `transformers` + `multimolecule`, CUDA 12.8. All teacher inference
and the cache are fp16; student training is fp32 with per-route adapters
(`adapter_dim: 96`). CPU is sufficient for the demo and the test suite.

## Testing

```bash
cd rna_head_alignment
pip install -e .
pytest            # or: PYTHONPATH=src python -m pytest
```

The suite (75 tests, ~25 s, CPU-only, no network) covers attention-extraction
shapes, cache round-trip and slim-cache equivalence, head2head losses/metrics/
gradient routing/DDP wiring, MLM utilities, manifest builders, task loaders,
and collation. Current status on the reference environment: **75 passed**.

## Notes

- Internal operational handoff documents used during development are kept in
  `docs/internal/`; they are not part of the reproducibility surface.
- All commands must be run from this directory (`rna_head_alignment/`); config
  paths are resolved relative to it. When running scripts directly (not the
  installed CLI), set `PYTHONPATH=$PWD/src`.
- If you regenerate demo/toy artifacts locally, nothing outside `data/demo/`
  and `outputs/demo/` is touched.

## Paper reproduction

All main-text and SI figures/tables can be regenerated from the published
derived-data package — see [REPRODUCTION.md](REPRODUCTION.md) for the
per-figure / per-table commands (`scripts/reproduction/`), prerequisites, and
verification against the Source Data CSVs.

- HF dataset `dataset-rna-head-distill` (figures/tables source data):
  <!-- TODO: fill after upload — https://huggingface.co/datasets/<org>/dataset-rna-head-distill -->
- HF model `rinalmo-micro-head2head` (trained student checkpoint):
  <!-- TODO: fill after upload — https://huggingface.co/<org>/rinalmo-micro-head2head -->

## Citation

TODO (added upon publication).

## License

MIT License. See [LICENSE](LICENSE).
