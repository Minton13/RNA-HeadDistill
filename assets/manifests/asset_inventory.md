# Model And Dataset Inventory

Generated: 2026-05-18

Sizes are local on-disk sizes. Parameter counts were computed from local weight tensor shapes where possible. Explicit-large models are staged but were not forward-loaded on the Windows staging machine.

## Models

Total staged model directory size: 37.737 GiB across 306 files.

| model | type | priority | local dir | disk GiB | files | params | layers | hidden | heads | offline forward | attentions |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `rnabert` | RNA | required | `assets/models/rnabert` | 0.0020 | 18 | 511,596 | 6 | 120 | 12 | yes | yes |
| `rnafm` | RNA | required | `assets/models/rnafm` | 0.3708 | 18 | 99,522,187 | 12 | 640 | 20 | yes | yes |
| `splicebert` | RNA | required | `assets/models/splicebert` | 0.0735 | 18 | 19,718,682 | 6 | 512 | 16 | yes | yes |
| `rinalmo_micro` | RNA | required | `assets/models/rinalmo-micro` | 0.1248 | 18 | 33,482,438 | 12 | 480 | 20 | yes | yes |
| `utrbert_3mer` | RNA | required | `assets/models/utrbert-3mer` | 0.3232 | 18 | 86,734,211 | 12 | 768 | 12 | yes | yes |
| `utrlm_mrl` | RNA | required | `assets/models/utrlm-mrl` | 0.0046 | 18 | 1,210,623 | 6 | 128 | 16 | yes | yes |
| `dnabert2` | DNA | required_control | `assets/models/dnabert2` | 0.4364 | 26 | 120,219,904 | 12 | 768 | 12 | yes | no |
| `hyenadna_small` | DNA | required_control | `assets/models/hyenadna-small` | 0.0062 | 20 | 1,641,344 | 2 | 256 | n/a | yes | no |
| `hyenadna_tiny` | DNA | server_optional | `assets/models/hyenadna-tiny` | 0.0017 | 20 | 436,352 | 2 | 128 | n/a | yes | no |
| `hyenadna_medium` | DNA | server_optional | `assets/models/hyenadna-medium` | 0.0123 | 20 | 3,278,080 | 4 | 256 | n/a | yes | no |
| `hyenadna_large` | DNA | server_optional | `assets/models/hyenadna-large` | 0.0245 | 20 | 6,551,552 | 8 | 256 | n/a | yes | no |
| `nucleotide_transformer_v2_500m` | DNA | server_optional | `assets/models/nucleotide-transformer-v2-500m` | 5.5620 | 26 | 498,346,364 | 29 | 1024 | 16 | yes | yes |
| `rinalmo_mega` | RNA | server_optional | `assets/models/rinalmo-mega` | 0.5516 | 18 | 148,045,456 | 30 | 640 | 20 | yes | yes |
| `rinalmo_giga` | RNA | explicit_large | `assets/models/rinalmo-giga` | 2.4248 | 18 | 650,878,757 | not loaded | not loaded | not loaded | skipped locally | not checked |
| `nucleotide_transformer_2_5b` | DNA | explicit_large | `assets/models/nucleotide-transformer-2.5b` | 27.8186 | 30 | 2,547,802,228 | 32 | 2560 | 20 | skipped locally | not checked |

## Normalized Downstream Datasets

Total staged data directory size: 10.287 GiB.

| task | type | primary path | records | mini records | split counts | disk GiB |
|---|---|---|---:|---:|---|---:|
| `secondary_structure_pair` | pairwise | `assets/data/secondary/secondary_full.jsonl` | 26,815 | 700 | train 21,377 / val 2,695 / test 2,743 | 0.0167 |
| `chemical_reactivity_token` | token regression | `assets/data/ribonanza/hf_TerminatorJ_RNA_chemical_ribonanza/train_data_QUICK_START.csv` | 335,616 | 2,000 | full split absent; mini train 2,000 | 0.5172 |
| `ncRNA_family_seqcls` | sequence classification | `assets/data/rfam/hf_multimolecule_rfam/data.parquet` | 20,051,822 | 2,000 | mini train 1,620 / val 193 / test 187 | 0.6768 |
| `rbp_binding_seqcls` | sequence classification | `assets/data/rbp_binding/rbp_binding_full.csv` | 1,225,584 | 2,000 | train 962,671 / val 120,152 / test 142,761 | 2.7814 |
| `splice_site_seqcls` | sequence classification | `assets/data/splice_site/splice_site_full.csv` | 45,619 | 667 | train 45,619 | 0.0282 |

## Source Snapshots And Supporting Assets

| asset | path | records / files | disk size |
|---|---|---:|---:|
| HF `multimolecule/bprna-90` | `assets/data/secondary/hf_multimolecule_bprna-90/data.parquet` | 28,370 records | 0.0024 GiB |
| HF `multimolecule/pdb-rna_secondary_structure` | `assets/data/secondary/hf_multimolecule_pdb-rna_secondary_structure/*.parquet` | 256 records / 4 parquet files | <0.001 GiB |
| HF `TerminatorJ/RNA_chemical_ribonanza` | `assets/data/ribonanza/hf_TerminatorJ_RNA_chemical_ribonanza/train_data_QUICK_START.csv` | 335,616 records | 0.5129 GiB |
| HF `multimolecule/rfam` | `assets/data/rfam/hf_multimolecule_rfam/data.parquet` | 20,051,822 records | 0.6762 GiB |
| HF `dnagpt/dna_splice_site_prediction` | `assets/data/splice_site/hf_dnagpt_dna_splice_site_prediction/data/train-00000-of-00001.parquet` | 45,619 records | 0.0082 GiB |
| RBP24 raw intervals | `assets/data/rbp_binding/raw/RBP24` | supporting raw files | 0.0692 GiB |
| RBPsuite2 raw peaks | `assets/data/rbp_binding/raw/RBPsuite2` | supporting raw files | 2.1353 GiB |
| UCSC hg19 reference | `assets/data/reference/hg19.fa` | reference FASTA | 2.9801 GiB |
| RNAscope Kaggle zip | `assets/data/rnascope_kaggle/raw/RNAscope-All-Datasets.zip` | 1 zip | 0.4631 GiB |
| RNAscope Kaggle extracted | `assets/data/rnascope_kaggle/extracted` | 47 files | 1.9396 GiB |
| RNAGlib RNA 3D graphs | `assets/data/rna3d/hf_luiswyss_rnaglib/huggingface_rna_cm/dataset` | 198 graph JSON / 196 PDB IDs | 0.0102 GiB |

## RNAscope Kaggle Benchmark Bundle

Official bundle: `rnascopedata/release`.

| benchmark task | local path under `assets/data/rnascope_kaggle/extracted` | records | files | size |
|---|---|---:|---:|---:|
| Secondary Structure Prediction | `1. Structure Tasks/(1) Second Structure Prediction/*.json` | 25,124 | 3 JSON | 15.83 MiB |
| Chemical Reactivity Prediction | `1. Structure Tasks/(2) Chemical Reactivity Prediction/chemical_Reactivity.json` | 167,808 | 1 JSON | 242.97 MiB |
| Contact Map Prediction | `1. Structure Tasks/(3) Contact Map Prediction/*.lmdb` | not counted | 3 LMDB dirs | 125.80 MiB |
| Binary Binding Prediction | `2. Interaction Tasks/(1) Binary Binding Prediction/*.json` | 552,682 | 22 JSON | 138.70 MiB |
| Systematic Binding Ranking | `2. Interaction Tasks/(2) Systematic Binding Ranking/*.json` | 882,985 | 3 JSON | 171.71 MiB |
| Binding Affinity Prediction | `2. Interaction Tasks/(3) Binding Affinity Prediction/*.json` | 3,071 | 2 JSON | 0.65 MiB |
| Coding Potential Prediction | `3. Function Tasks/mRNA Tasks/(1) Coding Potential Prediction/Coding-Potential.json` | 185,928 | 1 JSON | 408.04 MiB |
| mRNA Subcellular Localization | `3. Function Tasks/mRNA Tasks/(2) mRNA subcellular Localization/mRNA-Subcellular.json` | 17,298 | 1 JSON | 62.14 MiB |
| Ribosome Loading Prediction | `3. Function Tasks/mRNA Tasks/(3) Ribosome Loading Prediction/Ribosome-Loading.json` | 789,594 | 1 JSON | 113.82 MiB |
| ncRNA Category Classification | `3. Function Tasks/ncRNA Tasks/(1) ncRNA Category Classification/ncRNA-Category-Classification.json` | 45,447 | 1 JSON | 12.76 MiB |
| microRNA Subcellular Localization | `3. Function Tasks/ncRNA Tasks/(2) micaroRNA Subcellular Localization/microRNA-Subcellular.json` | 538 | 1 JSON | 0.09 MiB |
| gRNA Efficiency Prediction | `3. Function Tasks/ncRNA Tasks/(3) gRNA Efficiency Prediction/gRNA-Efficiency.json` | 21,245 | 1 JSON | 2.88 MiB |
| Splicing Site Prediction | `3. Function Tasks/pre-mRNA Tasks/(1) Splicing Site Prediction/*.json` | 1,049,978 | 2 JSON | 554.87 MiB |
| Splicing Event Prediction | `3. Function Tasks/pre-mRNA Tasks/(2) Splicing Event Prediction/Splicing-Event.json` | 101,373 | 1 JSON | 98.44 MiB |
| Polyadenylation Signal Prediction | `3. Function Tasks/pre-mRNA Tasks/(3) Polyadenylation Signal Prediction/Polyadenylation-Signal.json` | 108,341 | 1 JSON | 37.46 MiB |

Total counted JSON records in the RNAscope bundle: 3,951,412, plus Contact Map LMDB splits.

## RNA 3D Benchmark Assets

| asset | status | local path | records | unique structures | level |
|---|---|---|---:|---:|---|
| RNAGlib RNA 3D graph benchmark | staged | `assets/data/rna3d/hf_luiswyss_rnaglib` | 198 graph JSON | 196 PDB IDs | tertiary structure graph |
| RNA-Puzzles | metadata only | not staged | n/a | n/a | blind tertiary prediction targets |
| CASP RNA targets | metadata only | not staged | n/a | n/a | blind tertiary prediction targets |
| RNA 3D Hub non-redundant sets | metadata only | not staged | n/a | n/a | representative 3D structure sets |
| RNASolo | metadata only | not staged | n/a | n/a | cleaned PDB-derived RNA 3D structures |
| RNA3DB | metadata only | not staged | n/a | n/a | structurally dissimilar 3D splits |

Lineage and Sankey files:

- `assets/manifests/asset_lineage_inventory.csv`
- `assets/manifests/asset_lineage_inventory.json`
- `outputs/lineage/asset_lineage_sankey.html`
- `outputs/lineage/asset_lineage_sankey.svg`
