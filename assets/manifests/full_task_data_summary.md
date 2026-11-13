# Full Task Data Summary

- generated_at: `2026-05-16T13:29:17`

| task | records | primary_path | status |
|---|---:|---|---|
| `secondary_structure_pair` | `26815` | `assets\data\secondary\secondary_full.jsonl` | `ok` |
| `chemical_reactivity_token` | `335616` | `assets\data\ribonanza\hf_TerminatorJ_RNA_chemical_ribonanza\train_data_QUICK_START.csv` | `ok` |
| `ncRNA_family_seqcls` | `20051822` | `assets\data\rfam\hf_multimolecule_rfam` | `ok` |
| `rbp_binding_seqcls` | `1225584` | `assets\data\rbp_binding\rbp_binding_full.csv` | `ok` |
| `splice_site_seqcls` | `45619` | `assets\data\splice_site\splice_site_full.csv` | `ok` |

## Details
```json
{
  "generated_at": "2026-05-16T13:29:17",
  "hf_datasets": {
    "secondary_bprna90": {
      "repo_id": "multimolecule/bprna-90",
      "required": true,
      "kind": "secondary",
      "local_dir": "assets\\data\\secondary\\hf_multimolecule_bprna-90",
      "download_status": "ok",
      "attempts": [
        {
          "status": "ok",
          "endpoint": "local",
          "path": "assets\\data\\secondary\\hf_multimolecule_bprna-90",
          "method": "existing_local_files",
          "error": null
        }
      ],
      "size_bytes": 2582453,
      "size_gib": 0.0024
    },
    "secondary_pdb": {
      "repo_id": "multimolecule/pdb-rna_secondary_structure",
      "required": false,
      "kind": "secondary",
      "local_dir": "assets\\data\\secondary\\hf_multimolecule_pdb-rna_secondary_structure",
      "download_status": "ok",
      "attempts": [
        {
          "status": "ok",
          "endpoint": "local",
          "path": "assets\\data\\secondary\\hf_multimolecule_pdb-rna_secondary_structure",
          "method": "existing_local_files",
          "error": null
        }
      ],
      "size_bytes": 25560,
      "size_gib": 0.0
    },
    "rfam": {
      "repo_id": "multimolecule/rfam",
      "required": true,
      "kind": "rfam",
      "local_dir": "assets\\data\\rfam\\hf_multimolecule_rfam",
      "download_status": "ok",
      "attempts": [
        {
          "status": "ok",
          "endpoint": "local",
          "path": "assets\\data\\rfam\\hf_multimolecule_rfam",
          "method": "existing_local_files",
          "error": null
        }
      ],
      "size_bytes": 726084264,
      "size_gib": 0.6762
    },
    "ribonanza_hf_quickstart": {
      "repo_id": "TerminatorJ/RNA_chemical_ribonanza",
      "required": false,
      "kind": "ribonanza",
      "local_dir": "assets\\data\\ribonanza\\hf_TerminatorJ_RNA_chemical_ribonanza",
      "download_status": "ok",
      "attempts": [
        {
          "status": "ok",
          "endpoint": "local",
          "path": "assets\\data\\ribonanza\\hf_TerminatorJ_RNA_chemical_ribonanza",
          "method": "existing_local_files",
          "error": null
        }
      ],
      "size_bytes": 550746496,
      "size_gib": 0.5129
    },
    "splice_site_hf": {
      "repo_id": "dnagpt/dna_splice_site_prediction",
      "required": true,
      "kind": "splice_site",
      "local_dir": "assets\\data\\splice_site\\hf_dnagpt_dna_splice_site_prediction",
      "download_status": "ok",
      "attempts": [
        {
          "status": "ok",
          "endpoint": "local",
          "path": "assets\\data\\splice_site\\hf_dnagpt_dna_splice_site_prediction",
          "method": "existing_local_files",
          "error": null
        }
      ],
      "size_bytes": 8801239,
      "size_gib": 0.0082
    }
  },
  "downloads": {
    "rbp24": {
      "source": "ML-Bioinfo-CEITEC/ENNGene RBP24",
      "attempts": [
        {
          "status": "ok",
          "method": "existing",
          "url": "https://github.com/ML-Bioinfo-CEITEC/ENNGene/raw/master/RBP24/RBP24_input_files.tar.gz",
          "path": "assets\\data\\rbp_binding\\raw\\RBP24\\RBP24_input_files.tar.gz",
          "size_bytes": 15841159,
          "error": null
        },
        {
          "status": "ok",
          "method": "existing",
          "url": "https://github.com/ML-Bioinfo-CEITEC/ENNGene/raw/master/RBP24/RBP24_evaluation_(leaveout)_files.tar.gz",
          "path": "assets\\data\\rbp_binding\\raw\\RBP24\\RBP24_evaluation_(leaveout)_files.tar.gz",
          "size_bytes": 331061,
          "error": null
        }
      ]
    },
    "rbpsuite2": {
      "source": "Zenodo record 14215811",
      "attempts": [
        {
          "status": "ok",
          "method": "existing",
          "url": "https://zenodo.org/api/records/14215811/files/RBPsuite2.zip/content",
          "path": "assets\\data\\rbp_binding\\raw\\RBPsuite2.zip",
          "size_bytes": 244177949,
          "error": null
        }
      ]
    },
    "hg19": {
      "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz",
      "gz_path": "assets\\data\\reference\\hg19.fa.gz",
      "fa_path": "assets\\data\\reference\\hg19.fa",
      "download": {
        "status": "ok",
        "method": "existing",
        "url": "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz",
        "path": "assets\\data\\reference\\hg19.fa.gz",
        "size_bytes": 948731419,
        "error": null
      },
      "decompress": {
        "status": "ok",
        "method": "existing",
        "size_bytes": 3199905909
      }
    }
  },
  "secondary": {
    "prepared": true,
    "full_path": "assets\\data\\secondary\\secondary_full.jsonl",
    "mini_path": "assets\\data\\secondary\\secondary_mini.jsonl",
    "long_skipped_path": "assets\\data\\secondary\\secondary_full_long_skipped.jsonl",
    "invalid_skipped_path": "assets\\data\\secondary\\secondary_invalid_skipped.jsonl",
    "full_records": 26815,
    "mini_records": 700,
    "split_counts": {
      "train": 21377,
      "test": 2743,
      "val": 2695
    },
    "mini_split_counts": {
      "train": 500,
      "val": 100,
      "test": 100
    },
    "min_len": 11,
    "max_len": 512,
    "long_skipped": 1661,
    "invalid_skipped": 150,
    "sources": [
      "secondary_bprna90",
      "secondary_pdb"
    ]
  },
  "ribonanza": {
    "prepared": true,
    "full_path": "assets\\data\\ribonanza\\hf_TerminatorJ_RNA_chemical_ribonanza\\train_data_QUICK_START.csv",
    "full_records": 335616,
    "full_snapshot_path": "assets\\data\\ribonanza\\hf_TerminatorJ_RNA_chemical_ribonanza",
    "mini_path": "assets\\data\\ribonanza\\ribonanza_mini.csv",
    "mini_records": 2000,
    "mini_split_counts": {
      "train": 2000
    }
  },
  "rfam": {
    "prepared": true,
    "full_snapshot_path": "assets\\data\\rfam\\hf_multimolecule_rfam",
    "full_records": 20051822,
    "full_parquet_path": "assets\\data\\rfam\\hf_multimolecule_rfam\\data.parquet",
    "mini_path": "assets\\data\\rfam\\rfam_mini.csv",
    "mini_records": 2000,
    "mini_split_counts": {
      "train": 1620,
      "val": 193,
      "test": 187
    },
    "full_label_field": "family",
    "label_map_path": "assets\\data\\rfam\\rfam_family_label_map.csv",
    "num_classes": 4024,
    "mini_label_map_path": "assets\\data\\rfam\\rfam_family_label_map.csv",
    "mini_num_classes": 2000
  },
  "rbp_binding": {
    "prepared": true,
    "source": "RBP24 sequence labels extracted from BED coordinates against UCSC hg19",
    "interval_full_path": "assets\\data\\rbp_binding\\rbp_binding_intervals_full.csv",
    "interval_mini_path": "assets\\data\\rbp_binding\\rbp_binding_intervals_mini.csv",
    "interval_records": 1227141,
    "interval_mini_records": 2000,
    "interval_split_counts": {
      "train": 963896,
      "val": 120309,
      "test": 142936
    },
    "rbp_count": 24,
    "sequence_prepared": true,
    "full_path": "assets\\data\\rbp_binding\\rbp_binding_full.csv",
    "mini_path": "assets\\data\\rbp_binding\\rbp_binding_mini.csv",
    "skipped_path": "assets\\data\\rbp_binding\\rbp_binding_sequence_skipped.csv",
    "records": 1225584,
    "mini_records": 2000,
    "split_counts": {
      "train": 962671,
      "val": 120152,
      "test": 142761
    },
    "label_counts": {
      "0": 580112,
      "1": 645472
    },
    "min_len": 12,
    "max_len": 75,
    "skipped": 1557,
    "rbpsuite2_index_path": "assets\\data\\rbp_binding\\rbpsuite2_peak_files.csv",
    "rbpsuite2_peak_files": 198,
    "rbpsuite2_peak_records": 20678493
  },
  "splice_site": {
    "prepared": true,
    "full_path": "assets\\data\\splice_site\\splice_site_full.csv",
    "mini_path": "assets\\data\\splice_site\\splice_site_mini.csv",
    "records": 45619,
    "mini_records": 667,
    "split_counts": {
      "train": 45619
    },
    "class_counts": {
      "1": 9999,
      "0": 10000,
      "2": 25620
    },
    "binary_label_counts": {
      "1": 35619,
      "0": 10000
    },
    "source": "dnagpt/dna_splice_site_prediction"
  },
  "kaggle_rnascope": {
    "attempted": true,
    "prepared": true,
    "dataset_slug": "rnascopedata/release",
    "dataset_url": "https://www.kaggle.com/datasets/b0aeeed2f6b3dfd43c1ab33b58467a466308d056b6868d861ad00e1074ce384d",
    "download_url": "https://www.kaggle.com/datasets/b0aeeed2f6b3dfd43c1ab33b58467a466308d056b6868d861ad00e1074ce384d/download?datasetVersionNumber=1",
    "dataset_id": "7426236",
    "expected_zip_bytes": 497267067,
    "license": "CC BY-NC-SA 4.0",
    "requires_subscription": true,
    "credentials_ok": true,
    "credential_source": "~/.kaggle/kaggle.json",
    "zip_path": "assets\\data\\rnascope_kaggle\\raw\\RNAscope-All-Datasets.zip",
    "zip_size_bytes": 497267067,
    "extract_dir": "assets\\data\\rnascope_kaggle\\extracted",
    "extract_file_count": 47,
    "extract_size_bytes": 2082655662,
    "download_method": "kagglesdk_hash_link",
    "reason": null
  }
}
```
