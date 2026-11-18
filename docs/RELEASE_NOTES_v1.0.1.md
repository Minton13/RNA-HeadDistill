# RNA-HeadDistill v1.0.1 — reproducibility supplement

Supplements the unchanged [v1.0.0 original study artifacts](https://github.com/Minton13/RNA-HeadDistill/releases/tag/v1.0.0).

This release fixes CUDA attention conversion, makes screening failures visible, applies annotated split filtering before sample limits, and adds prediction-index collision checks and gated-config metric replay. No model checkpoint, reported experimental value or frozen Figure 1–4 is changed.

## New assets

- **Code and documentation:** reviewed overlay, tests, screening patch and reproduction/access documentation. Generated Python bytecode is excluded.
- **Metadata and Source Data:** 22-row provenance; checkpoint, license and metric indices; corrected prediction paths; 135 nine-configuration primary per-seed points; SourceData.xlsx (33 sheets / 8,117 rows) and matching CSVs.
- **Downstream predictions:** 241 unique files across five tasks / nine configurations, comprising 135 test and 106 existing validation files. P4g and A8-P4g use distinct directories. The 27 P4g hashes that failed in the earlier transfer now verify.

## Verification

- **88 tests passed**, including actual CUDA tensors on an RTX 3070.
- **241/241** packaged prediction-file sizes and SHA256 hashes verified; zero collisions.
- **408/408** per-seed metric checks MATCH, max difference 9.103828801926284e-15 (tolerance 1e-6): 273 original checks plus 135 primary-point checks, covering 135 unique cells. P4g and A8-P4g each cover 15/15 cells. This does not claim a complete rerun of bootstrap confidence intervals.
- The final three attachments are independently fully downloaded and SHA256 checked before publication; public downloads are rechecked after publication.

## Scope and remaining limits

Secondary-structure per-sequence predictions (32 GB) remain index + SHA256 only; the original replay table has 21 such unshipped cells. The server-wide 162 test / 133 existing validation counts include secondary-structure records and are not public file counts. There are 29 documented validation files absent at source (27 chemical-reactivity, two ncRNA).

Student checkpoints remain in the fixed v1.0.0 archive: best_val step 8,000 (`stage4b_paper_eval.pt`, paper evaluation) and latest step 20,000 (`stage4b_release.pt`). Existing student/MultiMolecule weight assets remain AGPL-3.0, PlantRNA-FM remains MIT, and the original derived-data asset remains MIT. NT-v2-500m and ModernGENA-base weights and the 8.8 TiB cache are not mirrored here.

Unverified DOI/license/snapshot fields remain explicitly marked. Author-created outputs/docs do not relicense third-party inputs or reference labels. Cache/raw-prediction retention and transfer policy remains an author-review draft. No DOI is invented.
