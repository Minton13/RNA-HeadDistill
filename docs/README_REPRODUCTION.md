# v1.0.1 reproducibility supplement (release candidate)

This supplement builds on the unchanged [v1.0.0 release](https://github.com/Minton13/RNA-HeadDistill/releases/tag/v1.0.0).
No model checkpoint, published measurement, or frozen Figure 1–4 was modified.
Plotting source is archived for provenance; no figure was regenerated during this release work.

## Code fixes

- Screening converts attention tensors with `.detach().cpu().numpy()`.
- Per-sample failures produce warning logs with model/task/sample identifiers and traceback.
- With `split_filter`, annotated JSONL split filtering occurs before the retained-sample limit. Without the option, existing prefix sampling is preserved.
- `post_filter_limit` caps the filtered pool; the train-only secondary configuration retains 100 valid train records.
- Prediction-index validation rejects missing files, SHA256 mismatches and configuration path collisions before metric replay.
- Re-ranking accepts `--atlas`, `--stored-candidates`, `--outdir` and optional `--per-model-root`; it no longer requires the private server monorepo layout.

## Public repository material

- `release-metadata/v1.0.1/`: 22-row provenance, checkpoint identities, license matrix, metric definitions and train-only evidence. Unverified DOI/license fields remain unverified.
- `source_data/v1.0.1/SourceData.xlsx`: 33 worksheets / 8,117 data rows; equivalent CSVs are in `source_data/v1.0.1/csv/`.
- `docs/DATA_ACCESS_AND_CACHE_POLICY_DRAFT.md`: author-review draft; it establishes no new retention or access commitment.

## Prediction-archive release gate: BLOCKED

The supplied server archive contains 214 unique prediction files: 120 test and 94 validation files across five tasks. The 324-row server index aliases 27 P4g and A8-P4g entries to the same paths. The actual bytes match A8-P4g; 27 P4g entries fail their indexed SHA256/size checks. These P4g bytes must be recovered and repackaged under distinct configuration directories before the prediction asset can be formally released. The original server index is preserved as evidence, not certified as a verified public archive.

The 273/273 passing metric checks cover 105 packaged cells from the v1.0.0 per-seed point table; they do not validate the colliding P4g/A8-P4g index entries or all nine configurations. The 21 secondary-structure cells in that replay table are deliberately index-only. No full secondary-structure per-sequence bootstrap reproduction is promised.

The server audit's global 162/162 test and 133/162 validation counts describe server index entries, not unique files in the transferred prediction archive. They must not be reported as public-download coverage.

## Local reproduction commands

Download and unpack the fixed v1.0.0 dataset:

```bash
curl -LO https://github.com/Minton13/RNA-HeadDistill/releases/download/v1.0.0/dataset-rna-head-distill_v1.0.0.tar.gz
tar -xzf dataset-rna-head-distill_v1.0.0.tar.gz
export RNA_HEAD_DISTILL_DATA="$PWD/dataset-rna-head-distill"
export PYTHONPATH="$PWD/src"
python -m pytest
python scripts/reproduction/reproduce_tables.py --outdir outputs/reproduction
python analysis/trainonly_secondary_rerank.py --outdir outputs/reproduction/trainonly
```

After obtaining a corrected prediction package, validate its index before replay:

```bash
python analysis/verify_prediction_archive.py --index release-metadata/v1.0.1/prediction_archive_index.csv --root PATH_TO_PAYLOAD --out index_check.json
python analysis/replay_downstream_metrics.py --pred-root PATH_TO_PAYLOAD/predictions --points-csv "$RNA_HEAD_DISTILL_DATA/metrics/downstream/downstream_recomputed_points.csv" --out replay_report
```

## Existing weight assets (unchanged)

- [Student archive](https://github.com/Minton13/RNA-HeadDistill/releases/download/v1.0.0/student-rinalmo-micro-head2head_v1.0.0.tar.gz): `stage4b_paper_eval.pt` = best validation at step 8,000 (paper evaluation); `stage4b_release.pt` = latest at step 20,000. AGPL-3.0.
- [MultiMolecule part 00](https://github.com/Minton13/RNA-HeadDistill/releases/download/v1.0.0/teacher-weights-multimolecule-agpl_v1.0.0.tar.gz.part-00), [part 01](https://github.com/Minton13/RNA-HeadDistill/releases/download/v1.0.0/teacher-weights-multimolecule-agpl_v1.0.0.tar.gz.part-01): seven mirrored snapshots, AGPL-3.0.
- [PlantRNA-FM](https://github.com/Minton13/RNA-HeadDistill/releases/download/v1.0.0/teacher-weight-plantrna-fm-mit_v1.0.0.tar.gz): MIT.
- NT-v2-500m and ModernGENA-base are not mirrored; use their indexed pinned upstream revisions subject to upstream access and licensing.
