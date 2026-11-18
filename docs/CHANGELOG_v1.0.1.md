# v1.0.1 release candidate

Supplements v1.0.0; no old tag, asset, checkpoint or frozen figure is changed.

## Fixed

- CUDA/gradient-carrying attention conversion uses detach → CPU → NumPy.
- Forward failures are logged instead of silently swallowed.
- Explicit split filtering precedes the sample cap, including for streamed pairwise JSONL input; legacy sampling remains the default.
- Missing required replay files fail instead of being treated as intentionally unshipped secondary-structure cells.
- Train-only re-ranking accepts public package paths instead of private monorepo paths.

## Added

- Screening regressions for CPU, CUDA, filtering order, legacy sampling and visible errors.
- Prediction archive/index verification with a regression reproducing the P4g/A8-P4g overwrite.
- Source Data workbook (33 sheets / 8,117 rows), CSVs, 22-row provenance and checkpoint/metric/license metadata.

## Release gate

The transferred prediction archive fails 27 P4g indexed hashes because P4g and A8-P4g paths collide. Full data release is held pending a corrected server package. Passing metric replay is limited to its 105 cells / 273 checks and does not certify those colliding configurations. See README_REPRODUCTION.md.
