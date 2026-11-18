# Local transfer verification — BLOCKED prediction package

All six top-level transfer checksums passed, including the three original assets. Every tar member was safely unpacked and matched its loose transfer copy. The server assets have not been modified.

The original prediction tarball has 214 unique files (120 test, 94 validation), not 162 test plus 133 validation unique public files. The 324-row index has 241 `present`, 54 `present_index_only`, and 29 `missing_on_disk` entries. Twenty-seven P4g/A8-P4g entries share destinations but have different SHA256 values. The supplied bytes match A8-P4g; 27 P4g entries mismatch. Index checks therefore report 27 hash mismatches plus 27 collision errors. These errors are additional to the disclosed index-only and missing-validation cases.

The 273 passing metric checks use the v1.0.0 per-seed table and do not cover the colliding P4g/A8-P4g configurations. Repackage both configurations under distinct paths, retain their original identities/checksums, and rerun both index verification and the corresponding published-metric checks before a full data release.
