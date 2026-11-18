# Local v1.0.1 integration record

The actual GitHub v1.0.0 and main baseline was 4ba39846482c65cdad64a501643b1609cdb45414, not the server's eb809c1 tree. The screening-file baseline matched the server's old SHA256, so the reviewed overlay was merged file by file while preserving the prior student-layer and table-reproduction fixes.

Local review additionally fixed loader-level filter-before-limit ordering, visible forward-error reporting, portable re-ranking paths and prediction-index validation. The server's two-hunk patch is not the final local code. The rebuilt patch and CODE_CHANGES.json record the final reviewed files; SERVER_CODE_CHANGES.json preserves the original server claims.

Do not apply this overlay to an unknown baseline. Do not replace v1.0.0 assets or tags. Do not include __pycache__, .pyc, private audit material, prediction tensors or model weights in a normal source commit. Server rev2 resolves the prediction collisions, adds gated per-seed replay and passes all 241 file hashes plus 408 metric checks. The local merge retains its additional filter-order, logging and index-validation fixes.
