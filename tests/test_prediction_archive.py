import csv
import hashlib
from rna_head_alignment.io.prediction_archive import verify_prediction_archive


def _index(tmp_path, rows):
    path = tmp_path / "index.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_prediction_index_detects_configuration_overwrite(tmp_path):
    content = b"A8 predictions"
    (tmp_path / "pred.bin").write_bytes(content)
    first = dict(task="task", config="P4g", seed="17", split="test", status="present",
                 packaged_as="pred.bin", sha256=hashlib.sha256(b"P4g predictions").hexdigest(), bytes=len(content))
    second = {**first, "config": "A8-P4g", "sha256": hashlib.sha256(content).hexdigest()}
    result = verify_prediction_archive(_index(tmp_path, [first, second]), tmp_path)
    assert result["verified_entries"] == 1
    assert {error["error"] for error in result["errors"]} == {"sha256_or_size_mismatch", "configuration_path_collision"}


def test_prediction_index_accepts_verified_and_disclosed_missing(tmp_path):
    content = b"predictions"
    (tmp_path / "pred.bin").write_bytes(content)
    first = dict(task="task", config="P0", seed="17", split="test", status="present",
                 packaged_as="pred.bin", sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
    missing = {**first, "split": "val", "status": "missing_on_disk", "packaged_as": ""}
    result = verify_prediction_archive(_index(tmp_path, [first, missing]), tmp_path)
    assert result["verified_entries"] == 1
    assert result["error_count"] == 0
