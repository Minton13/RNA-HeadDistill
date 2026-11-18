"""Missing required replay files must not silently pass as index-only data."""
import json
from pathlib import Path
import runpy
import sys

import pandas as pd
import pytest


@pytest.mark.parametrize("gated", [False, True])
def test_replay_missing_required_file_fails(tmp_path, monkeypatch, gated):
    main = tmp_path / "points.csv"
    task = "secondary_structure_pair" if gated else "splice_donor_seqcls"
    pd.DataFrame([dict(task=task, mode="vanilla_hidden", seed=17)]).to_csv(main, index=False)
    out = tmp_path / "report"
    arguments = ["replay", "--pred-root", str(tmp_path / "predictions"),
                 "--points-csv", str(main), "--out", str(out)]
    if gated:
        extra = tmp_path / "gated.csv"
        pd.DataFrame([dict(task="splice_donor_seqcls", mode_or_line="A8_P4g", seed=17,
                           metric="auroc", value=0.8)]).to_csv(extra, index=False)
        arguments += ["--gated-points-csv", str(extra)]
    monkeypatch.setattr(sys, "argv", arguments)
    script = Path(__file__).resolve().parents[1] / "analysis/replay_downstream_metrics.py"
    with pytest.raises(SystemExit) as result:
        runpy.run_path(str(script), run_name="__main__")
    assert result.value.code == 2
    summary = json.loads(out.with_suffix(".json").read_text())
    assert summary["errors"] == 1
    assert summary["not_packaged_cells"] == int(gated)
