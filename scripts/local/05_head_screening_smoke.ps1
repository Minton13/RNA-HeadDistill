$ErrorActionPreference = "Stop"

$env:PYTHONPATH = "$PWD\src"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"

$Python = if ($env:PYTHON) { $env:PYTHON } else { "python" }

function Invoke-Python {
  param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Python command failed with exit code ${LASTEXITCODE}: $Python $($Arguments -join ' ')"
  }
}

Invoke-Python -m rna_head_alignment.cli verify-data `
  --tasks configs/tasks.yaml `
  --output outputs/model_checks/data_check_smoke.csv `
  --limit 5

Invoke-Python -m rna_head_alignment.cli make-screening-jobs `
  --models configs/models.yaml `
  --tasks configs/tasks.yaml `
  --model-check outputs/model_checks/model_attention_check_rna_expanded.csv `
  --output outputs/head_screening/screening_jobs_smoke.csv

Invoke-Python -m rna_head_alignment.cli screen-heads `
  --model rnabert `
  --task secondary_structure_pair `
  --config configs/screening_local_smoke.yaml `
  --output outputs/head_screening/secondary_structure_pair/rnabert/head_atlas.csv `
  --limit 8 `
  --device auto

Invoke-Python -m rna_head_alignment.cli screen-heads `
  --model rnabert `
  --task chemical_reactivity_token `
  --config configs/screening_local_smoke.yaml `
  --output outputs/head_screening/chemical_reactivity_token/rnabert/head_atlas.csv `
  --limit 8 `
  --device auto

Invoke-Python -m rna_head_alignment.cli collect-head-screening `
  --input "outputs/head_screening/*/*/head_atlas.csv" `
  --output outputs/head_screening/all_head_atlas.csv

Invoke-Python -m rna_head_alignment.cli plot-head-heatmaps `
  --atlas outputs/head_screening/all_head_atlas.csv `
  --tasks configs/tasks.yaml `
  --output-dir outputs/head_screening/heatmaps `
  --vmin 0 `
  --vmax 1
