$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_DATASETS_OFFLINE = "1"
$env:TOKENIZERS_PARALLELISM = "false"

$Python = $env:RHA_PYTHON; if (-not $Python) { $Python = "python" }  # set RHA_PYTHON to your env interpreter
$PlanDir = "outputs\head_screening\runtime_plans\limit_1000_safety110"
$Jobs = Join-Path $PlanDir "screening_jobs_limit1000_under40h.csv"
$Report = "outputs\head_screening_limit1000\screening_run_report_limit1000_under40h.csv"

if (-not (Test-Path $Jobs)) {
  & $Python scripts\local\07_plan_screening_runtime.py --target-limit 1000 --budget-hours 40 --safety-factor 1.10 --output-dir $PlanDir
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "Running complete limit=1000 screening jobs, expected under 40h with safety factor 1.10"
Write-Host "Jobs: $Jobs"
Write-Host "Report: $Report"

& $Python -m rna_head_alignment.cli run-screening-jobs `
  --jobs $Jobs `
  --config configs\screening.yaml `
  --models configs\models.yaml `
  --tasks configs\tasks.yaml `
  --report $Report `
  --limit 1000 `
  --device auto `
  --time-budget-hours 40

if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Completed. Next: collect atlas and regenerate heatmaps from outputs/head_screening_limit1000."
