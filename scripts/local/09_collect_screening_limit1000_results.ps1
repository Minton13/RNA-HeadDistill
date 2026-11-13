$ErrorActionPreference = "Stop"

$ProjectRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..")
Set-Location $ProjectRoot

$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$Python = $env:RHA_PYTHON; if (-not $Python) { $Python = "python" }  # set RHA_PYTHON to your env interpreter
$Root = "outputs\head_screening_limit1000"
$Atlas = Join-Path $Root "all_head_atlas.csv"
$Heatmaps = Join-Path $Root "heatmaps"
$TeacherBanks = "outputs\teacher_banks_limit1000"

& $Python -m rna_head_alignment.cli collect-head-screening `
  --input "$Root/*/*/head_atlas.csv" `
  --output $Atlas
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python -m rna_head_alignment.cli plot-head-heatmaps `
  --atlas $Atlas `
  --tasks configs\tasks.yaml `
  --output-dir $Heatmaps `
  --vmin 0 `
  --vmax 1 `
  --top-k 100
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

New-Item -ItemType Directory -Force -Path $TeacherBanks | Out-Null
& $Python -m rna_head_alignment.cli propose-teacher-heads `
  --atlas $Atlas `
  --config configs\screening.yaml `
  --output "$TeacherBanks\teacher_head_candidates.csv" `
  --yaml "$TeacherBanks\teacher_bank_template.yaml"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Collected limit=1000 outputs under $Root"
