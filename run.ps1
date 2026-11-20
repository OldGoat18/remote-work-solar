<#
.SYNOPSIS
  Single entry point for the wlpr reproduction package.
.PARAMETER Mode
  env      print env setup instructions
  public   download/rebuild public census, weather and REPOS inputs
  stdstats rebuild the LV load input from the staged Standard Statistics ZIP
  verify   run 00 (manifest integrity)
  frozen   reproduce all results from data/derived/snapshot (default; no raw needed)
  raw      regenerate all derived from data/raw + data/frozen, then reproduce
  figures  rebuild only the result + validation figures from existing derived
  test     run the pytest gate suite
  all      = raw
#>
param([string]$Mode = "frozen")

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"

function Run-Step($script) {
  Write-Host "==> $script" -ForegroundColor Cyan
  python $script
  if ($LASTEXITCODE -ne 0) { throw "step failed: $script" }
}

# canonical ordered pipeline (steps materialise as they are ported)
$BUILD = @(
  "pipeline/00_verify_inputs.py",
  "pipeline/01_owner_ledger.py",
  "pipeline/02_component_product.py",
  "pipeline/03_survey_anchor.py",
  "pipeline/04_baseline_p0.py",
  "pipeline/05_pv_inputs.py",
  "pipeline/06_adwfh_v2.py",
  "pipeline/07_adwfh_emit.py",
  "pipeline/08_netload.py",
  "pipeline/09_indices.py",
  "pipeline/10_curves_summary_bands.py",
  "pipeline/11_storage_physics.py",
  "pipeline/12_r3.py"
)
$PUBLIC = @("pipeline/00_rebuild_public_inputs.py")
$STDSTATS = @("pipeline/00_rebuild_stdstats.py")
$FIGURES = @(
  "pipeline/13_figures.py",
  "pipeline/14_mobility.py",
  "pipeline/15_validation_suite.py"
)

switch ($Mode) {
  "env"     { Write-Host "conda env create -f environment.yml ; conda activate WFH" }
  "public"  { $env:WLPR_RUN_FROM = "raw"; foreach ($s in $PUBLIC) { Run-Step $s } }
  "stdstats" { $env:WLPR_RUN_FROM = "raw"; foreach ($s in $STDSTATS) { Run-Step $s } }
  "verify"  { $env:WLPR_RUN_FROM = "frozen"; Run-Step "pipeline/00_verify_inputs.py" }
  "test"    { python -m pytest tests/ -q }
  "figures" { $env:WLPR_RUN_FROM = "frozen"; foreach ($s in $FIGURES) { if (Test-Path $s) { Run-Step $s } } }
  "frozen"  { $env:WLPR_RUN_FROM = "frozen"; foreach ($s in ($BUILD + $FIGURES)) { if (Test-Path $s) { Run-Step $s } } }
  { $_ -in @("raw","all") } { $env:WLPR_RUN_FROM = "raw"; foreach ($s in ($PUBLIC + $STDSTATS + $BUILD + $FIGURES)) { if (Test-Path $s) { Run-Step $s } } }
  default   { throw "unknown mode '$Mode' (env|public|stdstats|verify|frozen|raw|figures|test|all)" }
}
Write-Host "run.ps1 ($Mode): DONE" -ForegroundColor Green
