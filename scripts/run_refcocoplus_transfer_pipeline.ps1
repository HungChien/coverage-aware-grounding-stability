param(
    [string]$Python = "python",
    [string]$Config = "config\operational_transfer_refcocoplus_v1.json",
    [string]$ResultRoot = "results\operational_transfer_refcocoplus_v1"
)

$ErrorActionPreference = "Stop"
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepositoryRoot

& $PSScriptRoot\run_frozen_operational_pipeline.ps1 `
    -Python $Python `
    -Config $Config `
    -ResultRoot $ResultRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts\analyse_operational_transfer.py `
    --target-root $ResultRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts\build_operational_artifact_manifest.py `
    --config $Config `
    --result-root $ResultRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Frozen RefCOCO+ transfer pipeline completed successfully."
