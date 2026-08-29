param(
    [string]$Python = "D:\Anaconda3\envs\ml-gpu\python.exe",
    [string]$Config = "config\operational_transfer_refl4_v1.json",
    [string]$ResultRoot = "results\operational_transfer_refl4_v1"
)

$ErrorActionPreference = "Continue"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepositoryRoot

& $PSScriptRoot\run_frozen_operational_pipeline.ps1 `
    -Python $Python `
    -Config $Config `
    -ResultRoot $ResultRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts\analyse_refl4_transfer.py `
    --target-root $ResultRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $Python scripts\build_operational_artifact_manifest.py `
    --config $Config `
    --result-root $ResultRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Frozen Ref-L4 transfer pipeline completed successfully."
