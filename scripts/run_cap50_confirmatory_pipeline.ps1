param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepositoryRoot
$env:TEMP = Join-Path $RepositoryRoot "results\runtime_tmp"
$env:TMP = $env:TEMP

$Runs = @(
    @{ Config = "config\operational_benchmark_owlv2_control_v1.json"; Root = "results\operational_benchmark_v1" },
    @{ Config = "config\operational_transfer_refcocoplus_owlv2_control_v1.json"; Root = "results\operational_transfer_refcocoplus_v1" },
    @{ Config = "config\operational_transfer_refl4_owlv2_control_v1.json"; Root = "results\operational_transfer_refl4_v1" }
)
$Models = @("groundingdino", "owlv2", "yoloworld")

foreach ($Run in $Runs) {
    foreach ($Model in $Models) {
        & $Python scripts\run_cap50_confirmatory.py `
            --config $Run.Config `
            --result-root $Run.Root `
            --model $Model `
            --resume
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

& $Python scripts\summarise_cap50_confirmatory.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "Cap-50 confirmatory pipeline completed successfully."
