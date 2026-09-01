param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepositoryRoot
$RuntimeTemp = Join-Path $RepositoryRoot "results\runtime_tmp"
$PytestBaseTemp = Join-Path $RuntimeTemp "pytest_owlv2_$PID"
New-Item -ItemType Directory -Force -Path $RuntimeTemp | Out-Null
$env:TEMP = $RuntimeTemp
$env:TMP = $RuntimeTemp
$Models = @("groundingdino", "yoloworld", "owlv2")
$Runs = @(
    @{
        Config = "config\operational_benchmark_owlv2_control_v1.json"
        Root = "results\operational_benchmark_v1"
    },
    @{
        Config = "config\operational_transfer_refcocoplus_owlv2_control_v1.json"
        Root = "results\operational_transfer_refcocoplus_v1"
    },
    @{
        Config = "config\operational_transfer_refl4_owlv2_control_v1.json"
        Root = "results\operational_transfer_refl4_v1"
    }
)

Write-Host "[1/7] Verifying code and frozen-config identity"
& $Python -m pytest tests -q -p no:cacheprovider --basetemp $PytestBaseTemp
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python scripts\assert_owlv2_control_configs.py
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python scripts\cache_owlv2_checkpoint.py --local-only
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
Write-Host "[2/7] Running OWLv2 on all registered manifests (resume-safe)"
foreach ($Run in $Runs) {
    & $Python scripts\run_operational_benchmark.py `
        --config $Run.Config `
        --model owlv2 `
        --output-root $Run.Root `
        --resume
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "[3/7] Validating all three models under the shared probes"
foreach ($Run in $Runs) {
    & $Python scripts\validate_operational_artifacts.py `
        --config $Run.Config `
        --result-root $Run.Root
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "[4/7] Rebuilding three-model primary and sensitivity analyses"
foreach ($Run in $Runs) {
    & $Python scripts\analyse_operational_benchmark.py `
        --config $Run.Config `
        --result-root $Run.Root
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $Python scripts\analyse_contract_sensitivity.py `
        --config $Run.Config `
        --result-root $Run.Root
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "[5/7] Rebuilding transfer analyses"
& $Python scripts\analyse_operational_transfer.py --models $Models
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& $Python scripts\analyse_refl4_transfer.py --models $Models
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[6/7] Rendering failure examples"
foreach ($Run in $Runs) {
    & $Python scripts\render_operational_failure_examples.py `
        --config $Run.Config `
        --result-root $Run.Root
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "[7/7] Hashing completed result artifacts"
foreach ($Run in $Runs) {
    & $Python scripts\build_operational_artifact_manifest.py `
        --config $Run.Config `
        --result-root $Run.Root
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "OWLv2 control extension completed successfully."
