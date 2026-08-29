param(
    [string]$Python = "D:\Anaconda3\envs\ml-gpu\python.exe",
    [string]$ResultRoot = "results\operational_benchmark_v1",
    [string]$Config = "config\operational_benchmark_v1.json"
)

$ErrorActionPreference = "Continue"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}
$RepositoryRoot = Split-Path -Parent $PSScriptRoot
Set-Location -LiteralPath $RepositoryRoot

Write-Host "[1/8] Verifying the frozen implementation"
& $Python -m pytest tests -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/8] Running GroundingDINO (resume-safe)"
& $Python scripts\run_operational_benchmark.py `
    --config $Config `
    --model groundingdino `
    --output-root $ResultRoot `
    --resume
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/8] Running YOLO-World (resume-safe)"
& $Python scripts\run_operational_benchmark.py `
    --config $Config `
    --model yoloworld `
    --output-root $ResultRoot `
    --resume
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[4/8] Validating trace completeness and all frozen identities"
& $Python scripts\validate_operational_artifacts.py `
    --config $Config `
    --result-root $ResultRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[5/8] Generating statistics, figures, and the English report"
& $Python scripts\analyse_operational_benchmark.py `
    --config $Config `
    --result-root $ResultRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[6/8] Running the post-primary output-contract sensitivity audit"
& $Python scripts\analyse_contract_sensitivity.py `
    --config $Config `
    --result-root $ResultRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[7/8] Rendering auditable examples of each operational failure type"
& $Python scripts\render_operational_failure_examples.py `
    --config $Config `
    --result-root $ResultRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[8/8] Hashing the complete reproducibility artifact set"
& $Python scripts\build_operational_artifact_manifest.py `
    --config $Config `
    --result-root $ResultRoot
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Frozen operational benchmark completed successfully."
