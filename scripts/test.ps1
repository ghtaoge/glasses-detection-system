param([switch]$RealModel)
$ErrorActionPreference = "Continue"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$commands = @(
  @(".venv\Scripts\python.exe", "-m", "ruff", "check", "backend"),
  @(".venv\Scripts\python.exe", "-m", "pytest", "-q"),
  @("npm.cmd", "--prefix", "frontend", "test"),
  @("npm.cmd", "--prefix", "frontend", "run", "build")
)
foreach ($command in $commands) {
  & $command[0] $command[1..($command.Length - 1)]
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
if ($RealModel) {
  & ".venv\Scripts\python.exe" -c "import onnxruntime, ultralytics; print('ML runtime available')"
  if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
Write-Host "Deterministic release checks passed."
