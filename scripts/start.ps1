$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

& ".venv\Scripts\alembic.exe" -c backend/alembic.ini upgrade head
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$backend = Start-Process -FilePath ".venv\Scripts\python.exe" -ArgumentList @(
  "-m", "uvicorn", "app.main:app", "--app-dir", "backend",
  "--host", "127.0.0.1", "--port", "8000", "--workers", "1"
) -PassThru -WindowStyle Hidden
$frontend = Start-Process -FilePath "npm.cmd" -ArgumentList @(
  "--prefix", "frontend", "run", "dev", "--", "--host", "127.0.0.1"
) -PassThru -WindowStyle Hidden

Write-Host "Application started: http://127.0.0.1:5173"
Write-Host "Press Ctrl+C to stop both services."
try {
  while (-not $backend.HasExited -and -not $frontend.HasExited) { Start-Sleep -Seconds 1 }
} finally {
  foreach ($process in @($backend, $frontend)) {
    if ($process -and -not $process.HasExited) { Stop-Process -Id $process.Id }
  }
}
