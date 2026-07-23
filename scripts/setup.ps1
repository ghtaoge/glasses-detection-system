$ErrorActionPreference = "Stop"
$repoRoot = Split-Path -Parent $PSScriptRoot
Set-Location $repoRoot

$python = Get-Command py -ErrorAction SilentlyContinue
if (-not $python) { throw "Python 3.11-3.13 is required." }
if (-not (Test-Path ".venv\Scripts\python.exe")) { py -m venv .venv }

& ".venv\Scripts\python.exe" -m pip install -e ".[dev,ml]"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Push-Location frontend
try { npm ci; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } } finally { Pop-Location }

if (-not (Test-Path ".env")) { Copy-Item -LiteralPath ".env.example" -Destination ".env" }
New-Item -ItemType Directory -Force -Path "data" | Out-Null
& ".venv\Scripts\alembic.exe" -c backend/alembic.ini upgrade head
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "Setup complete. Install a matching CUDA PyTorch build separately when needed."
Write-Host "Download yolo26n before training and YuNet before automatic annotation."
