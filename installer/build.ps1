# Build dist\LottoML (PyInstaller, CPU torch) and dist\LottoML-setup-*-cpu.exe (Inno Setup).
# Usage (from repo root): powershell -ExecutionPolicy Bypass -File installer\build.ps1
param(
    [string]$Python = "$env:USERPROFILE\lotto-build-venv\Scripts\python.exe",
    [string]$Iscc = "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe"
)
$ErrorActionPreference = "Stop"
$root = Split-Path $PSScriptRoot -Parent
Set-Location $root

& $Python -c "import torch, sys; sys.exit(0 if not torch.version.cuda else 'CUDA torch found: use the CPU build venv')"
if ($LASTEXITCODE -ne 0) { throw "build venv check failed" }

& $Python -m PyInstaller --noconfirm --clean --distpath dist --workpath build installer\lotto.spec
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed" }

& $Iscc installer\installer.iss
if ($LASTEXITCODE -ne 0) { throw "Inno Setup failed" }
Get-ChildItem dist\*.exe | Select-Object Name, @{n="MB"; e={[math]::Round($_.Length / 1MB)}}
