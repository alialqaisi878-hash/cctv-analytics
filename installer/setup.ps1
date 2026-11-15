# Prepares the Python environment for CCTV Analytics on Windows.
# Run by the installer after copying files; also available as "Repair" in the Start menu.
#  - gets a private Python 3.12 via uv (in <AppDir>\python) and a virtual environment in <AppDir>\.venv
#  - installs PyTorch (GPU build if an NVIDIA card is found, otherwise CPU) and the app requirements
#  - creates config.yaml from the example (never overwrites an existing one)
param(
    [string]$AppDir = (Split-Path -Parent $PSScriptRoot)
)

# "Continue", not "Stop": Windows PowerShell 5.1 turns any stderr output of a native program
# (python, pip) into a terminating error under "Stop". Native steps are checked via $LASTEXITCODE,
# and cmdlets that must succeed pass -ErrorAction Stop explicitly.
$ErrorActionPreference = "Continue"
$ProgressPreference = "SilentlyContinue"   # makes Invoke-WebRequest much faster
$AppDir = (Resolve-Path $AppDir).Path
$Log = Join-Path $AppDir "install.log"
$Host.UI.RawUI.WindowTitle = "CCTV Analytics - Setup"

# Only one setup at a time (the installer and the desktop shortcut can both start it).
try {
    $script:LockHandle = [System.IO.File]::Open((Join-Path $AppDir "setup.lock"), "OpenOrCreate", "ReadWrite", "None")
} catch {
    Write-Host "Setup is already running in another window. Please wait for it to finish." -ForegroundColor Yellow
    if (-not $env:CI) { Read-Host "Press Enter to close" | Out-Null }
    exit 0
}
try { Start-Transcript -Path $Log -Force -ErrorAction Stop | Out-Null } catch { }

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }

function Fail($msg) {
    Write-Host ""
    Write-Host "SETUP FAILED: $msg" -ForegroundColor Red
    Write-Host "Log file: $Log"
    Write-Host "Check your internet connection and run 'Repair CCTV Analytics' from the Start menu."
    try { Stop-Transcript | Out-Null } catch { }
    if (-not $env:CI) { Read-Host "Press Enter to close" | Out-Null }
    exit 1
}

try {
    Write-Host "CCTV Analytics setup. This downloads about 300-600 MB and can take 5-15 minutes." -ForegroundColor Green
    Write-Host "Install folder: $AppDir"

    # Python is provided by uv (https://github.com/astral-sh/uv) as a private copy inside the app
    # folder, so nothing depends on (or interferes with) other Pythons installed on this PC.
    $tools = Join-Path $AppDir "tools"
    $uv = Join-Path $tools "uv.exe"
    if (-not (Test-Path $uv)) {
        Step "Downloading uv (Python manager)"
        New-Item -ItemType Directory -Force $tools | Out-Null
        $uvZip = Join-Path $env:TEMP "uv-windows.zip"
        Invoke-WebRequest -Uri "https://github.com/astral-sh/uv/releases/latest/download/uv-x86_64-pc-windows-msvc.zip" `
            -OutFile $uvZip -UseBasicParsing -ErrorAction Stop
        Expand-Archive -Path $uvZip -DestinationPath $tools -Force -ErrorAction Stop
        Remove-Item $uvZip -ErrorAction SilentlyContinue
        if (-not (Test-Path $uv)) { Fail "uv.exe missing after download" }
    }
    $env:UV_PYTHON_INSTALL_DIR = Join-Path $AppDir "python"
    $env:UV_LINK_MODE = "copy"          # the uv cache and the app may be on different drives
    $env:UV_HTTP_TIMEOUT = "180"

    $venv = Join-Path $AppDir ".venv"
    $vpy = Join-Path $venv "Scripts\python.exe"
    # environments made by older versions of this setup (with whatever Python was on the PC) are rebuilt
    $cfgFile = Join-Path $venv "pyvenv.cfg"
    if ((Test-Path $venv) -and -not ((Test-Path $cfgFile) -and (Select-String -Path $cfgFile -Pattern "^uv\s*=" -Quiet))) {
        Step "Removing the old Python environment"
        Remove-Item -Recurse -Force $venv -ErrorAction SilentlyContinue
        if (Test-Path $venv) { Fail "could not delete $venv - close all CCTV Analytics windows and try again" }
    }

    Step "Preparing private Python 3.12"
    & $uv python install 3.12
    if ($LASTEXITCODE -ne 0) { Fail "could not download Python 3.12" }

    if (-not (Test-Path $vpy)) {
        Step "Creating private Python environment"
        & $uv venv $venv --python 3.12 --python-preference only-managed --seed
        if ($LASTEXITCODE -ne 0) { Fail "could not create the Python environment" }
    }

    $hasNvidia = $false
    try { $null = Get-Command nvidia-smi -ErrorAction Stop; & nvidia-smi -L *> $null; $hasNvidia = ($LASTEXITCODE -eq 0) } catch {}
    if ($env:CCTV_FORCE_CPU) { $hasNvidia = $false }

    & $vpy -c "import torch" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        if ($hasNvidia) {
            Step "NVIDIA graphics card found - installing PyTorch with GPU support (large download)"
            & $uv pip install --python $vpy torch torchvision --index-url https://download.pytorch.org/whl/cu126
            if ($LASTEXITCODE -ne 0) { Write-Host "GPU build failed, falling back to CPU build" -ForegroundColor Yellow; $hasNvidia = $false }
        }
        if (-not $hasNvidia) {
            Step "Installing PyTorch (CPU version)"
            & $uv pip install --python $vpy torch torchvision --index-url https://download.pytorch.org/whl/cpu
            if ($LASTEXITCODE -ne 0) { Fail "PyTorch installation failed" }
        }
    }

    Step "Installing program libraries"
    & $uv pip install --python $vpy -r (Join-Path $AppDir "requirements.txt")
    if ($LASTEXITCODE -ne 0) { Fail "installing requirements failed" }

    $cfg = Join-Path $AppDir "config.yaml"
    if (-not (Test-Path $cfg)) {
        Copy-Item (Join-Path $AppDir "config.example.yaml") $cfg -ErrorAction Stop
        Write-Host "Created config.yaml"
    }

    Step "Checking the installation"
    Push-Location $AppDir
    & $vpy -c "import cv2, ultralytics, fastapi, anthropic, app.server; print('OK')"
    $ok = $LASTEXITCODE
    Pop-Location
    if ($ok -ne 0) { Fail "self-check failed" }

    Write-Host ""
    Write-Host "Setup complete. Start 'CCTV Analytics' from the desktop or Start menu." -ForegroundColor Green
    try { Stop-Transcript | Out-Null } catch { }
    exit 0
} catch {
    Fail $_.Exception.Message
}
