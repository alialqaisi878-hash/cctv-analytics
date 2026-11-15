# Prepares the Python environment for CCTV Analytics on Windows.
# Run by the installer after copying files; also available as "Repair" in the Start menu.
#  - finds Python 3.10-3.13, or installs Python 3.12 for the current user
#  - creates a private virtual environment in <AppDir>\.venv
#  - installs PyTorch (GPU build if an NVIDIA card is found, otherwise CPU) and the app requirements
#  - creates config.yaml from the example (never overwrites an existing one)
param(
    [string]$AppDir = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # makes Invoke-WebRequest much faster
$AppDir = (Resolve-Path $AppDir).Path
$Log = Join-Path $AppDir "install.log"
Start-Transcript -Path $Log -Force | Out-Null
$Host.UI.RawUI.WindowTitle = "CCTV Analytics - Setup"

function Step($msg) { Write-Host ""; Write-Host "==> $msg" -ForegroundColor Cyan }

function Fail($msg) {
    Write-Host ""
    Write-Host "SETUP FAILED: $msg" -ForegroundColor Red
    Write-Host "Log file: $Log"
    Write-Host "Check your internet connection and run 'Repair CCTV Analytics' from the Start menu."
    Stop-Transcript | Out-Null
    if (-not $env:CI) { Read-Host "Press Enter to close" | Out-Null }
    exit 1
}

function Test-PythonCmd([string[]]$cmd) {
    try {
        $exe = $cmd[0]
        $rest = @()
        if ($cmd.Length -gt 1) { $rest = $cmd[1..($cmd.Length - 1)] }
        $out = & $exe @rest -c "import sys; print(sys.version_info[:2] >= (3,10) and sys.version_info[:2] <= (3,13))" 2>$null
        return ($LASTEXITCODE -eq 0 -and "$out".Trim() -eq "True")
    } catch { return $false }
}

function Find-Python {
    $candidates = @(
        @("py", "-3.12"), @("py", "-3.11"), @("py", "-3.13"), @("py", "-3.10"),
        @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"),
        @("python")
    )
    foreach ($c in $candidates) {
        if (Test-PythonCmd $c) {
            $exe = $c[0]
            $rest = @()
            if ($c.Length -gt 1) { $rest = $c[1..($c.Length - 1)] }
            return (& $exe @rest -c "import sys; print(sys.executable)").Trim()
        }
    }
    return $null
}

try {
    Write-Host "CCTV Analytics setup. This downloads about 300-600 MB and can take 5-15 minutes." -ForegroundColor Green
    Write-Host "Install folder: $AppDir"

    Step "Looking for Python 3.10-3.13"
    $py = Find-Python
    if (-not $py) {
        Step "Python not found - downloading Python 3.12"
        $pyUrl = "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe"
        $pyInstaller = Join-Path $env:TEMP "python-3.12-installer.exe"
        Invoke-WebRequest -Uri $pyUrl -OutFile $pyInstaller -UseBasicParsing
        Step "Installing Python 3.12 (for this user only)"
        $p = Start-Process -FilePath $pyInstaller -Wait -PassThru -ArgumentList `
            "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_launcher=1", "Include_test=0", "Include_doc=0"
        if ($p.ExitCode -ne 0) { Fail "Python installer exited with code $($p.ExitCode)" }
        Remove-Item $pyInstaller -ErrorAction SilentlyContinue
        $py = Find-Python
        if (-not $py) { Fail "Python was installed but could not be found." }
    }
    Write-Host "Using Python: $py"

    $venv = Join-Path $AppDir ".venv"
    $vpy = Join-Path $venv "Scripts\python.exe"
    if (-not (Test-Path $vpy)) {
        Step "Creating private Python environment"
        & $py -m venv $venv
        if ($LASTEXITCODE -ne 0) { Fail "could not create the virtual environment" }
    }

    Step "Updating pip"
    & $vpy -m pip install --upgrade pip --disable-pip-version-check -q
    if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed" }

    $hasNvidia = $false
    try { $null = Get-Command nvidia-smi -ErrorAction Stop; & nvidia-smi -L *> $null; $hasNvidia = ($LASTEXITCODE -eq 0) } catch {}
    if ($env:CCTV_FORCE_CPU) { $hasNvidia = $false }

    & $vpy -c "import torch" 2>$null
    if ($LASTEXITCODE -ne 0) {
        if ($hasNvidia) {
            Step "NVIDIA graphics card found - installing PyTorch with GPU support (large download)"
            & $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --disable-pip-version-check
            if ($LASTEXITCODE -ne 0) { Write-Host "GPU build failed, falling back to CPU build" -ForegroundColor Yellow; $hasNvidia = $false }
        }
        if (-not $hasNvidia) {
            Step "Installing PyTorch (CPU version)"
            & $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check
            if ($LASTEXITCODE -ne 0) { Fail "PyTorch installation failed" }
        }
    }

    Step "Installing program libraries"
    & $vpy -m pip install -r (Join-Path $AppDir "requirements.txt") --disable-pip-version-check
    if ($LASTEXITCODE -ne 0) { Fail "installing requirements failed" }

    $cfg = Join-Path $AppDir "config.yaml"
    if (-not (Test-Path $cfg)) {
        Copy-Item (Join-Path $AppDir "config.example.yaml") $cfg
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
    Stop-Transcript | Out-Null
    exit 0
} catch {
    Fail $_.Exception.Message
}
