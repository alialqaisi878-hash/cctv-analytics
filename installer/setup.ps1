# Prepares the Python environment for CCTV Analytics on Windows.
# Run by the installer after copying files; also available as "Repair" in the Start menu.
#  - finds Python 3.10-3.13, or installs Python 3.12 for the current user
#  - creates a private virtual environment in <AppDir>\.venv
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

function Get-PythonExe([string[]]$cmd) {
    # Returns the full path of a usable Python 3.10-3.13, or $null.
    # Usable = a normal installation with venv + ensurepip, not another app's private copy in AppData\Roaming.
    try {
        $exe = $cmd[0]
        $rest = @()
        if ($cmd.Length -gt 1) { $rest = $cmd[1..($cmd.Length - 1)] }
        $probe = "import sys, venv, ensurepip; v=sys.version_info[:2]; print(sys.executable if (3,10) <= v <= (3,13) else '')"
        $out = & $exe @rest -c $probe 2>$null
        if ($LASTEXITCODE -ne 0) { return $null }
        $path = "$out".Trim()
        if (-not $path) { return $null }
        if ($path.StartsWith($env:APPDATA, [System.StringComparison]::OrdinalIgnoreCase)) { return $null }
        return $path
    } catch { return $null }
}

function Find-Python {
    $candidates = @(
        @("py", "-3.12"), @("py", "-3.11"), @("py", "-3.13"), @("py", "-3.10"),
        @("$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"),
        @("$env:LOCALAPPDATA\Programs\Python\Python311\python.exe"),
        @("$env:ProgramFiles\Python312\python.exe"),
        @("$env:ProgramFiles\Python311\python.exe"),
        @("python")
    )
    foreach ($c in $candidates) {
        $p = Get-PythonExe $c
        if ($p) { return $p }
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
        Invoke-WebRequest -Uri $pyUrl -OutFile $pyInstaller -UseBasicParsing -ErrorAction Stop
        Step "Installing Python 3.12 (for this user only)"
        $p = Start-Process -FilePath $pyInstaller -Wait -PassThru -ErrorAction Stop -ArgumentList `
            "/quiet", "InstallAllUsers=0", "PrependPath=1", "Include_launcher=1", "Include_test=0", "Include_doc=0"
        if ($p.ExitCode -ne 0) { Fail "Python installer exited with code $($p.ExitCode)" }
        Remove-Item $pyInstaller -ErrorAction SilentlyContinue
        $py = Find-Python
        if (-not $py) { Fail "Python was installed but could not be found." }
    }
    Write-Host "Using Python: $py"

    $venv = Join-Path $AppDir ".venv"
    $vpy = Join-Path $venv "Scripts\python.exe"
    # rebuild an environment that was made from another app's private Python (older versions allowed that)
    $cfgFile = Join-Path $venv "pyvenv.cfg"
    if (Test-Path $cfgFile) {
        $venvHome = ((Get-Content $cfgFile | Where-Object { $_ -like "home*" }) -split "=", 2)[1]
        if ($venvHome -and $venvHome.Trim().StartsWith($env:APPDATA, [System.StringComparison]::OrdinalIgnoreCase)) {
            Step "Rebuilding the Python environment with a standard Python"
            Remove-Item -Recurse -Force $venv
        }
    }
    if (-not (Test-Path $vpy)) {
        Step "Creating private Python environment"
        & $py -m venv $venv
        if ($LASTEXITCODE -ne 0) { Fail "could not create the virtual environment" }
    }

    Step "Updating pip"
    & $vpy -m pip install --upgrade pip --disable-pip-version-check --timeout 60 --retries 5 -q
    if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed" }

    $hasNvidia = $false
    try { $null = Get-Command nvidia-smi -ErrorAction Stop; & nvidia-smi -L *> $null; $hasNvidia = ($LASTEXITCODE -eq 0) } catch {}
    if ($env:CCTV_FORCE_CPU) { $hasNvidia = $false }

    & $vpy -c "import torch" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        if ($hasNvidia) {
            Step "NVIDIA graphics card found - installing PyTorch with GPU support (large download)"
            & $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 --disable-pip-version-check --timeout 60 --retries 5
            if ($LASTEXITCODE -ne 0) { Write-Host "GPU build failed, falling back to CPU build" -ForegroundColor Yellow; $hasNvidia = $false }
        }
        if (-not $hasNvidia) {
            Step "Installing PyTorch (CPU version)"
            & $vpy -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu --disable-pip-version-check --timeout 60 --retries 5
            if ($LASTEXITCODE -ne 0) { Fail "PyTorch installation failed" }
        }
    }

    Step "Installing program libraries"
    & $vpy -m pip install -r (Join-Path $AppDir "requirements.txt") --disable-pip-version-check --timeout 60 --retries 5
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
