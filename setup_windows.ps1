# setup_windows.ps1 - one-time Rufus setup for Windows 11 + RTX 3090.
# Run from the repo root in PowerShell:   .\setup_windows.ps1
#
# Creates a venv, installs deps, checks ffmpeg, installs the optional
# Remotion renderer if Node.js is present, prints the ComfyUI/Kokoro
# commands you still need, then runs the health check.
# (Plain Write-Host lines only - no here-strings; Windows PowerShell 5.1
# fails to parse here-strings in files checked out with LF line endings.)

$ErrorActionPreference = "Stop"
Write-Host "=== Rufus Windows setup ===" -ForegroundColor Cyan

# 0. Prerequisites - fail with the exact install command instead of a cryptic error
$missing = @()
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    $missing += "winget install --id Python.Python.3.11 -e"
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    $missing += "winget install --id Git.Git -e"
}
if ($missing.Count -gt 0) {
    Write-Host "Missing prerequisites. Run these, then CLOSE and REOPEN PowerShell:" -ForegroundColor Red
    foreach ($m in $missing) { Write-Host "  $m" -ForegroundColor Yellow }
    Write-Host "(ffmpeg too, if you haven't:  winget install --id Gyan.FFmpeg -e)" -ForegroundColor Yellow
    exit 1
}

# 1. Python venv
if (-not (Test-Path ".\.venv")) {
    Write-Host "[1/5] Creating virtual environment (.venv)..."
    python -m venv .venv
} else {
    Write-Host "[1/5] .venv already exists - reusing."
}
. .\.venv\Scripts\Activate.ps1

# 2. Dependencies (core only — images come from ComfyUI, voice from Docker;
#    the heavy local-ML extras live in requirements-optional.txt)
Write-Host "[2/5] Installing Python dependencies (core)..."
python -m pip install --upgrade pip | Out-Null
python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "pip install failed — see the error above." -ForegroundColor Red
    exit 1
}
Write-Host "      Optional extras (in-process voice/music/images):  pip install -r requirements-optional.txt" -ForegroundColor DarkGray

# 3. ffmpeg check
Write-Host "[3/5] Checking ffmpeg..."
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Host "      ffmpeg found on PATH." -ForegroundColor Green
} else {
    Write-Host "      ffmpeg NOT on PATH." -ForegroundColor Yellow
    Write-Host "      Install it:  winget install --id Gyan.FFmpeg -e   (then reopen PowerShell)"
}

# 4. Remotion renderer (optional — ffmpeg is the default and needs nothing
#    extra; RUFUS_RENDERER=remotion switches to it, with automatic fallback
#    to ffmpeg on any failure, so skipping this is completely fine)
Write-Host "[4/5] Remotion renderer (optional)..."
if (Get-Command npm -ErrorAction SilentlyContinue) {
    if (Test-Path ".\remotion\node_modules") {
        Write-Host "      remotion\node_modules already installed - reusing." -ForegroundColor Green
    } else {
        Write-Host "      npm found - installing Remotion dependencies..."
        Push-Location .\remotion
        npm install
        $npmExit = $LASTEXITCODE
        Pop-Location
        if ($npmExit -ne 0) {
            Write-Host "      npm install failed - Remotion renderer won't be available (ffmpeg still works)." -ForegroundColor Yellow
        } else {
            Write-Host "      Remotion ready. Enable it with:" -ForegroundColor Green
            Write-Host '        $env:RUFUS_RENDERER = "remotion"'
        }
    }
} else {
    Write-Host "      Node.js not found - Remotion renderer skipped (ffmpeg still works)." -ForegroundColor Yellow
    Write-Host "      Want it later?  winget install --id OpenJS.NodeJS.LTS -e   then:  cd remotion; npm install"
}

# 5. External services you run yourself (GPU stack)
Write-Host "[5/5] External services (start these before a real run):" -ForegroundColor Cyan
Write-Host "  ComfyUI (images, Z-Image-Turbo on the 3090):"
Write-Host "    Launch ComfyUI with:  --listen   (default port 8188)"
Write-Host "    Export your Z-Image-Turbo workflow to config/stills_api.json (see scripts/comfy_client.py)"
Write-Host '    Then set:  $env:RUFUS_VIDEO_SOURCE = "comfy"'
Write-Host ""
Write-Host "  Kokoro-FastAPI (free natural voice, optional):"
Write-Host "    docker run -d -p 8880:8880 --gpus all ghcr.io/remsky/kokoro-fastapi-gpu:v0.2.2"
Write-Host '    Then set:  $env:RUFUS_TTS = "kokoro_api"'
Write-Host ""
Write-Host "  GPU acceleration for Whisper + NVENC encode:"
Write-Host '    $env:RUFUS_GPU = "1"'
Write-Host ""

Write-Host "Running health check..." -ForegroundColor Cyan
python scripts\health_check.py

Write-Host ""
Write-Host "Setup done. Daily run:  .\run.bat" -ForegroundColor Green
