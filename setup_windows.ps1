# setup_windows.ps1 - one-time Rufus setup for Windows 11 + RTX 3090.
# Run from the repo root in PowerShell:   .\setup_windows.ps1
#
# Creates a venv, installs deps, checks ffmpeg, prints the ComfyUI/Kokoro
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
    Write-Host "[1/4] Creating virtual environment (.venv)..."
    python -m venv .venv
} else {
    Write-Host "[1/4] .venv already exists - reusing."
}
. .\.venv\Scripts\Activate.ps1

# 2. Dependencies
Write-Host "[2/4] Installing Python dependencies..."
python -m pip install --upgrade pip | Out-Null
python -m pip install -r requirements.txt

# 3. ffmpeg check
Write-Host "[3/4] Checking ffmpeg..."
if (Get-Command ffmpeg -ErrorAction SilentlyContinue) {
    Write-Host "      ffmpeg found on PATH." -ForegroundColor Green
} else {
    Write-Host "      ffmpeg NOT on PATH." -ForegroundColor Yellow
    Write-Host "      Install it:  winget install --id Gyan.FFmpeg -e   (then reopen PowerShell)"
}

# 4. External services you run yourself (GPU stack)
Write-Host "[4/4] External services (start these before a real run):" -ForegroundColor Cyan
Write-Host "  ComfyUI (images, FLUX.1-dev on the 3090):"
Write-Host "    Launch ComfyUI with:  --listen   (default port 8188)"
Write-Host "    Put flux1-dev-fp8.safetensors in ComfyUI\models\checkpoints\"
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
