# setup_windows.ps1 — one-time Rufus setup for Windows 11 + RTX 3090.
# Run from the repo root in PowerShell:   .\setup_windows.ps1
#
# It creates a venv, installs deps, checks ffmpeg, prints the Docker/ComfyUI
# commands you still need, then runs the health check.

$ErrorActionPreference = "Stop"
Write-Host "=== Rufus Windows setup ===" -ForegroundColor Cyan

# 1. Python venv
if (-not (Test-Path ".\.venv")) {
    Write-Host "[1/4] Creating virtual environment (.venv)..."
    python -m venv .venv
} else {
    Write-Host "[1/4] .venv already exists — reusing."
}
.\.venv\Scripts\Activate.ps1

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
    Write-Host "      Download a build from https://www.gyan.dev/ffmpeg/builds/ ,"
    Write-Host "      unzip it, and add its \bin folder to your PATH, then reopen PowerShell."
}

# 4. External services you run yourself (GPU stack)
Write-Host "[4/4] External services (start these before a real run):" -ForegroundColor Cyan
Write-Host @"
  ComfyUI (images, FLUX.1-dev on the 3090):
    Launch ComfyUI with:  --listen   (default port 8188)
    Put flux1-dev-fp8.safetensors in ComfyUI\models\checkpoints\
    Then set:  `$env:RUFUS_VIDEO_SOURCE='comfy'

  Kokoro-FastAPI (free natural voice, optional):
    docker run -d -p 8880:8880 ghcr.io/remsky/kokoro-fastapi-cpu:v0.2.2
    Then set:  `$env:RUFUS_TTS='kokoro_api'

  GPU acceleration for Whisper + NVENC encode:
    `$env:RUFUS_GPU='1'
"@

Write-Host "Running health check..." -ForegroundColor Cyan
python scripts\health_check.py

Write-Host "`nSetup done. Daily run:  .\run.bat" -ForegroundColor Green
