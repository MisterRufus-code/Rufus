@echo off
REM run.bat — daily Rufus run on Windows 11 + RTX 3090.
REM Defaults to ComfyUI/FLUX images + GPU Whisper/NVENC. Edit the env lines to taste.

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

REM --- engine selection -------------------------------------------------------
set RUFUS_GPU=1
set RUFUS_VIDEO_SOURCE=comfy
REM Uncomment to use the free Kokoro voice (needs the Docker container on :8880):
REM set RUFUS_TTS=kokoro_api

REM --skip-upload renders without publishing. Drop it to upload (defaults private).
python scripts\main.py %*

pause
