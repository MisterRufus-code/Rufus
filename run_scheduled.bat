@echo off
REM run_scheduled.bat - the autonomous daily run, invoked by Windows Task Scheduler.
REM Differences from run.bat: no pause (a paused cmd hangs a scheduled task), and
REM no --skip-upload — this IS the product run. Safety comes from the pipeline's
REM own gates: script score >= 8/10, QC pass required, uploads default PRIVATE.

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

set RUFUS_GPU=1
set RUFUS_VIDEO_SOURCE=comfy
REM Uncomment for the Kokoro voice (needs the Docker container on :8880):
REM set RUFUS_TTS=kokoro_api

python scripts\main.py %*
