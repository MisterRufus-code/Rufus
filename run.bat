@echo off
REM run.bat — daily Rufus run on Windows 11 + RTX 3090.
REM Defaults to ComfyUI/FLUX images + GPU Whisper/NVENC. Edit the env lines to taste.

REM UTF-8 console: without this, em-dashes and quotes print as mojibake
REM (e.g. "ג€”") on Hebrew-locale Windows. Cosmetic in the console but the
REM same misreads can corrupt the tee'd log file.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

REM --- engine selection -------------------------------------------------------
set RUFUS_GPU=1
set RUFUS_VIDEO_SOURCE=comfy
REM Uncomment to use the free Kokoro voice (needs the Docker container on :8880):
REM set RUFUS_TTS=kokoro_api

REM --- Auto-launch the dashboard if it isn't already running -------------------
REM Only here (not in run_scheduled.bat) — headless Task Scheduler runs
REM shouldn't pop a browser window. curl.exe ships with Windows 10/11.
set DASH_CHECK=
for /f %%c in ('curl -s -o nul -w "%%{http_code}" http://localhost:8765 2^>nul') do set DASH_CHECK=%%c
if not "%DASH_CHECK%"=="200" (
    echo Starting Rufus dashboard...
    start "Rufus Dashboard" /min python scripts\dashboard.py
    timeout /t 2 /nobreak >nul
)
start "" http://localhost:8765

REM --skip-upload renders without publishing. Drop it to upload (defaults private).
python scripts\main.py %*

pause
