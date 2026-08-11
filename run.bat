@echo off
REM run.bat — daily Rufus run on Windows 11 + RTX 3090.
REM Defaults to ComfyUI/FLUX images + GPU Whisper/NVENC. Edit the env lines to taste.

REM UTF-8, in BOTH places it matters — they are different bugs.
REM   chcp + PYTHONIOENCODING : the CONSOLE and the tee'd log.
REM   PYTHONUTF8              : open() and Path.read_text(), which otherwise
REM                             use the ANSI code page. On this Hebrew-locale
REM                             box that is cp1255, and every config file here
REM                             is UTF-8, so an em-dash read out of
REM                             niches.json came back as "ג€”" — which is
REM                             exactly what a live run printed for a CTA that
REM                             then goes into the YouTube description. That
REM                             one is NOT cosmetic and chcp does not fix it;
REM                             the text is corrupt before it reaches stdout.
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

REM --- engine selection -------------------------------------------------------
set RUFUS_GPU=1
set RUFUS_VIDEO_SOURCE=comfy
REM Stills only, meanwhile — no Hunyuan/WAN/LTX/SVD motion pass on any beat.
REM Remove this line (or set it to 0) once motion is wanted back.
set RUFUS_STILLS_ONLY=1
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
