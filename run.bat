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

REM Resolve the interpreter EXPLICITLY. Do not rely on activate.bat having put
REM the venv on PATH: the "if exist" guard below it is silent, so a missing or
REM non-taking activation (a scheduled task under another profile, a venv whose
REM pyvenv.cfg no longer resolves) left bare "python" meaning the SYSTEM
REM interpreter — which has no Flask and no torch. That is a launcher quietly
REM running the wrong Python, which is worse than one that refuses.
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo(
    echo ERROR: %CD%\%PY% not found.
    echo The virtualenv is missing or was moved - venvs are not relocatable on
    echo Windows. Recreate it:  python -m venv .venv
    echo Then:                  .venv\Scripts\pip install -r requirements.txt
    echo Refusing to fall back to the system Python, which lacks this project packages.
    exit /b 9009
)

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
    start "Rufus Dashboard" /min "%PY%" scripts\dashboard.py
    timeout /t 2 /nobreak >nul
)
start "" http://localhost:8765

REM --skip-upload renders without publishing. Drop it to upload (defaults private).
"%PY%" scripts\main.py %*

pause
