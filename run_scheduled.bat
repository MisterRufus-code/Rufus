@echo off
REM run_scheduled.bat - the autonomous daily run, invoked by Windows Task Scheduler.
REM Differences from run.bat: no pause (a paused cmd hangs a scheduled task), and
REM no --skip-upload — this IS the product run. Safety comes from the pipeline's
REM own gates: script score >= 8/10, QC pass required, uploads default PRIVATE.

REM UTF-8 console — same fix as run.bat (mojibake in console + tee'd log).
chcp 65001 >nul
set PYTHONIOENCODING=utf-8
REM PYTHONUTF8: open()/read_text() default to the ANSI code page (cp1255 on a
REM Hebrew-locale box) — see run.bat for the em-dash corruption this prevents.
set PYTHONUTF8=1

cd /d "%~dp0"

REM Computed once, upfront, outside any conditional block — setting a variable
REM and reading it back with %var% inside the SAME parenthesized if-block is a
REM classic batch pitfall (the whole block is %-expanded once at parse time,
REM before it runs line-by-line, so a value set mid-block isn't seen later in
REM that same block without setlocal enabledelayedexpansion + !var! syntax).
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set TODAY=%%d

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

set RUFUS_GPU=1
set RUFUS_VIDEO_SOURCE=comfy
REM Stills only, meanwhile — no Hunyuan/WAN/LTX/SVD motion pass on any beat.
REM One switch instead of setting all four motion engines to 0 individually;
REM remove this line (or set it to 0) once motion is wanted back on the
REM automated runs. See scripts/svd_client.py's RUFUS_STILLS_ONLY docs.
set RUFUS_STILLS_ONLY=1
REM Uncomment for the Kokoro voice (needs the Docker container on :8880):
REM set RUFUS_TTS=kokoro_api

REM Push notification on FAILURE only (no news = good news, avoids daily spam).
REM Zero signup: pick any unguessable topic name below, then install the free
REM "ntfy" app (iOS/Android) and subscribe to that same topic name. Leave blank
REM to disable — everything below degrades to a no-op silently if unset.
set RUFUS_NTFY_TOPIC=

REM Feedback loop: pull yesterday's YouTube Analytics and refold "what worked"
REM into learnings.json BEFORE today's script gets written, so script_writer.py
REM can actually use it (was built but never scheduled — dormant until now).
REM Both are non-fatal: a failure here only prints a warning, never blocks the
REM main run below.
echo Updating analytics + feedback learnings...
"%PY%" scripts\analytics_fetcher.py
"%PY%" scripts\feedback_analyzer.py

REM --scheduled: pick today's niche from the schedule (rotates when the
REM schedule has more than one niche; identical to before with a single one).
REM Extra args from the task can still append via %*.
"%PY%" scripts\main.py --scheduled %*
set RUFUS_EXIT=%ERRORLEVEL%

REM KPI digest into the same daily log (report.py is a real owner-facing
REM tool — uploaded/held/failed, watch%% by niche — but nothing ran it
REM automatically until now). Non-fatal like the analytics step above.
"%PY%" scripts\report.py --weeks 1

if not "%RUFUS_EXIT%"=="0" (
    echo Rufus run FAILED - exit code %RUFUS_EXIT%
    if not "%RUFUS_NTFY_TOPIC%"=="" (
        curl -s -d "Rufus daily run FAILED (exit %RUFUS_EXIT%) - check logs\rufus_%TODAY%.log" "ntfy.sh/%RUFUS_NTFY_TOPIC%" >nul 2>&1
    )
)

REM Success digest is OPT-IN (RUFUS_NTFY_DAILY=1): default stays no-news-is-
REM good-news so the phone only buzzes on failure.
if "%RUFUS_EXIT%"=="0" if "%RUFUS_NTFY_DAILY%"=="1" if not "%RUFUS_NTFY_TOPIC%"=="" (
    curl -s -d "Rufus daily run OK - video rendered, see logs\rufus_%TODAY%.log for the KPI digest" "ntfy.sh/%RUFUS_NTFY_TOPIC%" >nul 2>&1
)

exit /b %RUFUS_EXIT%
