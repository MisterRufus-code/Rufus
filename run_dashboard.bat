@echo off
REM run_dashboard.bat - launcher for the always-on dashboard (serve.ps1 registers
REM this as an ONSTART scheduled task).
REM
REM Why a .bat instead of pointing schtasks straight at python.exe: a scheduled
REM task inherits NO working directory (it starts in C:\WINDOWS\system32) and
REM has NOWHERE to write stdout, so a dashboard that crashed on startup left no
REM trace at all - the exact situation this file exists to make debuggable.
REM Same pattern as run_scheduled.bat.

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
REM PYTHONUTF8: open()/read_text() default to the ANSI code page (cp1255 on a
REM Hebrew-locale box) — see run.bat for the em-dash corruption this prevents.
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

if not exist "logs" mkdir "logs"

REM The failure message goes to the LOG, not just the console. A scheduled task
REM has no console, which is the exact situation this file's header says it
REM exists to make debuggable — an error nobody can read is the same as no error.
if not exist "%PY%" (
    echo ==== dashboard FAILED TO START %DATE% %TIME% ==== >> logs\dashboard.log
    echo ERROR: "%CD%\%PY%" not found. >> logs\dashboard.log
    echo The virtualenv is missing or was moved - venvs are not relocatable on Windows. >> logs\dashboard.log
    echo Recreate it:  python -m venv .venv     then     .venv\Scripts\pip install -r requirements.txt >> logs\dashboard.log
    echo Refusing to fall back to the system Python, which lacks Flask and torch. >> logs\dashboard.log
    exit /b 9009
)

REM Stills only, meanwhile — the dashboard process's own env is what every
REM run IT launches inherits (_launch_run in dashboard.py copies os.environ),
REM so this is what makes /generate and /thumbnails runs (including a
REM partner's) default to stills-only too, not just run.bat/run_scheduled.bat.
REM Remove this line (or set it to 0) once motion is wanted back everywhere.
set RUFUS_STILLS_ONLY=1

echo. >> logs\dashboard.log
echo ==== dashboard starting %DATE% %TIME% ==== >> logs\dashboard.log

REM -u unbuffered: without it a crash traceback can sit in the buffer and never
REM reach the log file, which defeats the point of redirecting it.
"%PY%" -u scripts\dashboard.py >> logs\dashboard.log 2>&1

echo ==== dashboard exited (code %ERRORLEVEL%) %DATE% %TIME% ==== >> logs\dashboard.log
exit /b %ERRORLEVEL%
