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

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

REM Stills only, meanwhile — the dashboard process's own env is what every
REM run IT launches inherits (_launch_run in dashboard.py copies os.environ),
REM so this is what makes /generate and /thumbnails runs (including a
REM partner's) default to stills-only too, not just run.bat/run_scheduled.bat.
REM Remove this line (or set it to 0) once motion is wanted back everywhere.
set RUFUS_STILLS_ONLY=1

if not exist "logs" mkdir "logs"

echo. >> logs\dashboard.log
echo ==== dashboard starting %DATE% %TIME% ==== >> logs\dashboard.log

REM -u unbuffered: without it a crash traceback can sit in the buffer and never
REM reach the log file, which defeats the point of redirecting it.
python -u scripts\dashboard.py >> logs\dashboard.log 2>&1

echo ==== dashboard exited (code %ERRORLEVEL%) %DATE% %TIME% ==== >> logs\dashboard.log
exit /b %ERRORLEVEL%
