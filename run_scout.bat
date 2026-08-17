@echo off
REM run_scout.bat - one pass of the content scout, invoked by Task Scheduler.
REM
REM Deliberately NOT a daemon, for the same reason schedule_daily.ps1 registers
REM one task per trigger: a crash, a hung API call or a long pass can't take
REM down tomorrow's if each firing is its own process. "24/7" here means a
REM cheap pass every few hours that WRITES DOWN what it saw - the accumulating
REM table is the research, not a model left thinking.
REM
REM Renders nothing and uploads nothing. It writes a script proposal and stops;
REM a human approves it in the dashboard (/scout) and that starts a normal run.

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
REM PYTHONUTF8: without it stdout falls back to the system ANSI code page and a
REM tick mark in the output kills the process - see scripts/console.py.
set PYTHONUTF8=1

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" (
    echo(
    echo ERROR: %CD%\%PY% not found.
    echo The virtualenv is missing or was moved - venvs are not relocatable on
    echo Windows. Recreate it:  python -m venv .venv
    echo Then:                  .venv\Scripts\pip install -r requirements.txt
    echo Refusing to fall back to the system Python, which lacks this project's packages.
    exit /b 9009
)

REM The ceilings. An agent with a model, a schedule and no ceiling is a runaway
REM bill and a fight over the GPU. Both are overridable per-run; these are the
REM values a scheduled pass uses.
if "%RUFUS_SCOUT_MAX_PENDING%"=="" set RUFUS_SCOUT_MAX_PENDING=6
if "%RUFUS_SCOUT_MAX_COST%"=="" set RUFUS_SCOUT_MAX_COST=1.00

for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set TODAY=%%d
if not exist "logs" mkdir "logs"

echo Scout pass starting...
"%PY%" scripts\scout.py %* >> "logs\scout_%TODAY%.log" 2>&1
set RUFUS_EXIT=%ERRORLEVEL%
echo Scout pass finished (exit %RUFUS_EXIT%) - see logs\scout_%TODAY%.log
exit /b %RUFUS_EXIT%
