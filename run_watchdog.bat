@echo off
REM run_watchdog.bat - launcher for scripts\watchdog.py (serve.ps1 registers
REM this as an ONSTART scheduled task, same reasoning as run_dashboard.bat:
REM a bare schtasks /TR pointing at python.exe starts in C:\WINDOWS\system32
REM with no working directory and no log destination).
REM
REM THIS FILE IS WHY "always on" WAS NOT. The other four launchers were fixed
REM to resolve the interpreter explicitly and this one was left behind, still
REM running bare `python` after a silent `if exist` activation guard. Under a
REM scheduled task that activation does not take, so `python` meant the SYSTEM
REM interpreter, which has no `requests` - watchdog.py died on its import line
REM and the task went straight back to `Ready`.
REM
REM The consequence was invisible in exactly the wrong way: `serve.ps1 -Status`
REM showed "Rufus Watchdog registered (Ready)" and the owner read that as
REM healthy. Nothing was watching the dashboard, so when the dashboard stopped
REM it stayed stopped, and the thing built to notice was itself the thing that
REM had died first.

chcp 65001 >nul
set PYTHONIOENCODING=utf-8
REM PYTHONUTF8: open()/read_text() default to the ANSI code page (cp1255 on a
REM Hebrew-locale box) - see run.bat for the em-dash corruption this prevents.
set PYTHONUTF8=1

cd /d "%~dp0"

REM Resolve the interpreter EXPLICITLY - see the header. activate.bat still
REM runs for the rest of the environment; it just no longer decides which
REM interpreter executes.
if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"
set "PY=.venv\Scripts\python.exe"

if not exist "logs" mkdir "logs"

REM The failure message goes to the LOG, not just the console: a scheduled task
REM has no console, and an error nobody can read is the same as no error. That
REM is not hypothetical here - it is the whole reason this file's own failure
REM went unnoticed for as long as it did.
if not exist "%PY%" (
    echo ==== watchdog FAILED TO START %DATE% %TIME% ==== >> logs\watchdog.log
    echo ERROR: "%CD%\%PY%" not found. >> logs\watchdog.log
    echo The virtualenv is missing or was moved - venvs are not relocatable on Windows. >> logs\watchdog.log
    echo Recreate it:  python -m venv .venv     then     .venv\Scripts\pip install -r requirements.txt >> logs\watchdog.log
    echo Refusing to fall back to the system Python, which lacks requests. >> logs\watchdog.log
    exit /b 9009
)

echo. >> logs\watchdog.log
echo ==== watchdog starting %DATE% %TIME% ==== >> logs\watchdog.log

REM -u unbuffered: without it a crash traceback can sit in the buffer and never
REM reach the log file, which defeats the point of redirecting it.
"%PY%" -u scripts\watchdog.py >> logs\watchdog.log 2>&1
set "RC=%ERRORLEVEL%"

echo ==== watchdog exited (code %RC%) %DATE% %TIME% ==== >> logs\watchdog.log

REM A watchdog that exits is a silent loss of supervision - nothing else is
REM watching it. Say so to whoever is standing here, the same way the dashboard
REM launcher does.
if not "%RC%"=="0" (
    echo.
    echo ==== the watchdog exited with code %RC%. Last lines of logs\watchdog.log:
    echo.
    powershell -NoProfile -Command "Get-Content 'logs\watchdog.log' -Tail 20" 2>nul
    echo.
    echo Full log: %CD%\logs\watchdog.log
    echo.
)
exit /b %RC%
