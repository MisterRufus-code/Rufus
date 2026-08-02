@echo off
REM run_watchdog.bat - launcher for scripts\watchdog.py (serve.ps1 registers
REM this as an ONSTART scheduled task, same reasoning as run_dashboard.bat:
REM a bare schtasks /TR pointing at python.exe starts in C:\WINDOWS\system32
REM with no working directory and no log destination).

chcp 65001 >nul
set PYTHONIOENCODING=utf-8

cd /d "%~dp0"

if exist ".venv\Scripts\activate.bat" call ".venv\Scripts\activate.bat"

if not exist "logs" mkdir "logs"

python -u scripts\watchdog.py >> logs\watchdog.log 2>&1
exit /b %ERRORLEVEL%
