# schedule_daily.ps1 - register (or remove) the daily autonomous Rufus run.
#
#   .\schedule_daily.ps1                 # run every day at 13:00
#   .\schedule_daily.ps1 -Time "09:30"   # pick your own hour (24h HH:MM)
#   .\schedule_daily.ps1 -Unregister     # stop the daily runs
#
# Uses Windows Task Scheduler (schtasks) - no extra software. The task runs
# run_scheduled.bat, which renders + uploads (private) behind the pipeline's
# quality gates. Plain Write-Host only - no here-strings (PS 5.1/LF parse bug).

param(
    [string]$Time = "13:00",
    [switch]$Unregister
)

$TaskName = "Rufus Daily Short"
$RunBat   = Join-Path $PSScriptRoot "run_scheduled.bat"

if ($Unregister) {
    schtasks /Delete /TN "$TaskName" /F
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Daily task removed." -ForegroundColor Green
    } else {
        Write-Host "No task named '$TaskName' found (already removed?)." -ForegroundColor Yellow
    }
    exit $LASTEXITCODE
}

if (-not (Test-Path $RunBat)) {
    Write-Host "run_scheduled.bat not found next to this script - run from the repo root." -ForegroundColor Red
    exit 1
}

if ($Time -notmatch '^([01]?\d|2[0-3]):[0-5]\d$') {
    Write-Host "Invalid -Time '$Time' - use 24h HH:MM, e.g. -Time `"09:30`"" -ForegroundColor Red
    exit 1
}

schtasks /Create /TN "$TaskName" /TR "`"$RunBat`"" /SC DAILY /ST $Time /F
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create the task - see the schtasks error above." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Scheduled: one Short every day at $Time." -ForegroundColor Green
Write-Host "The PC must be ON with ComfyUI running at that hour."
Write-Host ""
Write-Host "Verify:   schtasks /Query /TN `"$TaskName`""
Write-Host "Run now:  schtasks /Run /TN `"$TaskName`""
Write-Host "Logs:     logs\rufus_YYYYMMDD.log   (every run is recorded)"
Write-Host "Output:   media_library\output\     (video + .qc.json report)"
Write-Host "Remove:   .\schedule_daily.ps1 -Unregister"
