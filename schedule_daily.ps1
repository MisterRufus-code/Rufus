# schedule_daily.ps1 - register (or remove) the daily autonomous Rufus run(s).
#
#   .\schedule_daily.ps1                              # one run/day at 13:00
#   .\schedule_daily.ps1 -Time "09:30"                # one run/day, your hour
#   .\schedule_daily.ps1 -Times "09:00,13:00,17:00,20:00,23:00"   # 5 runs/day
#   .\schedule_daily.ps1 -Unregister                  # stop ALL Rufus daily runs
#
# Uses Windows Task Scheduler (schtasks) - no extra software. Each trigger is
# its OWN scheduled task (not an in-process loop) - a crash or a long render
# running past its neighbor's start time can't take down the rest of the day,
# since Task Scheduler just fires each one independently. run_scheduled.bat
# renders behind the pipeline's quality gates; approval/upload now happens
# separately in the dashboard (RUFUS_AUTO_UPLOAD=1 restores old behavior).
# Plain Write-Host only - no here-strings (PS 5.1/LF parse bug).
#
# Multiple runs/day on ONE channel share the same topic pool and used-seed
# history - at 5/day you'll burn through money_history's ~155 Wikipedia
# topics in about a month instead of five. Nothing breaks (the pipeline
# just tries harder sources / eventually hits the wisdom-quote fallback),
# but it's worth knowing before you pick a high number.

param(
    [string]$Time = "13:00",
    [string]$Times = "",
    [switch]$Unregister
)

$TaskPrefix = "Rufus Daily Short"
$RunBat     = Join-Path $PSScriptRoot "run_scheduled.bat"

if ($Unregister) {
    $all = schtasks /Query /FO CSV 2>$null | ConvertFrom-Csv
    $mine = $all | Where-Object { $_.TaskName -like "*$TaskPrefix*" }
    if (-not $mine) {
        Write-Host "No '$TaskPrefix*' tasks found (already removed?)." -ForegroundColor Yellow
        exit 0
    }
    foreach ($t in $mine) {
        schtasks /Delete /TN "$($t.TaskName)" /F | Out-Null
        Write-Host "Removed: $($t.TaskName)" -ForegroundColor Green
    }
    exit 0
}

if (-not (Test-Path $RunBat)) {
    Write-Host "run_scheduled.bat not found next to this script - run from the repo root." -ForegroundColor Red
    exit 1
}

# -Times (comma-separated) wins if given; otherwise fall back to single -Time.
$timeList = if ($Times) { $Times -split "," | ForEach-Object { $_.Trim() } } else { @($Time) }

foreach ($t in $timeList) {
    if ($t -notmatch '^([01]?\d|2[0-3]):[0-5]\d$') {
        Write-Host "Invalid time '$t' - use 24h HH:MM, e.g. 09:30" -ForegroundColor Red
        exit 1
    }
}

# Clear any previously-registered Rufus daily tasks first, so re-running this
# script with a different count/-Times doesn't leave stale extra triggers
# behind (e.g. going from 5 times down to 2).
$existing = schtasks /Query /FO CSV 2>$null | ConvertFrom-Csv | Where-Object { $_.TaskName -like "*$TaskPrefix*" }
foreach ($t in $existing) {
    schtasks /Delete /TN "$($t.TaskName)" /F | Out-Null
}

$i = 1
foreach ($t in $timeList) {
    $taskName = "$TaskPrefix $i"
    schtasks /Create /TN "$taskName" /TR "`"$RunBat`"" /SC DAILY /ST $t /F | Out-Null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Failed to create '$taskName' - see the schtasks error above." -ForegroundColor Red
        exit 1
    }
    Write-Host "Scheduled: $taskName at $t" -ForegroundColor Green
    $i++
}

Write-Host ""
Write-Host "$($timeList.Count) run(s)/day registered." -ForegroundColor Green
Write-Host "The PC must be ON with ComfyUI running at each of those hours."
Write-Host ""
Write-Host "Verify:   schtasks /Query /FO LIST /TN `"$TaskPrefix 1`""
Write-Host "Run now:  schtasks /Run /TN `"$TaskPrefix 1`""
Write-Host "Logs:     logs\rufus_YYYYMMDD.log   (every run is recorded)"
Write-Host "Review:   python scripts\dashboard.py   (approve/reject queued videos)"
Write-Host "Remove:   .\schedule_daily.ps1 -Unregister"
