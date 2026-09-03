# schedule_scout.ps1 - register (or remove) the content scout's recurring pass.
#
#   .\schedule_scout.ps1                 # every 4 hours
#   .\schedule_scout.ps1 -Hours 6        # every 6 hours
#   .\schedule_scout.ps1 -Unregister     # stop it
#
# WHAT THE SCOUT DOES, and what it deliberately does not. Each pass looks at the
# channels in config\competitors.json, scores every recent upload against its
# OWN channel's median, records what it saw, and - when something beat its
# channel and this channel hasn't covered it - researches it and writes one
# script proposal. It renders nothing and uploads nothing. You approve a
# proposal in the dashboard (/scout) and that starts an ordinary run.
#
# WHY A SCHEDULED TASK AND NOT A DAEMON, same as schedule_daily.ps1: each
# firing is its own process, so a hung API call or a crash can't take down the
# next one. "Researching for days" is the accumulating table of observations,
# not a model left thinking - which is what makes running it around the clock
# affordable at all.
#
# COST. Each pass is a handful of cheap API calls plus, at most, one scripted
# proposal. run_scout.bat sets RUFUS_SCOUT_MAX_COST (default $1.00/day across
# every pass) and RUFUS_SCOUT_MAX_PENDING (default 6), and the scout stands
# down entirely while a render is using the GPU.
#
# Plain Write-Host only - no here-strings (PS 5.1/LF parse bug).

param(
    [int]$Hours = 4,
    [switch]$Unregister
)

$TaskName = "Rufus Scout"
$RunBat   = Join-Path $PSScriptRoot "run_scout.bat"

if ($Unregister) {
    $all = schtasks /Query /FO CSV 2>$null | ConvertFrom-Csv
    $mine = $all | Where-Object { $_.TaskName -like "*$TaskName*" }
    if (-not $mine) {
        Write-Host "No '$TaskName' task found (already removed?)." -ForegroundColor Yellow
        exit 0
    }
    foreach ($t in $mine) {
        schtasks /Delete /TN "$($t.TaskName)" /F | Out-Null
        Write-Host "Removed: $($t.TaskName)" -ForegroundColor Green
    }
    exit 0
}

if (-not (Test-Path $RunBat)) {
    Write-Host "run_scout.bat not found next to this script - run from the repo root." -ForegroundColor Red
    exit 1
}

if ($Hours -lt 1 -or $Hours -gt 24) {
    Write-Host "Hours must be 1-24. Every 4 hours is six passes a day, which is" -ForegroundColor Red
    Write-Host "plenty: competitor channels do not publish faster than that." -ForegroundColor Red
    exit 1
}

$competitors = Join-Path $PSScriptRoot "config\competitors.json"
if (-not (Test-Path $competitors)) {
    Write-Host "config\competitors.json does not exist yet." -ForegroundColor Yellow
    Write-Host "Copy config\competitors.json.example to it and put 5-15 channel"
    Write-Host "ids in it, or every pass will do nothing and say so."
    Write-Host ""
}

schtasks /Delete /TN "$TaskName" /F 2>$null | Out-Null
schtasks /Create /TN "$TaskName" /TR "`"$RunBat`"" /SC HOURLY /MO $Hours /F | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to create '$TaskName' - see the schtasks error above." -ForegroundColor Red
    exit 1
}

Write-Host "Scheduled: $TaskName every $Hours hour(s)." -ForegroundColor Green
Write-Host ""
Write-Host "It proposes; you decide. Nothing renders or uploads on its own."
Write-Host ""
Write-Host "Verify:   schtasks /Query /FO LIST /TN `"$TaskName`""
Write-Host "Run now:  schtasks /Run /TN `"$TaskName`""
Write-Host "Dry run:  .venv\Scripts\python.exe scripts\scout.py --once --dry-run"
Write-Host "Logs:     logs\scout_YYYYMMDD.log"
Write-Host "Review:   the dashboard's Scout page"
Write-Host "Remove:   .\schedule_scout.ps1 -Unregister"
