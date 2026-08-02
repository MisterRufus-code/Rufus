# serve.ps1 - turn this PC into the always-on Rufus server.
#
#   .\serve.ps1                  # register: dashboard + watchdog start at boot
#   .\serve.ps1 -Tailscale       # ...and publish it to your tailnet over https
#   .\serve.ps1 -Status          # what's registered / running / reachable
#   .\serve.ps1 -Unregister      # remove the boot tasks (leaves data alone)
#
# What "always on" means here: two Windows scheduled tasks that run AT STARTUP
# under your account, before anyone logs in - the dashboard itself, and the
# watchdog that restarts it if it stops answering. Nothing is installed as a
# real Windows service (that needs nssm/srvany); a startup task is what gets
# you the same practical result with only built-in tools.
#
# The dashboard stays bound to 127.0.0.1 and is published with `tailscale
# serve`. It is never bound to 0.0.0.0 and never port-forwarded - see
# scripts\dashboard.py's header for why that distinction is load-bearing.
#
# Plain Write-Host only - no here-strings (PS 5.1/LF parse bug), same as
# schedule_daily.ps1.

param(
    [switch]$Tailscale,
    [switch]$Status,
    [switch]$Unregister
)

$DashTask  = "Rufus Dashboard"
$WatchTask = "Rufus Watchdog"
$Root      = $PSScriptRoot
$Port      = if ($env:RUFUS_DASHBOARD_PORT) { $env:RUFUS_DASHBOARD_PORT } else { "8765" }

function Get-PythonExe {
    $venv = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $venv) { return $venv }
    return "python"
}

# ---------------------------------------------------------------- status ----
if ($Status) {
    Write-Host ""
    Write-Host "Rufus server status" -ForegroundColor Cyan
    Write-Host "-------------------"

    foreach ($t in @($DashTask, $WatchTask)) {
        $q = schtasks /Query /TN "$t" /FO LIST 2>$null
        if ($LASTEXITCODE -eq 0) {
            $state = ($q | Select-String "Status:").ToString().Split(":")[1].Trim()
            Write-Host ("{0,-18} registered ({1})" -f $t, $state) -ForegroundColor Green
        } else {
            Write-Host ("{0,-18} not registered" -f $t) -ForegroundColor Yellow
        }
    }

    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/healthz" -TimeoutSec 5 -UseBasicParsing
        if ($r.StatusCode -eq 200) {
            Write-Host ("{0,-18} answering on port {1}" -f "dashboard", $Port) -ForegroundColor Green
        }
    } catch {
        Write-Host ("{0,-18} NOT answering on port {1}" -f "dashboard", $Port) -ForegroundColor Red
    }

    if (Get-Command tailscale -ErrorAction SilentlyContinue) {
        Write-Host ""
        Write-Host "Tailscale serve:" -ForegroundColor Cyan
        tailscale serve status
    }

    Write-Host ""
    Write-Host "Users with access:" -ForegroundColor Cyan
    & (Get-PythonExe) (Join-Path $Root "scripts\auth.py") list
    exit 0
}

# ------------------------------------------------------------ unregister ----
if ($Unregister) {
    foreach ($t in @($DashTask, $WatchTask)) {
        schtasks /Delete /TN "$t" /F 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host "Removed: $t" -ForegroundColor Green
        } else {
            Write-Host "Not registered: $t" -ForegroundColor Yellow
        }
    }
    if (Get-Command tailscale -ErrorAction SilentlyContinue) {
        Write-Host ""
        Write-Host "Tailscale is still serving the dashboard if you enabled it."
        Write-Host "Stop that separately with:  tailscale serve --bg off"
    }
    exit 0
}

# -------------------------------------------------------------- register ----
$Python = Get-PythonExe
if ($Python -eq "python") {
    Write-Host "No .venv found - using system python. Create one with:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv; .\.venv\Scripts\pip install -r requirements.txt"
    Write-Host ""
}

# Auth first. An always-on dashboard with no users file is a dashboard that
# trusts every tailnet visitor as the owner, which is the exact failure this
# whole setup exists to avoid.
$UsersFile = Join-Path $Root "config\users.json"
if (-not (Test-Path $UsersFile)) {
    Write-Host "No config\users.json yet - creating the owner account." -ForegroundColor Cyan
    & $Python (Join-Path $Root "scripts\auth.py") init
    Write-Host ""
    Write-Host "Save the link above. Add your partner with:" -ForegroundColor Cyan
    Write-Host "  python scripts\auth.py add james --role partner"
    Write-Host ""
}

$DashCmd  = "`"$Python`" `"$Root\scripts\dashboard.py`""
$WatchCmd = "`"$Python`" `"$Root\scripts\watchdog.py`""

schtasks /Create /TN "$DashTask" /TR "$DashCmd" /SC ONSTART /RL HIGHEST /F | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to register '$DashTask' - run this PowerShell as Administrator." -ForegroundColor Red
    exit 1
}
Write-Host "Registered: $DashTask (starts at boot)" -ForegroundColor Green

schtasks /Create /TN "$WatchTask" /TR "$WatchCmd" /SC ONSTART /RL HIGHEST /F | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to register '$WatchTask'." -ForegroundColor Red
    exit 1
}
Write-Host "Registered: $WatchTask (restarts the dashboard if it stops)" -ForegroundColor Green

# Start them now so you don't have to reboot to test any of this.
schtasks /Run /TN "$DashTask"  2>$null | Out-Null
schtasks /Run /TN "$WatchTask" 2>$null | Out-Null
Start-Sleep -Seconds 3

try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/healthz" -TimeoutSec 5 -UseBasicParsing
    Write-Host "Dashboard is answering on http://localhost:$Port" -ForegroundColor Green
} catch {
    Write-Host "Dashboard not answering yet - check logs\dashboard.log in a moment." -ForegroundColor Yellow
}

# ------------------------------------------------------------- tailscale ----
if ($Tailscale) {
    if (-not (Get-Command tailscale -ErrorAction SilentlyContinue)) {
        Write-Host ""
        Write-Host "Tailscale is not installed or not on PATH." -ForegroundColor Red
        Write-Host "Install from https://tailscale.com/download, sign in, then re-run:"
        Write-Host "  .\serve.ps1 -Tailscale"
        exit 1
    }
    Write-Host ""
    Write-Host "Publishing the dashboard to your tailnet..." -ForegroundColor Cyan
    tailscale serve --bg $Port
    if ($LASTEXITCODE -ne 0) {
        Write-Host "tailscale serve failed - is this machine signed in? (tailscale status)" -ForegroundColor Red
        exit 1
    }
    tailscale serve status
    Write-Host ""
    Write-Host "Only devices signed into YOUR tailnet can reach that URL," -ForegroundColor Green
    Write-Host "and they still need a personal sign-in token to get past /login."
}

Write-Host ""
Write-Host "Done. The dashboard now starts with Windows." -ForegroundColor Green
Write-Host ""
Write-Host "Add your partner:   python scripts\auth.py add james --role partner"
Write-Host "Check everything:   .\serve.ps1 -Status"
Write-Host "Daily video runs:   .\schedule_daily.ps1 -Times `"09:00,17:00`""
Write-Host "Stop serving:       .\serve.ps1 -Unregister"
Write-Host ""
Write-Host "For the box to answer at 3am it must not sleep. Check with:" -ForegroundColor Yellow
Write-Host "  powercfg /query SCHEME_CURRENT SUB_SLEEP"
Write-Host "  powercfg /change standby-timeout-ac 0     (never sleep on AC)"
