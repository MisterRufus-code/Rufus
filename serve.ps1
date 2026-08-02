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

$DashTask   = "Rufus Dashboard"
$WatchTask  = "Rufus Watchdog"
$Root       = $PSScriptRoot
$Port       = if ($env:RUFUS_DASHBOARD_PORT) { $env:RUFUS_DASHBOARD_PORT } else { "8765" }
$UrlFile    = Join-Path $Root "config\dashboard_url.txt"
$DashBat    = Join-Path $Root "run_dashboard.bat"
$WatchBat   = Join-Path $Root "run_watchdog.bat"

function Get-PythonExe {
    $venv = Join-Path $Root ".venv\Scripts\python.exe"
    if (Test-Path $venv) { return $venv }
    return "python"
}

function Test-Dependencies($Python) {
    # The single most common failure of this whole setup: no .venv, so this
    # silently falls back to system python, which usually does NOT have
    # flask/filelock installed (they're declared in requirements.txt, which
    # nothing auto-installs into system python). The dashboard task then dies
    # on the very first import, and without this check the only symptom is
    # "not answering" with no indication why.
    $check = & $Python -c "import flask, filelock, requests" 2>&1
    return $LASTEXITCODE -eq 0
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
        $log = Join-Path $Root "logs\dashboard.log"
        if (Test-Path $log) {
            Write-Host "Last lines of logs\dashboard.log:" -ForegroundColor Yellow
            Get-Content $log -Tail 15 | ForEach-Object { Write-Host "  $_" }
        }
    }

    if (Test-Path $UrlFile) {
        Write-Host ""
        Write-Host ("Public URL:        {0}" -f (Get-Content $UrlFile -Raw).Trim()) -ForegroundColor Cyan
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
    Write-Host "No .venv found - using system python." -ForegroundColor Yellow
} else {
    Write-Host "Using $Python" -ForegroundColor Cyan
}

if (-not (Test-Dependencies $Python)) {
    Write-Host ""
    Write-Host "Missing packages (flask/filelock/requests) under $Python." -ForegroundColor Red
    Write-Host "This is why the dashboard task fails silently without this check." -ForegroundColor Red
    Write-Host ""
    if ($Python -eq "python") {
        Write-Host "Recommended - create a venv and install into THAT instead:" -ForegroundColor Yellow
        Write-Host "  python -m venv .venv"
        Write-Host "  .\.venv\Scripts\pip install -r requirements.txt"
        Write-Host ""
        Write-Host "Or install into system python right now:" -ForegroundColor Yellow
        Write-Host "  pip install -r requirements.txt"
    } else {
        Write-Host "Install into the venv:" -ForegroundColor Yellow
        Write-Host "  .\.venv\Scripts\pip install -r requirements.txt"
    }
    Write-Host ""
    Write-Host "Re-run .\serve.ps1 after that." -ForegroundColor Yellow
    exit 1
}

# --------------------------------------------------------------- tailscale --
# Enabled BEFORE the owner account is created / the sign-in link is printed,
# so that link can carry the real tailnet URL instead of localhost, which is
# useless on a phone. This was silently wrong before: auth.py had no way to
# know a tailnet URL existed at all.
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
    $serveOut = tailscale serve --bg $Port 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-Host "tailscale serve failed - is this machine signed in? (tailscale status)" -ForegroundColor Red
        Write-Host $serveOut
        exit 1
    }
    $statusOut = tailscale serve status 2>&1
    $statusOut | ForEach-Object { Write-Host $_ }

    # Pull the https URL out of `tailscale serve status` output, e.g.:
    #   https://rufus.tail635959.ts.net/
    #   |-- proxy http://127.0.0.1:8765
    $tailnetUrl = ($statusOut | Select-String -Pattern 'https://\S+' | Select-Object -First 1)
    if ($tailnetUrl) {
        $url = ($tailnetUrl.Matches[0].Value).TrimEnd('/')
        Set-Content -Path $UrlFile -Value $url -NoNewline
        $env:RUFUS_DASHBOARD_URL = $url   # so THIS session's auth.py calls below use it too
        Write-Host ""
        Write-Host "Saved public URL to config\dashboard_url.txt: $url" -ForegroundColor Green
        Write-Host "(sign-in links, and Discord/ntfy notifications, will use this from now on)"
    } else {
        Write-Host ""
        Write-Host "Could not parse the tailnet URL from 'tailscale serve status' output." -ForegroundColor Yellow
        Write-Host "Sign-in links will fall back to localhost until you set it by hand:"
        Write-Host '  "https://your-machine.your-tailnet.ts.net" | Set-Content config\dashboard_url.txt'
    }
    Write-Host ""
    Write-Host "Only devices signed into YOUR tailnet can reach that URL," -ForegroundColor Green
    Write-Host "and they still need a personal sign-in token to get past /login."
}

# Auth. An always-on dashboard with no users file trusts every tailnet
# visitor as the owner, which is the exact failure this whole setup exists
# to avoid.
$UsersFile = Join-Path $Root "config\users.json"
if (-not (Test-Path $UsersFile)) {
    Write-Host ""
    Write-Host "No config\users.json yet - creating the owner account." -ForegroundColor Cyan
    & $Python (Join-Path $Root "scripts\auth.py") init
    Write-Host ""
    Write-Host "Save the link above. Add your partner with:" -ForegroundColor Cyan
    Write-Host "  python scripts\auth.py add james --role partner"
    Write-Host ""
}

# Scheduled tasks point at .bat wrappers, not python.exe directly: a bare
# `schtasks /TR "python.exe ... dashboard.py"` has no log destination, so a
# crash on startup (exactly what happens without the deps this script just
# checked for) leaves zero trace anywhere. The .bat sets the working
# directory and redirects stdout/stderr to logs\dashboard.log.
schtasks /Create /TN "$DashTask" /TR "`"$DashBat`"" /SC ONSTART /RL HIGHEST /F | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to register '$DashTask' - run this PowerShell as Administrator." -ForegroundColor Red
    exit 1
}
Write-Host "Registered: $DashTask (starts at boot)" -ForegroundColor Green

schtasks /Create /TN "$WatchTask" /TR "`"$WatchBat`"" /SC ONSTART /RL HIGHEST /F | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to register '$WatchTask'." -ForegroundColor Red
    exit 1
}
Write-Host "Registered: $WatchTask (restarts the dashboard if it stops)" -ForegroundColor Green

# Start them now so you don't have to reboot to test any of this.
schtasks /Run /TN "$DashTask"  2>$null | Out-Null
schtasks /Run /TN "$WatchTask" 2>$null | Out-Null
Start-Sleep -Seconds 4

try {
    $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/healthz" -TimeoutSec 5 -UseBasicParsing
    Write-Host "Dashboard is answering on http://localhost:$Port" -ForegroundColor Green
} catch {
    Write-Host "Dashboard not answering yet." -ForegroundColor Yellow
    $log = Join-Path $Root "logs\dashboard.log"
    if (Test-Path $log) {
        Write-Host "Last lines of logs\dashboard.log:" -ForegroundColor Yellow
        Get-Content $log -Tail 15 | ForEach-Object { Write-Host "  $_" }
    } else {
        Write-Host "logs\dashboard.log doesn't exist yet either - the task may not have run." -ForegroundColor Yellow
        Write-Host "Try running it directly to see the error:  .\run_dashboard.bat"
    }
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
