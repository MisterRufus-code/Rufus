# serve.ps1 - turn this PC into the always-on Rufus server.
#
#   .\serve.ps1                  # register: dashboard + watchdog start at boot
#   .\serve.ps1 -Tailscale       # ...and publish it to your tailnet over https
#   .\serve.ps1 -Status          # what's registered / running / reachable
#   .\serve.ps1 -Restart         # start both tasks again without rebooting
#   .\serve.ps1 -Unregister      # remove the boot tasks (leaves data alone)
#
# What "always on" means here, precisely: two Windows scheduled tasks with an
# AT STARTUP trigger, running under your account - the dashboard itself, and
# the watchdog that restarts it if it stops answering. Nothing is installed as
# a real Windows service (that needs nssm/srvany); a startup task is what gets
# you the closest practical thing with only built-in tools.
#
# THE LIMIT OF THAT, stated because the previous version of this comment
# claimed "before anyone logs in" and that is not what schtasks creates without
# a stored password. A task registered with no /RU runs with an interactive
# token, so it starts when you log on rather than at the boot prompt. On a box
# somebody signs into, the difference is a few seconds; on one that reboots
# unattended overnight, the dashboard is down until the next logon.
#
# To get true pre-logon start, re-register the two tasks under the SYSTEM
# account by hand:
#     schtasks /Create /TN "Rufus Dashboard" /TR "<path>\run_dashboard.bat" /SC ONSTART /RU SYSTEM /RL HIGHEST /F
# It is not the default here on purpose: SYSTEM has its own profile, so every
# %USERPROFILE% cache the pipeline uses (HuggingFace weights, Whisper models)
# resolves somewhere else and gets re-downloaded on the first run the dashboard
# launches. That is a fair trade for a headless box and a bad surprise for a
# desktop, so it is a decision to make rather than one to inherit.
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
    [switch]$Restart,
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

    # READY IS NOT RUNNING. This block used to print "registered (Ready)" in
    # green for both tasks, and that is how a dead watchdog looked healthy for
    # days: Task Scheduler says "Running" while a process is alive and "Ready"
    # once it has exited. For a task that is supposed to run forever, Ready IS
    # the failure — it means the thing started, quit, and is now merely
    # eligible to start again at the next boot. Green on that is the status
    # screen agreeing with the problem.
    #
    # /V adds Last Run Time and Last Result, which is the part that says WHY:
    # 9009 is the venv guard in the .bat, 1 is a Python traceback, 0 is a
    # process that decided on its own to stop.
    foreach ($t in @($DashTask, $WatchTask)) {
        $q = schtasks /Query /TN "$t" /FO LIST /V 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host ("{0,-18} not registered" -f $t) -ForegroundColor Yellow
            continue
        }
        $state = (($q | Select-String "^Status:") | Select-Object -First 1)
        $state = if ($state) { $state.ToString().Split(":", 2)[1].Trim() } else { "unknown" }
        $last  = (($q | Select-String "^Last Run Time:") | Select-Object -First 1)
        $last  = if ($last) { $last.ToString().Split(":", 2)[1].Trim() } else { "" }
        $code  = (($q | Select-String "^Last Result:") | Select-Object -First 1)
        $code  = if ($code) { $code.ToString().Split(":", 2)[1].Trim() } else { "" }

        if ($state -eq "Running") {
            Write-Host ("{0,-18} running" -f $t) -ForegroundColor Green
        } elseif ($state -eq "unknown") {
            # schtasks output is localized. On a non-English Windows there is
            # no "Status:" line to find, and calling that "NOT running" would
            # be inventing a failure — the opposite mistake to the one this
            # block was written to fix, and just as misleading.
            Write-Host ("{0,-18} registered; state unreadable (localized schtasks output)" -f $t) -ForegroundColor Yellow
        } else {
            Write-Host ("{0,-18} NOT running (task state: {1})" -f $t, $state) -ForegroundColor Red
            if ($last) { Write-Host ("{0,-18}   last run {1}, exit code {2}" -f "", $last, $code) }
            switch ($code) {
                "9009" { Write-Host ("{0,-18}   exit 9009 = the .venv interpreter was missing. Recreate it:" -f "") -ForegroundColor Yellow
                         Write-Host ("{0,-18}   python -m venv .venv ; .\.venv\Scripts\pip install -r requirements.txt" -f "") }
                "3"    { Write-Host ("{0,-18}   exit 3 = port {1} was already held by another process." -f "", $Port) -ForegroundColor Yellow }
            }
        }
    }

    try {
        $r = Invoke-WebRequest -Uri "http://127.0.0.1:$Port/healthz" -TimeoutSec 5 -UseBasicParsing
        if ($r.StatusCode -eq 200) {
            Write-Host ("{0,-18} answering on port {1}" -f "dashboard", $Port) -ForegroundColor Green
        }
    } catch {
        Write-Host ("{0,-18} NOT answering on port {1}" -f "dashboard", $Port) -ForegroundColor Red
        foreach ($name in @("dashboard", "watchdog")) {
            $log = Join-Path $Root "logs\$name.log"
            if (Test-Path $log) {
                Write-Host ""
                Write-Host "Last lines of logs\$name.log:" -ForegroundColor Yellow
                Get-Content $log -Tail 12 | ForEach-Object { Write-Host "  $_" }
            }
        }
        # BOTH logs, not just the dashboard's. When the dashboard is down the
        # first question is why the watchdog did not bring it back, and that
        # answer has never been in dashboard.log.
        Write-Host ""
        Write-Host "Start them now with:  .\serve.ps1 -Restart" -ForegroundColor Cyan
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

# --------------------------------------------------------------- restart ----
# The command -Status now tells you to run. Both tasks are ONSTART, so without
# this the only documented way to bring them back was a reboot, and the honest
# answer to "the dashboard is down" cannot be "restart Windows".
if ($Restart) {
    foreach ($t in @($DashTask, $WatchTask)) {
        schtasks /End /TN "$t" 2>$null | Out-Null
        schtasks /Run /TN "$t" 2>$null | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Write-Host ("Started: {0}" -f $t) -ForegroundColor Green
        } else {
            Write-Host ("Could not start {0} - is it registered? (.\serve.ps1)" -f $t) -ForegroundColor Red
        }
    }
    Start-Sleep -Seconds 5
    try {
        Invoke-WebRequest -Uri "http://127.0.0.1:$Port/healthz" -TimeoutSec 5 -UseBasicParsing | Out-Null
        Write-Host ("dashboard answering on port {0}" -f $Port) -ForegroundColor Green
    } catch {
        Write-Host ("dashboard still NOT answering on port {0} - .\serve.ps1 -Status has the reason" -f $Port) -ForegroundColor Red
    }
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
Write-Host "Registered: $DashTask (starts when you log on)" -ForegroundColor Green

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
Write-Host "Done. The dashboard now starts when you log on." -ForegroundColor Green
Write-Host "(Not before that: these tasks run as you, not as SYSTEM - see the header"
Write-Host " of this script for how to change that and what it costs.)"
Write-Host ""
Write-Host "Add your partner:   python scripts\auth.py add james --role partner"
Write-Host "Check everything:   .\serve.ps1 -Status"
Write-Host "Bring it back up:   .\serve.ps1 -Restart"
Write-Host "Daily video runs:   .\schedule_daily.ps1 -Times `"09:00,17:00`""
Write-Host "Stop serving:       .\serve.ps1 -Unregister"
Write-Host ""
Write-Host "For the box to answer at 3am it must not sleep. Check with:" -ForegroundColor Yellow
Write-Host "  powercfg /query SCHEME_CURRENT SUB_SLEEP"
Write-Host "  powercfg /change standby-timeout-ac 0     (never sleep on AC)"
