#!/usr/bin/env python3
"""
watchdog.py — keep the Rufus dashboard (and optionally ComfyUI) answering.

The point of "the PC is the server" is that the dashboard is reachable at
02:00 on a Sunday without anyone logging into Windows. A Task Scheduler entry
starts it at boot, but nothing noticed if it later died — a crashed Flask
process left the tailnet URL dead until someone happened to try it, which is
precisely when you can't fix it.

This polls /healthz and restarts what stopped answering. It is deliberately
dumb: no supervision tree, no state machine, just "is it up, and if not, start
it again," because a watchdog that can itself get stuck is worse than none.

Run it as its own scheduled task at boot (serve.ps1 registers both):
    python scripts/watchdog.py

Environment:
  RUFUS_DASHBOARD_PORT     8765
  RUFUS_WATCHDOG_INTERVAL  60      seconds between checks
  RUFUS_WATCHDOG_COMFY     0/1     also restart ComfyUI when it stops answering
  COMFY_HOST               http://localhost:8188
  COMFY_START_CMD          command used to start ComfyUI (required for the above)
  RUFUS_WATCHDOG_NOTIFY    1       ping the owner when something is restarted
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import requests

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(Path(__file__).parent))

# A restart that immediately dies restarts forever and buries the real error in
# a loop. After this many consecutive failures the watchdog stops trying that
# service and says so — a loud stop beats a silent spin.
MAX_CONSECUTIVE_FAILURES = 5


def _interval() -> int:
    try:
        return max(10, int(os.environ.get("RUFUS_WATCHDOG_INTERVAL", "60")))
    except ValueError:
        return 60


def _dashboard_url() -> str:
    port = os.environ.get("RUFUS_DASHBOARD_PORT", "8765")
    return f"http://127.0.0.1:{port}"


def _dashboard_alive() -> bool:
    """/healthz is unauthenticated precisely so this check works without
    holding a token — see dashboard.PUBLIC_ENDPOINTS."""
    try:
        r = requests.get(f"{_dashboard_url()}/healthz", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def _start_dashboard() -> subprocess.Popen | None:
    cmd = [sys.executable, str(ROOT / "scripts" / "dashboard.py")]
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "dashboard.log"
    try:
        logf = open(log_path, "ab")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        print(f"[watchdog] started dashboard (pid {proc.pid}) → logs/dashboard.log")
        return proc
    except Exception as e:
        print(f"[watchdog] could not start the dashboard: {e}")
        return None


def _comfy_enabled() -> bool:
    return (os.environ.get("RUFUS_WATCHDOG_COMFY", "0").strip().lower()
            in ("1", "true", "yes", "on"))


def _comfy_alive() -> bool:
    host = os.environ.get("COMFY_HOST", "http://localhost:8188").rstrip("/")
    try:
        return requests.get(f"{host}/system_stats", timeout=5).status_code == 200
    except Exception:
        return False


def _start_comfy() -> bool:
    """Start ComfyUI from COMFY_START_CMD.

    No default command on purpose: ComfyUI lives wherever the owner installed
    it, and guessing a path would produce a watchdog that silently "restarts"
    nothing every minute.
    """
    cmd = (os.environ.get("COMFY_START_CMD") or "").strip()
    if not cmd:
        print("[watchdog] ComfyUI is down but COMFY_START_CMD is not set — "
              "cannot restart it (set it to e.g. "
              r'"C:\ComfyUI\run_nvidia_gpu.bat")')
        return False
    try:
        subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL)
        print(f"[watchdog] started ComfyUI via COMFY_START_CMD")
        return True
    except Exception as e:
        print(f"[watchdog] could not start ComfyUI: {e}")
        return False


def _notify(what: str) -> None:
    if os.environ.get("RUFUS_WATCHDOG_NOTIFY", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    try:
        import notify as notify_mod
        notify_mod.send("Rufus: service restarted", what, priority="normal")
    except Exception:
        pass


def main() -> int:
    interval = _interval()
    watch_comfy = _comfy_enabled()
    print(f"[watchdog] polling every {interval}s "
          f"(dashboard{' + ComfyUI' if watch_comfy else ''}) — Ctrl+C to stop")

    failures = {"dashboard": 0, "comfy": 0}
    # ComfyUI takes a while to load models; don't declare it dead and stack a
    # second copy on top of one that's still starting.
    grace_until = {"dashboard": 0.0, "comfy": 0.0}

    while True:
        now = time.time()

        if now >= grace_until["dashboard"] and failures["dashboard"] < MAX_CONSECUTIVE_FAILURES:
            if not _dashboard_alive():
                print("[watchdog] dashboard not answering — restarting")
                started = _start_dashboard() is not None
                failures["dashboard"] = 0 if started else failures["dashboard"] + 1
                grace_until["dashboard"] = time.time() + 30
                if started:
                    _notify("The dashboard stopped answering and was restarted.")
                elif failures["dashboard"] >= MAX_CONSECUTIVE_FAILURES:
                    msg = (f"Dashboard failed to start {MAX_CONSECUTIVE_FAILURES}x "
                           f"in a row — giving up. See logs/dashboard.log")
                    print(f"[watchdog] {msg}")
                    _notify(msg)
            else:
                failures["dashboard"] = 0

        if watch_comfy and now >= grace_until["comfy"] and failures["comfy"] < MAX_CONSECUTIVE_FAILURES:
            if not _comfy_alive():
                print("[watchdog] ComfyUI not answering — restarting")
                started = _start_comfy()
                failures["comfy"] = 0 if started else failures["comfy"] + 1
                grace_until["comfy"] = time.time() + 180   # model loading is slow
                if started:
                    _notify("ComfyUI stopped answering and was restarted.")
                elif failures["comfy"] >= MAX_CONSECUTIVE_FAILURES:
                    msg = f"ComfyUI failed to start {MAX_CONSECUTIVE_FAILURES}x in a row — giving up."
                    print(f"[watchdog] {msg}")
                    _notify(msg)
            else:
                failures["comfy"] = 0

        time.sleep(interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[watchdog] stopped")
        sys.exit(0)
