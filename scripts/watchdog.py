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

Run it as its own scheduled task at boot (serve.ps1 registers both), ALWAYS
through run_watchdog.bat rather than by pointing the task at an interpreter:

    python scripts/watchdog.py

WHO WATCHES THIS. Nothing does, which is why its own startup has to be
bulletproof and why the .bat exists. It was not: the task ran bare `python`,
which under a scheduled task is the system interpreter, which has no
`requests` — so this file died on line 33 every time, in a console nobody
sees, and `serve.ps1 -Status` reported the task as "registered (Ready)" in
green. The dashboard then stayed down for as long as it liked. A supervisor
that fails to start is indistinguishable from one with nothing to do, unless
somebody makes the difference visible.

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

# ...but it stops for THIS long, not for good. Giving up permanently means the
# service stays down until a human runs something, and the entire premise of
# this file is that it is 02:00 and there is no human. Most of the causes clear
# on their own or get cleared without anyone thinking about the watchdog: a
# port freed when a stale process is killed, a venv reinstalled, a disk that
# had filled. Backing off for half an hour keeps the failure loud and the
# recovery automatic.
GIVE_UP_COOLDOWN_S = 1800


def _say(msg: str) -> None:
    """Every line stamped. This log is only ever read after the fact, and the
    first question is always "when did it stop" — which an unstamped line
    cannot answer, however clearly it is worded."""
    print(f"[watchdog {time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


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


# A process that exits this fast never served anything — it hit an import
# error, a held port, or a missing config. Long enough that a slow Flask boot
# on a cold disk is not mistaken for a crash.
STILLBORN_S = 4.0


def _start_dashboard() -> subprocess.Popen | None:
    """Start it, and only call that a success if it is still alive after a few
    seconds.

    SPAWNING IS NOT STARTING. Popen returns a process object for a program that
    dies on its next instruction, so a dashboard exiting instantly — port 8765
    already held (dashboard.py exits 3), a missing dependency, a broken config
    — counted as a successful restart every single minute. failures never
    incremented, MAX_CONSECUTIVE_FAILURES never tripped, nothing was ever
    notified, and the log filled with "started dashboard" lines describing a
    thing that was not running. A watchdog that reports success while the
    service is down is worse than no watchdog: it is a watchdog people believe.
    """
    cmd = [sys.executable, str(ROOT / "scripts" / "dashboard.py")]
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "dashboard.log"
    try:
        logf = open(log_path, "ab")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf,
                                stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    except Exception as e:
        _say(f"could not start the dashboard: {e}")
        return None
    try:
        rc = proc.wait(timeout=STILLBORN_S)
    except subprocess.TimeoutExpired:
        _say(f"started dashboard (pid {proc.pid}) → logs/dashboard.log")
        return proc
    hint = {3: "port already held by another process",
            9009: "the .venv interpreter is missing"}.get(rc, "")
    _say(f"the dashboard exited immediately (code {rc})"
         f"{' — ' + hint if hint else ''} — see logs/dashboard.log")
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
        _say("ComfyUI is down but COMFY_START_CMD is not set — "
             "cannot restart it (set it to e.g. "
             r'"C:\ComfyUI\run_nvidia_gpu.bat")')
        return False
    try:
        subprocess.Popen(cmd, shell=True, stdin=subprocess.DEVNULL)
        _say("started ComfyUI via COMFY_START_CMD")
        return True
    except Exception as e:
        _say(f"could not start ComfyUI: {e}")
        return False


def _notify(what: str) -> None:
    if os.environ.get("RUFUS_WATCHDOG_NOTIFY", "1").strip().lower() in ("0", "false", "no", "off"):
        return
    try:
        import notify as notify_mod
        notify_mod.send("Rufus: service restarted", what, priority="normal")
    except Exception:
        pass


def _supervise(name: str, alive, start, state: dict, grace_s: float,
               restarted_text: str, log: str) -> None:
    """One service, one poll. Mutates `state` in place.

    Both services had the same twenty lines written out twice, differing in
    three strings and one number, which is how the ComfyUI branch quietly kept
    an older version of the give-up logic than the dashboard branch.
    """
    now = time.time()
    if now < state["grace_until"]:
        return
    if alive():
        if state["failures"] or state["given_up"]:
            _say(f"{name} is answering again")
        state["failures"] = 0
        state["given_up"] = False
        return
    if state["given_up"]:
        # Cooling off after too many failed starts. Try once more, quietly.
        state["given_up"] = False
        state["failures"] = 0
        _say(f"{name} is still down — trying again after the cooldown")
    _say(f"{name} not answering — restarting")
    if start():
        state["failures"] = 0
        state["grace_until"] = time.time() + grace_s
        _notify(restarted_text)
        return
    state["failures"] += 1
    state["grace_until"] = time.time() + grace_s
    if state["failures"] >= MAX_CONSECUTIVE_FAILURES:
        state["given_up"] = True
        state["grace_until"] = time.time() + GIVE_UP_COOLDOWN_S
        msg = (f"{name} failed to start {MAX_CONSECUTIVE_FAILURES}x in a row. "
               f"Pausing {GIVE_UP_COOLDOWN_S // 60} minutes, then trying "
               f"again. See {log}")
        _say(msg)
        _notify(msg)


def main() -> int:
    interval = _interval()
    watch_comfy = _comfy_enabled()
    _say(f"polling every {interval}s "
         f"(dashboard{' + ComfyUI' if watch_comfy else ''}) — Ctrl+C to stop")

    # ComfyUI takes a while to load models; don't declare it dead and stack a
    # second copy on top of one that's still starting.
    state = {
        "dashboard": {"failures": 0, "grace_until": 0.0, "given_up": False},
        "comfy": {"failures": 0, "grace_until": 0.0, "given_up": False},
    }

    while True:
        _supervise("dashboard", _dashboard_alive, _start_dashboard,
                   state["dashboard"], 30,
                   "The dashboard stopped answering and was restarted.",
                   "logs/dashboard.log")
        if watch_comfy:
            _supervise("ComfyUI", _comfy_alive, _start_comfy,
                       state["comfy"], 180,
                       "ComfyUI stopped answering and was restarted.",
                       "logs/comfy.log")
        time.sleep(interval)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[watchdog] stopped")
        sys.exit(0)
