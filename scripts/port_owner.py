#!/usr/bin/env python3
"""
port_owner.py — which process is holding this port, and is it one of ours.

THE OUTAGE THIS EXISTS FOR. The dashboard was down for ten hours. Everything
that could have said why was already running and already knew half of it:

    serve.ps1 -Status   "dashboard NOT answering on port 8765"
    dashboard.py        "port 8765 is already in use"
    watchdog.py         "the dashboard exited immediately (code 3)
                         — port already held by another process"

Three components, each holding one piece, none of them naming the process. The
owner ran -Restart three times against a port squatted on by a python that had
stopped serving, and the only way anybody found out was by typing
Get-NetTCPConnection by hand. It took one command. Nothing in this repo ran it.

Worse, the advice was actively wrong. dashboard.py prints "a dashboard is
almost certainly running already — open http://localhost:8765, that IS this
dashboard", which is exactly the sentence you must not print when the health
check next door has already proved nothing is answering there.

WHAT MAKES THIS SAFE TO ACT ON. is_rufus_dashboard() is deliberately narrow:
it says yes only when the command line actually names dashboard.py. "It took
my port" is not grounds to end a process nobody has identified — the port could
be held by something the owner started on purpose, and a supervisor that kills
strangers to free a socket is worse than one that stays down and says so.

NO NEW DEPENDENCY. psutil would make this four lines, but requirements.txt is
floors-only on purpose and this is one netstat parse. Every branch here is
best-effort: on any failure at all it returns None, because a diagnostic that
raises turns a readable outage into an unreadable one.

    python scripts/port_owner.py 8765        say who holds it
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath, PureWindowsPath

# Long enough for a cold `Get-CimInstance` on a busy box, short enough that a
# hung query cannot wedge the caller. The watchdog calls into here on a 60s
# poll and dashboard.py calls it on the startup path; neither can afford to
# block, and both prefer "unknown" to "later".
_TIMEOUT_S = 8


def _is_windows() -> bool:
    """The platform switch, in one named place.

    Not `os.name == "nt"` at each call site: a test that wants the Windows
    branch would have to monkeypatch `os.name` itself, and `pathlib` reads that
    global to choose its path flavour — so faking it turns every later Path()
    in the session into "cannot instantiate WindowsPath on your system". One
    function is both easier to fake and impossible to fake destructively.
    """
    return os.name == "nt"


def _run(cmd: list[str] | str, *, shell: bool = False) -> str:
    """Command output, or "" for anything that goes wrong. Never raises."""
    try:
        out = subprocess.run(cmd, shell=shell, capture_output=True,
                             timeout=_TIMEOUT_S, text=True,
                             errors="replace")
    except Exception:
        return ""
    return (out.stdout or "") + (out.stderr or "")


def _pid_windows(port: int) -> int | None:
    """The PID LISTENING on this port, from `netstat -ano`.

    Parsed rather than asked for by name because `Get-NetTCPConnection` is a
    PowerShell cmdlet — reaching it from Python means spawning powershell.exe,
    which is an order of magnitude slower and is not present on every image.
    netstat has been in Windows since NT and its LISTENING lines are stable:

        TCP    127.0.0.1:8765   0.0.0.0:0    LISTENING    1888

    The address is matched with an anchored `:port` so that port 8765 is never
    satisfied by 18765 or by a remote address that happens to end in it.
    """
    text = _run(["netstat", "-ano", "-p", "TCP"])
    if not text:
        text = _run(["netstat", "-ano"])
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        if "LISTEN" not in parts[3].upper():
            continue
        local = parts[1]
        if not re.search(rf"[:\]]{port}$", local):
            continue
        try:
            return int(parts[-1])
        except ValueError:
            continue
    return None


def _pid_posix(port: int) -> int | None:
    if not shutil.which("lsof"):
        return None
    text = _run(["lsof", "-nP", "-iTCP:%d" % port, "-sTCP:LISTEN", "-t"])
    for line in text.splitlines():
        line = line.strip()
        if line.isdigit():
            return int(line)
    return None


def _details_windows(pid: int) -> dict:
    """Executable path and full command line for a PID.

    The command line is the whole point — `Get-Process` gives a name of
    "python" for the venv interpreter, the system interpreter, and a script
    that has nothing to do with Rufus alike, and "python" is not something you
    may end a process over.
    """
    ps = ("Get-CimInstance Win32_Process -Filter \"ProcessId=%d\" | "
          "ForEach-Object { $_.ExecutablePath; $_.CommandLine }" % pid)
    text = _run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps])
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    exe = lines[0] if lines else ""
    cmdline = lines[1] if len(lines) > 1 else ""
    # One line back means the process has an ExecutablePath and no readable
    # CommandLine, or the reverse. Guessing which would produce a cmdline that
    # is really a path, and is_rufus_dashboard() would then match on it.
    if len(lines) == 1 and not PureWindowsPath(exe).name:
        exe, cmdline = "", lines[0]
    return {"exe": exe, "cmdline": cmdline}


def _details_posix(pid: int) -> dict:
    exe = ""
    try:
        exe = os.readlink(f"/proc/{pid}/exe")
    except OSError:
        pass
    cmdline = ""
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
        cmdline = " ".join(raw.decode("utf-8", "replace").split("\0")).strip()
    except OSError:
        pass
    if not cmdline:
        cmdline = _run(["ps", "-o", "args=", "-p", str(pid)]).strip()
    if not exe:
        exe = cmdline.split(" ")[0] if cmdline else ""
    return {"exe": exe, "cmdline": cmdline}


def holder(port: int) -> dict | None:
    """Who is listening on `port`: {pid, exe, cmdline, name}, or None.

    None means "could not find out", never "nothing is there" — the caller
    that needs to know whether the port is occupied should test the socket,
    which is what dashboard._port_taken already does. Conflating the two would
    make an unreadable `netstat` look like a free port.
    """
    try:
        pid = (_pid_windows(port) if _is_windows() else _pid_posix(port))
    except Exception:
        return None
    if not pid:
        return None
    try:
        info = (_details_windows(pid) if _is_windows() else _details_posix(pid))
    except Exception:
        info = {"exe": "", "cmdline": ""}
    info["pid"] = pid
    # The flavour is decided by the platform we just queried, NOT by the one
    # this interpreter happens to run on. `Path(r"C:\x\python.exe").name` on
    # POSIX is the entire string — backslash is an ordinary character there —
    # so the ambient Path would report a whole Windows path as the process
    # "name" and every status line built from it would be unreadable.
    flavour = PureWindowsPath if _is_windows() else PurePosixPath
    info["name"] = flavour(info.get("exe") or "").name
    return info


def is_rufus_dashboard(info: dict | None) -> bool:
    """Whether this process is positively a Rufus dashboard.

    POSITIVELY. Everything about this function's callers — serve.ps1 -Restart
    and the watchdog — ends the process it says yes to, so the only acceptable
    failure direction is "not sure, leave it alone". A python with an
    unreadable command line reads as not-sure and is left running; the caller
    then names it and stops, which is a bad afternoon rather than an ended
    process that belonged to somebody else.
    """
    if not info:
        return False
    cmdline = (info.get("cmdline") or "").replace("\\", "/").lower()
    if not cmdline:
        return False
    return "dashboard.py" in cmdline


def describe(info: dict | None) -> str:
    """One line naming the holder, for a log or a status screen."""
    if not info:
        return "could not identify the process holding the port"
    bits = [f"pid {info.get('pid')}"]
    if info.get("name"):
        bits.append(info["name"])
    line = " · ".join(bits)
    if info.get("exe"):
        line += f"\n    exe     {info['exe']}"
    if info.get("cmdline"):
        line += f"\n    command {info['cmdline']}"
    return line


def end(pid: int) -> bool:
    """Terminate a process by pid. True only if it is actually gone."""
    try:
        if _is_windows():
            _run(["taskkill", "/PID", str(pid), "/F", "/T"])
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
    except Exception:
        return False
    return not _alive(pid)


def _alive(pid: int) -> bool:
    if _is_windows():
        out = _run(["tasklist", "/FI", f"PID eq {pid}", "/NH"])
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main(argv: list[str] | None = None) -> int:
    import console
    console.force_utf8()
    args = list(argv if argv is not None else sys.argv[1:])
    port = int(args[0]) if args else int(
        os.environ.get("RUFUS_DASHBOARD_PORT", "8765"))
    info = holder(port)
    if not info:
        print(f"[port] nothing identifiable is listening on {port}")
        return 1
    print(f"[port] {port} is held by {describe(info)}")
    print(f"[port] {'this IS' if is_rufus_dashboard(info) else 'this is NOT'} "
          f"a Rufus dashboard")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
