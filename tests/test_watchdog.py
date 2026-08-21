"""The supervisor had no tests, and it was down for days without anyone
knowing.

Two defects, both of the same family: something reported success it had not
earned. The launcher ran a bare `python` that could not import `requests`, so
the watchdog died instantly while the scheduled task said "Ready" — the
launcher half is covered in test_launchers.py. The half here is `_start_dashboard`
counting a process that exited a millisecond later as a successful restart,
which meant the failure counter never moved, the give-up branch never ran, and
nothing was ever notified.
"""

import subprocess
import sys
import time
from pathlib import Path

import pytest

# The real Popen, kept before any monkeypatching: patching
# `watchdog.subprocess.Popen` patches the subprocess module itself, so a fake
# that calls subprocess.Popen calls its own replacement.
_POPEN = subprocess.Popen

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import watchdog  # noqa: E402


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    """Four real seconds per start attempt makes a test suite nobody runs."""
    monkeypatch.setattr(watchdog, "STILLBORN_S", 0.2)
    monkeypatch.setattr(watchdog, "_notify", lambda *a, **k: None)


def _fresh() -> dict:
    return {"failures": 0, "grace_until": 0.0, "given_up": False}


# ── a start is not a start until it is still running ─────────────────────────

def test_a_process_that_dies_immediately_is_not_a_successful_start(monkeypatch,
                                                                   tmp_path):
    """dashboard.py exits 3 when the port is already held. Popen still hands
    back a process object, so the old code logged "started dashboard" once a
    minute forever while nothing was listening."""
    monkeypatch.setattr(watchdog, "ROOT", tmp_path)
    monkeypatch.setattr(watchdog.subprocess, "Popen",
                        lambda *a, **k: _POPEN(
                            [sys.executable, "-c", "raise SystemExit(3)"]))
    assert watchdog._start_dashboard() is None


def test_a_process_still_alive_after_the_grace_window_is_a_start(monkeypatch,
                                                                 tmp_path):
    monkeypatch.setattr(watchdog, "ROOT", tmp_path)
    proc = None

    def _spawn(*a, **k):
        nonlocal proc
        proc = _POPEN([sys.executable, "-c", "import time; time.sleep(30)"])
        return proc

    monkeypatch.setattr(watchdog.subprocess, "Popen", _spawn)
    try:
        assert watchdog._start_dashboard() is not None
    finally:
        if proc:
            proc.kill()


# 9009 is deliberately absent: POSIX masks exit codes to a byte, so a child
# raising SystemExit(9009) is seen as 49 and the case cannot be reproduced
# through a real subprocess here. It is a Windows-only code anyway (the .bat
# launchers' venv guard), and its mapping is covered by the table itself.
@pytest.mark.parametrize("code,expected", [
    (3, "already holds the port"),
    (4, "not answering"),
])
def test_the_exit_code_is_translated(monkeypatch, tmp_path, capsys, code, expected):
    """A bare number sends someone reading the log to a search engine instead
    of to the fix.

    3 AND 4 ARE DELIBERATELY DIFFERENT SENTENCES, because they need opposite
    responses. 3 is "a healthy dashboard is already there" — leave it alone,
    the service is up. 4 is "something holds the port and serves nothing" —
    which is the ten-hour outage, and the only one of the two this file may
    act on."""
    monkeypatch.setattr(watchdog, "ROOT", tmp_path)
    monkeypatch.setattr(watchdog, "_free_the_port", lambda: False)
    monkeypatch.setattr(watchdog.subprocess, "Popen",
                        lambda *a, **k: _POPEN(
                            [sys.executable, "-c", f"raise SystemExit({code})"]))
    watchdog._start_dashboard()
    assert expected in capsys.readouterr().out


# ── the loop ─────────────────────────────────────────────────────────────────

def test_a_live_service_is_left_alone():
    state = _fresh()
    started = []
    watchdog._supervise("dashboard", lambda: True, lambda: started.append(1),
                        state, 30, "restarted", "log")
    assert not started
    assert state["failures"] == 0


def test_a_dead_service_is_restarted():
    state = _fresh()
    started = []
    watchdog._supervise("dashboard", lambda: False,
                        lambda: (started.append(1), True)[1],
                        state, 30, "restarted", "log")
    assert started
    assert state["failures"] == 0
    assert state["grace_until"] > time.time()


def test_failures_accumulate_only_when_the_start_fails():
    state = _fresh()
    for _ in range(3):
        state["grace_until"] = 0.0
        watchdog._supervise("dashboard", lambda: False, lambda: False,
                            state, 0, "restarted", "log")
    assert state["failures"] == 3


def test_it_stops_hammering_after_too_many_failures():
    """A restart that dies instantly, retried every 60s forever, buries the
    real error under thousands of identical lines."""
    state = _fresh()
    attempts = []
    for _ in range(watchdog.MAX_CONSECUTIVE_FAILURES + 3):
        state["grace_until"] = 0.0 if not state["given_up"] else state["grace_until"]
        watchdog._supervise("dashboard", lambda: False,
                            lambda: (attempts.append(1), False)[1],
                            state, 0, "restarted", "log")
    assert state["given_up"]
    assert len(attempts) == watchdog.MAX_CONSECUTIVE_FAILURES


def test_giving_up_is_a_cooldown_and_not_a_surrender():
    """The premise of this file is that it is 02:00 and nobody is awake. A
    permanent stop means the service stays down until a human types something,
    which is the situation the watchdog exists to avoid — and most causes (a
    freed port, a reinstalled venv, a disk that had filled) clear without
    anyone thinking about the watchdog at all."""
    state = _fresh()
    for _ in range(watchdog.MAX_CONSECUTIVE_FAILURES):
        state["grace_until"] = 0.0
        watchdog._supervise("dashboard", lambda: False, lambda: False,
                            state, 0, "restarted", "log")
    assert state["given_up"]
    # The cooldown is real: nothing is attempted while it runs.
    assert state["grace_until"] > time.time() + watchdog.GIVE_UP_COOLDOWN_S - 60
    attempts = []
    watchdog._supervise("dashboard", lambda: False, lambda: attempts.append(1),
                        state, 0, "restarted", "log")
    assert not attempts, "still inside the cooldown"
    # ...and once it expires, it tries again by itself.
    state["grace_until"] = 0.0
    watchdog._supervise("dashboard", lambda: False,
                        lambda: (attempts.append(1), True)[1],
                        state, 30, "restarted", "log")
    assert attempts


def test_recovery_clears_the_failure_state(capsys):
    state = {"failures": 4, "grace_until": 0.0, "given_up": True}
    watchdog._supervise("dashboard", lambda: True, lambda: True,
                        state, 30, "restarted", "log")
    assert state["failures"] == 0 and not state["given_up"]
    assert "answering again" in capsys.readouterr().out


def test_the_grace_window_is_respected():
    """Flask takes a moment to bind; declaring it dead one second after
    starting it stacks a second copy onto a port the first is about to take."""
    state = {"failures": 0, "grace_until": time.time() + 60, "given_up": False}
    started = []
    watchdog._supervise("dashboard", lambda: False, lambda: started.append(1),
                        state, 30, "restarted", "log")
    assert not started


# ── the log ──────────────────────────────────────────────────────────────────

def test_every_line_is_timestamped(capsys):
    """This log is only ever read after the fact, and the first question is
    always when it stopped."""
    watchdog._say("hello")
    out = capsys.readouterr().out
    assert time.strftime("%Y-%m-%d") in out


# ── freeing a port from a dashboard that is provably dead ──────────────────
#
# THE ONE THING ONLY THE WATCHDOG KNOWS. _supervise calls start() only after
# alive() came back False, and exit code 4 means dashboard.py found the port
# held by something that does not answer /healthz. Those two facts together say
# the holder is dead weight — and nothing was putting them together. The
# dashboard sat down for ten hours behind exactly that process while this file
# retried the identical failing action every sixty seconds.
#
# It can now end a process, so every guard below is load-bearing.

DASHBOARD_HOLDER = {"pid": 1888, "name": "python.exe",
                    "exe": r"C:\Python311\python.exe",
                    "cmdline": r"python.exe scripts\dashboard.py"}
STRANGER = {"pid": 4242, "name": "python.exe",
            "exe": r"C:\Python311\python.exe",
            "cmdline": "python.exe train_lora.py"}


@pytest.fixture(autouse=True)
def _reset_clear_cooldown(monkeypatch):
    monkeypatch.setattr(watchdog, "_LAST_PORT_CLEAR", 0.0)


def _holder(monkeypatch, info, ended=None):
    import port_owner
    monkeypatch.setattr(port_owner, "holder", lambda port: info)
    monkeypatch.setattr(port_owner, "end",
                        lambda pid: (ended.append(pid), True)[1] if ended is not None
                        else True)
    return port_owner


def test_a_stale_dashboard_on_the_port_is_ended(monkeypatch, capsys):
    ended = []
    _holder(monkeypatch, DASHBOARD_HOLDER, ended)
    monkeypatch.setattr(watchdog, "_busy_reason", lambda: "")
    monkeypatch.setattr(watchdog, "_notify", lambda *a, **k: None)
    assert watchdog._free_the_port() is True
    assert ended == [1888]


def test_a_process_that_is_not_ours_is_never_ended(monkeypatch, capsys):
    """"It took my port" is not grounds to kill something that might be the
    owner's own work. Report it and stop."""
    ended = []
    _holder(monkeypatch, STRANGER, ended)
    monkeypatch.setattr(watchdog, "_busy_reason", lambda: "")
    monkeypatch.setattr(watchdog, "_notify", lambda *a, **k: None)
    assert watchdog._free_the_port() is False
    assert ended == []
    assert "Not ending it" in capsys.readouterr().out


def test_a_holder_that_cannot_be_identified_is_never_ended(monkeypatch):
    ended = []
    _holder(monkeypatch, None, ended)
    monkeypatch.setattr(watchdog, "_busy_reason", lambda: "")
    assert watchdog._free_the_port() is False
    assert ended == []


def test_a_dashboard_that_says_it_is_busy_is_left_working(monkeypatch, capsys):
    """A YouTube upload of a 25MB file answers nothing for minutes. From
    outside that is indistinguishable from a wedged process, and the wrong
    guess loses the upload."""
    ended = []
    _holder(monkeypatch, DASHBOARD_HOLDER, ended)
    monkeypatch.setattr(watchdog, "_busy_reason", lambda: "uploading video #7, 42s ago")
    assert watchdog._free_the_port() is False
    assert ended == []
    assert "waiting rather than killing" in capsys.readouterr().out


def test_it_will_not_end_a_second_process_in_one_cooldown(monkeypatch):
    """A watchdog that ends a process, fails to start, and ends the next one is
    not a supervisor — it is a loop with a body count."""
    ended = []
    _holder(monkeypatch, DASHBOARD_HOLDER, ended)
    monkeypatch.setattr(watchdog, "_busy_reason", lambda: "")
    monkeypatch.setattr(watchdog, "_notify", lambda *a, **k: None)
    assert watchdog._free_the_port() is True
    assert watchdog._free_the_port() is False
    assert ended == [1888]


def test_a_failed_kill_is_reported_as_a_failure(monkeypatch, capsys):
    """Reporting success would have the caller start a replacement, which
    would hit the same held port."""
    import port_owner
    monkeypatch.setattr(port_owner, "holder", lambda port: DASHBOARD_HOLDER)
    monkeypatch.setattr(port_owner, "end", lambda pid: False)
    monkeypatch.setattr(watchdog, "_busy_reason", lambda: "")
    monkeypatch.setattr(watchdog, "_notify", lambda *a, **k: None)
    assert watchdog._free_the_port() is False
    assert "could not end pid 1888" in capsys.readouterr().out


def test_only_exit_4_frees_the_port(monkeypatch, tmp_path):
    """Exit 3 means a HEALTHY dashboard is already there. Ending that one would
    take down a working service to start an identical one."""
    calls = []
    monkeypatch.setattr(watchdog, "ROOT", tmp_path)
    monkeypatch.setattr(watchdog, "_free_the_port", lambda: calls.append(1))
    for code in (3, 9009, 1):
        monkeypatch.setattr(watchdog.subprocess, "Popen",
                            lambda *a, _c=code, **k: _POPEN(
                                [sys.executable, "-c", f"raise SystemExit({_c})"]))
        watchdog._start_dashboard()
    assert calls == []


# ── the busy marker ────────────────────────────────────────────────────────

def test_a_marker_left_by_a_crash_stops_protecting_the_corpse(monkeypatch, tmp_path):
    """A marker nothing cleaned up is not a busy dashboard, it is a crashed
    one — which is the case this whole guard stands in front of."""
    import paths
    monkeypatch.setattr(paths, "log_dir", lambda: tmp_path)
    stale = time.time() - watchdog.BUSY_MARKER_MAX_S - 60
    (tmp_path / ".dashboard_busy").write_text(f"uploading|{stale}", encoding="utf-8")
    assert watchdog._busy_reason() == ""


def test_a_fresh_marker_reports_what_it_is_doing(monkeypatch, tmp_path):
    import paths
    monkeypatch.setattr(paths, "log_dir", lambda: tmp_path)
    (tmp_path / ".dashboard_busy").write_text(
        f"uploading video #7|{time.time()}", encoding="utf-8")
    assert "uploading video #7" in watchdog._busy_reason()


def test_no_marker_at_all_is_not_busy(monkeypatch, tmp_path):
    import paths
    monkeypatch.setattr(paths, "log_dir", lambda: tmp_path)
    assert watchdog._busy_reason() == ""


def test_an_unreadable_marker_is_not_busy(monkeypatch, tmp_path):
    """Fail toward acting, not toward paralysis: a corrupt marker must not
    protect a wedged process forever."""
    import paths
    monkeypatch.setattr(paths, "log_dir", lambda: tmp_path)
    (tmp_path / ".dashboard_busy").write_text("garbage", encoding="utf-8")
    assert watchdog._busy_reason() == ""
