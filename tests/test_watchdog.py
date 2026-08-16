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


def test_the_exit_code_is_translated(monkeypatch, tmp_path, capsys):
    """3 and 9009 are the two the owner will actually hit — a held port and a
    missing venv. A bare number sends someone reading the log to a search
    engine instead of to the fix."""
    monkeypatch.setattr(watchdog, "ROOT", tmp_path)
    monkeypatch.setattr(watchdog.subprocess, "Popen",
                        lambda *a, **k: _POPEN(
                            [sys.executable, "-c", "raise SystemExit(3)"]))
    watchdog._start_dashboard()
    assert "port already held" in capsys.readouterr().out


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
