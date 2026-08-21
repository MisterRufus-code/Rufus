"""The module that names the process holding the port.

The dashboard was down for ten hours behind a port squatted on by a python
that had stopped serving. Three components each knew half of it — the port is
held, nothing answers — and none of them named the process. These tests cover
the parsing (which is where a diagnostic quietly starts lying) and the identity
check (which is where a supervisor starts ending processes that are not its
own).
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import port_owner  # noqa: E402


NETSTAT = """
Active Connections

  Proto  Local Address          Foreign Address        State           PID
  TCP    0.0.0.0:135            0.0.0.0:0              LISTENING       1128
  TCP    127.0.0.1:8765         0.0.0.0:0              LISTENING       1888
  TCP    127.0.0.1:18765        0.0.0.0:0              LISTENING       4242
  TCP    127.0.0.1:53201        127.0.0.1:8765         ESTABLISHED     9001
  TCP    [::1]:8188             [::]:0                 LISTENING       7777
"""


def _netstat(monkeypatch, text=NETSTAT):
    monkeypatch.setattr(port_owner, "_run", lambda *a, **k: text)


# ── reading netstat ─────────────────────────────────────────────────────────

def test_the_listening_pid_is_found(monkeypatch):
    _netstat(monkeypatch)
    assert port_owner._pid_windows(8765) == 1888


def test_a_longer_port_number_is_not_a_match(monkeypatch):
    """18765 ends in 8765. An unanchored search finds it, and the caller then
    reports — or ends — an unrelated process."""
    _netstat(monkeypatch)
    assert port_owner._pid_windows(8765) != 4242


def test_a_connection_TO_the_port_is_not_the_holder(monkeypatch):
    """The ESTABLISHED row has :8765 as its FOREIGN address — that is a
    browser talking to the dashboard, not the dashboard."""
    _netstat(monkeypatch)
    assert port_owner._pid_windows(8765) != 9001


def test_an_ipv6_local_address_still_parses(monkeypatch):
    _netstat(monkeypatch)
    assert port_owner._pid_windows(8188) == 7777


def test_a_port_nobody_holds_is_none(monkeypatch):
    _netstat(monkeypatch)
    assert port_owner._pid_windows(9999) is None


def test_unreadable_netstat_output_is_none_not_a_crash(monkeypatch):
    _netstat(monkeypatch, "")
    assert port_owner._pid_windows(8765) is None


def test_a_failing_command_never_raises(monkeypatch):
    def _boom(*a, **k):
        raise OSError("no netstat here")
    monkeypatch.setattr(port_owner.subprocess, "run", _boom)
    assert port_owner._run(["netstat"]) == ""


# ── deciding whether we may end it ──────────────────────────────────────────
#
# Every caller of is_rufus_dashboard() ENDS the process it says yes to. The
# only acceptable failure direction is "not sure, leave it alone".

def test_a_dashboard_is_recognised():
    assert port_owner.is_rufus_dashboard(
        {"pid": 1888, "cmdline": r'C:\Rufus\.venv\Scripts\python.exe -u scripts\dashboard.py'})


def test_a_forward_slash_path_is_recognised_too():
    assert port_owner.is_rufus_dashboard(
        {"pid": 3, "cmdline": "/usr/bin/python3 scripts/dashboard.py"})


def test_a_python_that_is_not_ours_is_left_alone():
    """"python" is the name of the venv interpreter, the system interpreter and
    somebody's unrelated script alike. It is not something you may end a
    process over."""
    assert not port_owner.is_rufus_dashboard(
        {"pid": 77, "name": "python.exe", "cmdline": "python.exe train_lora.py"})


def test_an_unreadable_command_line_is_left_alone():
    """A python whose command line could not be read reads as not-sure. The
    caller names it and stops — a bad afternoon, rather than an ended process
    that belonged to somebody else."""
    assert not port_owner.is_rufus_dashboard(
        {"pid": 1888, "name": "python.exe", "exe": r"C:\Python311\python.exe",
         "cmdline": ""})


def test_nothing_at_all_is_left_alone():
    assert not port_owner.is_rufus_dashboard(None)
    assert not port_owner.is_rufus_dashboard({})


# ── what the caller shows a human ───────────────────────────────────────────

def test_the_description_names_the_pid_and_the_command():
    line = port_owner.describe({"pid": 1888, "name": "python.exe",
                                "exe": r"C:\Python311\python.exe",
                                "cmdline": r"python.exe scripts\dashboard.py"})
    assert "1888" in line
    assert "python.exe" in line
    assert "dashboard.py" in line


def test_an_unknown_holder_says_so_rather_than_printing_none():
    assert "could not identify" in port_owner.describe(None)


# ── holder() as a whole ─────────────────────────────────────────────────────

def test_holder_returns_none_when_nothing_can_be_identified(monkeypatch):
    """None means "could not find out", never "the port is free" — the caller
    that needs occupancy tests the socket. Conflating the two would make an
    unreadable netstat look like a port nobody is using."""
    monkeypatch.setattr(port_owner, "_is_windows", lambda: True)
    monkeypatch.setattr(port_owner, "_pid_windows", lambda p: None)
    assert port_owner.holder(8765) is None


def test_holder_survives_a_pid_whose_details_cannot_be_read(monkeypatch):
    """Knowing the pid and nothing else is still worth reporting — and it must
    NOT then read as a dashboard."""
    monkeypatch.setattr(port_owner, "_is_windows", lambda: True)
    monkeypatch.setattr(port_owner, "_pid_windows", lambda p: 1888)
    monkeypatch.setattr(port_owner, "_details_windows",
                        lambda pid: (_ for _ in ()).throw(OSError("nope")))
    info = port_owner.holder(8765)
    assert info["pid"] == 1888
    assert not port_owner.is_rufus_dashboard(info)


def test_holder_fills_in_the_process_name_from_the_exe(monkeypatch):
    monkeypatch.setattr(port_owner, "_is_windows", lambda: True)
    monkeypatch.setattr(port_owner, "_pid_windows", lambda p: 1888)
    monkeypatch.setattr(port_owner, "_details_windows", lambda pid: {
        "exe": r"C:\Users\ddani\AppData\Local\Programs\Python\Python311\python.exe",
        "cmdline": r"python.exe scripts\dashboard.py"})
    info = port_owner.holder(8765)
    assert info["name"] == "python.exe"
    assert port_owner.is_rufus_dashboard(info)


# ── ending it ───────────────────────────────────────────────────────────────

def test_ending_reports_failure_when_the_process_survives(monkeypatch):
    """A kill that did not kill must not be reported as one: the caller starts
    a replacement on the strength of this, and it would hit the same held
    port."""
    monkeypatch.setattr(port_owner, "_is_windows", lambda: True)
    monkeypatch.setattr(port_owner, "_run", lambda *a, **k: "")
    monkeypatch.setattr(port_owner, "_alive", lambda pid: True)
    assert port_owner.end(1888) is False


def test_ending_reports_success_when_it_is_gone(monkeypatch):
    monkeypatch.setattr(port_owner, "_is_windows", lambda: True)
    monkeypatch.setattr(port_owner, "_run", lambda *a, **k: "")
    monkeypatch.setattr(port_owner, "_alive", lambda pid: False)
    assert port_owner.end(1888) is True


def test_ending_a_pid_that_cannot_be_signalled_is_false_not_a_crash(monkeypatch):
    monkeypatch.setattr(port_owner, "_is_windows", lambda: False)
    monkeypatch.setattr(port_owner.os, "kill",
                        lambda *a: (_ for _ in ()).throw(PermissionError()))
    assert port_owner.end(1) is False
