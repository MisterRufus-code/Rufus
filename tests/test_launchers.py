"""A launcher must never quietly run the wrong Python.

All three .bat files used to do this:

    if exist ".venv\\Scripts\\activate.bat" call ".venv\\Scripts\\activate.bat"
    ...
    python -u scripts\\dashboard.py

The `if exist` guard is silent. When activation is absent or does not take — a
scheduled task under another profile, a venv whose pyvenv.cfg no longer
resolves (venvs are not relocatable on Windows) — bare `python` becomes the
SYSTEM interpreter, which has no Flask and no torch. The dashboard then dies
instantly and the scheduled task returns to `Ready`, which is exactly what the
owner's `netstat` and `Get-ScheduledTask` showed.

Fail-open with no fail-loud is fail-silent, again.
"""

from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
LAUNCHERS = ["run.bat", "run_scheduled.bat", "run_dashboard.bat"]


def _body(name: str) -> str:
    """Launcher text with REM comment lines dropped — prose about the bug must
    not read as the bug."""
    return "\n".join(
        line for line in (ROOT / name).read_text(encoding="utf-8").splitlines()
        if not line.strip().upper().startswith("REM")
    )


@pytest.mark.parametrize("name", LAUNCHERS)
def test_no_launcher_invokes_a_bare_python(name):
    import re

    offenders = [
        f"{name}:{n}: {line.strip()}"
        for n, line in enumerate(_body(name).splitlines(), 1)
        if re.search(r"(?<![\\\w\"])python(?:\.exe)?\s+(?:-\w+\s+)*scripts\\", line)
    ]
    assert not offenders, (
        "spawn the venv interpreter explicitly; a bare `python` is the system "
        "one when activation does not take:\n  " + "\n  ".join(offenders))


@pytest.mark.parametrize("name", LAUNCHERS)
def test_every_launcher_resolves_the_venv_interpreter(name):
    body = _body(name)
    assert 'set "PY=.venv\\Scripts\\python.exe"' in body
    assert '"%PY%"' in body


@pytest.mark.parametrize("name", LAUNCHERS)
def test_a_missing_venv_refuses_instead_of_falling_back(name):
    body = _body(name)
    assert 'if not exist "%PY%"' in body
    assert "exit /b 9009" in body, "must exit non-zero so the task result shows it"


def test_the_dashboard_writes_its_failure_where_it_can_be_read():
    """A scheduled task has no console — the whole reason this .bat exists per
    its own header. An error nobody can read is the same as no error."""
    body = _body("run_dashboard.bat")
    fail_block = body[body.index('if not exist "%PY%"'):]
    fail_block = fail_block[:fail_block.index("exit /b 9009")]
    assert "logs\\dashboard.log" in fail_block


def test_the_dashboard_creates_its_log_dir_before_writing_the_failure():
    """Ordering bug waiting to happen: writing the error into logs\\ before
    logs\\ exists loses the error."""
    body = _body("run_dashboard.bat")
    assert body.index('if not exist "logs" mkdir "logs"') < body.index('if not exist "%PY%"')


@pytest.mark.parametrize("name", LAUNCHERS)
def test_activation_is_kept_but_is_not_the_mechanism(name):
    """activate.bat still runs for the rest of the environment — it just no
    longer decides which interpreter executes."""
    body = _body(name)
    assert "activate.bat" in body
    assert body.index("activate.bat") < body.index('set "PY=')


@pytest.mark.parametrize("name", LAUNCHERS)
def test_utf8_mode_survived_the_edit(name):
    """PYTHONUTF8 is what stops config em-dashes decoding as cp1255 mojibake —
    see AGENTS.md. It must not be lost to an unrelated launcher change."""
    assert "PYTHONUTF8=1" in _body(name)
