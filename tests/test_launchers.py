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
LAUNCHERS = ["run.bat", "run_scheduled.bat", "run_dashboard.bat",
             "run_wan_fast.bat"]


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


# ── run_wan_fast.bat: the eight-variable path, as one file ───────────────────

def _wan_fast() -> str:
    return (ROOT / "run_wan_fast.bat").read_text(encoding="utf-8")


def test_it_clears_the_variable_that_silently_disables_motion():
    """RUFUS_FRAMES_PER_BEAT>1 means "cut between stills", which BYPASSES every
    motion engine. A live run printed "motion engines bypassed" while every
    other setting said motion was wanted — the value was a leftover from an
    earlier shell. Setting the rest without clearing this one is a no-op."""
    assert "set RUFUS_FRAMES_PER_BEAT=\n" in _wan_fast().replace("\r\n", "\n")


def test_it_asks_for_the_hero_beat_and_text_to_video_together():
    """RUFUS_T2V alone does nothing — text-to-video only renders the hero
    beat, so both have to be set or the run silently stays on stills."""
    body = _wan_fast()
    assert "set RUFUS_BEAT_MOTION=hero" in body
    assert "set RUFUS_T2V=1" in body
    assert "set RUFUS_STILLS_ONLY=0" in body


def test_the_frame_count_is_a_valid_wan_length():
    """Wan wants 4n+1. 50 is not a legal frame count and fails inside ComfyUI,
    long after the stills have been paid for."""
    import re
    n = int(re.search(r"set RUFUS_T2V_FRAMES=(\d+)", _wan_fast()).group(1))
    assert (n - 1) % 4 == 0, f"{n} is not 4n+1"


def test_it_refuses_to_start_when_the_template_is_not_exported():
    """The whole point of the preflight. A missing export is only rejected by
    ComfyUI at submit time — after the entire stills phase has run."""
    body = _wan_fast()
    assert "comfy_doctor.py wan_t2v" in body
    assert "if errorlevel 1" in body
    assert "exit /b 2" in body


def test_the_preflight_runs_before_main():
    body = _wan_fast()
    assert body.index("comfy_doctor.py") < body.index("scripts\\main.py")


def test_it_says_steps_are_not_settable_here():
    """comfy_template.prepare() substitutes prompt, image, seed and dims only.
    Someone tuning speed will look for a steps variable in this file; it has to
    tell them where the setting actually lives instead of leaving them to
    invent RUFUS_T2V_STEPS."""
    body = _wan_fast()
    assert "steps" in body.lower()
    assert "prepare()" in body


def test_it_warns_when_the_checkout_is_behind_origin():
    """Observed three times in one session: a doctor report was read, acted on
    and discussed while the box was several commits behind, so the fixes being
    described were not the code being run. A stale checkout produces output
    that looks completely normal — git is the only thing that can say."""
    body = _wan_fast()
    assert "git fetch" in body
    assert "rev-list --count HEAD..@{u}" in body
    assert "behind origin" in body


def test_it_warns_but_never_pulls():
    """Pulling someone's repository out from under them mid-run is not a
    launcher's business. Knowing is."""
    # Drop comments AND echoed advice — telling the owner to pull is the whole
    # point; what must not appear is an executed one.
    body = "\n".join(
        line for line in _wan_fast().splitlines()
        if not line.strip().upper().startswith(("REM", "ECHO")))
    assert "git pull" not in body
    assert "git reset" not in body
    assert "git checkout" not in body


def test_the_staleness_check_cannot_stop_the_run():
    """It is information, not a gate — someone deliberately testing an older
    commit must not be blocked by it."""
    body = _wan_fast()
    behind = body.index("behind origin")
    preflight = body.index("comfy_doctor.py")
    between = body[behind:preflight]
    assert "exit /b" not in between


def test_the_fast_launcher_names_its_tts_backend():
    """Unset, the run opens by trying ElevenLabs and failing:

        [tts] ElevenLabs failed (… is a library (premade) voice — free accounts
              cannot use those via the API …) — falling back to Kokoro

    The fallback works, so this costs a round trip and prints a failure on a run
    where nothing is wrong — the kind of noise that trains someone to ignore
    the log."""
    assert "set RUFUS_TTS=" in _wan_fast()
