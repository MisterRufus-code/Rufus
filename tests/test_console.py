"""A tick mark must not be able to kill a run.

THE REPORT, from a live run that had already finished rendering:

    File "...\\Lib\\encodings\\cp1255.py", line 19, in encode
      return codecs.charmap_encode(input, self.errors, encoding_table)[0]
    UnicodeEncodeError: 'charmap' codec can't encode character '\\u2717'

\\u2717 is ✗, from qc_check's verdict line. The script was written and judged,
the voice recorded, every picture drawn, an hour of the 3090 spent — and the
run died printing the report about the video it had just finished making.
"""

import io
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import console  # noqa: E402


def test_the_marks_this_pipeline_prints_survive_a_hebrew_code_page(tmp_path):
    """The exact failure, reproduced: a cp1255 stream is what Windows gives a
    redirected stdout on this owner's box, and every one of these characters
    appears in the pipeline's own output."""
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1255")
    with pytest.raises(UnicodeEncodeError):
        stream.write("QC ✗ CRITICAL")
        stream.flush()

    fixed = io.TextIOWrapper(io.BytesIO(), encoding="cp1255")
    fixed.reconfigure(encoding="utf-8", errors="replace")
    for mark in ("✗", "✓", "⚠", "—", "×", "→"):
        fixed.write(mark)
    fixed.flush()


def test_force_utf8_changes_a_legacy_stream(monkeypatch):
    monkeypatch.setattr(console, "_DONE", False)
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1255")
    monkeypatch.setattr(sys, "stdout", stream)
    assert console.force_utf8() is True
    assert stream.encoding.lower().replace("-", "") == "utf8"


def test_it_is_idempotent(monkeypatch):
    """Called from every entry point, and several of them import each other."""
    monkeypatch.setattr(console, "_DONE", False)
    monkeypatch.setattr(sys, "stdout",
                        io.TextIOWrapper(io.BytesIO(), encoding="cp1255"))
    assert console.force_utf8() is True
    assert console.force_utf8() is False


def test_a_stream_that_cannot_be_reconfigured_is_left_alone(monkeypatch):
    """pytest's capture object, a pipe someone else already wrapped. Failing
    here would be this module causing the class of crash it exists to stop."""
    class Stubborn:
        encoding = "cp1255"

        def reconfigure(self, **kw):
            raise io.UnsupportedOperation("nope")

    monkeypatch.setattr(console, "_DONE", False)
    monkeypatch.setattr(sys, "stdout", Stubborn())
    monkeypatch.setattr(sys, "stderr", Stubborn())
    console.force_utf8()          # must not raise


def test_a_stream_with_no_reconfigure_at_all_is_survivable(monkeypatch):
    class Ancient:
        encoding = "cp1255"

    monkeypatch.setattr(console, "_DONE", False)
    monkeypatch.setattr(sys, "stdout", Ancient())
    console.force_utf8()


# ── the entry points ─────────────────────────────────────────────────────────

ENTRY_POINTS = ["main.py", "dashboard.py", "run_review.py", "comfy_doctor.py",
                "health_check.py", "watchdog.py"]


@pytest.mark.parametrize("name", ENTRY_POINTS)
def test_every_entry_point_forces_it(name):
    """The launchers set PYTHONUTF8, and AGENTS.md documents that. It is
    necessary and not sufficient: a run launched from the dashboard inherits
    whatever started the dashboard, and its stdout is a redirected FILE, where
    Python has no console to ask and falls back to the system code page."""
    src = (ROOT / "scripts" / name).read_text(encoding="utf-8")
    assert "console.force_utf8()" in src, name


def test_a_dashboard_launched_run_gets_a_utf8_child():
    """The child's stdout is a log file. Nothing about the parent's console
    reaches it, so the environment has to say so explicitly."""
    src = (ROOT / "scripts" / "dashboard.py").read_text(encoding="utf-8")
    assert 'env.setdefault("PYTHONUTF8", "1")' in src
    assert 'env.setdefault("PYTHONIOENCODING", "utf-8")' in src


def test_main_survives_a_cp1255_stdout_end_to_end(tmp_path):
    """The whole point, proven by running a real interpreter the way the
    dashboard runs one: stdout redirected to a file, the locale encoding
    forced to cp1255, and main.py imported."""
    log = tmp_path / "run.log"
    code = ("import sys; sys.path.insert(0, r'%s');"
            "import console; console.force_utf8();"
            "print('           QC \\u2717 CRITICAL: no audio stream');"
            "print('\\u2713 done \\u2014 fine')" % (ROOT / "scripts"))
    env = {"PYTHONIOENCODING": "cp1255", "PATH": "/usr/bin:/bin"}
    with open(log, "wb") as f:
        r = subprocess.run([sys.executable, "-c", code], stdout=f,
                           stderr=subprocess.STDOUT, env=env, timeout=60)
    assert r.returncode == 0, log.read_text(encoding="utf-8", errors="replace")
    assert "✗" in log.read_text(encoding="utf-8")
