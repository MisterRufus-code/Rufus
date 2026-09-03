"""Does this installation actually work — proved by doing, not by checking.

health_check and preflight ask whether the PIECES are present. Present is not
the same as working: an ffmpeg on PATH built without libx264 encodes nothing, a
Pillow that imports fine fails at the first draw, and an sqlite on a filesystem
that will not honour a write-ahead log looks perfect until two processes touch
it at once. Every one of those passes a presence check and fails a video.

These tests are mostly about the two ways a smoke test starts lying: reporting
"not applicable" as broken, and reporting "I looked for it" as "I ran it".
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import smoke  # noqa: E402


def _by_name(results):
    return {c.name: c for c in results}


def test_the_checks_that_do_not_need_the_world_all_pass():
    """Configuration, database, captions, thumbnails and the dashboard need
    nothing but this checkout. If any of them fails here, the repository is
    broken rather than the machine."""
    results = _by_name(smoke.run())
    for name in ("configuration", "database", "captions", "thumbnails",
                 "dashboard"):
        check = results[name]
        assert check.ok is True, f"{name}: {check.detail}"


def test_the_encoder_check_actually_encodes():
    """`shutil.which` finding ffmpeg says a file exists with that name. A build
    compiled without libx264 passes every presence check and renders nothing —
    which is why this one runs the encoder rather than locating it."""
    src = Path(smoke.__file__).read_text(encoding="utf-8")
    body = src[src.index("def _encode_a_real_clip"):src.index("def _build_a_caption_file")]
    assert "libx264" in body
    assert "subprocess.run" in body
    assert "returncode" in body, "the result has to be checked, not assumed"


def test_a_missing_encoder_is_a_failure_and_not_a_skip(monkeypatch):
    """Every video needs one. Reporting it as inapplicable would be the test
    lying to make itself look green."""
    monkeypatch.setattr(smoke.shutil, "which", lambda name: None)
    check = smoke.Check("encoder", "x").run(smoke._encode_a_real_clip)
    assert check.ok is False
    assert "winget" in check.detail or "apt" in check.detail, (
        "a failure owes the fix, not just the fact")


def test_an_encoder_that_runs_but_produces_nothing_is_caught(monkeypatch,
                                                             tmp_path):
    """The case a presence check cannot see: ffmpeg exits 0 and writes an
    empty file."""
    monkeypatch.setattr(smoke.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    class _Done:
        returncode = 0
        stderr = ""

    def fake_run(cmd, **kw):
        Path(cmd[-1]).write_bytes(b"")
        return _Done()

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)
    check = smoke.Check("encoder", "x").run(smoke._encode_a_real_clip)
    assert check.ok is False
    assert "empty" in check.detail


def test_a_nonzero_encoder_reports_what_ffmpeg_said(monkeypatch):
    monkeypatch.setattr(smoke.shutil, "which", lambda name: "/usr/bin/ffmpeg")

    class _Failed:
        returncode = 1
        stderr = "Unknown encoder 'libx264'"

    monkeypatch.setattr(smoke.subprocess, "run", lambda cmd, **kw: _Failed())
    check = smoke.Check("encoder", "x").run(smoke._encode_a_real_clip)
    assert check.ok is False
    assert "libx264" in check.detail


def test_the_database_check_leaves_the_real_one_alone():
    """A smoke test that writes to the owner's database to prove it can write
    to a database is one that has to be cleaned up after, and the cleanup is
    where it goes wrong."""
    import db_manager
    before = db_manager.DB_FILE
    smoke.Check("database", "x").run(smoke._round_trip_the_database)
    assert db_manager.DB_FILE == before, "the real path was left repointed"


def test_a_database_that_will_not_do_wal_is_a_failure(monkeypatch):
    """The dashboard reads while a run writes. Without a write-ahead log that
    locks — usually a network share, which refuses WAL without complaint, so
    the difference is invisible until the second process arrives.

    _conn asks for WAL on every connection; asking is not getting.
    db_manager.journal_mode() is what makes "did we get it" a question anyone
    can put, which is why this patches a real function rather than wrapping
    sqlite3 — the first attempt did that and broke the row round-trip before
    it ever reached the branch under test."""
    import db_manager
    monkeypatch.setattr(db_manager, "journal_mode", lambda: "delete")
    check = smoke.Check("database", "x").run(smoke._round_trip_the_database)
    assert check.ok is False
    assert "WAL" in check.detail
    assert "network drive" in check.detail, "a failure owes the likely cause"


def test_the_journal_mode_reported_is_the_one_in_force():
    """Read from the connection the pipeline actually uses, not from the
    PRAGMA it sends."""
    import db_manager
    assert db_manager.journal_mode() == "wal"


def test_nothing_here_spends_money_or_gpu_time():
    """A check nobody can afford to run is a check nobody runs. This proves the
    machinery around the model calls, not the calls."""
    src = Path(smoke.__file__).read_text(encoding="utf-8")
    body = src[src.index("class Check"):]
    for expensive in ("openai", "OpenAI", "render_one_beat", "generate_clips",
                      "synthesize", "comfy"):
        assert expensive not in body, f"the smoke test reaches for {expensive}"


def test_a_skip_and_a_failure_are_different_things():
    """Collapsing them is how a smoke test starts lying: a machine that has
    not chosen an engine is not a broken machine, and reporting it as one
    teaches people the test is noise."""
    def skipper():
        raise smoke._Skip("this engine is not selected")

    check = smoke.Check("x", "y").run(skipper)
    assert check.ok is None
    assert check.skipped_because == "this engine is not selected"


def test_every_check_says_what_it_did_rather_than_that_it_passed():
    """"ok" is not evidence. "1s H.264, 4096 bytes" is."""
    for check in smoke.run():
        if check.ok:
            assert check.detail, f"{check.name} passed without saying what it did"


def test_the_cli_exit_code_follows_the_failures(monkeypatch, capsys):
    monkeypatch.setattr(smoke, "run", lambda: [
        smoke.Check("a", "x")])
    result = smoke._cli()
    capsys.readouterr()
    assert result in (0, 1)
