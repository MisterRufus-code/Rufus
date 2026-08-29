"""Snapshots of rufus.db, taken safely and verified before they are believed.

WHAT IS IN THAT FILE. Not videos — those are on disk and could be re-rendered.
What the database holds is every JUDGEMENT: which of three scripts a person
preferred and which two they passed over, which draw won each shot, what each
published video scored, which YouTube ids belong to this channel. This
project's own account of itself is that the labelled preference pairs are the
product. Nothing backed them up.

Most of what follows is about the difference between a backup and a belief.
"""

import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import backup  # noqa: E402
import db_manager  # noqa: E402


@pytest.fixture(autouse=True)
def sandbox(tmp_path, monkeypatch):
    """A database and a backup directory of our own. A backup module that
    hard-coded paths would cheerfully snapshot the developer's real database
    during a test run — and, worse, restore over it."""
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "rufus.db")
    monkeypatch.setattr(backup, "BACKUP_DIR", tmp_path / "backups")
    db_manager.init_db()
    return tmp_path


def _a_video(score=8):
    return db_manager.save_video(niche="n", script_hook="h", scene_desc="s",
                                 video_file="a.mp4", score=score)


def test_a_snapshot_carries_the_decisions_and_not_just_the_schema():
    _a_video()
    _a_video()
    path = backup.snapshot(reason="test")
    assert path and path.exists()
    with sqlite3.connect(str(path)) as c:
        assert c.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 2


def test_it_uses_sqlites_own_backup_and_not_a_file_copy():
    """A WAL-mode database copied with the filesystem while a run is writing
    produces a snapshot whose main file and -wal sidecar disagree — a backup
    that restores into corruption. The online backup API exists for exactly
    this and is the whole reason this is not three lines of shutil."""
    src = Path(backup.__file__).read_text(encoding="utf-8")
    assert ".backup(" in src
    body = src[src.index("def snapshot("):src.index("def snapshots(")]
    assert "shutil.copy" not in body


def test_an_unsound_snapshot_is_deleted_rather_than_kept(monkeypatch):
    """A bad file in the backup directory is worse than an empty backup
    directory, because it is counted as protection."""
    _a_video()
    monkeypatch.setattr(backup, "_verify", lambda p: (False, "torn"))
    assert backup.snapshot(reason="bad") is None
    assert backup.snapshots() == [], "the bad file was left behind"


def test_an_empty_file_does_not_pass_as_a_backup(tmp_path):
    """Valid SQLite and empty is what you get from snapshotting a database
    that was never initialised, and from the outside it looks exactly like a
    good backup."""
    empty = tmp_path / "empty.db"
    sqlite3.connect(str(empty)).close()
    ok, detail = backup._verify(empty)
    assert not ok and "empty" in detail


def test_no_database_is_reported_rather_than_producing_an_empty_one(sandbox):
    (sandbox / "rufus.db").unlink()
    assert backup.snapshot() is None


def test_a_failed_snapshot_never_raises_at_the_caller(monkeypatch):
    """Called from a dashboard button and from before-a-risky-operation hooks.
    A safety feature that stops the thing it protects is not one."""
    monkeypatch.setattr(backup.sqlite3, "connect",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("disk")))
    assert backup.snapshot(reason="x") is None


def test_old_snapshots_are_pruned_so_the_disk_survives():
    """An unbounded backup directory on the same disk as the thing it protects
    is a way to run out of space during a render — a new failure introduced by
    the safety feature."""
    _a_video()
    for i in range(6):
        backup.snapshot(reason=f"r{i}", keep=3)
    assert len(backup.snapshots()) == 3


def test_the_newest_snapshots_are_the_ones_kept():
    _a_video()
    kept = [backup.snapshot(reason=f"r{i}", keep=2) for i in range(4)]
    names = {p.name for p in backup.snapshots()}
    assert kept[-1].name in names and kept[0].name not in names


# ── restoring, which is where a mistake costs everything ────────────────────

def test_restoring_brings_the_decisions_back():
    _a_video(score=9)
    snap = backup.snapshot(reason="good")
    # Something goes wrong afterwards.
    with sqlite3.connect(str(db_manager.DB_FILE)) as c:
        c.execute("DELETE FROM videos")
    assert db_manager.video_by_id(1) is None
    backup.restore(snap)
    assert db_manager.video_by_id(1)["score"] == 9


def test_the_database_being_replaced_is_never_deleted():
    """Restoring the wrong snapshot is a mistake somebody makes at 2am, and it
    has to be undoable."""
    _a_video()
    snap = backup.snapshot(reason="s")
    _a_video()
    displaced = backup.restore(snap)
    assert displaced.exists()
    with sqlite3.connect(str(displaced)) as c:
        assert c.execute("SELECT COUNT(*) FROM videos").fetchone()[0] == 2, (
            "the displaced file must be the database as it was")


def test_the_sidecars_move_with_the_database_and_are_not_deleted(sandbox):
    """THE BUG THIS TEST WAS WRITTEN TO CATCH. Deleting them does clear the
    way for the restored database — and strips the displaced one of every
    committed transaction still sitting in its write-ahead log, so the "we kept
    your old data" file will not open at all. A safety copy discovered to be
    empty on the day it is needed is worse than no safety copy."""
    _a_video()
    snap = backup.snapshot(reason="s")
    wal = sandbox / "rufus.db-wal"
    wal.write_bytes(b"stale")
    displaced = backup.restore(snap)
    assert not wal.exists(), "nothing stale beside the restored database"
    assert displaced.with_name(displaced.name + "-wal").exists(), (
        "the displaced database kept its own journal")


def test_an_unsound_snapshot_is_refused_rather_than_restored(tmp_path):
    """The one moment a corrupt file must not be trusted is the moment it is
    about to overwrite a working database."""
    junk = tmp_path / "junk.db"
    junk.write_bytes(b"this is not a database")
    with pytest.raises(ValueError):
        backup.restore(junk)
    assert db_manager.DB_FILE.exists()


def test_restoring_something_that_is_not_there_raises_rather_than_returns():
    """A caller who ignored a falsy return would be operating on a database
    they think they replaced."""
    with pytest.raises(FileNotFoundError):
        backup.restore(Path("nowhere.db"))


def test_the_reason_is_written_into_the_filename():
    """"Which of these twelve files is the one I took before the experiment"
    is the question you have at the moment you need one."""
    _a_video()
    p = backup.snapshot(reason="before the style change")
    assert "beforethestylechange" in p.name.replace("-", "")


def test_backups_are_never_committed():
    """Every published id and every human decision, in a repository that may
    be handed to somebody."""
    ignore = (Path(__file__).parent.parent / ".gitignore").read_text(
        encoding="utf-8")
    assert "backups/" in ignore
    assert "rufus.replaced-" in ignore


# ── and it happens without anybody remembering to ───────────────────────────

def test_the_daily_snapshot_happens_once_and_then_stops():
    """A button somebody presses before doing something risky is the wrong
    shape for this: the losses that matter are the ones nobody saw coming, and
    by definition nobody pressed a button first."""
    _a_video()
    first = backup.snapshot_daily()
    assert first is not None
    assert backup.snapshot_daily() is None, "twice in one day is waste"
    assert len(backup.snapshots()) == 1


def test_a_failing_snapshot_never_stops_the_thing_it_protects(monkeypatch):
    """Fired from dashboard import and from the top of every run. A safety
    feature that keeps the dashboard from starting is not one."""
    monkeypatch.setattr(backup, "snapshot", lambda **kw: None)
    assert backup.snapshot_daily() is None


def test_both_entry_points_take_it():
    """Whichever the owner opens first. The dashboard may sit untouched for
    days on a machine that renders nightly, and vice versa."""
    root = Path(__file__).parent.parent / "scripts"
    for name in ("dashboard.py", "main.py"):
        src = (root / name).read_text(encoding="utf-8")
        assert "snapshot_daily()" in src, name


def test_importing_the_dashboard_does_not_write_a_backup():
    """WHERE THE FIRST VERSION PUT IT, AND WHY CI WAS RIGHT TO REFUSE IT.
    Importing a module must not write to disk: a tool that merely inspects the
    file, a test that imports every script, an editor's autocomplete would each
    have taken a snapshot and printed a line about it — and the suite has a
    test asserting that importing every module under both formats prints
    nothing, which is exactly the guard that caught it.

    It passed locally only because a backups/ directory already existed there,
    so the call no-opped and said nothing. A clean machine exposed it
    immediately, which is what a clean machine is for."""
    src = (Path(__file__).parent.parent / "scripts" / "dashboard.py").read_text(
        encoding="utf-8")
    at_import = src[:src.index("if __name__ ==")]
    for line in at_import.splitlines():
        stripped = line.strip()
        if (not stripped or stripped.startswith(("#", '"', "'", "def "))
                or line.startswith((" ", "\t"))):
            continue        # a comment, or inside some function's body
        assert "snapshot_daily()" not in stripped, (
            f"the daily snapshot runs at import time: {stripped}")
        assert "_daily_snapshot()" not in stripped, (
            f"the daily snapshot runs at import time: {stripped}")
    after = src[src.index("if __name__ =="):]
    assert "_daily_snapshot()" in after, (
        "and it still has to happen when the dashboard actually starts")


def test_the_restore_button_addresses_snapshots_by_name_not_by_path():
    """A filename from a form that could address anything on disk is how a
    restore button becomes a way to read arbitrary files — the same reasoning
    that makes gallery stills load by row id."""
    src = (Path(__file__).parent.parent / "scripts" / "dashboard.py").read_text(
        encoding="utf-8")
    body = src[src.index("def system_restore("):]
    body = body[:body.index("\n@app.route")]
    assert "backup.snapshots()" in body and "p.name == name" in body
    assert "Path(" not in body, "no path from the form reaches the filesystem"
