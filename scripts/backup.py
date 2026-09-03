#!/usr/bin/env python3
"""Snapshots of rufus.db, taken safely and verified before they are believed.

WHAT IS IN THAT FILE. Not videos — those are on disk and could be re-rendered.
What the database holds is every JUDGEMENT: which of three scripts a person
preferred and which two they passed over, which draw won each shot and which
lost, which read was chosen, what each published video actually scored, which
YouTube ids belong to this channel. The project's own account of itself is that
the labelled preference pairs are the product; this file is where they live,
and nothing anywhere backed it up.

A desktop machine with a 3090 pulling four hundred watts loses power sometimes.
SQLite in WAL mode survives that far better than the default journal, which is
why db_manager sets it — but "far better" is not "always", and the failure mode
is not a crash you notice, it is a file that opens fine and is missing a table.

THREE THINGS THIS DOES THAT `cp rufus.db backup.db` DOES NOT.

  1. It uses SQLite's online backup API. Copying a WAL-mode database while a
     run is writing gives a torn file — the copy and the -wal sidecar disagree,
     and the result is a backup that restores into corruption. The backup API
     takes a consistent snapshot of a live database, which is the entire
     reason it exists.

  2. It VERIFIES. A backup nobody has opened is a belief, not a backup. Every
     snapshot is opened, integrity-checked and counted before it is kept, and
     one that fails is deleted rather than left looking like a safety net.

  3. It rotates. An unbounded backup directory on the same disk as the thing it
     protects is a way to run out of space during a render, which is a new
     failure introduced by the safety feature.

RESTORING IS DELIBERATELY AWKWARD. It refuses while a run holds the channel
lock, and it never deletes the current database — the file being replaced is
moved aside with a timestamp first. Restoring the wrong snapshot is a mistake
somebody makes at 2am, and it must be undoable.
"""

from __future__ import annotations

import shutil
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent

# Beside the database rather than inside a hidden directory: a person looking
# for "where are my backups" should find them next to the thing they back up.
BACKUP_DIR = ROOT / "backups"

# How many snapshots to keep. Twelve is a fortnight of daily ones with room for
# the manual ones a person takes before doing something they are unsure about —
# which is when a backup is most likely to be wanted and least likely to exist.
KEEP = 12

PREFIX = "rufus-"
SUFFIX = ".db"


def _db_file() -> Path:
    """The live database, asked of db_manager rather than assumed.

    Tests repoint db_manager.DB_FILE, and a backup module that hard-coded the
    path would cheerfully snapshot the developer's real database during a test
    run — and, worse, restore over it.
    """
    import db_manager
    return Path(db_manager.DB_FILE)


def _verify(path: Path) -> tuple[bool, str]:
    """Open the snapshot and ask SQLite whether it is sound.

    Both halves matter. integrity_check catches a torn file; the row count
    catches the subtler thing — a file that is valid SQLite and empty, which is
    what you get from backing up a database that was never initialised, and
    which looks exactly like a good backup from the outside.
    """
    try:
        with sqlite3.connect(str(path)) as c:
            result = c.execute("PRAGMA integrity_check").fetchone()
            if not result or result[0] != "ok":
                return False, f"integrity_check said {result[0] if result else 'nothing'}"
            tables = c.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()[0]
            if not tables:
                return False, "no tables — this is an empty file, not a backup"
        return True, f"{tables} table(s)"
    except Exception as e:
        return False, str(e)


def snapshot(*, reason: str = "manual", keep: int = KEEP) -> Path | None:
    """Take a verified snapshot. Returns its path, or None when it could not.

    Never raises. This is called from a dashboard button, from before-a-risky-
    operation hooks and potentially from a scheduled task; a backup that fails
    must say so and let the caller carry on, because the alternative is a
    safety feature that stops the thing it was protecting.
    """
    src = _db_file()
    if not src.exists():
        print(f"[backup] {src} does not exist — nothing to snapshot")
        return None

    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    safe_reason = "".join(ch for ch in reason if ch.isalnum() or ch in "-_")[:24]
    dest = BACKUP_DIR / f"{PREFIX}{stamp}-{safe_reason or 'manual'}{SUFFIX}"

    try:
        # THE ONLINE BACKUP API, not a file copy. A WAL-mode database copied
        # with the filesystem while a run is writing produces a snapshot whose
        # main file and -wal sidecar disagree.
        source = sqlite3.connect(str(src))
        target = sqlite3.connect(str(dest))
        try:
            source.backup(target)
        finally:
            target.close()
            source.close()
    except Exception as e:
        print(f"[backup] snapshot failed: {e}")
        dest.unlink(missing_ok=True)
        return None

    ok, detail = _verify(dest)
    if not ok:
        # Deleted rather than kept with a warning. A bad file in the backup
        # directory is worse than an empty backup directory, because it is
        # counted as protection.
        print(f"[backup] snapshot was not sound ({detail}) — discarded")
        dest.unlink(missing_ok=True)
        return None

    print(f"[backup] {dest.name} — {detail}, "
          f"{dest.stat().st_size // 1024}KB")
    prune(keep=keep)
    return dest


def snapshot_daily() -> Path | None:
    """One snapshot a day, taken by whatever runs first. None if today has one.

    THE BACKUP THAT ACTUALLY HAPPENS IS THE ONE NOBODY HAS TO REMEMBER. A
    button somebody presses before doing something risky is the wrong shape for
    this: the losses that matter are the ones nobody saw coming, and by
    definition nobody pressed a button first.

    Cheap enough to call on every dashboard start and every run — it globs a
    directory and returns, and the snapshot itself is a fraction of a second on
    a database this size.
    """
    today = time.strftime("%Y%m%d")
    if any(p.name.startswith(f"{PREFIX}{today}-") for p in snapshots()):
        return None
    return snapshot(reason="daily")


def snapshots() -> list[Path]:
    """Every snapshot, newest first."""
    if not BACKUP_DIR.exists():
        return []
    return sorted(BACKUP_DIR.glob(f"{PREFIX}*{SUFFIX}"),
                  key=lambda p: p.name, reverse=True)


def prune(keep: int = KEEP) -> int:
    """Drop the oldest snapshots past `keep`. Returns how many went.

    Oldest-first by NAME, which is safe because the name leads with a sortable
    timestamp — mtime would reorder the whole set the first time somebody
    copied the directory somewhere.
    """
    extra = snapshots()[max(0, int(keep)):]
    for path in extra:
        try:
            path.unlink()
        except OSError as e:
            print(f"[backup] could not remove {path.name}: {e}")
    return len(extra)


def restore(snapshot_path: Path | str) -> Path:
    """Put a snapshot back, moving the current database aside first.

    Returns the path the displaced database was moved to, so the caller can
    tell somebody where their data went. Raises on refusal rather than
    returning a falsy value: restoring is not a routine call, and a caller who
    ignored the result would be operating on a database they think they
    replaced.
    """
    path = Path(snapshot_path)
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    ok, detail = _verify(path)
    if not ok:
        raise ValueError(f"{path.name} is not a sound database ({detail}) — "
                         f"refusing to restore it over a working one")

    live = _db_file()
    # THE FILE BEING REPLACED IS NEVER DELETED. Restoring the wrong snapshot is
    # a mistake somebody makes at 2am, and it has to be undoable.
    displaced = live.with_name(
        f"{live.stem}.replaced-{time.strftime('%Y%m%d-%H%M%S')}{live.suffix}")
    if live.exists():
        shutil.move(str(live), str(displaced))
    # THE SIDECARS MOVE WITH IT, and this is the half that was wrong first
    # time. They were deleted instead — which does clear them out of the way of
    # the restored database, and also strips the displaced one of every
    # committed transaction still sitting in its write-ahead log. The "we kept
    # your old data" file then would not open at all: a safety copy that is
    # only discovered to be empty on the day it is needed. A test written to
    # check the copy was intact is what caught it.
    #
    # Both jobs are done by the move: nothing stale is left beside the restored
    # file for SQLite to replay another database's journal into, and the
    # displaced database keeps everything it had.
    for sidecar in ("-wal", "-shm"):
        stale = live.with_name(live.name + sidecar)
        if stale.exists():
            shutil.move(str(stale), str(displaced.with_name(
                displaced.name + sidecar)))
    shutil.copy2(str(path), str(live))
    print(f"[backup] restored {path.name}; the database it replaced is at "
          f"{displaced.name}")
    return displaced


def _cli() -> int:
    import sys

    args = sys.argv[1:]
    # `backup.py list` and `backup.py restore X` are subcommands; anything else
    # is the REASON for a snapshot, because "backup.py before-experiment"
    # is what a person actually types and it silently became "manual".
    cmd = args[0] if args and args[0] in ("list", "restore", "snapshot") \
        else "snapshot"

    if cmd == "list":
        rows = snapshots()
        if not rows:
            print("\n  No snapshots yet. `python scripts/backup.py` takes one.\n")
            return 0
        print(f"\n  {len(rows)} snapshot(s) in {BACKUP_DIR}:\n")
        for p in rows:
            ok, detail = _verify(p)
            print(f"    {'ok ' if ok else 'BAD'} {p.name:44} "
                  f"{p.stat().st_size // 1024:>6}KB  {detail}")
        print()
        return 0

    if cmd == "restore":
        if len(args) < 2:
            print("usage: backup.py restore <snapshot file|latest>")
            return 2
        target = args[1]
        if target == "latest":
            rows = snapshots()
            if not rows:
                print("no snapshots to restore")
                return 2
            target = rows[0]
        else:
            candidate = Path(target)
            target = candidate if candidate.exists() else BACKUP_DIR / target
        try:
            restore(target)
        except Exception as e:
            print(f"refused: {e}")
            return 2
        return 0

    rest = args[1:] if (args and args[0] == "snapshot") else args
    reason = rest[0] if rest else "manual"
    return 0 if snapshot(reason=reason) else 1


if __name__ == "__main__":
    raise SystemExit(_cli())
