"""The permanent review record has a size, and it has to be visible.

media_library/debug/ is exempt from housekeeping BY DESIGN — it is the quality
record, not a cache, and _housekeeping says so: "kept forever and never
auto-deleted here. Disk usage is the tradeoff; prune it by hand if it grows too
large."

That decision is right and these tests do not reverse it. The problem is that
it shipped without a number attached: "prune it by hand if it grows too large"
is only actionable if you can see that it has. Three things make it sharp on
the owner's machine — the system drive has single-digit GB free,
RUFUS_BEAT_MOTION=cut writes three stills per beat instead of one, and a disk
that fills fails the render at the very end, after the GPU time is spent.

So: report always, prune only when explicitly capped.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import main as rufus_main


def _run_dir(root: Path, name: str, kb: int) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "keyframe.png").write_bytes(b"x" * kb * 1024)
    return d


def test_usage_counts_runs_and_bytes(monkeypatch, tmp_path):
    monkeypatch.setattr(rufus_main.paths, "debug_root", lambda: tmp_path)
    _run_dir(tmp_path, "20260812-aaa", 100)
    _run_dir(tmp_path, "20260812-bbb", 200)

    total, runs = rufus_main._debug_usage()
    assert runs == 2
    assert total >= 300 * 1024


def test_usage_on_a_missing_root_is_zero(monkeypatch, tmp_path):
    monkeypatch.setattr(rufus_main.paths, "debug_root", lambda: tmp_path / "absent")
    assert rufus_main._debug_usage() == (0, 0)


def test_the_size_is_reported(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rufus_main.paths, "debug_root", lambda: tmp_path)
    _run_dir(tmp_path, "20260812-aaa", 100)

    rufus_main._report_debug_usage()
    out = capsys.readouterr().out
    assert "debug record" in out
    assert "1 run(s)" in out


def test_a_nearly_full_disk_is_called_out(monkeypatch, tmp_path, capsys):
    """A render that fills the disk fails after the GPU time is already spent —
    the warning has to arrive at the start of the run, not the end."""
    monkeypatch.setattr(rufus_main.paths, "debug_root", lambda: tmp_path)
    _run_dir(tmp_path, "20260812-aaa", 10)

    class _Usage:
        free = int(3 * 1024 ** 3)

    monkeypatch.setattr(rufus_main.shutil, "disk_usage", lambda p: _Usage)
    rufus_main._report_debug_usage()
    out = capsys.readouterr().out
    assert "only 3.0 GB free" in out
    assert "RUFUS_DEBUG_MAX_GB" in out


def test_nothing_is_reported_when_there_is_no_record(monkeypatch, tmp_path, capsys):
    monkeypatch.setattr(rufus_main.paths, "debug_root", lambda: tmp_path)
    rufus_main._report_debug_usage()
    assert capsys.readouterr().out == ""


# ── pruning is opt-in, and never touches a run awaiting review ───────────────

def test_pruning_is_off_unless_capped(monkeypatch, tmp_path):
    monkeypatch.delenv("RUFUS_DEBUG_MAX_GB", raising=False)
    monkeypatch.setattr(rufus_main.paths, "debug_root", lambda: tmp_path)
    _run_dir(tmp_path, "20260812-aaa", 100)

    assert rufus_main._housekeep_debug() == 0
    assert (tmp_path / "20260812-aaa").exists()


def test_a_junk_cap_is_ignored_not_crashed(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("RUFUS_DEBUG_MAX_GB", "lots")
    monkeypatch.setattr(rufus_main.paths, "debug_root", lambda: tmp_path)
    _run_dir(tmp_path, "20260812-aaa", 10)

    assert rufus_main._housekeep_debug() == 0
    assert "not a number" in capsys.readouterr().out


def test_oldest_runs_go_first(monkeypatch, tmp_path):
    import os as _os

    monkeypatch.setenv("RUFUS_DEBUG_MAX_GB", "0.0002")   # ~210 KB
    monkeypatch.setattr(rufus_main.paths, "debug_root", lambda: tmp_path)

    old = _run_dir(tmp_path, "20260101-old", 150)
    new = _run_dir(tmp_path, "20260812-new", 150)
    _os.utime(old, (1_000_000, 1_000_000))

    class _Rows(list):
        def fetchall(self): return list(self)

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a): return _Rows()

    monkeypatch.setitem(sys.modules, "db_manager",
                        type("M", (), {"_conn": staticmethod(lambda: _Conn())}))

    rufus_main._housekeep_debug()
    assert not old.exists()
    assert new.exists()


def test_a_run_awaiting_review_is_never_pruned(monkeypatch, tmp_path):
    """Same protection _housekeep_output gives a pending mp4 — the reviewer
    needs the keyframes and the report to judge it."""
    monkeypatch.setenv("RUFUS_DEBUG_MAX_GB", "0.00001")
    monkeypatch.setattr(rufus_main.paths, "debug_root", lambda: tmp_path)

    pending = _run_dir(tmp_path, "20260812-pending", 200)
    done    = _run_dir(tmp_path, "20260812-done", 200)

    class _Rows(list):
        def fetchall(self): return list(self)

    class _Conn:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a): return _Rows([("20260812-pending",)])

    monkeypatch.setitem(sys.modules, "db_manager",
                        type("M", (), {"_conn": staticmethod(lambda: _Conn())}))

    rufus_main._housekeep_debug()
    assert pending.exists()
    assert not done.exists()


def test_a_db_failure_prunes_nothing(monkeypatch, tmp_path, capsys):
    """Never delete a run whose review status cannot be confirmed."""
    monkeypatch.setenv("RUFUS_DEBUG_MAX_GB", "0.00001")
    monkeypatch.setattr(rufus_main.paths, "debug_root", lambda: tmp_path)
    d = _run_dir(tmp_path, "20260812-aaa", 100)

    def _boom():
        raise RuntimeError("db locked")

    monkeypatch.setitem(sys.modules, "db_manager",
                        type("M", (), {"_conn": staticmethod(_boom)}))

    assert rufus_main._housekeep_debug() == 0
    assert d.exists()
    assert "debug prune skipped" in capsys.readouterr().out
