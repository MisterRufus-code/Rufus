"""Which build is this, and which build made that video.

Two uses. The first is support: "which version are you running?" is the opening
question of every conversation about software somebody paid for, and nothing in
this tree could answer it.

The second is why this file earns its place. The standing complaint about this
project — written into DIRECTION.md and half the module docstrings — is code
running ahead of evidence: changes made on judgement with no way to tell
afterwards whether they helped. The measure pages compared videos against each
other while the variable that changed most between them, the code, was recorded
nowhere. These tests hold the line that makes the comparison trustworthy: the
stamp is measured rather than declared, and the comparison stays silent until
it has enough behind it to mean something.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import version  # noqa: E402


@pytest.fixture(autouse=True)
def _fresh_cache():
    version._CACHE.clear()
    yield
    version._CACHE.clear()


def test_the_version_is_written_by_a_person():
    """Nothing derives it from tags or dates. A version that moves on its own
    says a release happened when one did not."""
    assert version.VERSION.count(".") == 2
    assert all(part.isdigit() for part in version.VERSION.split("."))


def test_the_commit_is_read_from_the_repository_and_not_stored():
    """A build fingerprint a stale constant could get wrong is worse than none,
    because it is believed."""
    src = Path(version.__file__).read_text(encoding="utf-8")
    assert "rev-parse" in src
    assert version.commit(), "this checkout has git, so a commit is knowable"


def test_a_copy_with_no_git_says_so_instead_of_guessing(monkeypatch, tmp_path):
    """A zip or an installer payload. Empty rather than "dev" or "unknown": a
    placeholder that reads like a commit is exactly what a support
    conversation must not be handed."""
    monkeypatch.setattr(version, "_git", lambda *a: "")
    monkeypatch.setattr(version, "BUILD_FILE", tmp_path / "BUILD")
    assert version.commit() == ""
    assert version.stamp() == f"Rufus {version.VERSION}"


def test_a_copy_made_without_git_can_still_carry_what_it_was_made_from(
        monkeypatch, tmp_path):
    monkeypatch.setattr(version, "_git", lambda *a: "")
    build = tmp_path / "BUILD"
    build.write_text("commit: deadbee\n", encoding="utf-8")
    monkeypatch.setattr(version, "BUILD_FILE", build)
    assert version.commit() == "deadbee"


def test_local_edits_are_reported_and_not_smoothed_over(monkeypatch):
    """"0.5.0 at a1b2c3d" and "0.5.0 at a1b2c3d with local edits" are different
    claims, and a support conversation that mistakes the second for the first
    wastes everybody's afternoon."""
    monkeypatch.setattr(version, "_git",
                        lambda *a: "abc1234" if a[0] == "rev-parse"
                        else " M scripts/main.py")
    assert version.dirty() is True
    assert "+local" in version.stamp()


# ── and it reaches the row, which is the half that pays for the file ────────

def test_every_video_records_the_build_that_made_it(tmp_path, monkeypatch):
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "v.db")
    dbm.init_db()
    vid = dbm.save_video(niche="n", script_hook="h", scene_desc="s",
                         video_file="a.mp4", score=8)
    import sqlite3
    with sqlite3.connect(str(dbm.DB_FILE)) as c:
        stamped = c.execute("SELECT rufus_version FROM videos WHERE id=?",
                            (vid,)).fetchone()[0]
    assert stamped == version.stamp()


def test_a_version_that_cannot_be_read_does_not_lose_the_video(tmp_path,
                                                               monkeypatch):
    """Fail-open, and None rather than a default string: a wrong build on a
    real row would be believed later, which is worse than an empty one."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "v2.db")
    monkeypatch.setattr(dbm, "_version_stamp", lambda: None)
    dbm.init_db()
    assert dbm.save_video(niche="n", script_hook="h", scene_desc="s",
                          video_file="a.mp4", score=8)


def test_one_build_is_not_a_comparison(tmp_path, monkeypatch):
    """Silent until it has something to say. This project's whole complaint is
    acting on evidence that was not there."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "one.db")
    dbm.init_db()
    for _ in range(8):
        dbm.save_video(niche="n", script_hook="h", scene_desc="s",
                       video_file="a.mp4", score=7)
    rows = dbm.score_by_version()
    assert len(rows) == 1, "one build produced them all"


def test_a_build_with_too_few_videos_is_dropped_rather_than_caveated(
        tmp_path, monkeypatch):
    """An average over two videos rendered next to an average over forty
    invites exactly the reading it cannot support."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "few.db")
    dbm.init_db()
    monkeypatch.setattr(dbm, "_version_stamp", lambda: "Rufus 0.4.0")
    for _ in range(6):
        dbm.save_video(niche="n", script_hook="h", scene_desc="s",
                       video_file="a.mp4", score=6)
    monkeypatch.setattr(dbm, "_version_stamp", lambda: "Rufus 0.5.0")
    for _ in range(2):
        dbm.save_video(niche="n", script_hook="h", scene_desc="s",
                       video_file="a.mp4", score=9)
    versions = [r["version"] for r in dbm.score_by_version()]
    assert versions == ["Rufus 0.4.0"], (
        "the two-video build must not appear beside a six-video one")


def test_two_real_builds_are_compared_newest_first(tmp_path, monkeypatch):
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "two.db")
    dbm.init_db()
    monkeypatch.setattr(dbm, "_version_stamp", lambda: "Rufus 0.4.0")
    for _ in range(5):
        dbm.save_video(niche="n", script_hook="h", scene_desc="s",
                       video_file="a.mp4", score=6)
    monkeypatch.setattr(dbm, "_version_stamp", lambda: "Rufus 0.5.0")
    for _ in range(5):
        dbm.save_video(niche="n", script_hook="h", scene_desc="s",
                       video_file="a.mp4", score=9)
    rows = dbm.score_by_version()
    assert [r["version"] for r in rows] == ["Rufus 0.5.0", "Rufus 0.4.0"]
    assert rows[0]["avg_score"] == 9.0 and rows[1]["avg_score"] == 6.0


def test_videos_from_before_builds_were_recorded_are_counted_and_named(
        tmp_path, monkeypatch):
    """"Why does this only cover forty of my ninety videos" deserves an answer
    on the page rather than a silent omission."""
    import db_manager as dbm
    monkeypatch.setattr(dbm, "DB_FILE", tmp_path / "old.db")
    dbm.init_db()
    monkeypatch.setattr(dbm, "_version_stamp", lambda: None)
    for _ in range(3):
        dbm.save_video(niche="n", script_hook="h", scene_desc="s",
                       video_file="a.mp4", score=5)
    assert dbm.videos_without_a_version() == 3


def test_the_build_is_at_the_foot_of_every_page():
    """A footer rather than a page: the answer is needed while you are looking
    at whatever went wrong, not after navigating to find it."""
    import dashboard
    for path in ("/", "/measure", "/create"):
        page = dashboard.app.test_client().get(path).get_data(as_text=True)
        assert version.stamp() in page, path


def test_the_footer_does_not_break_the_tail_it_is_attached_to():
    """It is computed once at import, which is correct rather than a shortcut:
    this app does not hot-reload, so new code means a restart and a restart
    re-imports the module. An earlier attempt computed it on use via a str
    subclass and broke `PAGE_TAIL.startswith("</main>")` — a string whose str
    value disagrees with itself is a trap for every future reader."""
    import dashboard
    assert isinstance(dashboard.PAGE_TAIL, str)
    assert dashboard.PAGE_TAIL.endswith("</body></html>")
    assert version.stamp() in dashboard.PAGE_TAIL
