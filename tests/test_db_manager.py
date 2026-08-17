"""Tests for db_manager.py — schema migrations and the hold_reason column
(added so the dashboard can show WHY a video wasn't auto-uploaded)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import db_manager


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "test.db")
    db_manager.init_db()
    return db_manager


def test_init_db_creates_hold_reason_column(isolated_db):
    with isolated_db._conn() as c:
        cols = {row[1] for row in c.execute("PRAGMA table_info(videos)")}
    assert "hold_reason" in cols


def test_init_db_creates_seed_url_column(isolated_db):
    with isolated_db._conn() as c:
        cols = {row[1] for row in c.execute("PRAGMA table_info(videos)")}
    assert "seed_url" in cols


def test_init_db_idempotent(isolated_db):
    """Running init_db twice (every startup does) must not raise even though
    hold_reason already exists from the first call."""
    isolated_db.init_db()
    isolated_db.init_db()


def test_save_video_persists_hold_reason(isolated_db):
    vid = isolated_db.save_video(
        niche="finance", script_hook="Hook", scene_desc="s",
        video_file="v.mp4", score=5,
        hold_reason="score 5/10 < 8/10 threshold",
    )
    with isolated_db._conn() as c:
        row = c.execute("SELECT hold_reason FROM videos WHERE id=?", (vid,)).fetchone()
    assert row[0] == "score 5/10 < 8/10 threshold"


def test_save_video_persists_seed_url(isolated_db):
    """The real source link (a Wikipedia/Stack Exchange URL) — needed for
    youtube_uploader.post_source_comment — must survive the DB round-trip,
    not just the short seed_source label."""
    vid = isolated_db.save_video(
        niche="money_history", script_hook="Hook", scene_desc="s",
        video_file="v.mp4", score=8,
        seed_source="history.stackexchange.com",
        seed_url="https://history.stackexchange.com/questions/1234",
    )
    with isolated_db._conn() as c:
        row = c.execute("SELECT seed_url FROM videos WHERE id=?", (vid,)).fetchone()
    assert row[0] == "https://history.stackexchange.com/questions/1234"


def test_save_video_seed_url_defaults_to_none(isolated_db):
    vid = isolated_db.save_video(niche="finance", script_hook="H", scene_desc="s",
                                 video_file="v.mp4")
    with isolated_db._conn() as c:
        row = c.execute("SELECT seed_url FROM videos WHERE id=?", (vid,)).fetchone()
    assert row[0] is None


def test_save_video_hold_reason_defaults_to_none(isolated_db):
    """A clean, auto-uploaded video must record NULL, not an empty string —
    the dashboard treats NULL as 'uploaded cleanly'."""
    vid = isolated_db.save_video(
        niche="finance", script_hook="Hook", scene_desc="s",
        video_file="v.mp4", score=9,
    )
    with isolated_db._conn() as c:
        row = c.execute("SELECT hold_reason FROM videos WHERE id=?", (vid,)).fetchone()
    assert row[0] is None


# ── Approval queue: upload_status + description ────────────────────────────

def test_new_video_defaults_to_pending(isolated_db):
    vid = isolated_db.save_video(niche="finance", script_hook="H", scene_desc="s",
                                 video_file="v.mp4", score=9)
    with isolated_db._conn() as c:
        row = c.execute("SELECT upload_status FROM videos WHERE id=?", (vid,)).fetchone()
    assert row[0] == "pending"


def test_backfill_marks_already_uploaded_rows_approved(isolated_db, tmp_path, monkeypatch):
    """A pre-existing row with a youtube_id (uploaded under the old auto-upload
    flow, before this column existed) must not show as 'pending' just because
    the new column's DEFAULT applies to it too."""
    with isolated_db._conn() as c:
        c.execute("INSERT INTO videos (niche, youtube_id) VALUES ('finance', 'ytXYZ')")
    isolated_db.init_db()   # re-run migrations, including the backfill
    with isolated_db._conn() as c:
        row = c.execute("SELECT upload_status FROM videos WHERE youtube_id='ytXYZ'").fetchone()
    assert row[0] == "approved"


def test_save_video_persists_description(isolated_db):
    vid = isolated_db.save_video(niche="finance", script_hook="H", scene_desc="s",
                                 video_file="v.mp4", score=9,
                                 description="A description with #hashtags")
    with isolated_db._conn() as c:
        row = c.execute("SELECT description FROM videos WHERE id=?", (vid,)).fetchone()
    assert row[0] == "A description with #hashtags"


def test_update_metadata_updates_title_and_description(isolated_db):
    vid = isolated_db.save_video(niche="finance", script_hook="H", scene_desc="s",
                                 video_file="v.mp4", score=9,
                                 title="Old title", description="Old desc")
    isolated_db.update_metadata(vid, title="New title", description="New desc")
    with isolated_db._conn() as c:
        row = c.execute("SELECT title, description FROM videos WHERE id=?", (vid,)).fetchone()
    assert row == ("New title", "New desc")


def test_update_metadata_partial_update_leaves_other_field(isolated_db):
    vid = isolated_db.save_video(niche="finance", script_hook="H", scene_desc="s",
                                 video_file="v.mp4", score=9,
                                 title="Old title", description="Old desc")
    isolated_db.update_metadata(vid, title="New title")
    with isolated_db._conn() as c:
        row = c.execute("SELECT title, description FROM videos WHERE id=?", (vid,)).fetchone()
    assert row == ("New title", "Old desc")


def test_set_upload_status_transitions(isolated_db):
    vid = isolated_db.save_video(niche="finance", script_hook="H", scene_desc="s",
                                 video_file="v.mp4", score=9)
    isolated_db.set_upload_status(vid, "approved")
    with isolated_db._conn() as c:
        row = c.execute("SELECT upload_status FROM videos WHERE id=?", (vid,)).fetchone()
    assert row[0] == "approved"
    isolated_db.set_upload_status(vid, "rejected")
    with isolated_db._conn() as c:
        row = c.execute("SELECT upload_status FROM videos WHERE id=?", (vid,)).fetchone()
    assert row[0] == "rejected"


# ── waiting to go live ───────────────────────────────────────────────────────

def test_a_scheduled_video_is_findable(tmp_path, monkeypatch):
    """A scheduled upload looked identical, from every page in the dashboard,
    to one that is private forever — and that difference is the whole question
    of whether the channel is publishing."""
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "t.db")
    db_manager.init_db()
    vid = db_manager.save_video(niche="money_history", script_hook="h",
                                scene_desc="s", video_file="v.mp4", score=9)
    db_manager.set_publish_at(vid, "2099-01-01T12:00:00Z")
    rows = db_manager.scheduled()
    assert [r["id"] for r in rows] == [vid]
    assert rows[0]["publish_at"] == "2099-01-01T12:00:00Z"


def test_a_publish_time_that_has_passed_is_not_still_waiting(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "t.db")
    db_manager.init_db()
    vid = db_manager.save_video(niche="money_history", script_hook="h",
                                scene_desc="s", video_file="v.mp4", score=9)
    db_manager.set_publish_at(vid, "2000-01-01T12:00:00Z")
    assert db_manager.scheduled() == []


def test_a_public_upload_carries_no_publish_time(tmp_path, monkeypatch):
    """publishAt is only valid on a private upload — YouTube's rule, not ours
    — so empty is the honest answer for a public one, not a missing row."""
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "t.db")
    db_manager.init_db()
    vid = db_manager.save_video(niche="money_history", script_hook="h",
                                scene_desc="s", video_file="v.mp4", score=9)
    db_manager.set_publish_at(vid, "")
    assert db_manager.scheduled() == []


def test_the_soonest_is_first(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "t.db")
    db_manager.init_db()
    ids = []
    for when in ("2099-03-01T12:00:00Z", "2099-01-01T12:00:00Z"):
        v = db_manager.save_video(niche="n", script_hook="h", scene_desc="s",
                                  video_file="v.mp4", score=9)
        db_manager.set_publish_at(v, when)
        ids.append(v)
    assert [r["id"] for r in db_manager.scheduled()] == [ids[1], ids[0]]
