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
