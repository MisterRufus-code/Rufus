"""Tests for review_recent.py — prepping recent videos for a /watch-driven
quality review (claude-video plugin, https://github.com/bradautomates/claude-video)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import db_manager
import review_recent as rr


@pytest.fixture
def isolated_db(tmp_path, monkeypatch):
    monkeypatch.setattr(db_manager, "DB_FILE", tmp_path / "test.db")
    db_manager.init_db()
    return db_manager


def test_recent_for_review_orders_newest_first(isolated_db):
    isolated_db.save_video(niche="finance", script_hook="H1", scene_desc="s",
                           video_file="v1.mp4", score=6)
    isolated_db.save_video(niche="finance", script_hook="H2", scene_desc="s",
                           video_file="v2.mp4", score=8)
    rows = rr.recent_for_review(n=5)
    assert [r["script_hook"] for r in rows] == ["H2", "H1"]


def test_recent_for_review_respects_n(isolated_db):
    for i in range(5):
        isolated_db.save_video(niche="finance", script_hook=f"H{i}", scene_desc="s",
                               video_file=f"v{i}.mp4")
    rows = rr.recent_for_review(n=2)
    assert len(rows) == 2


def test_recent_for_review_filters_by_channel(isolated_db):
    isolated_db.save_video(niche="finance", script_hook="A", scene_desc="s",
                           video_file="a.mp4", channel="main_en")
    isolated_db.save_video(niche="finance", script_hook="B", scene_desc="s",
                           video_file="b.mp4", channel="side_channel")
    rows = rr.recent_for_review(n=10, channel="side_channel")
    assert len(rows) == 1
    assert rows[0]["script_hook"] == "B"


def test_recent_for_review_filters_by_niche(isolated_db):
    isolated_db.save_video(niche="finance", script_hook="A", scene_desc="s",
                           video_file="a.mp4")
    isolated_db.save_video(niche="money_history", script_hook="B", scene_desc="s",
                           video_file="b.mp4")
    rows = rr.recent_for_review(n=10, niche="money_history")
    assert len(rows) == 1
    assert rows[0]["script_hook"] == "B"


def test_recent_for_review_empty_db_returns_empty(isolated_db):
    assert rr.recent_for_review(n=5) == []


# ── format_block ──────────────────────────────────────────────────────────────

def test_format_block_flags_missing_file(isolated_db, tmp_path):
    vid = isolated_db.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                                 video_file=str(tmp_path / "nope.mp4"), score=7,
                                 script_full="The full script text.")
    row = rr.recent_for_review(n=1)[0]
    block = rr.format_block(row)
    assert "MISSING ON DISK" in block
    assert "The full script text." in block
    assert "score=7" in block


def test_format_block_no_missing_flag_when_file_exists(isolated_db, tmp_path):
    video = tmp_path / "real.mp4"
    video.write_bytes(b"x")
    isolated_db.save_video(niche="finance", script_hook="Hook", scene_desc="s",
                           video_file=str(video), score=9)
    row = rr.recent_for_review(n=1)[0]
    block = rr.format_block(row)
    assert "MISSING ON DISK" not in block
    assert str(video) in block


def test_format_block_falls_back_to_hook_when_no_full_script(isolated_db, tmp_path):
    video = tmp_path / "real.mp4"
    video.write_bytes(b"x")
    isolated_db.save_video(niche="finance", script_hook="Just the hook line",
                           scene_desc="s", video_file=str(video))
    row = rr.recent_for_review(n=1)[0]
    block = rr.format_block(row)
    assert "Just the hook line" in block


def test_format_block_shows_unscored_for_null_score(isolated_db, tmp_path):
    video = tmp_path / "real.mp4"
    video.write_bytes(b"x")
    isolated_db.save_video(niche="finance", script_hook="H", scene_desc="s",
                           video_file=str(video), score=None)
    row = rr.recent_for_review(n=1)[0]
    assert "score=unscored" in rr.format_block(row)


def test_format_block_shows_zero_score_explicitly(isolated_db, tmp_path):
    """score=0 (a real, terrible score) must print '0', not be confused with
    unscored — only a NULL score means 'never scored'."""
    video = tmp_path / "real.mp4"
    video.write_bytes(b"x")
    isolated_db.save_video(niche="finance", script_hook="H", scene_desc="s",
                           video_file=str(video), score=0)
    row = rr.recent_for_review(n=1)[0]
    assert "score=0" in rr.format_block(row)
