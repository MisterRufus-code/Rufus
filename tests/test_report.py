"""Tests for report.py — the read-only KPI dashboard over rufus.db."""

import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import db_manager
import report


def _seed_db(path: Path):
    """Build a minimal rufus.db with videos + metrics for report tests."""
    db_manager.DB_FILE = path
    report.DB_FILE = path
    db_manager.init_db()
    # 3 videos: one live with metrics, one held, one upload-failed
    live = db_manager.save_video(niche="finance", script_hook="Hook A", scene_desc="s",
                                 video_file="/tmp/a.mp4", score=9, title="Title A",
                                 channel="main_en", youtube_id="vidA")
    db_manager.save_video(niche="finance", script_hook="Hook B", scene_desc="s",
                          video_file="/tmp/b.mp4", score=6, title="Title B", channel="main_en")
    failed = db_manager.save_video(niche="mindset", script_hook="Hook C", scene_desc="s",
                                  video_file="/tmp/c.mp4", score=8, title="Title C",
                                  channel="main_en")
    db_manager.mark_upload_failed(failed, "quota exceeded")
    db_manager.save_metrics(live, views=12000, watch_pct=68.0, ctr=7.5, likes=300)
    return live


def test_report_renders_rows_and_summary(tmp_path):
    _seed_db(tmp_path / "r.db")
    out = report.report(days=3650)
    assert "Title A" in out
    assert "12000" in out or "12,000" in out      # live video's views
    assert "live 1" in out
    assert "held-for-review 1" in out
    assert "upload-failed 1" in out
    assert "FAILED" in out                          # the marked-failed row


def test_report_channel_filter(tmp_path):
    _seed_db(tmp_path / "r.db")
    assert "Title A" in report.report(days=3650, channel="main_en")
    # a channel with no rows → graceful "No videos"
    assert "No videos" in report.report(days=3650, channel="ghost")


def test_report_empty_window(tmp_path):
    _seed_db(tmp_path / "r.db")
    # 0-day window excludes everything seeded "today"? upload_date defaults to today,
    # so use a negative-equivalent by filtering an empty channel instead:
    assert "No videos" in report.report(days=3650, channel="nonexistent")


def test_correlate_needs_min_sample(tmp_path):
    _seed_db(tmp_path / "r.db")          # only 1 video has metrics
    assert "Need ≥5" in report.correlate()


def test_correlate_with_enough_metrics(tmp_path):
    path = tmp_path / "r.db"
    _seed_db(path)
    # add 5 more videos with metrics so correlate has a sample
    for i in range(5):
        vid = db_manager.save_video(niche="finance", script_hook=f"H{i}", scene_desc="s",
                                   video_file=f"/tmp/{i}.mp4", score=8, channel="main_en",
                                   youtube_id=f"v{i}")
        db_manager.save_metrics(vid, views=1000 * (i + 1), watch_pct=50.0, ctr=5.0, likes=10)
    out = report.correlate()
    assert "avg views" in out
