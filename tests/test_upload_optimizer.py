"""Tests for upload time optimizer and A/B thumbnail tracker."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def tmp_ab_db(tmp_path, monkeypatch):
    db = tmp_path / "ab_thumbnails.json"
    import src.upload_optimizer as opt
    monkeypatch.setattr(opt, "_AB_DB_PATH", db)
    return db


class TestBestUploadTime:
    def test_returns_future_datetime(self):
        from src.upload_optimizer import best_upload_time
        result = best_upload_time("finance")
        assert isinstance(result, datetime)
        assert result > datetime.utcnow()

    def test_returns_datetime_for_unknown_niche(self):
        from src.upload_optimizer import best_upload_time
        result = best_upload_time("xyzzy_unknown_9999")
        assert isinstance(result, datetime)
        assert result > datetime.utcnow()

    def test_uses_niche_schedule_weekday(self):
        from src.upload_optimizer import best_upload_time
        result = best_upload_time("finance")
        # finance schedule has Tue/Thu/Sat — verify it's one of those
        assert result.weekday() in (1, 3, 5)  # Tue=1, Thu=3, Sat=5


class TestNextOccurrence:
    def test_next_occurrence_is_future(self):
        from src.upload_optimizer import _next_occurrence
        for wd in range(7):
            dt = _next_occurrence(wd, 14)
            assert dt > datetime.utcnow()
            assert dt.hour == 14
            assert dt.minute == 0

    def test_same_day_same_hour_gives_next_week(self):
        from src.upload_optimizer import _next_occurrence
        now = datetime.utcnow()
        dt = _next_occurrence(now.weekday(), now.hour - 1 if now.hour > 0 else 0)
        # If current hour is past the target, must be next week
        if now.hour > 0:
            assert (dt - now).days >= 6


class TestABThumbnailTracker:
    def test_record_and_load(self, tmp_ab_db):
        from src.upload_optimizer import record_thumbnail_variant, _load_ab_db
        record_thumbnail_variant("vid123", "A", "face_close_up", "/tmp/thumb_a.jpg")
        db = _load_ab_db()
        assert len(db) == 1
        assert db[0]["video_id"] == "vid123"
        assert db[0]["variant"] == "A"

    def test_two_variants_stored(self, tmp_ab_db):
        from src.upload_optimizer import record_thumbnail_variant, _load_ab_db
        record_thumbnail_variant("vid123", "A", "face_close_up", "/tmp/a.jpg")
        record_thumbnail_variant("vid123", "B", "text_heavy", "/tmp/b.jpg")
        db = _load_ab_db()
        assert len(db) == 2

    def test_update_performance_sets_winner(self, tmp_ab_db):
        from src.upload_optimizer import (
            record_thumbnail_variant, update_thumbnail_performance, _load_ab_db
        )
        record_thumbnail_variant("vid456", "A", "face_close_up", "/tmp/a.jpg")
        record_thumbnail_variant("vid456", "B", "text_heavy", "/tmp/b.jpg")
        update_thumbnail_performance("vid456", "A", ctr=0.08, performance_score=0.7)
        update_thumbnail_performance("vid456", "B", ctr=0.05, performance_score=0.5)
        db = _load_ab_db()
        entries = {e["variant"]: e for e in db if e["video_id"] == "vid456"}
        assert entries["A"]["winner"] is True
        assert entries["B"]["winner"] is False

    def test_best_thumbnail_style_returns_winner(self, tmp_ab_db):
        from src.upload_optimizer import (
            record_thumbnail_variant, update_thumbnail_performance, best_thumbnail_style
        )
        record_thumbnail_variant("vid789", "A", "face_close_up", "/tmp/a.jpg")
        record_thumbnail_variant("vid789", "B", "text_heavy", "/tmp/b.jpg")
        update_thumbnail_performance("vid789", "A", ctr=0.10, performance_score=0.8)
        update_thumbnail_performance("vid789", "B", ctr=0.04, performance_score=0.3)
        best = best_thumbnail_style("finance")
        assert best == "face_close_up"

    def test_best_thumbnail_style_none_when_no_data(self, tmp_ab_db):
        from src.upload_optimizer import best_thumbnail_style
        result = best_thumbnail_style("finance")
        assert result is None

    def test_atomic_write_creates_file(self, tmp_ab_db):
        from src.upload_optimizer import record_thumbnail_variant
        record_thumbnail_variant("vid_atomic", "A", "style_x", "/tmp/x.jpg")
        assert tmp_ab_db.exists()
        data = json.loads(tmp_ab_db.read_text())
        assert len(data) == 1
