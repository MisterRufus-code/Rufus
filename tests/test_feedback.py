"""Tests for ML feedback store — record, load, render records."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

import config


@pytest.fixture()
def tmp_feedback_db(tmp_path, monkeypatch):
    db = tmp_path / "feedback.json"
    monkeypatch.setattr(config, "FEEDBACK_DB_PATH", db)
    return db


def _make_feedback(**kwargs):
    from src.ml.feedback import VideoFeedback
    defaults = dict(
        video_id="abc123",
        title="Test Video",
        niche="finance",
        topic="investing",
        prompt_model="mistral",
    )
    defaults.update(kwargs)
    return VideoFeedback(**defaults)


class TestRecordFeedback:
    def test_record_saves_to_db(self, tmp_feedback_db):
        from src.ml.feedback import record_feedback, load_all_feedback
        fb = _make_feedback()
        with patch("src.database.vector_store.get_vector_store", side_effect=Exception("no store")):
            record_feedback(fb)
        records = load_all_feedback()
        assert len(records) == 1
        assert records[0].video_id == "abc123"

    def test_multiple_records_append(self, tmp_feedback_db):
        from src.ml.feedback import record_feedback, load_all_feedback
        for i in range(3):
            with patch("src.database.vector_store.get_vector_store", side_effect=Exception("no store")):
                record_feedback(_make_feedback(video_id=f"vid_{i}"))
        assert len(load_all_feedback()) == 3


class TestRecordRender:
    def test_render_record_saved(self, tmp_feedback_db):
        from src.ml.feedback import record_render, _load_db
        record_render(topic="test", niche="finance", asset_ids=["a1", "a2"])
        db = _load_db()
        assert len(db) == 1
        assert db[0]["is_render_only"] is True
        assert db[0]["asset_ids"] == ["a1", "a2"]

    def test_render_record_loadable_as_feedback(self, tmp_feedback_db):
        from src.ml.feedback import record_render, load_all_feedback
        record_render(topic="test", niche="finance", asset_ids=["a1"],
                      entropy_score=0.72, keywords_used=["money", "investing"])
        records = load_all_feedback()
        assert len(records) == 1
        assert records[0].niche == "finance"

    def test_extra_fields_do_not_crash_load(self, tmp_feedback_db):
        """Render records have extra fields — load_all_feedback must not crash."""
        from src.ml.feedback import _load_db, _save_db, load_all_feedback
        raw = [{
            "video_id": "local_123", "title": "test", "niche": "tech",
            "topic": "AI", "prompt_model": "mistral",
            "entropy_score": 0.8, "keywords_used": ["AI"], "is_render_only": True,
            "UNKNOWN_FUTURE_FIELD": "should be ignored",
        }]
        _save_db(raw)
        records = load_all_feedback()
        assert len(records) == 1


class TestPerformanceScore:
    def test_all_zero_is_zero(self):
        fb = _make_feedback(ctr=0.0, retention_rate=0.0, views=0)
        assert fb.performance_score == 0.0

    def test_max_score_is_one(self):
        fb = _make_feedback(ctr=0.10, retention_rate=1.0, views=10_000)
        assert fb.performance_score <= 1.0

    def test_score_increases_with_better_metrics(self):
        low = _make_feedback(ctr=0.01, retention_rate=0.2, views=100)
        high = _make_feedback(ctr=0.08, retention_rate=0.7, views=5000)
        assert high.performance_score > low.performance_score
