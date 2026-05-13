"""Extended feedback tests — thread safety, summary, feedback_summary."""

from __future__ import annotations

import threading
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
        video_id="abc123", title="Test", niche="finance",
        topic="investing", prompt_model="mistral",
    )
    defaults.update(kwargs)
    return VideoFeedback(**defaults)


class TestThreadSafety:
    def test_concurrent_writes_no_data_loss(self, tmp_feedback_db):
        from src.ml.feedback import record_feedback, load_all_feedback
        errors = []

        def write_record(i):
            try:
                with patch("src.database.vector_store.get_vector_store",
                           side_effect=Exception("no store")):
                    record_feedback(_make_feedback(video_id=f"vid_{i:04d}"))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_record, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        records = load_all_feedback()
        assert len(records) == 20

    def test_concurrent_render_records_no_loss(self, tmp_feedback_db):
        from src.ml.feedback import record_render, load_all_feedback
        errors = []

        def write_render(i):
            try:
                record_render(topic=f"topic_{i}", niche="finance", asset_ids=[f"a{i}"])
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write_render, args=(i,)) for i in range(15)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        records = load_all_feedback()
        assert len(records) == 15


class TestFeedbackSummary:
    def test_empty_db_returns_zero_total(self, tmp_feedback_db):
        from src.ml.feedback import feedback_summary
        result = feedback_summary()
        assert result["total"] == 0

    def test_summary_contains_avg_performance(self, tmp_feedback_db):
        from src.ml.feedback import record_feedback, feedback_summary
        for i in range(3):
            with patch("src.database.vector_store.get_vector_store",
                       side_effect=Exception("no store")):
                record_feedback(_make_feedback(
                    video_id=f"v{i}", ctr=0.05, retention_rate=0.5, views=1000
                ))
        s = feedback_summary()
        assert s["total"] == 3
        assert 0 < s["avg_performance"] <= 1.0

    def test_best_niche_identified(self, tmp_feedback_db):
        from src.ml.feedback import record_feedback, feedback_summary
        for i in range(2):
            with patch("src.database.vector_store.get_vector_store",
                       side_effect=Exception("no store")):
                record_feedback(_make_feedback(
                    video_id=f"fin_{i}", niche="finance", ctr=0.08, retention_rate=0.7, views=5000
                ))
        with patch("src.database.vector_store.get_vector_store",
                   side_effect=Exception("no store")):
            record_feedback(_make_feedback(
                video_id="tech_1", niche="tech", ctr=0.02, retention_rate=0.3, views=500
            ))
        s = feedback_summary()
        assert s["best_niche"] == "finance"

    def test_niche_breakdown_in_summary(self, tmp_feedback_db):
        from src.ml.feedback import record_feedback, feedback_summary
        with patch("src.database.vector_store.get_vector_store",
                   side_effect=Exception("no store")):
            record_feedback(_make_feedback(video_id="v1", niche="finance"))
        s = feedback_summary()
        assert "finance" in s["niches"]


class TestAtomicWrite:
    def test_no_tmp_file_after_save(self, tmp_feedback_db):
        from src.ml.feedback import record_feedback
        with patch("src.database.vector_store.get_vector_store",
                   side_effect=Exception("no store")):
            record_feedback(_make_feedback())
        # The .tmp file should be cleaned up after atomic replace
        assert not (tmp_feedback_db.with_suffix(".tmp")).exists()
