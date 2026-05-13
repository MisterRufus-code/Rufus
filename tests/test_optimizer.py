"""Tests for ML optimization layer — feedback analysis, prompt prefix, keyword context."""

from __future__ import annotations

from unittest.mock import patch, MagicMock

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
        video_id="v1", title="Test", niche="finance",
        topic="investing", prompt_model="mistral",
    )
    defaults.update(kwargs)
    return VideoFeedback(**defaults)


class TestAnalyseFeedback:
    def test_empty_db_returns_defaults(self, tmp_feedback_db):
        from src.ml.optimizer import analyse_feedback
        result = analyse_feedback()
        assert result.best_niches == []
        assert result.best_models == []
        assert "No feedback data" in result.suggestions[0]

    def test_single_record_no_crash(self, tmp_feedback_db):
        from src.ml.optimizer import analyse_feedback
        from src.ml.feedback import record_feedback
        with patch("src.database.vector_store.get_vector_store", side_effect=Exception()):
            record_feedback(_make_feedback(ctr=0.05, retention_rate=0.5, views=1000))
        result = analyse_feedback()
        assert result.best_niches[0][0] == "finance"

    def test_low_ctr_suggests_thumbnails(self, tmp_feedback_db):
        from src.ml.optimizer import analyse_feedback
        from src.ml.feedback import record_feedback
        for i in range(3):
            with patch("src.database.vector_store.get_vector_store", side_effect=Exception()):
                record_feedback(_make_feedback(video_id=f"v{i}", ctr=0.01))
        result = analyse_feedback()
        combined = " ".join(result.suggestions).lower()
        assert "ctr" in combined or "thumbnail" in combined

    def test_low_retention_suggests_shorter(self, tmp_feedback_db):
        from src.ml.optimizer import analyse_feedback
        from src.ml.feedback import record_feedback
        for i in range(3):
            with patch("src.database.vector_store.get_vector_store", side_effect=Exception()):
                record_feedback(_make_feedback(video_id=f"v{i}", retention_rate=0.2))
        result = analyse_feedback()
        combined = " ".join(result.suggestions).lower()
        assert "retention" in combined

    def test_multi_niche_best_niche_identified(self, tmp_feedback_db):
        from src.ml.optimizer import analyse_feedback
        from src.ml.feedback import record_feedback
        for i in range(3):
            with patch("src.database.vector_store.get_vector_store", side_effect=Exception()):
                record_feedback(_make_feedback(video_id=f"fin{i}", niche="finance",
                                               ctr=0.08, retention_rate=0.7, views=5000))
        for i in range(3):
            with patch("src.database.vector_store.get_vector_store", side_effect=Exception()):
                record_feedback(_make_feedback(video_id=f"tech{i}", niche="tech",
                                               ctr=0.02, retention_rate=0.3, views=500))
        result = analyse_feedback()
        assert result.best_niches[0][0] == "finance"

    def test_recommended_entropy_threshold_in_range(self, tmp_feedback_db):
        from src.ml.optimizer import analyse_feedback
        from src.ml.feedback import record_feedback
        with patch("src.database.vector_store.get_vector_store", side_effect=Exception()):
            record_feedback(_make_feedback(retention_rate=0.6, views=1000))
        result = analyse_feedback()
        assert 0.3 <= result.recommended_entropy_threshold <= 1.0


class TestGetOptimizedPromptPrefix:
    def test_empty_db_returns_default(self, tmp_feedback_db):
        from src.ml.optimizer import get_optimized_prompt_prefix
        result = get_optimized_prompt_prefix("finance")
        assert isinstance(result, str)
        assert len(result) > 10

    def test_with_data_includes_niche(self, tmp_feedback_db):
        from src.ml.optimizer import get_optimized_prompt_prefix
        from src.ml.feedback import record_feedback
        with patch("src.database.vector_store.get_vector_store", side_effect=Exception()):
            record_feedback(_make_feedback(ctr=0.07, retention_rate=0.65, views=2000))
        result = get_optimized_prompt_prefix("finance")
        assert "finance" in result.lower()

    def test_wrong_niche_returns_default(self, tmp_feedback_db):
        from src.ml.optimizer import get_optimized_prompt_prefix
        from src.ml.feedback import record_feedback
        with patch("src.database.vector_store.get_vector_store", side_effect=Exception()):
            record_feedback(_make_feedback(niche="tech"))
        result = get_optimized_prompt_prefix("finance")  # no finance data
        assert isinstance(result, str)


class TestGetKeywordContext:
    def test_empty_db_returns_empty_string(self, tmp_feedback_db):
        from src.ml.optimizer import get_keyword_context
        result = get_keyword_context("finance", "investing")
        assert result == ""

    def test_returns_string(self, tmp_feedback_db):
        from src.ml.optimizer import get_keyword_context
        result = get_keyword_context("finance", "investing")
        assert isinstance(result, str)
