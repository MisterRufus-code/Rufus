"""Tests for psychology hook scoring engine."""

from __future__ import annotations

import pytest


class TestScoreHook:
    def test_empty_hook_scores_low(self):
        from src.psychology.hooks import score_hook
        score = score_hook("")
        assert score.total < 3.0

    def test_loss_aversion_hook_scores_higher(self):
        from src.psychology.hooks import score_hook
        weak = score_hook("This video is about money")
        strong = score_hook("Stop making this money mistake — you're losing thousands")
        assert strong.total > weak.total

    def test_open_loop_detected(self):
        from src.psychology.hooks import score_hook
        s = score_hook("What most people don't know about investing will shock you")
        assert s.breakdown.get("open_loop", 0) > 0

    def test_specificity_numbers_detected(self):
        from src.psychology.hooks import score_hook
        s = score_hook("5 investing habits that made me $50,000 in 30 days")
        assert s.breakdown.get("specificity", 0) > 0

    def test_second_person_detected(self):
        from src.psychology.hooks import score_hook
        s = score_hook("You are wasting your money on these 3 mistakes")
        assert s.breakdown.get("second_person", 0) > 0

    def test_max_score_is_ten(self):
        from src.psychology.hooks import score_hook
        perfect = score_hook(
            "Stop wasting your money — 5 secrets most people never discover about investing"
        )
        assert perfect.total <= 10.0

    def test_short_hook_penalised(self):
        from src.psychology.hooks import score_hook
        s = score_hook("Stop now")
        assert "short" in " ".join(s.suggestions).lower() or s.total < 5.0

    def test_suggestions_generated_for_missing_principles(self):
        from src.psychology.hooks import score_hook
        s = score_hook("A video about money")
        assert len(s.suggestions) > 0

    def test_score_hooks_sorted_best_first(self):
        from src.psychology.hooks import score_hooks
        hooks = [
            "A video about stuff",
            "Stop wasting your money — 5 secrets most people never discover",
            "Interesting facts about finance",
        ]
        scored = score_hooks(hooks)
        assert scored[0].total >= scored[1].total >= scored[2].total


class TestBestHook:
    def test_best_hook_returns_string(self, monkeypatch):
        from src.psychology import hooks
        # Patch Ollama call — return pre-written hooks
        import httpx
        from unittest.mock import MagicMock, patch

        fake_hooks = "\n".join([
            f"{i+1}. Stop making this money mistake — you're losing thousands" if i == 0
            else f"{i+1}. A basic finance tip number {i}"
            for i in range(10)
        ])
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": fake_hooks}
        mock_response.raise_for_status.return_value = None

        with patch("src.task_lock.task_lock"), \
             patch("httpx.post", return_value=mock_response):
            result = hooks.best_hook("investing", "finance")

        assert isinstance(result, str)
        assert len(result) > 5
