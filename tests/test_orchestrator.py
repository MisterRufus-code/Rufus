"""Tests for pipeline orchestrator — path sanitization, pre-flight guard."""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


class TestPathSanitization:
    def test_special_chars_stripped_from_topic(self):
        """Safe topic must not contain filesystem-dangerous characters."""
        dangerous = "../../etc/passwd; rm -rf /"
        safe = re.sub(r"[^\w\s-]", "", dangerous).strip().replace(" ", "_")[:40]
        assert ".." not in safe
        assert "/" not in safe
        assert ";" not in safe

    def test_topic_with_normal_chars_preserved(self):
        topic = "How to invest in stocks"
        safe = re.sub(r"[^\w\s-]", "", topic).strip().replace(" ", "_")[:40]
        assert "invest" in safe
        assert "stocks" in safe

    def test_empty_topic_becomes_video(self):
        topic = "!@#$%^&*()"
        safe = re.sub(r"[^\w\s-]", "", topic).strip().replace(" ", "_")[:40] or "video"
        assert safe == "video"

    def test_long_topic_truncated_to_40_chars(self):
        topic = "a" * 100
        safe = re.sub(r"[^\w\s-]", "", topic).strip().replace(" ", "_")[:40]
        assert len(safe) <= 40

    def test_unicode_topic_handled(self):
        topic = "投资理财 — personal finance 🚀"
        safe = re.sub(r"[^\w\s-]", "", topic).strip().replace(" ", "_")[:40]
        # Must not raise; must be a usable directory name
        assert isinstance(safe, str)
        assert len(safe) <= 40


class TestPreflightGuard:
    def test_pipeline_aborts_on_preflight_failure(self, tmp_path):
        from src.pipeline.orchestrator import run_pipeline
        with patch("src.preflight.run_preflight", side_effect=SystemExit(1)):
            result = run_pipeline(
                topic="test",
                output_dir=tmp_path,
                download_footage=False,
            )
        assert result.success is False
        assert any("Pre-flight" in e for e in result.errors)


class TestPipelineResult:
    def test_result_has_required_fields(self):
        from src.pipeline.orchestrator import PipelineResult
        r = PipelineResult(topic="test", niche="finance")
        assert r.topic == "test"
        assert r.success is False
        assert isinstance(r.errors, list)
        assert isinstance(r.asset_ids_used, list)
        assert r.entropy_score == 0.0
