"""Tests for subtitle SRT generation and overlay logic."""

from __future__ import annotations

import pytest


class TestBuildSrt:
    def test_basic_sections_generate_srt(self):
        from src.pipeline.subtitles import build_srt
        sections = [{"script": "First section content here."}]
        result = build_srt(sections, total_duration=10.0)
        assert "1" in result
        assert "00:00:00" in result

    def test_empty_sections_returns_empty_string(self):
        from src.pipeline.subtitles import build_srt
        result = build_srt([], total_duration=10.0)
        assert result == ""

    def test_whitespace_only_sections_skipped(self):
        from src.pipeline.subtitles import build_srt
        sections = [{"script": "   "}, {"script": "\n\t"}, {"script": "Real content"}]
        result = build_srt(sections, total_duration=10.0)
        assert "Real content" in result
        # Only one real segment — no ZeroDivisionError
        assert result != ""

    def test_zero_duration_returns_empty(self):
        from src.pipeline.subtitles import build_srt
        result = build_srt([{"script": "content"}], total_duration=0.0)
        assert result == ""

    def test_hook_and_cta_included(self):
        from src.pipeline.subtitles import build_srt
        result = build_srt(
            [{"script": "Main body"}],
            total_duration=15.0,
            hook="Opening hook here",
            call_to_action="Subscribe now",
        )
        assert "Opening hook here" in result
        assert "Subscribe now" in result

    def test_sections_evenly_distributed(self):
        from src.pipeline.subtitles import build_srt
        sections = [{"script": f"Section {i}"} for i in range(4)]
        result = build_srt(sections, total_duration=20.0)
        # 4 sections over 20s = 5s each — check timestamps present
        assert "00:00:05" in result or "00:00:04" in result  # approximate


class TestSecondsToSrtTime:
    def test_zero(self):
        from src.pipeline.subtitles import _seconds_to_srt_time
        assert _seconds_to_srt_time(0) == "00:00:00,000"

    def test_one_minute_thirty(self):
        from src.pipeline.subtitles import _seconds_to_srt_time
        assert _seconds_to_srt_time(90.5) == "00:01:30,500"

    def test_one_hour(self):
        from src.pipeline.subtitles import _seconds_to_srt_time
        assert _seconds_to_srt_time(3600) == "01:00:00,000"
