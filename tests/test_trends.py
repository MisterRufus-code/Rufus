"""Tests for trending topic fetching and selection."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestFetchTrendsRss:
    def test_returns_list_on_success(self):
        from src.trends import fetch_trends_rss
        fake_xml = """<?xml version="1.0" encoding="UTF-8"?>
        <rss><channel>
          <item><title>Bitcoin price</title></item>
          <item><title>Stock market crash</title></item>
        </channel></rss>"""
        mock_resp = MagicMock()
        mock_resp.text = fake_xml
        mock_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=mock_resp):
            result = fetch_trends_rss()
        assert "Bitcoin price" in result
        assert "Stock market crash" in result

    def test_returns_empty_on_network_error(self):
        from src.trends import fetch_trends_rss
        with patch("httpx.get", side_effect=Exception("timeout")):
            result = fetch_trends_rss()
        assert result == []

    def test_returns_empty_on_invalid_xml(self):
        from src.trends import fetch_trends_rss
        mock_resp = MagicMock()
        mock_resp.text = "this is not xml <<<"
        mock_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=mock_resp):
            result = fetch_trends_rss()
        assert result == []


class TestFetchRedditHot:
    def test_returns_post_titles(self):
        from src.trends import fetch_reddit_hot
        fake_json = {
            "data": {"children": [
                {"data": {"title": "How I saved $10k in one year", "score": 100}},
                {"data": {"title": "[Meta] moderator post", "score": 50}},  # should be skipped
                {"data": {"title": "Best passive income strategies 2024", "score": 200}},
            ]}
        }
        mock_resp = MagicMock()
        mock_resp.json.return_value = fake_json
        mock_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=mock_resp):
            result = fetch_reddit_hot("finance")
        assert "How I saved $10k in one year" in result
        assert not any(t.startswith("[") for t in result)

    def test_returns_empty_on_error(self):
        from src.trends import fetch_reddit_hot
        with patch("httpx.get", side_effect=Exception("network")):
            result = fetch_reddit_hot("finance")
        assert result == []


class TestSelectBestTopic:
    def test_returns_ollama_choice(self):
        from src.trends import select_best_topic
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"response": "How to invest in index funds"}
        mock_resp.raise_for_status.return_value = None
        with patch("httpx.post", return_value=mock_resp):
            result = select_best_topic(["topic A", "topic B"], niche="finance")
        assert result == "How to invest in index funds"

    def test_falls_back_to_first_topic_on_ollama_failure(self):
        from src.trends import select_best_topic
        with patch("httpx.post", side_effect=Exception("ollama down")):
            result = select_best_topic(["first topic", "second topic"], niche="finance")
        assert result == "first topic"

    def test_returns_fallback_on_empty_list(self):
        from src.trends import select_best_topic
        result = select_best_topic([], niche="finance")
        assert isinstance(result, str)
        assert len(result) > 0
