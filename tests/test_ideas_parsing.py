"""Tests for LLM output parsing — the most failure-prone part of the system."""

from __future__ import annotations

import pytest


class TestCleanJsonStr:
    def test_fixes_unescaped_newlines_in_strings(self):
        from src.ideas import _clean_json_str
        import json
        # Simulate what Ollama actually returns — a literal newline embedded in a JSON value
        raw = '{"key": "line one\nline two"}'
        cleaned = _clean_json_str(raw)
        # After cleaning it must be parseable as valid JSON
        result = json.loads(cleaned)
        assert "key" in result

    def test_valid_json_unchanged_structure(self):
        from src.ideas import _clean_json_str
        import json
        raw = '{"title": "Hello World", "score": 5}'
        result = json.loads(_clean_json_str(raw))
        assert result["title"] == "Hello World"


class TestParseJsonResponse:
    def test_plain_json_object(self):
        from src.ideas import _parse_json_response
        raw = '{"title": "Test", "hook": "A hook"}'
        result = _parse_json_response(raw)
        assert result["title"] == "Test"

    def test_json_in_markdown_fence(self):
        from src.ideas import _parse_json_response
        raw = '```json\n{"title": "Test", "hook": "Hook"}\n```'
        result = _parse_json_response(raw)
        assert result["title"] == "Test"

    def test_json_array(self):
        from src.ideas import _parse_json_response
        raw = '[{"title": "A"}, {"title": "B"}]'
        result = _parse_json_response(raw)
        assert isinstance(result, list)
        assert result[0]["title"] == "A"

    def test_json_buried_in_prose(self):
        from src.ideas import _parse_json_response
        raw = 'Here is the result:\n{"title": "My Video", "hook": "Watch this"}\nEnd.'
        result = _parse_json_response(raw)
        assert result.get("title") == "My Video"

    def test_malformed_raises_or_returns_empty(self):
        from src.ideas import _parse_json_response
        raw = "This is not JSON at all."
        # The function raises ValueError on completely unparseable input — that's acceptable
        try:
            result = _parse_json_response(raw)
            assert isinstance(result, (dict, list))
        except ValueError:
            pass  # expected behaviour for fully unparseable LLM output


class TestGenerateSearchKeywords:
    def test_returns_list_of_strings(self, monkeypatch):
        from src import ideas
        monkeypatch.setattr(ideas, "_ollama_generate",
                            lambda *a, **kw: '["keyword one", "keyword two", "keyword three"]')
        monkeypatch.setattr(ideas, "_check_ollama", lambda: True)
        result = ideas.generate_search_keywords("investing", "finance", count=3)
        assert isinstance(result, list)
        assert all(isinstance(k, str) for k in result)

    def test_falls_back_on_exception(self, monkeypatch):
        from src import ideas
        monkeypatch.setattr(ideas, "_ollama_generate", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("Ollama down")))
        monkeypatch.setattr(ideas, "_check_ollama", lambda: False)
        result = ideas.generate_search_keywords("investing", "finance", count=3)
        assert isinstance(result, list)
        assert len(result) >= 1
