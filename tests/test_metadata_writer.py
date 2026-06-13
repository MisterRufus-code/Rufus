"""Tests for metadata_writer.py — GPT title/description with a hard legacy
fallback. The GPT path is exercised via a stubbed client; the fallback path
needs no network."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import metadata_writer as mw

SCRIPT = "Banks made $176B while you earned nothing.\nHere is the spread.\nSave this."
HOOK   = "Banks made $176B while you earned nothing."


# ── Legacy fallback (no key / API failure) ───────────────────────────────────────

def test_legacy_fallback_when_no_key(monkeypatch):
    monkeypatch.setattr(mw, "_load_key", lambda: "")
    meta = mw.generate_metadata(SCRIPT, "finance", {"cta": "Follow."}, ["#finance", "#Shorts"])
    assert meta["title"] == HOOK                 # first line, original behavior
    assert "#finance #Shorts" in meta["description"]
    assert meta["tags"] == ["finance", "Shorts"]


def test_legacy_fallback_on_api_exception(monkeypatch):
    monkeypatch.setattr(mw, "_load_key", lambda: "sk-test")

    class _Boom:
        def __init__(self, *a, **k): raise RuntimeError("network down")
    monkeypatch.setattr("openai.OpenAI", _Boom, raising=False)

    meta = mw.generate_metadata(SCRIPT, "finance", {"cta": "Follow."})
    assert meta["title"] == HOOK                 # fell back cleanly, no raise


# ── Title clamp + hook-duplication rejection ─────────────────────────────────────

def test_clamp_title_enforces_length():
    long = "This is a very very very very very very very long YouTube title indeed"
    out  = mw._clamp_title(long, HOOK)
    assert 0 < len(out) <= mw.TITLE_MAX


def test_clamp_title_rejects_hook_duplicate():
    # title that just repeats the spoken hook → "" signals caller to fall back
    assert mw._clamp_title(HOOK, HOOK) == ""


def test_clamp_title_keeps_distinct_title():
    distinct = "The bank spread nobody explains to savers"
    assert mw._clamp_title(distinct, HOOK) == distinct


# ── GPT path via stubbed client ──────────────────────────────────────────────────

class _FakeResp:
    def __init__(self, content):
        self.choices = [type("C", (), {"message": type("M", (), {"content": content})()})()]

class _FakeClient:
    def __init__(self, content): self._content = content
    @property
    def chat(self): return self
    @property
    def completions(self): return self
    def create(self, **kwargs): return _FakeResp(self._content)


def test_gpt_path_builds_description_and_tags(monkeypatch):
    monkeypatch.setattr(mw, "_load_key", lambda: "sk-test")
    payload = ('{"title": "The bank spread that quietly drains savers", '
               '"description_lead": "How banks profit on your deposits.", '
               '"tags": ["finance", "banking", "savings", "money"]}')
    monkeypatch.setattr("openai.OpenAI", lambda api_key=None: _FakeClient(payload), raising=False)

    meta = mw.generate_metadata(SCRIPT, "finance", {"cta": "Follow."}, ["#finance", "#Shorts"])
    assert meta["title"] == "The bank spread that quietly drains savers"
    assert meta["description"].startswith("How banks profit")
    assert SCRIPT in meta["description"]          # full script still embedded
    assert "finance" in meta["tags"] and len(meta["tags"]) <= 15


def test_gpt_hook_duplicate_falls_back(monkeypatch):
    monkeypatch.setattr(mw, "_load_key", lambda: "sk-test")
    payload = f'{{"title": "{HOOK}", "description_lead": "x", "tags": ["a"]}}'
    monkeypatch.setattr("openai.OpenAI", lambda api_key=None: _FakeClient(payload), raising=False)

    meta = mw.generate_metadata(SCRIPT, "finance", {"cta": "Follow."})
    assert meta["title"] == HOOK                 # clamp rejected dup → legacy (same text here)
    # legacy description has no GPT lead prefix
    assert meta["description"].startswith(SCRIPT)
