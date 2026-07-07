"""Tests for research.py — topic-relevance filtering (SE + RSS) and the fix for
off-topic seeds slipping through general-interest sources (e.g. history.SE
surfacing a Viking-insult question for the money_history niche)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import research


# ── SE_TOPIC_FILTER_RE itself ─────────────────────────────────────────────────────

def test_money_history_topic_filter_matches_monetary_terms():
    rex = research.SE_TOPIC_FILTER_RE["money_history"]
    for phrase in ("How did the gold standard work?", "Roman coin debasement",
                   "Why did Weimar hyperinflation happen?", "medieval banking origins",
                   "what caused the tax revolt", "silver mining in Potosí"):
        assert rex.search(phrase), f"should match: {phrase}"


def test_money_history_topic_filter_rejects_the_actual_bug_case():
    rex = research.SE_TOPIC_FILTER_RE["money_history"]
    # The exact off-topic question that slipped through in production
    assert not rex.search("How would a 16-year-old girl from Cleopatra's era curse?")
    assert not rex.search("What did Vikings call cowards?")


def test_inherently_ontopic_niches_have_no_filter():
    # money.SE / workplace.SE are on-topic by definition — no filter needed
    assert "finance" not in research.SE_TOPIC_FILTER_RE
    assert "business" not in research.SE_TOPIC_FILTER_RE


# ── fetch_stackexchange_story applies the filter ─────────────────────────────────

def _se_response(items):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"items": items}
    return resp


def _se_item(title, body_words=120, score=100):
    return {
        "title": title,
        "body": "word " * body_words,   # clears SE_MIN_BODY_LEN
        "score": score,
        "link": f"https://history.stackexchange.com/q/{abs(hash(title))}",
    }


def test_fetch_stackexchange_rejects_offtopic_history_question():
    items = [_se_item("How would a 16-year-old girl from Cleopatra's era curse someone? "
                      "I saved this for my novel")]
    with patch.object(research.httpx, "get", return_value=_se_response(items)):
        result = research.fetch_stackexchange_story("money_history")
    assert result is None


def test_fetch_stackexchange_accepts_ontopic_history_question():
    items = [_se_item("How did Rome's silver coin debase over 200 years?")]
    with patch.object(research.httpx, "get", return_value=_se_response(items)):
        result = research.fetch_stackexchange_story("money_history")
    assert result is not None
    assert "coin" in result["title"].lower()


def test_fetch_stackexchange_finance_niche_unaffected_by_topic_filter():
    # money.SE has no topic filter — a plain personal-finance-shaped question
    # (passing TITLE_STORY_RE) must still pass with no topic_re applied.
    items = [_se_item("I saved $50,000 in two years working night shifts")]
    with patch.object(research.httpx, "get", return_value=_se_response(items)):
        result = research.fetch_stackexchange_story("finance")
    assert result is not None


# ── Wikipedia seed source (self-directed, grounded) ──────────────────────────────

def _wiki_response(extract, title="Nixon shock"):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"title": title, "extract": extract}
    return resp


def test_fetch_wikipedia_returns_grounded_seed(monkeypatch, tmp_path):
    topics = tmp_path / "wiki_topics.json"
    topics.write_text('{"money_history": ["Nixon shock"]}')
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)

    extract = ("The Nixon shock was a series of economic measures taken by "
               "United States President Richard Nixon in 1971, including the "
               "unilateral cancellation of the direct international "
               "convertibility of the United States dollar to gold. " * 2)
    with patch.object(research.httpx, "get", return_value=_wiki_response(extract)):
        seed = research.fetch_wikipedia_story("money_history")

    assert seed is not None
    assert seed["type"] == "wikipedia"
    assert "Nixon" in seed["content"]
    assert seed["url"] == "https://en.wikipedia.org/wiki/Nixon_shock"


def test_fetch_wikipedia_uses_wiki_headers_not_reddit_ua(monkeypatch, tmp_path):
    """Wikipedia's API rejects/deprioritizes browser-spoofed User-Agents (403)
    — regression test for the real 403 Forbidden bug seen in production. The
    Wikipedia call must use WIKI_HEADERS, never the Chrome-spoofing REDDIT_HEADERS."""
    topics = tmp_path / "wiki_topics.json"
    topics.write_text('{"money_history": ["Nixon shock"]}')
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)

    extract = "x" * (research.WIKI_MIN_EXTRACT + 10)
    with patch.object(research.httpx, "get", return_value=_wiki_response(extract)) as get:
        research.fetch_wikipedia_story("money_history")

    _, kwargs = get.call_args
    assert kwargs["headers"] == research.WIKI_HEADERS
    assert kwargs["headers"] != research.REDDIT_HEADERS
    assert "Chrome" not in kwargs["headers"]["User-Agent"]


def test_fetch_wikipedia_skips_used_topics(monkeypatch, tmp_path):
    topics = tmp_path / "wiki_topics.json"
    topics.write_text('{"money_history": ["Nixon shock"]}')
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)

    used = {"wiki:https://en.wikipedia.org/wiki/Nixon_shock"}
    with patch.object(research.httpx, "get") as get:
        seed = research.fetch_wikipedia_story("money_history", used_ids=used)
    assert seed is None
    get.assert_not_called()          # dedup happens before any network fetch


def test_fetch_wikipedia_rejects_short_extract(monkeypatch, tmp_path):
    topics = tmp_path / "wiki_topics.json"
    topics.write_text('{"money_history": ["Stub article"]}')
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)

    with patch.object(research.httpx, "get", return_value=_wiki_response("Too short.")):
        assert research.fetch_wikipedia_story("money_history") is None


def test_fetch_wikipedia_none_for_unlisted_niche(monkeypatch, tmp_path):
    topics = tmp_path / "wiki_topics.json"
    topics.write_text('{"money_history": ["Nixon shock"]}')
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)
    assert research.fetch_wikipedia_story("motivation") is None


def test_wikipedia_seed_id_uses_url():
    seed = {"type": "wikipedia", "url": "https://en.wikipedia.org/wiki/Denarius"}
    assert research._seed_id(seed) == "wiki:https://en.wikipedia.org/wiki/Denarius"


def test_wiki_topics_config_valid_and_substantial():
    import json as _json
    topics = _json.loads((Path(__file__).parent.parent / "config" / "wiki_topics.json").read_text())
    assert len(topics["money_history"]) >= 150   # enough runway that daily uploads
                                                  # don't exhaust fresh topics for months
    assert all(isinstance(t, str) and t for t in topics["money_history"])


def test_wiki_topics_no_duplicates():
    import json as _json
    topics = _json.loads((Path(__file__).parent.parent / "config" / "wiki_topics.json").read_text())
    money_history = topics["money_history"]
    assert len(money_history) == len(set(money_history))


# ── RUFUS_SKIP_REDDIT / get_seed source order ─────────────────────────────────────

def _niches_fixture(tmp_path, subreddits=None):
    niches = tmp_path / "niches.json"
    niches.write_text(json.dumps({
        "active": "money_history",
        "niches": {"money_history": {"subreddits": subreddits or ["badeconomics"]}},
    }))
    return niches


def test_skip_reddit_env_var_recognizes_truthy_values(monkeypatch):
    for val in ("1", "true", "True", "yes", "on"):
        monkeypatch.setenv("RUFUS_SKIP_REDDIT", val)
        assert research._skip_reddit() is True
    for val in ("0", "false", "", "off"):
        monkeypatch.setenv("RUFUS_SKIP_REDDIT", val)
        assert research._skip_reddit() is False


def test_get_seed_skips_reddit_when_flag_set(monkeypatch, tmp_path):
    monkeypatch.setenv("RUFUS_SKIP_REDDIT", "1")
    monkeypatch.setattr(research, "NICHES_FILE", _niches_fixture(tmp_path))
    monkeypatch.setattr(research, "USED_SEEDS_FILE", tmp_path / "used_seeds.json")

    se_seed = {"type": "stackexchange", "source": "history.SE",
               "title": "Why did Rome debase its coinage?", "content": "...", "url": "http://x"}
    with patch.object(research, "fetch_reddit_story") as reddit_mock, \
         patch.object(research, "fetch_stackexchange_story", return_value=se_seed), \
         patch.object(research, "_mark_seed_used"):
        seed = research.get_seed("money_history")

    reddit_mock.assert_not_called()
    assert seed["type"] == "stackexchange"


def test_get_seed_tries_reddit_when_flag_unset(monkeypatch, tmp_path):
    monkeypatch.delenv("RUFUS_SKIP_REDDIT", raising=False)
    monkeypatch.setattr(research, "NICHES_FILE", _niches_fixture(tmp_path))
    monkeypatch.setattr(research, "USED_SEEDS_FILE", tmp_path / "used_seeds.json")

    with patch.object(research, "fetch_reddit_story", return_value=None) as reddit_mock, \
         patch.object(research, "fetch_stackexchange_story", return_value=None), \
         patch.object(research, "fetch_wikipedia_story", return_value=None), \
         patch.object(research, "fetch_rss_story", return_value=None), \
         patch.object(research, "fetch_hackernews_story", return_value=None), \
         patch.object(research, "pick_wisdom_quote", return_value={"type": "wisdom", "source": "x"}):
        research.get_seed("money_history")

    reddit_mock.assert_called_once_with("badeconomics", used_ids=set())
