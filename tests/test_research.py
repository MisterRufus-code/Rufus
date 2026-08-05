"""Tests for research.py — topic-relevance filtering (SE + RSS) and the fix for
off-topic seeds slipping through general-interest sources (e.g. history.SE
surfacing a Viking-insult question for the money_history niche)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

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


def test_fetch_stackexchange_accepts_academic_history_question_with_no_story_shape():
    """The actual live bug, not a coincidental pass: a realistic academic
    history.SE title with no dollar sign, no digit, and none of
    TITLE_STORY_RE's narrative verbs — on-topic, well-sourced, and exactly
    what this niche needs, but structurally unable to ever pass a filter that
    wants a first-person story shape. Symptom in production: 0 of 60 items
    passed, every single run — money.SE/workplace.SE's story-shape
    requirement doesn't fit an academic Q&A site at all."""
    items = [_se_item("Why did medieval European coin debasement violate usury laws?")]
    assert not research.TITLE_STORY_RE.search(items[0]["title"]), \
        "test setup: this title must NOT accidentally satisfy the old gate"
    with patch.object(research.httpx, "get", return_value=_se_response(items)):
        result = research.fetch_stackexchange_story("money_history")
    assert result is not None


def test_fetch_stackexchange_still_rejects_offtopic_even_with_story_shape():
    """topic_re is still the real gate for history.SE — a narrative-shaped
    but off-topic question (Vikings, not money) must still be rejected."""
    items = [_se_item("I lost my sword and $12 in the Viking raid last year")]
    with patch.object(research.httpx, "get", return_value=_se_response(items)):
        result = research.fetch_stackexchange_story("money_history")
    assert result is None


def test_fetch_stackexchange_money_niche_still_requires_story_shape():
    """The other half of the fix that must NOT change: money.SE/workplace.SE
    have no topic_re, so TITLE_STORY_RE stays the only quality gate there —
    an on-topic but non-narrative question must still be rejected."""
    items = [_se_item("What is the difference between a Roth and Traditional IRA?")]
    assert not research.TITLE_STORY_RE.search(items[0]["title"])
    with patch.object(research.httpx, "get", return_value=_se_response(items)):
        result = research.fetch_stackexchange_story("finance")
    assert result is None


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


# ── Manual topic injection (backlog item #6) ──────────────────────────────────

def _wiki_search_response(hits):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.json.return_value = {"query": {"search": hits}}
    return resp


def test_fetch_wikipedia_by_title_exact_match_fast_path(monkeypatch):
    extract = "Bretton Woods was a 1944 conference. " * 10
    # Stub the full-text fetch so this test counts only the RESOLUTION calls
    # (its actual subject: fast path vs search fallback), not the extra
    # grounding-corpus fetch every successful resolution now makes.
    monkeypatch.setattr(research, "fetch_wikipedia_fulltext", lambda t: "")
    with patch.object(research.httpx, "get",
                      return_value=_wiki_response(extract, title="Bretton Woods")) as get:
        seed = research.fetch_wikipedia_by_title("Bretton Woods")
    assert seed is not None
    assert seed["type"] == "wikipedia"
    assert seed["title"] == "Bretton Woods"
    assert "1944" in seed["content"]
    # exact-title fast path: only ONE call (the summary endpoint), no search fallback
    assert get.call_count == 1


def test_fetch_wikipedia_by_title_falls_back_to_search(monkeypatch):
    """An imprecise query ('bretton woods conference' lowercase, or a partial
    phrase) must not just fail — it should find the real article via search."""
    monkeypatch.setattr(research, "fetch_wikipedia_fulltext", lambda t: "")
    summary_404 = MagicMock()
    summary_404.raise_for_status.side_effect = Exception("404")
    search_hit = _wiki_search_response([{"title": "Bretton Woods Conference"}])
    good_summary = _wiki_response("The Bretton Woods Conference of 1944. " * 10,
                                  title="Bretton Woods Conference")

    with patch.object(research.httpx, "get",
                      side_effect=[summary_404, search_hit, good_summary]) as get:
        seed = research.fetch_wikipedia_by_title("bretton woods conference thing")

    assert seed is not None
    assert seed["title"] == "Bretton Woods Conference"
    assert get.call_count == 3   # failed direct attempt, search, then the resolved title


def test_fetch_wikipedia_by_title_returns_none_when_nothing_matches():
    summary_404 = MagicMock()
    summary_404.raise_for_status.side_effect = Exception("404")
    empty_search = _wiki_search_response([])
    with patch.object(research.httpx, "get",
                      side_effect=[summary_404, empty_search]):
        seed = research.fetch_wikipedia_by_title("complete gibberish query xyz123")
    assert seed is None


def test_fetch_wikipedia_by_title_empty_query_returns_none_no_network():
    with patch.object(research.httpx, "get") as get:
        seed = research.fetch_wikipedia_by_title("   ")
    assert seed is None
    get.assert_not_called()


def test_fetch_wikipedia_by_title_rejects_short_extract_then_tries_search():
    short = "Too short."
    search_hit = _wiki_search_response([{"title": "Real Article"}])
    good = _wiki_response("A real, long enough extract about the topic. " * 10,
                          title="Real Article")
    with patch.object(research.httpx, "get",
                      side_effect=[_wiki_response(short), search_hit, good]):
        seed = research.fetch_wikipedia_by_title("some query")
    assert seed is not None
    assert seed["title"] == "Real Article"


# ── Full-article grounding corpus ─────────────────────────────────────────────
# The REST summary endpoint returns only the lead paragraph — too thin to
# fact-check a 110-word script against, which capped run after run at 5/10
# with "not supported by the source material".

def _fulltext_response(body):
    r = MagicMock()
    r.raise_for_status.return_value = None
    r.json.return_value = {"query": {"pages": {"123": {"extract": body}}}}
    return r


def test_fetch_wikipedia_fulltext_returns_article_body():
    body = "Lead paragraph. " + ("Section body sentence. " * 50)
    with patch.object(research.httpx, "get", return_value=_fulltext_response(body)):
        got = research.fetch_wikipedia_fulltext("Some_Article")
    assert "Section body sentence." in got
    assert len(got) > 200


def test_fetch_wikipedia_fulltext_truncates_long_articles():
    body = "x. " * 20_000                      # way past the cap
    with patch.object(research.httpx, "get", return_value=_fulltext_response(body)):
        got = research.fetch_wikipedia_fulltext("Long")
    assert len(got) <= research.WIKI_FULLTEXT_CHARS


def test_fetch_wikipedia_fulltext_empty_on_failure():
    with patch.object(research.httpx, "get", side_effect=Exception("network")):
        assert research.fetch_wikipedia_fulltext("X") == ""


def test_wikipedia_story_prefers_fulltext_over_summary(monkeypatch, tmp_path):
    """The seed's content must be the real article body, not the lead-paragraph
    summary — that's the corpus the fact gate verifies the script against."""
    summary = "A short lead paragraph about the topic that clears the minimum. " * 4
    full    = "Much longer real article body with specifics. " * 40

    topics = tmp_path / "wiki_topics.json"
    topics.write_text(json.dumps({"money_history": ["Some Article"]}))
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)
    monkeypatch.setattr(research, "replenish_wiki_topics", lambda n: None)
    monkeypatch.setattr(research, "_unused_wiki_topic_count", lambda n, u: 99)
    monkeypatch.setattr(research, "fetch_wikipedia_fulltext", lambda t: full)

    with patch.object(research.httpx, "get", return_value=_wiki_response(summary)):
        seed = research.fetch_wikipedia_story("money_history", used_ids=set())

    assert seed is not None
    assert seed["content"] == full          # full body won, not the summary


def test_wikipedia_story_falls_back_to_summary_when_fulltext_fails(monkeypatch, tmp_path):
    summary = "A short lead paragraph about the topic that clears the minimum. " * 4
    topics = tmp_path / "wiki_topics.json"
    topics.write_text(json.dumps({"money_history": ["Some Article"]}))
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)
    monkeypatch.setattr(research, "replenish_wiki_topics", lambda n: None)
    monkeypatch.setattr(research, "_unused_wiki_topic_count", lambda n, u: 99)
    monkeypatch.setattr(research, "fetch_wikipedia_fulltext", lambda t: "")   # fetch failed

    with patch.object(research.httpx, "get", return_value=_wiki_response(summary)):
        seed = research.fetch_wikipedia_story("money_history", used_ids=set())

    assert seed is not None
    assert seed["content"].startswith("A short lead paragraph")   # summary kept


# ── Trend-driven topic selection ──────────────────────────────────────────────

def test_fetch_trending_wikipedia_resolves_rising_query(monkeypatch):
    """A rising Google Trends query is resolved to a real Wikipedia article, so
    the video's SUBJECT tracks demand — with the query recorded as provenance."""
    monkeypatch.setattr(research, "_trending_queries",
                        lambda n: ["gold standard collapse"])
    fake = {"type": "wikipedia", "source": "Wikipedia", "title": "Gold standard",
            "content": "x" * 300, "url": "https://en.wikipedia.org/wiki/Gold_standard"}
    monkeypatch.setattr(research, "fetch_wikipedia_by_title", lambda q: fake)

    seed = research.fetch_trending_wikipedia("money_history", used_ids=set())
    assert seed["title"] == "Gold standard"
    assert seed["trend_query"] == "gold standard collapse"


def test_fetch_trending_wikipedia_skips_used_and_tries_next(monkeypatch):
    used = {"wiki:https://en.wikipedia.org/wiki/Gold_standard"}
    seen = {"type": "wikipedia", "source": "Wikipedia", "title": "Gold standard",
            "content": "x" * 300, "url": "https://en.wikipedia.org/wiki/Gold_standard"}
    fresh = {"type": "wikipedia", "source": "Wikipedia", "title": "Hyperinflation",
             "content": "x" * 300, "url": "https://en.wikipedia.org/wiki/Hyperinflation"}
    monkeypatch.setattr(research, "_trending_queries", lambda n: ["gold", "hyperinflation"])
    monkeypatch.setattr(research, "fetch_wikipedia_by_title",
                        lambda q: seen if q == "gold" else fresh)

    seed = research.fetch_trending_wikipedia("money_history", used_ids=used)
    assert seed["title"] == "Hyperinflation"   # skipped the already-used one


def test_fetch_trending_wikipedia_none_when_no_trends(monkeypatch):
    monkeypatch.setattr(research, "_trending_queries", lambda n: [])
    assert research.fetch_trending_wikipedia("money_history", used_ids=set()) is None


def test_get_seed_prefers_trending_topic_when_enabled(monkeypatch):
    """With RUFUS_TREND_TOPICS on (default), a resolved trending topic is used
    BEFORE the Reddit/SE/random-Wikipedia chain."""
    monkeypatch.setenv("RUFUS_TREND_TOPICS", "1")
    monkeypatch.setattr(research, "_load_niche", lambda: ({"subreddits": ["x"]}, "money_history"))
    monkeypatch.setattr(research, "_load_used_seeds", lambda: [])
    monkeypatch.setattr(research, "get_trending_context", lambda n: None)
    monkeypatch.setattr(research, "_mark_seed_used", lambda s: None)
    trend_seed = {"type": "wikipedia", "source": "Wikipedia", "title": "Roman denarius",
                  "content": "x" * 300, "url": "https://en.wikipedia.org/wiki/Denarius",
                  "trend_query": "roman coins"}
    monkeypatch.setattr(research, "fetch_trending_wikipedia", lambda n, used_ids=None: trend_seed)
    # If the chain were reached it would blow up — assert it is NOT.
    monkeypatch.setattr(research, "fetch_reddit_story",
                        lambda *a, **k: (_ for _ in ()).throw(AssertionError("reached chain")))

    seed = research.get_seed(niche_name="money_history")
    assert seed["title"] == "Roman denarius"


def test_get_seed_trend_disabled_falls_to_chain(monkeypatch):
    """RUFUS_TREND_TOPICS=0 must skip the trend-first path entirely."""
    monkeypatch.setenv("RUFUS_TREND_TOPICS", "0")
    monkeypatch.setattr(research, "_load_niche", lambda: ({"subreddits": []}, "money_history"))
    monkeypatch.setattr(research, "_load_used_seeds", lambda: [])
    monkeypatch.setattr(research, "get_trending_context", lambda n: None)
    monkeypatch.setattr(research, "_mark_seed_used", lambda s: None)
    monkeypatch.setattr(research, "_skip_reddit", lambda: True)
    called = {"trend": False}
    def _trend(*a, **k):
        called["trend"] = True
        return {"type": "wikipedia", "title": "X", "content": "x" * 300, "url": "u"}
    monkeypatch.setattr(research, "fetch_trending_wikipedia", _trend)
    monkeypatch.setattr(research, "fetch_stackexchange_story", lambda *a, **k: None)
    wiki_seed = {"type": "wikipedia", "source": "Wikipedia", "title": "Chain topic",
                 "content": "x" * 300, "url": "https://en.wikipedia.org/wiki/Chain"}
    monkeypatch.setattr(research, "fetch_wikipedia_story", lambda *a, **k: wiki_seed)

    seed = research.get_seed(niche_name="money_history")
    assert called["trend"] is False        # trend path never invoked
    assert seed["title"] == "Chain topic"


def test_get_seed_with_topic_bypasses_auto_source_chain(monkeypatch, tmp_path):
    """--topic must not fall through Reddit/StackExchange/random-Wikipedia —
    it goes straight to the resolved topic, and marks it used like any seed."""
    monkeypatch.setattr(research, "NICHES_FILE", None, raising=False)

    def fake_load_niche():
        return {"subreddits": []}, "money_history"
    monkeypatch.setattr(research, "_load_niche", fake_load_niche)
    monkeypatch.setattr(research, "get_trending_context", lambda name: "")

    marked = []
    monkeypatch.setattr(research, "_mark_seed_used", lambda s: marked.append(s))

    fake_seed = {"type": "wikipedia", "source": "Wikipedia", "title": "Bretton Woods",
                "content": "x" * 300, "url": "https://en.wikipedia.org/wiki/Bretton_Woods"}
    monkeypatch.setattr(research, "fetch_wikipedia_by_title", lambda q: fake_seed)

    seed = research.get_seed(topic="Bretton Woods")
    assert seed["title"] == "Bretton Woods"
    assert marked == [fake_seed]


def test_get_seed_with_topic_raises_when_unresolvable(monkeypatch):
    monkeypatch.setattr(research, "fetch_wikipedia_by_title", lambda q: None)
    with pytest.raises(RuntimeError, match="No Wikipedia article found"):
        research.get_seed(niche_name="money_history", topic="complete gibberish xyz")


# ── Topic pool auto-replenish (prevents 5-videos/day from exhausting it) ──────

def test_unused_wiki_topic_count_excludes_used(monkeypatch, tmp_path):
    topics = tmp_path / "wiki_topics.json"
    topics.write_text('{"money_history": ["A", "B", "C"]}')
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)
    used = {"wiki:https://en.wikipedia.org/wiki/A"}
    assert research._unused_wiki_topic_count("money_history", used) == 2


def test_unused_wiki_topic_count_missing_niche_is_zero(monkeypatch, tmp_path):
    topics = tmp_path / "wiki_topics.json"
    topics.write_text('{"money_history": ["A"]}')
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)
    assert research._unused_wiki_topic_count("nonexistent_niche", set()) == 0


def test_replenish_noop_without_openai_key(monkeypatch, tmp_path):
    topics = tmp_path / "wiki_topics.json"
    topics.write_text('{"money_history": ["A"]}')
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)
    monkeypatch.setattr(research, "_load_keys", lambda: {})
    with patch.object(research.httpx, "get") as get:
        added = research.replenish_wiki_topics("money_history")
    assert added == 0
    get.assert_not_called()   # no key -> no network calls at all


def test_replenish_validates_before_trusting_gpt(monkeypatch, tmp_path):
    """GPT invents plausible-sounding titles that don't exist often enough to
    matter — only titles that actually resolve on Wikipedia get appended."""
    topics = tmp_path / "wiki_topics.json"
    topics.write_text('{"money_history": ["Existing Topic"]}')
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)
    monkeypatch.setattr(research, "_load_keys", lambda: {"openai": "sk-real"})

    class FakeResp:
        class choices:
            pass
    fake_msg = MagicMock()
    fake_msg.content = "Real Article\nFake Nonexistent Article\nAnother Real One"
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=fake_msg)]

    fake_openai_client = MagicMock()
    fake_openai_client.chat.completions.create.return_value = fake_completion

    def fake_wiki_get(url, **kwargs):
        resp = MagicMock()
        if "Fake_Nonexistent_Article" in url:
            resp.raise_for_status.side_effect = Exception("404")
        else:
            resp.raise_for_status = MagicMock()
            resp.json.return_value = {"extract": "A real long extract. " * 15}
        return resp

    with patch("openai.OpenAI", return_value=fake_openai_client), \
         patch.object(research.httpx, "get", side_effect=fake_wiki_get):
        added = research.replenish_wiki_topics("money_history", count=3)

    assert added == 2   # only the two real articles
    data = json.loads(topics.read_text())
    assert "Real Article" in data["money_history"]
    assert "Another Real One" in data["money_history"]
    assert "Fake Nonexistent Article" not in data["money_history"]
    assert "Existing Topic" in data["money_history"]   # original list preserved


def test_replenish_never_duplicates_existing_topics(monkeypatch, tmp_path):
    topics = tmp_path / "wiki_topics.json"
    topics.write_text('{"money_history": ["Already Here"]}')
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)
    monkeypatch.setattr(research, "_load_keys", lambda: {"openai": "sk-real"})

    fake_msg = MagicMock()
    fake_msg.content = "Already Here\nNew One"
    fake_completion = MagicMock()
    fake_completion.choices = [MagicMock(message=fake_msg)]
    fake_openai_client = MagicMock()
    fake_openai_client.chat.completions.create.return_value = fake_completion

    def fake_wiki_get(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"extract": "A real long extract. " * 15}
        return resp

    with patch("openai.OpenAI", return_value=fake_openai_client), \
         patch.object(research.httpx, "get", side_effect=fake_wiki_get):
        research.replenish_wiki_topics("money_history", count=2)

    data = json.loads(topics.read_text())
    assert data["money_history"].count("Already Here") == 1   # not duplicated


def test_replenish_returns_zero_and_does_not_crash_on_gpt_error(monkeypatch, tmp_path):
    topics = tmp_path / "wiki_topics.json"
    topics.write_text('{"money_history": ["A"]}')
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)
    monkeypatch.setattr(research, "_load_keys", lambda: {"openai": "sk-real"})

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("API down")
    with patch("openai.OpenAI", return_value=fake_client):
        added = research.replenish_wiki_topics("money_history")
    assert added == 0


def test_fetch_wikipedia_story_triggers_replenish_when_pool_low(monkeypatch, tmp_path):
    topics = tmp_path / "wiki_topics.json"
    topics.write_text('{"money_history": ["Nixon shock"]}')   # 1 topic < threshold
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)

    called = []
    monkeypatch.setattr(research, "replenish_wiki_topics", lambda n, count=40: called.append(n))
    extract = "x" * (research.WIKI_MIN_EXTRACT + 10)
    with patch.object(research.httpx, "get", return_value=_wiki_response(extract)):
        research.fetch_wikipedia_story("money_history")

    assert called == ["money_history"]


def test_fetch_wikipedia_story_skips_replenish_when_pool_healthy(monkeypatch, tmp_path):
    topics = tmp_path / "wiki_topics.json"
    many = [f"Topic {i}" for i in range(research.WIKI_REPLENISH_THRESHOLD + 10)]
    topics.write_text(json.dumps({"money_history": many}))
    monkeypatch.setattr(research, "WIKI_TOPICS_FILE", topics)

    called = []
    monkeypatch.setattr(research, "replenish_wiki_topics", lambda n, count=40: called.append(n))
    extract = "x" * (research.WIKI_MIN_EXTRACT + 10)
    with patch.object(research.httpx, "get", return_value=_wiki_response(extract)):
        research.fetch_wikipedia_story("money_history")

    assert called == []
