"""The StackExchange source could never succeed, and said so quietly.

EVERY log the owner has sent — across months of runs — carries the same line:

    [research] SE history: 60 items, none passed quality filter

That is not intermittent bad luck, and reading it as bad luck is what let it
run for months. The request was

    /2.3/questions?order=desc&sort=votes&site=history&filter=withbody&pagesize=60

with no query and no page: the site's all-time top 60, identical on every call,
forever. For history.SE those are the famous questions — wars, empires, daily
life — and almost none are about money. A fixed input meeting a fixed filter
gives a fixed result, so the source cost one HTTP call per run and could not
ever return a seed.

It matters more than one source: with Reddit also failing on every run
("Expecting value: line 2 column 5"), Wikipedia was carrying the channel alone
against a blacklist that had already grown past 100 burned seeds.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import research  # noqa: E402


def test_the_request_asks_for_the_topic_not_the_sites_greatest_hits():
    url, described = research._se_url("history", "money_history")
    assert "search/advanced" in url
    assert "q=" in url
    assert any(q.split()[0] in url for q in research.SE_TOPIC_QUERIES["money_history"])
    assert described.startswith('"')


def test_the_request_moves_between_runs():
    """A deterministic query is what made 'none passed' permanent."""
    seen = {research._se_url("history", "money_history")[0] for _ in range(60)}
    assert len(seen) > 1, "same URL every run — the original bug"


def test_pages_stay_inside_the_configured_depth():
    for _ in range(60):
        url, _d = research._se_url("history", "money_history")
        page = int(re.search(r"page=(\d+)", url).group(1))
        assert 1 <= page <= research.SE_MAX_PAGE


def test_an_inherently_on_topic_site_needs_no_query_but_still_rotates():
    """money.SE questions are all about money already — but re-reading the same
    top 60 for a year is its own dead end."""
    urls = {research._se_url("money", "finance")[0] for _ in range(60)}
    assert all("search/advanced" not in u for u in urls)
    assert len(urls) > 1


def test_every_niche_with_an_se_site_resolves_to_a_url():
    for niche, site in research.SE_NICHE_SITES.items():
        if not site:
            continue
        url, described = research._se_url(site, niche)
        assert url.startswith("https://api.stackexchange.com/")
        assert f"site={site}" in url and described


def _items(n, **over):
    base = {"score": 99, "title": "Why did Roman denarii lose silver?",
            "body": "<p>" + "money and trade. " * 40 + "</p>",
            "link": "https://history.stackexchange.com/q/1"}
    return [dict(base, link=f"{base['link']}{i}", **over) for i in range(n)]


def _run(monkeypatch, items):
    class R:
        status_code = 200
        @staticmethod
        def raise_for_status(): pass
        @staticmethod
        def json(): return {"items": items}
    monkeypatch.setattr(research.httpx, "get", lambda *a, **kw: R())
    return research.fetch_stackexchange_story("money_history", set())


def test_an_empty_result_names_the_reason(monkeypatch, capsys):
    """"none passed quality filter" looks like bad luck. "58 offtopic" is a
    dead source — and the difference between those two readings is the whole
    bug."""
    _run(monkeypatch, _items(5, title="Who won the Battle of Hastings?",
                             body="<p>" + "swords and horses. " * 40 + "</p>"))
    out = capsys.readouterr().out
    assert "none usable" in out
    assert "offtopic" in out


def test_a_short_body_is_reported_as_a_length_rejection(monkeypatch, capsys):
    _run(monkeypatch, _items(3, body="<p>too short</p>"))
    assert "length" in capsys.readouterr().out


def test_a_usable_batch_says_how_many_survived(monkeypatch, capsys):
    seed = _run(monkeypatch, _items(4))
    assert seed and seed["type"] == "stackexchange"
    assert "usable" in capsys.readouterr().out


# ── Reddit was not "unreachable" — it was unauthenticated ───────────────────

def test_the_reddit_advice_is_printed_once_per_run_not_per_subreddit(capsys):
    research._reddit_warned = False
    for _ in range(5):                      # five subreddits, as in every log
        research._warn_reddit_unauthenticated()
    out = capsys.readouterr().out
    assert out.count("prefs/apps") == 1


def test_the_advice_names_the_two_keys_and_the_stake(capsys):
    research._reddit_warned = False
    research._warn_reddit_unauthenticated()
    out = capsys.readouterr().out
    assert "reddit_client_id" in out and "reddit_client_secret" in out
    assert "Wikipedia is the only one left" in out


def test_the_log_no_longer_calls_a_permanent_block_unreachable():
    src = (Path(__file__).parent.parent / "scripts" / "research.py").read_text(
        encoding="utf-8")
    assert "reddit unreachable for" not in src
    assert "reddit blocked r/" in src


# ── The threshold has to match the population it filters ─────────────────────

def test_a_topical_query_gets_a_lower_bar_than_the_top_list():
    """Rotating to a topical search fixed the frozen input and immediately
    exposed the threshold: 'SE history ["banking", page 1]: 46 items, none
    usable (45 score, 1 length)'. 40+ is ordinary on an all-time-top list and
    unreachable in a topical search."""
    import research

    assert research._se_min_score("money_history") == research.SE_MIN_SCORE_TOPICAL
    assert research.SE_MIN_SCORE_TOPICAL < research.SE_MIN_SCORE


def test_sites_without_a_topical_query_keep_the_original_bar():
    """money.SE / workplace.SE still ask for the top list, where 40+ is the
    right bar — lowering it there would let weak questions through."""
    import research

    assert research._se_min_score("finance") == research.SE_MIN_SCORE
    assert research._se_min_score("business") == research.SE_MIN_SCORE


def test_an_unknown_niche_gets_the_conservative_bar():
    import research

    assert research._se_min_score("does_not_exist") == research.SE_MIN_SCORE


def test_the_threshold_choice_follows_the_query_shape():
    """One function decides both, so they cannot drift apart — the bug was
    exactly a query and a threshold that no longer described each other."""
    import research

    for niche in research.SE_NICHE_SITES:
        topical = bool(research.SE_TOPIC_QUERIES.get(niche))
        expected = (research.SE_MIN_SCORE_TOPICAL if topical
                    else research.SE_MIN_SCORE)
        assert research._se_min_score(niche) == expected
