"""Historic newspapers, and the disambiguation page that shouldn't have shipped.

Reddit has needed OAuth for months — five of six seed sources blocked on every
run — so what actually feeds this channel is Stack Exchange when it answers and
Wikipedia when it doesn't. Wikipedia returns an OVERVIEW of a subject, and an
overview is the one shape the pipeline cannot use: "Sterling was the
fourth-most-traded currency in 2022" has no date, no street and nobody standing
in it. That seed produced "The secret? Its historical resilience and trust."

The same run also shows the failure this file's second half is about:

    [research] using Wikipedia article: "Potosi"
    → Quote: "Potosí or Potosi may refer to the following topics, whose names
      generally origin" — Wikipedia

A disambiguation page is a list of links. Nothing on it can be fact-checked and
nothing on it is a moment.

Tests mock at the HTTP boundary — no network, per AGENTS.md.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import research  # noqa: E402


# ── the OCR legibility gate ──────────────────────────────────────────────────

_CLEAN = ("The failure of the Philadelphia and Reading Railroad was announced "
          "yesterday afternoon and the news spread rapidly through the "
          "financial district. ") * 10
_SOUP = "tlie ii 0f rn iiii tbe 1 8 .. ,, tlie rn ii 0f " * 60


def test_a_readable_column_passes():
    assert research._ocr_is_legible(_CLEAN)


def test_microfilm_soup_is_rejected():
    """19th-century OCR fails as many short non-words, not few long ones. A
    rejected page costs one HTTP call; an accepted bad one costs a video."""
    assert not research._ocr_is_legible(_SOUP)


def test_a_page_too_short_to_hold_a_story_is_rejected():
    assert not research._ocr_is_legible("Bank failure. See page four.")


def test_empty_and_none_are_safe():
    assert not research._ocr_is_legible("")
    assert not research._ocr_is_legible(None)


# ── the window: which part of a six-column page we send ──────────────────────

def test_the_window_lands_on_the_story_not_the_masthead():
    page = ("MASTHEAD ads ads " * 40) + " a run on the bank began at noon " + ("tail " * 300)
    win = research._ocr_window(page, "run on the bank")
    assert "run on the bank began at noon" in win
    assert len(win) <= research.NEWS_WINDOW_CHARS


def test_the_window_anchors_on_the_longest_query_word():
    """Searching from "run" lands on "running"/"drunk"/"runaway" long before
    the story; "bank" is the token that marks it."""
    page = ("running drunk runaway " * 60) + " BANK DIRECTORS MET " + ("x " * 400)
    assert "BANK DIRECTORS MET" in research._ocr_window(page, "run on the bank")


def test_a_missing_query_still_returns_the_top_of_the_page():
    """Fail-open: no hit means the caller gets something, not nothing."""
    assert research._ocr_window("a b c " * 200, "zzzzzz").startswith("a b c")


# ── date and place, the two things this source exists for ────────────────────

def test_the_packed_date_becomes_readable():
    date, place = research._news_date_place(
        {"date": "18930220", "city": ["Philadelphia"], "state": ["Pennsylvania"]})
    assert date == "1893-02-20"
    assert place == "Philadelphia, Pennsylvania"


def test_missing_place_fields_do_not_raise():
    assert research._news_date_place({}) == ("", "")


def test_scalar_city_and_state_are_accepted_too():
    """The API returns lists; a scalar must not crash the source."""
    assert research._news_date_place(
        {"date": "1893", "city": "Philadelphia", "state": "Pennsylvania"}) == (
            "1893", "Philadelphia, Pennsylvania")


# ── the fetch itself ─────────────────────────────────────────────────────────

class _Resp:
    def __init__(self, payload):
        self._p = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._p


def _item(**over):
    base = {"id": "/lccn/sn83025121/1893-02-21/ed-1/seq-1/",
            "date": "18930221", "title": "The Philadelphia Record",
            "city": ["Philadelphia"], "state": ["Pennsylvania"],
            "ocr_eng": _CLEAN}
    base.update(over)
    return base


def test_a_good_page_becomes_a_seed_with_its_date_and_town(monkeypatch):
    monkeypatch.setattr(research.httpx, "get",
                        lambda *a, **k: _Resp({"items": [_item()]}))
    seed = research.fetch_newspaper_story("money_history", used_ids=set())
    assert seed["type"] == "newspaper"
    assert "1893-02-21" in seed["content"]
    assert "Philadelphia" in seed["content"]
    assert seed["url"].startswith("https://chroniclingamerica.loc.gov/lccn/")


def test_an_unreadable_page_is_skipped_not_returned(monkeypatch):
    monkeypatch.setattr(research.httpx, "get",
                        lambda *a, **k: _Resp({"items": [_item(ocr_eng=_SOUP)]}))
    assert research.fetch_newspaper_story("money_history", used_ids=set()) is None


def test_an_already_used_page_is_skipped(monkeypatch):
    monkeypatch.setattr(research.httpx, "get",
                        lambda *a, **k: _Resp({"items": [_item()]}))
    used = {"news:https://chroniclingamerica.loc.gov"
            "/lccn/sn83025121/1893-02-21/ed-1/seq-1/"}
    assert research.fetch_newspaper_story("money_history", used_ids=used) is None


def test_a_dead_endpoint_is_loud_and_non_fatal(monkeypatch, capsys):
    """The LoC retired one host already. "newspapers unavailable" alone would
    read as a network blip for months — the line has to name the cause."""
    def _boom(*a, **k):
        raise OSError("connection refused")
    monkeypatch.setattr(research.httpx, "get", _boom)
    assert research.fetch_newspaper_story("money_history", used_ids=set()) is None
    out = capsys.readouterr().out
    assert "LoC endpoint moved" in out
    assert "RUFUS_NEWSPAPERS=0" in out


def test_a_niche_with_no_newspaper_vocabulary_is_a_silent_noop(monkeypatch):
    def _boom(*a, **k):
        raise AssertionError("must not make a request")
    monkeypatch.setattr(research.httpx, "get", _boom)
    assert research.fetch_newspaper_story("stoicism", used_ids=set()) is None


def test_the_source_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("RUFUS_NEWSPAPERS", "0")
    def _boom(*a, **k):
        raise AssertionError("must not make a request")
    monkeypatch.setattr(research.httpx, "get", _boom)
    assert research.fetch_newspaper_story("money_history", used_ids=set()) is None


def test_the_query_moves_between_runs(monkeypatch):
    """A fixed query meeting a fixed filter is what made StackExchange dead for
    months. Same mistake, same file — do not repeat it."""
    seen = set()
    monkeypatch.setattr(research.httpx, "get",
                        lambda url, *a, **k: seen.add(url) or _Resp({"items": []}))
    for _ in range(40):
        research.fetch_newspaper_story("money_history", used_ids=set())
    assert len(seen) > 1


def test_the_seed_gets_a_stable_id_so_it_is_never_reused():
    seed = {"type": "newspaper", "url": "https://x/lccn/1", "title": "t"}
    assert research._seed_id(seed) == "news:https://x/lccn/1"


# ── the disambiguation page that shipped ─────────────────────────────────────

def test_the_potosi_page_is_recognised():
    assert research._is_disambiguation(
        {}, "Potosí or Potosi may refer to the following topics, whose names "
            "generally originate from the city in Bolivia.")


def test_the_explicit_api_type_is_recognised():
    assert research._is_disambiguation({"type": "disambiguation"}, "anything")


def test_a_real_article_is_untouched():
    assert not research._is_disambiguation(
        {"type": "standard"},
        "The Panic of 1893 was an economic depression in the United States "
        "that began in February 1893 and lasted until 1897.")


@pytest.mark.parametrize("head", [
    "Sterling may refer to several currencies.",
    "Mint may also refer to the plant.",
])
def test_both_phrasings_are_caught(head):
    assert research._is_disambiguation({}, head)
