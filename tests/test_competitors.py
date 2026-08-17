"""Which of the neighbours' videos actually did something.

THE IDEA THE WHOLE MODULE RESTS ON: raw view counts are not evidence. Fifty
thousand views on a channel that averages two hundred thousand is a flop.
Twenty thousand on a channel that averages three thousand is the most
interesting thing that happened that week. Sorting other people's videos by
view count tells you which channels are big, which you already knew.

So everything is scored against its OWN channel's median, and the tests below
are mostly about that arithmetic being right — because it is the number that
decides what the scout reads, and it is checkable without a network.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import competitors  # noqa: E402


def _vids(channel: str, views: list[int], title_prefix: str = "v"):
    return [{"video_id": f"{channel}{i}", "channel_id": channel,
             "channel_title": channel.upper(), "title": f"{title_prefix}{i}",
             "published_at": "2026-08-01T00:00:00Z", "views": v,
             "duration": "PT2M"} for i, v in enumerate(views)]


# ── the ratio ────────────────────────────────────────────────────────────────

def test_a_video_is_scored_against_its_own_channel():
    """The whole point: 400 views on a channel that medians 100 is a 4x, and
    the same 400 on a channel that medians 800 is half a normal day."""
    # Flat baselines on purpose: with six mixed values the median is the mean
    # of the middle two, and a fixture whose expected number needs working out
    # is testing the arithmetic of the fixture.
    small = _vids("small", [100, 100, 100, 100, 100, 400])
    big = _vids("big", [800, 800, 800, 800, 800, 400])
    scored = {(v["channel_id"], v["title"]): v["outperformance"]
              for v in competitors.score(small + big)}
    assert scored[("small", "v5")] == 4.0
    assert scored[("big", "v5")] == 0.5


def test_a_channel_with_too_few_uploads_gets_no_ratio():
    """With two videos each is either the median or twice it, and both numbers
    are noise. A made-up baseline is worse than none, because the scout would
    act on it."""
    scored = competitors.score(_vids("tiny", [100, 900]))
    assert all(v["outperformance"] == 0.0 for v in scored)
    assert all(v["channel_median"] == 0 for v in scored)


def test_the_median_ignores_the_outlier_that_a_mean_would_not():
    """One viral video would drag a mean up and make every ordinary upload on
    that channel look like a failure."""
    scored = competitors.score(_vids("c", [100, 100, 100, 100, 100, 1_000_000]))
    ordinary = [v for v in scored if v["views"] == 100]
    assert all(v["outperformance"] == 1.0 for v in ordinary)


def test_outperformers_are_strongest_first_and_thresholded():
    scored = competitors.score(_vids("c", [100, 100, 100, 100, 100, 250, 900]))
    hits = competitors.outperformers(scored)
    assert [v["views"] for v in hits] == [900, 250]
    assert competitors.outperformers(scored, threshold=100.0) == []


def test_a_normal_week_is_not_a_finding():
    """A threshold that flags a third of every channel's uploads is describing
    variance, and this repo has walked back two checks that fired on most of
    what they looked at."""
    scored = competitors.score(_vids("c", [95, 100, 105, 98, 102, 110]))
    assert competitors.outperformers(scored) == []
    assert "none above" in competitors.describe(scored)


def test_scoring_an_empty_list_is_not_an_error():
    assert competitors.score([]) == []
    assert competitors.describe([]) == "nothing observed"


def test_a_video_with_no_views_does_not_break_the_median():
    scored = competitors.score(_vids("c", [0, 100, 100, 100, 100, 100]))
    assert all("outperformance" in v for v in scored)


# ── the config ───────────────────────────────────────────────────────────────

def test_a_missing_channel_list_says_what_to_do(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(competitors, "COMPETITORS_FILE", tmp_path / "nope.json")
    assert competitors.channels() == []
    assert "competitors.json.example" in capsys.readouterr().out


def test_an_unreadable_channel_list_is_not_an_exception(tmp_path, monkeypatch, capsys):
    """This runs inside a scheduled task. A raise here is a dead scout and a
    log nobody reads."""
    bad = tmp_path / "competitors.json"
    bad.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(competitors, "COMPETITORS_FILE", bad)
    assert competitors.channels() == []
    assert "unreadable" in capsys.readouterr().out


def test_an_empty_channel_list_says_so(tmp_path, monkeypatch, capsys):
    f = tmp_path / "competitors.json"
    f.write_text(json.dumps({"channels": []}), encoding="utf-8")
    monkeypatch.setattr(competitors, "COMPETITORS_FILE", f)
    assert competitors.channels() == []
    assert "no channels" in capsys.readouterr().out


def test_the_channel_list_is_read(tmp_path, monkeypatch):
    f = tmp_path / "competitors.json"
    f.write_text(json.dumps({"channels": ["UCaaa", " UCbbb ", ""]}),
                 encoding="utf-8")
    monkeypatch.setattr(competitors, "COMPETITORS_FILE", f)
    assert competitors.channels() == ["UCaaa", "UCbbb"]


def test_the_example_is_a_real_example():
    """A template that does not parse is one nobody can copy."""
    root = Path(__file__).parent.parent
    raw = json.loads((root / "config" / "competitors.json.example")
                     .read_text(encoding="utf-8"))
    assert isinstance(raw.get("channels"), list) and raw["channels"]


# ── the contract ─────────────────────────────────────────────────────────────

def test_no_youtube_access_is_an_empty_pass_not_a_crash(monkeypatch, capsys):
    monkeypatch.setattr(competitors, "channels", lambda: ["UCaaa"])
    monkeypatch.setattr(competitors, "_service", lambda: None)
    assert competitors.observe() == []


def test_a_channel_with_too_little_history_is_skipped(monkeypatch, capsys):
    """Reported per channel rather than silently: a watched channel producing
    nothing is something the owner chose and may want to un-choose."""
    monkeypatch.setattr(competitors, "channels", lambda: ["UCaaa"])
    monkeypatch.setattr(competitors, "_service", lambda: object())
    monkeypatch.setattr(competitors, "_uploads_playlist", lambda yt, c: "UU")
    monkeypatch.setattr(competitors, "_recent_video_ids",
                        lambda yt, p, n: ["a", "b"])
    monkeypatch.setattr(competitors, "_stats",
                        lambda yt, ids: _vids("UCaaa", [10, 20]))
    assert competitors.observe() == []
    assert "not enough for a baseline" in capsys.readouterr().out


def test_a_full_pass_scores_what_it_found(monkeypatch):
    monkeypatch.setattr(competitors, "channels", lambda: ["UCaaa"])
    monkeypatch.setattr(competitors, "_service", lambda: object())
    monkeypatch.setattr(competitors, "_uploads_playlist", lambda yt, c: "UU")
    monkeypatch.setattr(competitors, "_recent_video_ids",
                        lambda yt, p, n: list("abcdef"))
    monkeypatch.setattr(competitors, "_stats", lambda yt, ids:
                        _vids("UCaaa", [100, 100, 100, 100, 100, 500]))
    got = competitors.observe()
    assert len(got) == 6
    assert max(v["outperformance"] for v in got) == 5.0
