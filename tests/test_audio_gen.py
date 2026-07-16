"""Tests for audio_gen.py – timestamp formatting and word clustering."""

from types import SimpleNamespace

import pytest

from audio_gen import _cluster_words, _ts


def test_ts_formats_under_one_minute():
    assert _ts(3.45)  == "0:00:03.45"


def test_ts_formats_minutes():
    assert _ts(75.0)  == "0:01:15.00"


def test_ts_zero():
    assert _ts(0.0)   == "0:00:00.00"


def _word(text: str, start: float, end: float):
    return SimpleNamespace(word=text, start=start, end=end)


def _segment(words):
    return SimpleNamespace(words=words)


def test_cluster_words_basic_pairing():
    """CLUSTER_SIZE=1 → one word per subtitle line."""
    seg = _segment([
        _word("hello", 0.0, 0.5),
        _word("world", 0.5, 1.0),
        _word("foo",   1.0, 1.5),
        _word("bar",   1.5, 2.0),
    ])
    clusters = list(_cluster_words([seg], audio_dur=5.0))
    assert len(clusters) == 4
    assert clusters[0] == (0.0, 0.5, "HELLO")
    assert clusters[1] == (0.5, 1.0, "WORLD")
    assert clusters[2] == (1.0, 1.5, "FOO")
    # The FINAL caption now intentionally lingers up to CAPTION_GAP_FILL_S
    # past its word (holds the last caption instead of blanking early).
    import audio_gen as ag
    assert clusters[3][0] == 1.5
    assert clusters[3][1] == pytest.approx(min(2.0 + ag.CAPTION_GAP_FILL_S, 10.0))
    assert clusters[3][2] == "BAR"


def test_cluster_words_clips_to_audio_dur():
    """Clusters past audio_dur must be dropped (fix for the v2.1 bug)."""
    seg = _segment([
        _word("a", 0.0, 0.5),
        _word("b", 0.5, 1.0),
        _word("c", 1.5, 2.0),
        _word("d", 2.0, 5.0),   # extends WAY past audio_dur
    ])
    clusters = list(_cluster_words([seg], audio_dur=2.5))

    # Last cluster end must be clipped to audio_dur
    assert all(end <= 2.5 for _, end, _ in clusters)
    # No cluster should start past audio_dur
    assert all(start < 2.5 for start, _, _ in clusters)


def test_cluster_words_empty_input():
    assert list(_cluster_words([], audio_dur=10.0)) == []


def test_cluster_words_strips_whitespace_in_word_text():
    """Each word gets its own subtitle, stripped and uppercased."""
    seg = _segment([
        _word("  hi  ",  0.0, 0.3),
        _word(" there ", 0.3, 0.6),
    ])
    clusters = list(_cluster_words([seg], audio_dur=2.0))
    assert clusters[0][2] == "HI"
    assert clusters[1][2] == "THERE"


# ── Caption precision fixes (fresh-eyes audit) ────────────────────────────────

def test_ts_rolls_over_seconds_correctly():
    """59.998 % 60 formatted as %05.2f produced '0:00:60.00' — an invalid ASS
    timestamp. Integer centisecond math must roll it into the next minute."""
    import audio_gen as ag
    assert ag._ts(59.998) == "0:01:00.00"
    assert ag._ts(0) == "0:00:00.00"
    assert ag._ts(61.5) == "0:01:01.50"
    assert ag._ts(3600.0) == "1:00:00.00"


def _fake_segments(word_times):
    """[(word, start, end), ...] → whisper-like segments structure."""
    import types
    words = [types.SimpleNamespace(word=w, start=s, end=e) for w, s, e in word_times]
    return [types.SimpleNamespace(words=words)]


def test_cluster_words_bridges_gaps_no_strobing():
    """Whisper ends each word the instant sound stops — captions displayed for
    exactly that window strobe (blank screen at every inter-word pause). Each
    caption must extend to the next word's start (capped), never overlapping."""
    import audio_gen as ag
    segs = _fake_segments([("Money", 0.0, 0.3), ("never", 0.9, 1.2), ("sleeps", 1.3, 1.6)])
    events = list(ag._cluster_words(segs, audio_dur=10.0))

    assert events[0][1] == pytest.approx(0.9)   # extended to next word's start
    assert events[1][1] == pytest.approx(1.3)
    # no overlap: each end <= next start
    assert events[0][1] <= events[1][0]


def test_cluster_words_gap_fill_is_capped():
    # A long silence must not keep a stale caption on screen forever.
    import audio_gen as ag
    segs = _fake_segments([("Wait", 0.0, 0.3), ("for", 5.0, 5.2), ("it", 5.3, 5.5)])
    events = list(ag._cluster_words(segs, audio_dur=10.0))
    assert events[0][1] == pytest.approx(0.3 + ag.CAPTION_GAP_FILL_S)


def test_cluster_words_enforces_minimum_duration():
    import audio_gen as ag
    segs = _fake_segments([("a", 1.0, 1.01), ("blink", 3.0, 3.4)])
    events = list(ag._cluster_words(segs, audio_dur=10.0))
    assert events[0][1] - events[0][0] >= ag.CAPTION_MIN_S


def test_sentence_ends_skips_abbreviations():
    """'the U.S. dollar' was counted as a sentence end — scene cuts and whoosh
    SFX landed mid-sentence. Abbreviation periods must not count."""
    import audio_gen as ag
    segs = _fake_segments([
        ("The", 0.0, 0.2), ("U.S.", 0.3, 0.6), ("dollar", 0.7, 1.0),
        ("collapsed.", 1.1, 1.5),
        ("Mr.", 2.0, 2.2), ("Ponzi", 2.3, 2.6), ("smiled.", 2.7, 3.0),
    ])
    ends = ag._sentence_ends(segs)
    assert 0.6 not in ends      # U.S. is not a sentence end
    assert 2.2 not in ends      # Mr. is not a sentence end
    assert 1.5 in ends
    assert 3.0 in ends
