"""Tests for audio_gen.py – timestamp formatting and word clustering."""

from types import SimpleNamespace

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
    seg = _segment([
        _word("hello", 0.0, 0.5),
        _word("world", 0.5, 1.0),
        _word("foo",   1.0, 1.5),
        _word("bar",   1.5, 2.0),
    ])
    clusters = list(_cluster_words([seg], audio_dur=5.0))
    assert len(clusters) == 2
    assert clusters[0] == (0.0, 1.0, "HELLO WORLD")
    assert clusters[1] == (1.0, 2.0, "FOO BAR")


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
    seg = _segment([
        _word("  hi  ",  0.0, 0.3),
        _word(" there ", 0.3, 0.6),
    ])
    clusters = list(_cluster_words([seg], audio_dur=2.0))
    assert clusters[0][2] == "HI THERE"
