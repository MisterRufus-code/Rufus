"""Tests for audio_gen.py – timestamp formatting and word clustering."""

import os
from pathlib import Path
from types import SimpleNamespace

import audio_gen

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
    """'the U.S. dollar' was counted as a sentence end — scene cuts and cut
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


# ── Cut-sound gain ───────────────────────────────────────────────────────────

def test_bubble_gain_default_is_audible_but_small(monkeypatch):
    """The whoosh this replaces sat at 0.02 after five rounds of feedback,
    which was right for a sheet of filtered noise on every cut. A bubble is one
    short rounded tone, and at 0.02 it would not be there at all — the owner
    asked for the sound, not for a placeholder. It still plays on every cut
    (23x in a 38-second video), so it stays well under hit and riser."""
    import importlib
    import audio_gen as ag
    monkeypatch.delenv("RUFUS_BUBBLE_GAIN", raising=False)
    monkeypatch.delenv("RUFUS_WHOOSH_GAIN", raising=False)
    importlib.reload(ag)
    assert ag.SFX_BUBBLE_GAIN == pytest.approx(0.05)
    importlib.reload(ag)


def test_bubble_gain_env_override(monkeypatch):
    import importlib
    import audio_gen as ag
    monkeypatch.setenv("RUFUS_BUBBLE_GAIN", "0")
    importlib.reload(ag)
    try:
        assert ag.SFX_BUBBLE_GAIN == 0.0
    finally:
        monkeypatch.delenv("RUFUS_BUBBLE_GAIN", raising=False)
        importlib.reload(ag)


def test_the_old_whoosh_variable_still_works_and_says_so(monkeypatch, capsys):
    """Someone who tuned RUFUS_WHOOSH_GAIN is tuning THIS layer. Ignoring
    their setting silently would be the worst of both — so it is honoured, and
    announced."""
    import importlib
    import audio_gen as ag
    monkeypatch.delenv("RUFUS_BUBBLE_GAIN", raising=False)
    monkeypatch.setenv("RUFUS_WHOOSH_GAIN", "0.11")
    try:
        importlib.reload(ag)
        assert ag.SFX_BUBBLE_GAIN == pytest.approx(0.11)
        assert "the whoosh is gone" in capsys.readouterr().out
    finally:
        monkeypatch.delenv("RUFUS_WHOOSH_GAIN", raising=False)
        importlib.reload(ag)


# ── Whisper CPU load: offline retry (live "Server disconnected" failure) ──────

def test_whisper_cpu_load_retries_from_local_cache(monkeypatch):
    """Constructing WhisperModel re-checks HuggingFace; a transient network
    drop killed a whole render even though the model was cached. The CPU
    loader must retry with local_files_only=True."""
    import audio_gen as ag

    calls = []

    class FakeModel:
        def __init__(self, name, device=None, compute_type=None,
                     local_files_only=False):
            calls.append(local_files_only)
            if not local_files_only:
                raise RuntimeError("Server disconnected without sending a response.")

    monkeypatch.setattr(ag, "WhisperModel", FakeModel)
    model = ag._load_whisper_cpu("small")
    assert isinstance(model, FakeModel)
    assert calls == [False, True]      # online attempt, then cached-only retry


def test_whisper_cpu_load_no_retry_when_first_attempt_works(monkeypatch):
    import audio_gen as ag

    calls = []

    class FakeModel:
        def __init__(self, name, device=None, compute_type=None,
                     local_files_only=False):
            calls.append(local_files_only)

    monkeypatch.setattr(ag, "WhisperModel", FakeModel)
    ag._load_whisper_cpu("small")
    assert calls == [False]


# ── a nine-minute filtergraph does not fit on a command line ─────────────────

def test_a_shorts_graph_still_goes_on_the_command_line(tmp_path):
    """Byte-identical to what ships today: same argument, same graph. A
    24-picture Short's graph is about 6,000 characters and has never come
    close to a limit."""
    args = audio_gen._filter_complex_args("[0:v]null[vout]", tmp_path / "o.mp4")
    assert args[0] == "-filter_complex"
    assert args[1] == "[0:v]null[vout]"


def test_a_long_graph_moves_into_a_file(tmp_path):
    """Windows' CreateProcess takes 32,767 characters for the WHOLE command
    line, and the graph is one argument. A 150-picture render's graph measures
    about 44,000 — the render would have died at the last step of the night,
    after every image was drawn and the voice recorded, with an error about a
    filename being too long that names nothing in the pipeline."""
    out = tmp_path / "video.mp4"
    graph = "x" * (audio_gen._FILTER_ARG_MAX + 1)
    args = audio_gen._filter_complex_args(graph, out)
    assert args[0] == "-filter_complex_script"
    written = Path(args[1])
    assert written.read_text(encoding="utf-8") == graph
    assert written.name.startswith("video.mp4"), "kept beside the render it made"


def test_the_graph_file_survives_the_render_for_diagnosis(tmp_path):
    """When a render fails the graph IS the diagnosis, and forty thousand
    characters is not something anyone reads out of a traceback."""
    out = tmp_path / "video.mp4"
    audio_gen._filter_complex_args("y" * 9000, out)
    assert (tmp_path / "video.mp4.filtergraph.txt").exists()


def test_an_unwritable_directory_does_not_fail_the_render(tmp_path, capsys):
    """Fail-open: the long command line may still work on this platform, and
    dying here would fail a render that could have run."""
    args = audio_gen._filter_complex_args(
        "z" * 9000, tmp_path / "nope" / "deep" / "video.mp4")
    assert args[0] == "-filter_complex"
    assert "could not write the filtergraph" in capsys.readouterr().out


def test_the_real_long_form_graph_is_over_the_line(monkeypatch):
    """The measurement this is all based on, kept as a test so the number is
    not a memory: 150 clips at nine minutes."""
    monkeypatch.setenv("RUFUS_FORMAT", "long")
    import importlib
    importlib.reload(audio_gen)
    n, dur = 150, 540.0
    bounds = [round(i * dur / n, 3) for i in range(1, n)]
    lens = audio_gen._xfade_input_lengths(bounds, dur)
    fc = audio_gen._video_filter_complex(
        lens, bounds, dur, 2112, 1188, 54, "eq=contrast=1.1", "a.ass",
        "fonts", "#FFC53D", grades=["eq=contrast=1.1"] * n)
    assert len(fc) > 32_767, len(fc)
    assert len(fc) > audio_gen._FILTER_ARG_MAX


def test_a_command_near_the_windows_ceiling_says_so(capsys):
    cmd = ["ffmpeg"] + ["-i", "x" * 200] * 200
    size = audio_gen._warn_if_command_is_too_long(cmd)
    assert size > audio_gen._CMDLINE_WARN
    out = capsys.readouterr().out
    assert "32767" in out and "SD_CLIPS" in out


def test_an_ordinary_command_says_nothing(capsys):
    audio_gen._warn_if_command_is_too_long(["ffmpeg", "-i", "a.mp4", "out.mp4"])
    assert capsys.readouterr().out == ""


# ── the hook read a person already chose ────────────────────────────────────

def test_the_chosen_hook_tone_overrides_beat_zero(monkeypatch):
    """/voice records the opening line in three tones and one is picked;
    RUFUS_HOOK_TONE carries that answer into the render."""
    import emotional_map
    monkeypatch.setenv("RUFUS_HOOK_TONE", "tension")
    tones = ["neutral", "curiosity", "weight"]
    picked = emotional_map.normalise(os.environ["RUFUS_HOOK_TONE"])
    tones[0] = picked
    assert tones == ["tension", "curiosity", "weight"]


def test_an_unknown_hook_tone_falls_back_to_neutral():
    """normalise is the guard: a tone name that is not in the map must not
    reach the grade filters, where an unknown key is a KeyError mid-render."""
    import emotional_map
    assert emotional_map.normalise("shouty") == emotional_map.NEUTRAL
    assert emotional_map.normalise("tension") == "tension"
