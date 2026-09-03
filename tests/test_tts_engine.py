"""Tests for tts_engine.py — the pauses and the prosody a tone earns."""

from pathlib import Path

import pytest

from tts_engine import _pause_seconds


def test_pause_after_full_stop():
    assert _pause_seconds("The bank called.") == 0.26


def test_pause_after_question_or_exclaim():
    assert _pause_seconds("Then what happened?") == 0.30
    assert _pause_seconds("She lost everything!") == 0.30


def test_pause_after_dramatic_beat():
    assert _pause_seconds("Until the day it stopped—") == 0.32
    assert _pause_seconds("He waited...") == 0.32


def test_pause_after_comma_is_shortest():
    assert _pause_seconds("First, the market crashed,") == 0.14


def test_pause_trims_trailing_whitespace_before_checking_punctuation():
    assert _pause_seconds("The bank called.   \n") == 0.26


def test_pause_empty_chunk_defaults_light():
    assert _pause_seconds("") == 0.15


def test_pause_no_recognized_punctuation_defaults_light():
    assert _pause_seconds("no punctuation here") == 0.15


# ── a nine-minute script is longer than one request allows ──────────────────

def test_a_short_is_still_one_request():
    """A 40-second Short is ~650 characters and never came near the limit.
    Nothing about the existing channel may change."""
    import tts_engine
    assert len(tts_engine._paragraph_batches("A short script.", 4800)) == 1


def test_a_long_script_is_split_and_every_piece_fits():
    """~7,500 characters would have been rejected outright — the format change
    turning a working voice into a failed render, on the one stage with no
    fallback of its own."""
    import tts_engine
    script = "\n\n".join(f"Paragraph {i}. " + "word " * 120 for i in range(10))
    batches = tts_engine._paragraph_batches(script, tts_engine.ELEVEN_MAX_CHARS)
    assert len(batches) > 1
    assert all(len(b) <= tts_engine.ELEVEN_MAX_CHARS for b in batches)


def test_nothing_is_lost_in_the_split():
    import tts_engine
    script = "\n\n".join(f"Paragraph {i}." for i in range(40))
    batches = tts_engine._paragraph_batches(script, 200)
    rejoined = " ".join(" ".join(b.split()) for b in batches)
    assert rejoined == " ".join(script.split())


def test_it_cuts_on_paragraphs_and_never_mid_sentence():
    """A join at a sentence boundary is inaudible; a join mid-clause is a
    stutter the listener hears and cannot explain."""
    import tts_engine
    script = "\n\n".join("Sentence one. Sentence two." for _ in range(20))
    for b in tts_engine._paragraph_batches(script, 120):
        assert b.strip().endswith("."), b[-40:]


def test_an_oversized_single_paragraph_still_goes_somewhere():
    import tts_engine
    batches = tts_engine._paragraph_batches("x" * 1000, 200)
    assert batches and all(len(b) <= 200 for b in batches)


def test_the_long_writer_separates_sections_with_blank_lines():
    """Kokoro chunks on blank lines and breathes between chunks. A
    single-newline join hands it one 1,300-word block to read flat, with no
    pause at any of the seams the outline worked to create — and gives this
    splitter nothing to cut on either."""
    import inspect
    import longform_writer
    src = inspect.getsource(longform_writer.write)
    assert '"\\n\\n".join([_opening(outline)] + body)' in src


# ── the Edge backend says each beat in its own voice ─────────────────────────
#
# Edge is the DEFAULT backend and was the one branch handed no tone at all.
# `synthesize(script, out, tones)` carried them; `_kokoro` received them and
# spent them on silence; `_edge` was called with the whole script and one
# global rate. Six tones computed per run, one delivery.
#
# These tests do not touch the network. They record what would have been
# requested, because the thing worth pinning is which text was sent with which
# prosody — not what Microsoft's voice sounds like.

import tts_engine  # noqa: E402


@pytest.fixture
def edge_calls(monkeypatch, tmp_path):
    """Capture (text, prosody) per request instead of synthesizing."""
    calls = []

    async def _fake(script, out_path, prosody=None):
        calls.append((script, prosody))
        Path(out_path).write_bytes(b"ID3" + script.encode()[:8])

    monkeypatch.setattr(tts_engine, "_edge_async", _fake)
    monkeypatch.setattr(tts_engine, "_silence_mp3", lambda s, p: False)
    return calls


def test_an_all_neutral_script_is_still_one_request(edge_calls, tmp_path):
    """Per-beat costs a round trip per beat. tones_from_plan fails open to
    all-neutral, so the common case must not start paying for nothing."""
    tts_engine._edge("A. B. C.", tmp_path / "o.mp3",
                     ["neutral", "neutral"], ["A.", "B."])
    assert len(edge_calls) == 1
    assert edge_calls[0][0] == "A. B. C."


def test_no_tones_at_all_is_the_path_it_always_took(edge_calls, tmp_path):
    tts_engine._edge("A. B.", tmp_path / "o.mp3")
    assert len(edge_calls) == 1


def test_a_mixed_script_is_said_beat_by_beat(edge_calls, tmp_path):
    tts_engine._edge("A. B.", tmp_path / "o.mp3",
                     ["weight", "revelation"], ["A.", "B."])
    assert [c[0] for c in edge_calls] == ["A.", "B."]


def test_each_beat_carries_its_own_tone(edge_calls, tmp_path):
    """The whole point. A weight beat and a revelation beat in one video have
    to leave with different numbers on them."""
    tts_engine._edge("A. B.", tmp_path / "o.mp3",
                     ["weight", "revelation"], ["A.", "B."])
    weight, revelation = edge_calls[0][1], edge_calls[1][1]
    assert weight != revelation
    assert int(weight["pitch"].rstrip("Hz")) < int(revelation["pitch"].rstrip("Hz"))


def test_the_owners_rate_setting_survives_the_split(edge_calls, tmp_path, monkeypatch):
    """RUFUS_EDGE_RATE is +6% by default. Splitting into beats must not
    silently reset the voice to its unmodified speed."""
    monkeypatch.setattr(tts_engine, "EDGE_RATE", "+20%")
    tts_engine._edge("A. B.", tmp_path / "o.mp3",
                     ["neutral", "weight"], ["A.", "B."])
    assert edge_calls[0][1]["rate"] == "+20%"


@pytest.mark.parametrize("rate", ["fast", "", "20", None])
def test_an_unparseable_rate_setting_does_not_break_the_voice(rate, monkeypatch):
    """An owner who typed something odd into RUFUS_EDGE_RATE should get a
    working voice at the default speed, not a failed render."""
    monkeypatch.setattr(tts_engine, "EDGE_RATE", rate)
    assert isinstance(tts_engine._edge_base_rate_pct(), int)


def test_fewer_tones_than_beats_falls_back_rather_than_misaligning(edge_calls, tmp_path):
    """A tone landing on the wrong sentence is worse than no tone at all —
    the video would emphasise the line before the turn."""
    tts_engine._edge("A. B. C.", tmp_path / "o.mp3",
                     ["weight"], ["A.", "B.", "C."])
    assert len(edge_calls) == 1


def test_a_single_beat_is_not_worth_splitting(edge_calls, tmp_path):
    tts_engine._edge("A.", tmp_path / "o.mp3", ["weight"], ["A."])
    assert len(edge_calls) == 1


def test_an_empty_beat_is_skipped_not_sent(edge_calls, tmp_path):
    tts_engine._edge("A. B.", tmp_path / "o.mp3",
                     ["weight", "revelation", "tension"], ["A.", "  ", "B."])
    assert [c[0] for c in edge_calls] == ["A.", "B."]


def test_the_beats_are_joined_into_one_file(edge_calls, tmp_path):
    out = tmp_path / "o.mp3"
    tts_engine._edge("A. B.", out, ["weight", "revelation"], ["A.", "B."])
    assert out.exists() and out.stat().st_size > 0


def test_the_per_beat_temp_files_are_cleaned_up(edge_calls, tmp_path):
    out = tmp_path / "o.mp3"
    tts_engine._edge("A. B.", out, ["weight", "revelation"], ["A.", "B."])
    assert [p.name for p in tmp_path.iterdir()] == ["o.mp3"]
