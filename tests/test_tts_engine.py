"""Tests for tts_engine.py — punctuation-driven pause shaping for Kokoro."""

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
