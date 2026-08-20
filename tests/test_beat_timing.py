"""One beat, one picture, on the beat.

WHAT WAS WRONG. Short.tsx sized every shot identically:

    seqFrames = ceil((durationInFrames + (n-1)*transFrames) / n)

the runtime divided evenly by the number of clips. A beat whose sentence takes
two seconds and a beat whose sentence takes six got the same time on screen, so
the picture drifted further out of step with the narration the longer the video
ran — by the end it was illustrating a sentence that had already been said.

The FFmpeg renderer never had this: _plan_cuts snaps its cuts to sentence ends.
Both renderers ship the same channel and only one of them was cutting on the
voice — and the one that was not is the one this channel renders with.

These tests are mostly about the FAIL-OPEN being whole. A partial alignment
would put some pictures on the voice and leave the rest on the even grid, and a
video that is right for six shots and wrong for four reads worse than one that
is uniformly approximate.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import audio_gen  # noqa: E402


def _spoken(pairs):
    return [{"text": t, "start": s, "end": s + 0.3} for t, s in pairs]


WORDS = _spoken([
    ("The", 0.0), ("bank", 0.4), ("called.", 0.9),
    ("Nobody", 2.0), ("answered.", 2.6),
    ("It", 5.0), ("closed", 5.4), ("by", 5.9), ("Friday.", 6.2),
])
BEATS = ["The bank called.", "Nobody answered.", "It closed by Friday."]


# ── the spans ────────────────────────────────────────────────────────────────

def test_each_beat_starts_where_its_first_word_is_spoken():
    assert audio_gen.beat_spans(BEATS, WORDS, 8.0) == \
        [(0.0, 2.0), (2.0, 5.0), (5.0, 8.0)]


def test_the_spans_are_gap_free_and_never_overlap():
    """A gap is a frame of nothing; an overlap is two pictures claiming the
    same second. Either is visible."""
    spans = audio_gen.beat_spans(BEATS, WORDS, 8.0)
    for (s1, e1), (s2, _) in zip(spans, spans[1:]):
        assert e1 == s2, f"{e1} != {s2}"
        assert e1 > s1


def test_the_last_beat_runs_to_the_end_of_the_audio():
    """Anything else leaves the final sentence over a black frame or cuts it
    off mid-word."""
    assert audio_gen.beat_spans(BEATS, WORDS, 8.0)[-1][1] == 8.0


def test_a_long_sentence_gets_more_time_than_a_short_one():
    """The entire point. Under the even division these were identical.

    A fixture where the two happen to come out equal proves nothing, so this
    one is deliberately lopsided: two seconds of speech against six.
    """
    words = _spoken([("Short", 0.0), ("one.", 0.6),
                     ("A", 2.0), ("much", 3.0), ("longer", 4.5),
                     ("sentence", 6.0), ("here.", 7.0)])
    spans = audio_gen.beat_spans(["Short one.", "A much longer sentence here."],
                                 words, 8.0)
    assert spans == [(0.0, 2.0), (2.0, 8.0)]
    assert (spans[1][1] - spans[1][0]) == 3 * (spans[0][1] - spans[0][0])


def test_punctuation_and_case_do_not_break_the_match():
    """Whisper writes "called." and the script may have "called" — matching on
    raw text would fail on the first full stop in the video."""
    beats = ["the BANK called", "nobody answered", "it closed by friday"]
    assert len(audio_gen.beat_spans(beats, WORDS, 8.0)) == 3


def test_a_dropped_filler_word_is_tolerated():
    """Whisper loses the occasional short word. Giving up on the whole video
    for one of them would mean this never runs in practice."""
    words = _spoken([
        ("The", 0.0), ("bank", 0.4), ("called.", 0.9),
        ("Nobody", 2.0), ("answered.", 2.6),
        ("uh", 4.6), ("It", 5.0), ("closed", 5.4), ("by", 5.9), ("Friday.", 6.2),
    ])
    assert len(audio_gen.beat_spans(BEATS, words, 8.0)) == 3


# ── giving up, in one piece ──────────────────────────────────────────────────

def test_an_unfindable_beat_abandons_every_span(capsys):
    """Not just its own. Six shots on the voice and four on the grid is worse
    than ten that are uniformly approximate."""
    beats = ["The bank called.", "Something never said aloud.", "It closed by Friday."]
    assert audio_gen.beat_spans(beats, WORDS, 8.0) == []
    assert "falling back to even shot lengths" in capsys.readouterr().out


def test_a_zero_length_shot_is_refused(capsys):
    """A picture nobody sees, and a crossfade with nothing to fade from.

    Whisper does hand back words sharing a timestamp on very short syllables,
    so this is a transcript shape to survive rather than a hypothetical.
    """
    words = _spoken([("The", 0.0), ("bank", 0.0), ("The", 0.0), ("bank", 0.0)])
    assert audio_gen.beat_spans(["The bank", "The bank"], words, 4.0) == []
    assert "no duration in the transcript" in capsys.readouterr().out


@pytest.mark.parametrize("beats,words,dur", [
    ([], WORDS, 8.0),
    (BEATS, [], 8.0),
    (BEATS, WORDS, 0.0),
    ([""], WORDS, 8.0),
])
def test_nothing_to_align_is_an_empty_list_not_an_exception(beats, words, dur):
    assert audio_gen.beat_spans(beats, words, dur) == []


def test_a_transcript_of_only_punctuation_is_not_alignable():
    assert audio_gen.beat_spans(BEATS, _spoken([("—", 0.0), (".", 0.5)]), 8.0) == []


# ── the composition uses them ────────────────────────────────────────────────

def test_the_renderer_only_sends_spans_when_a_beat_is_a_clip():
    """With several stills to one beat the two lists do not correspond, and a
    span list that does not line up with the clips is worse than none."""
    src = (Path(__file__).parent.parent / "scripts" / "remotion_renderer.py") \
        .read_text(encoding="utf-8")
    assert "if beats and len(beats) == len(clip_names):" in src
    assert '"beatSpans":' in src


def test_the_composition_no_longer_divides_the_runtime_evenly():
    src = (Path(__file__).parent.parent / "remotion" / "src" / "Short.tsx") \
        .read_text(encoding="utf-8")
    assert "beatSpans" in src
    assert "framesFor(i)" in src, "the sequences still take a fixed length"
    assert "durationInFrames={framesFor(i)}" in src


def test_the_composition_still_has_the_even_division_to_fall_back_to():
    """Fail-open means the old behaviour has to still be there."""
    src = (Path(__file__).parent.parent / "remotion" / "src" / "Short.tsx") \
        .read_text(encoding="utf-8")
    assert "evenFrames" in src
    assert "?? evenFrames" in src


def test_the_composition_refuses_a_span_list_of_the_wrong_length():
    src = (Path(__file__).parent.parent / "remotion" / "src" / "Short.tsx") \
        .read_text(encoding="utf-8")
    assert "beatSpans.length !== n" in src
