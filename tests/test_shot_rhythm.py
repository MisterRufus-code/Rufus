"""How long each picture stays, and where the cut lands.

THE REPORT: "the last video switched images too fast — maybe a little bit
less pictures or better timing, show the perfect scene at the perfect moment."

The measurement behind it, from that run's own cut list:

    gaps: 3.9 2.4 1.2 1.2 1.5 2.4 1.2 1.2 1.2 1.7 1.2 2.7 1.3 1.2 1.4 1.2
          2.0 1.2 1.2 1.2 1.6 1.2 1.2 1.4

Thirteen of twenty-four shots sat EXACTLY on MIN_SEG. That is not an edit, it
is a clamp: the planner had been handed more pictures than the narration had
pauses to put them on, and spent the remainder at the minimum.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import audio_gen  # noqa: E402
import emotional_map  # noqa: E402


def _gaps(cuts, total):
    marks = [0.0] + list(cuts) + [total]
    return [round(b - a, 2) for a, b in zip(marks, marks[1:])]


# ── the floor ───────────────────────────────────────────────────────────────

def test_a_shot_is_never_a_flash():
    """Below about 1.5s a picture reads as a flash rather than a shot."""
    assert audio_gen.MIN_SEG >= 1.5


def test_more_pictures_than_the_audio_can_hold_are_dropped():
    """Better to render fewer than to show them all too briefly to register.
    The beat count is decided from word count before the voice exists, so the
    finished audio can be shorter than that assumed."""
    assert audio_gen._max_shots(38.0) == int(38.0 // audio_gen.MIN_SEG)
    cuts = audio_gen._plan_cuts([], 10.0, 40)
    assert len(cuts) + 1 <= audio_gen._max_shots(10.0)
    assert all(g >= audio_gen.MIN_SEG - 0.01 for g in _gaps(cuts, 10.0))


def test_no_shot_lands_under_the_floor_on_a_dense_request():
    ends = [round(x * 0.7, 2) for x in range(2, 55)]
    cuts = audio_gen._plan_cuts(ends, 38.0, 22)
    assert all(g >= audio_gen.MIN_SEG - 0.01 for g in _gaps(cuts, 38.0)), \
        _gaps(cuts, 38.0)


# ── the rhythm ──────────────────────────────────────────────────────────────

def test_the_beat_that_carries_the_story_holds_longest():
    """An even grid gives the number, the turn and the closing line the same
    duration as "and then this happened", which a viewer reads as a slideshow
    however good the pictures are."""
    tones = ["tension", "neutral", "curiosity", "neutral",
             "revelation", "weight", "neutral", "resolution"]
    ends = [round(x * 0.7, 2) for x in range(2, 55)]
    gaps = _gaps(audio_gen._plan_cuts(ends, 38.0, len(tones), tones), 38.0)
    # gaps[i] is the shot for beat i.
    assert gaps[4] > gaps[3], "the revelation should outlast the beat before it"
    assert gaps[5] > gaps[6], "consequence should outlast plain narration"
    assert max(gaps[1:]) == gaps[4], gaps


def test_the_rhythm_is_uneven_on_purpose():
    tones = ["neutral", "neutral", "revelation", "neutral", "weight",
             "neutral", "resolution", "neutral"]
    ends = [round(x * 0.7, 2) for x in range(2, 55)]
    gaps = _gaps(audio_gen._plan_cuts(ends, 38.0, len(tones), tones), 38.0)
    assert max(gaps[1:]) - min(gaps[1:]) > 1.0, gaps


def test_no_tones_is_exactly_the_even_grid_it_replaced():
    """The weighting improves the rhythm; it is never required for having one."""
    even = audio_gen._tone_grid(2.0, 38.0, 6, None)
    spans = [round(b - a, 3) for a, b in zip([2.0] + even, even)]
    assert len(set(spans)) == 1, spans


def test_a_wrong_length_tone_list_is_ignored_rather_than_trusted():
    """tones comes from a separate plan; a mismatch means they describe
    different beats, and lining them up anyway would stretch the wrong ones."""
    assert audio_gen._tone_grid(2.0, 38.0, 6, ["revelation"]) == \
        audio_gen._tone_grid(2.0, 38.0, 6, None)


def test_every_tone_has_a_hold_weight():
    for tone in emotional_map.TONES:
        assert emotional_map.hold_weight(tone) > 0


def test_the_holds_stay_mild():
    """This is a fast-cut channel. The point is that the reveal breathes and
    the connective tissue does not, not that the edit lurches."""
    weights = [emotional_map.hold_weight(t) for t in emotional_map.TONES]
    assert max(weights) / min(weights) <= 2.0, weights


def test_the_renderer_passes_the_tones_it_already_planned():
    src = Path(audio_gen.__file__).read_text(encoding="utf-8")
    assert "_plan_cuts(_snap_points, audio_dur, n, plan_tones)" in src
