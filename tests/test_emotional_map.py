"""One per-beat tone, read by every stage that decides how a beat feels.

Before this, three systems each decided that independently and none could see
the other two: storyboard picked the shot, edit_director picked the camera
move, audio_gen graded the whole video with a single global `ffmpeg_eq` from
niches.json. Three opinions about one video that never compared notes.

The load-bearing property is that none of it can break a render. Every path
through here degrades to NEUTRAL, which grades to the niche's own base look —
i.e. exactly the video the pipeline shipped before this module existed.
"""

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import emotional_map as em


# ------------------------------------------------------------ the vocabulary

def test_every_tone_grades_and_weights():
    for tone in em.TONES:
        assert isinstance(em.grade_filter(tone), str)
        assert em.grade_filter(tone)
        assert em.sfx_weight(tone) > 0


def test_neutral_is_the_identity():
    """A neutral beat must reproduce the niche's own look untouched — that is
    what makes 'no tones came back' indistinguishable from before."""
    g = em.grade_filter("neutral", 1.1, 1.0)
    assert "contrast=1.100" in g
    assert "saturation=1.000" in g
    assert "brightness=0.000" in g
    assert "gamma=1.000" in g


def test_neutral_emits_no_colorbalance_node():
    """A zero-shift colorbalance still costs a pass over every frame."""
    assert "colorbalance" not in em.grade_filter("neutral")
    assert "colorbalance" in em.grade_filter("tension")


def test_tones_are_actually_distinguishable():
    """Vocabulary without render differences is vocabulary without meaning."""
    grades = {em.grade_filter(t) for t in em.TONES}
    assert len(grades) == len(em.TONES)


def test_the_emotional_axis_points_the_right_way():
    """Cold/hard for tension, warm/saturated for revelation — if these ever
    invert, every video is graded backwards and nothing else would catch it."""
    def field(tone, name):
        return float(re.search(rf"\b{name}=(-?\d+\.\d+)", em.grade_filter(tone)).group(1))

    assert field("tension", "saturation") < field("neutral", "saturation")
    assert field("tension", "contrast")   > field("neutral", "contrast")
    assert field("revelation", "saturation") > field("neutral", "saturation")
    assert field("weight", "brightness")  < field("neutral", "brightness")
    assert field("resolution", "contrast") < field("neutral", "contrast")

    # blue up / red down is cold; the reverse is warm
    assert "rm=-" in em.grade_filter("tension")
    assert "rm=0" in em.grade_filter("revelation")


def test_sfx_weight_follows_the_tone():
    assert em.sfx_weight("revelation") > em.sfx_weight("neutral")
    assert em.sfx_weight("resolution") < em.sfx_weight("neutral")
    assert em.sfx_weight("neutral") == 1.0


# ------------------------------------------------------------- fail-open

@pytest.mark.parametrize("junk", [
    None, "", "  ", 42, [], {}, "REVELATION!!", "mystery_building", object(),
])
def test_anything_unrecognised_becomes_neutral(junk):
    assert em.normalise(junk) == "neutral"
    assert em.grade_filter(junk) == em.grade_filter("neutral")
    assert em.sfx_weight(junk) == 1.0


def test_case_and_whitespace_are_forgiven():
    assert em.normalise("  Revelation ") == "revelation"


@pytest.mark.parametrize("plan", [
    None, {}, {"beats": None}, {"beats": "nope"}, {"beats": [1, 2, 3]},
    {"beats": [{"motion": "push_in"}]}, "not a dict",
])
def test_a_broken_plan_grades_every_beat_neutral(plan):
    assert em.tones_from_plan(plan, 4) == ["neutral"] * 4


def test_a_short_plan_still_covers_every_beat():
    plan = {"beats": [{"tone": "tension"}, {"tone": "revelation"}]}
    assert em.tones_from_plan(plan, 5) == [
        "tension", "revelation", "neutral", "neutral", "neutral"]


def test_a_long_plan_is_truncated_not_crashed():
    plan = {"beats": [{"tone": "tension"}] * 9}
    assert em.tones_from_plan(plan, 2) == ["tension", "tension"]


def test_zero_beats_is_not_an_error():
    assert em.tones_from_plan({"beats": [{"tone": "tension"}]}, 0) == []


# ------------------------------------------------- clamping a hostile niche

def test_an_extreme_niche_grade_cannot_crush_the_picture():
    """A niche shipping ffmpeg_eq=contrast=1.4 plus a tension beat must not
    multiply into an unwatchable frame."""
    g = em.grade_filter("tension", base_contrast=3.0, base_saturation=3.0)
    contrast   = float(re.search(r"contrast=(\d+\.\d+)", g).group(1))
    saturation = float(re.search(r"saturation=(\d+\.\d+)", g).group(1))
    assert contrast <= 1.60
    assert saturation <= 1.80


def test_a_zero_saturation_niche_stays_valid():
    g = em.grade_filter("revelation", base_contrast=0.1, base_saturation=0.0)
    assert float(re.search(r"contrast=(\d+\.\d+)", g).group(1)) >= 0.70
    assert float(re.search(r"saturation=(\d+\.\d+)", g).group(1)) >= 0.30


def test_describe_says_when_nothing_came_back():
    assert "grading unchanged" in em.describe(["neutral", "neutral"])
    assert "grading unchanged" not in em.describe(["tension", "neutral"])
    assert em.describe([]) == "no beats"


# ------------------------------------------------- wired into the ffmpeg path

def test_the_grade_lands_in_the_clip_filter():
    import audio_gen

    grade = em.grade_filter("tension")
    part  = audio_gen._ken_burns_part(0, 5.0, 1188, 2112, 96, grade)
    assert grade in part
    # order matters: grade after the crop, before format conversion
    assert part.index("crop=") < part.index("eq=") < part.index("format=yuv420p")


def test_no_grade_reproduces_the_old_filter_exactly():
    """The default argument is the regression guarantee for every caller that
    does not pass grades."""
    import audio_gen

    assert (audio_gen._ken_burns_part(1, 5.0, 1188, 2112, 96)
            == audio_gen._ken_burns_part(1, 5.0, 1188, 2112, 96, ""))


def test_each_clip_gets_its_own_grade():
    import audio_gen

    grades = [em.grade_filter("tension"), em.grade_filter("revelation")]
    fc = audio_gen._video_filter_complex_concat(
        [3.0, 3.0], 6.0, 1188, 2112, 96,
        "eq=contrast=1.1:saturation=1.0", "sub.ass", "fonts", "#FFCC00", grades)
    assert grades[0] in fc
    assert grades[1] in fc


def test_missing_grades_do_not_break_the_graph():
    """Fewer grades than clips must not IndexError mid-render."""
    import audio_gen

    fc = audio_gen._video_filter_complex_concat(
        [3.0, 3.0, 3.0], 9.0, 1188, 2112, 96,
        "eq=contrast=1.1:saturation=1.0", "sub.ass", "fonts", "#FFCC00",
        [em.grade_filter("tension")])
    assert "[v2]" in fc


def test_base_eq_is_parsed_from_the_niche():
    import audio_gen

    assert audio_gen._parse_base_eq("eq=contrast=1.25:saturation=0.9") == (1.25, 0.9)
    assert audio_gen._parse_base_eq("") == (1.0, 1.0)
    assert audio_gen._parse_base_eq("hue=h=90") == (1.0, 1.0)


# ------------------------------------------------------- the director's tone

def test_the_director_asks_for_a_tone():
    import edit_director

    prompt = edit_director._prompt(["a beat", "another beat"])
    assert "TONE" in prompt
    for tone in em.TONES:
        assert tone in prompt


def test_an_unknown_tone_degrades_but_a_bad_motion_still_rejects():
    """Tone is cosmetic, motion is executable — refusing a whole edit plan over
    a tone would trade a working feature for a new one."""
    import edit_director

    plan = {"peak_beat": 1, "beats": [
        {"n": 1, "motion": "push_in", "intensity": "normal",
         "tone": "mystery_building", "emphasis": []}]}
    cleaned = edit_director._clean(plan, 1)
    assert cleaned is not None
    assert cleaned["beats"][0]["tone"] == "neutral"

    plan["beats"][0]["motion"] = "orbit"
    assert edit_director._clean(plan, 1) is None


def test_a_valid_tone_survives_cleaning():
    import edit_director

    plan = {"peak_beat": 1, "beats": [
        {"n": 1, "motion": "hold_still", "intensity": "strong",
         "tone": "revelation", "emphasis": ["1494"]}]}
    assert edit_director._clean(plan, 1)["beats"][0]["tone"] == "revelation"


def test_the_plan_is_memoised_so_both_renderers_see_one_edit(monkeypatch):
    """Remotion asks, fails, FFmpeg asks again — inside one run. Two calls at
    temperature 0.7 would grade the video against a different edit than the one
    the director chose."""
    import edit_director

    monkeypatch.setattr(edit_director, "_plan_cache", {})
    calls = []

    def fake(beats):
        calls.append(tuple(beats))
        return {"peak_beat": 1, "beats": [
            {"n": 1, "motion": "push_in", "intensity": "normal",
             "tone": "tension", "emphasis": []}]}

    monkeypatch.setattr(edit_director, "_direct_uncached", fake)
    first  = edit_director.direct(["one beat"])
    second = edit_director.direct(["one beat"])

    assert len(calls) == 1
    assert first is second

    edit_director.direct(["a different beat"])
    assert len(calls) == 2


def test_a_none_plan_is_cached_too(monkeypatch):
    """The failure path is the common one when no key is configured — it must
    not retry the model once per renderer."""
    import edit_director

    monkeypatch.setattr(edit_director, "_plan_cache", {})
    calls = []
    monkeypatch.setattr(edit_director, "_direct_uncached",
                        lambda beats: calls.append(1) or None)

    assert edit_director.direct(["b"]) is None
    assert edit_director.direct(["b"]) is None
    assert len(calls) == 1


# ------------------------------------------------------------- sfx weighting

def test_sfx_gain_is_weighted_by_the_beat_it_introduces():
    """The riser into a revelation should be audible; a whoosh should not step
    on a resolution beat's closing line."""
    import audio_gen

    base = audio_gen.SFX_WHOOSH_GAIN
    assert em.sfx_weight("revelation") * base > base
    assert em.sfx_weight("resolution") * base < base


def test_no_tones_reproduces_the_original_gains():
    """Every weight is 1.0 without an edit plan, so the mix is byte-identical
    to the one that shipped before the map existed."""
    assert em.sfx_weight("neutral") == 1.0
