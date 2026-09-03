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
    """The riser into a revelation should be audible; a bubble should not step
    on a resolution beat's closing line."""
    import audio_gen

    base = audio_gen.SFX_BUBBLE_GAIN
    assert em.sfx_weight("revelation") * base > base
    assert em.sfx_weight("resolution") * base < base


def test_no_tones_reproduces_the_original_gains():
    """Every weight is 1.0 without an edit plan, so the mix is byte-identical
    to the one that shipped before the map existed."""
    assert em.sfx_weight("neutral") == 1.0


# --------------------------------------------------------------- film grain

def test_grain_is_temporal_not_static():
    """Static grain looks like a dirty lens; grain that changes per frame looks
    like film. allf=t is the whole difference."""
    assert "allf=t" in em.grain_filter("tension")


def test_grain_rides_along_in_the_grade():
    assert "noise=" in em.grade_filter("tension")


def test_grain_can_be_turned_off(monkeypatch):
    monkeypatch.setenv("RUFUS_FILM_GRAIN", "0")
    assert em.grain_filter("tension") == ""
    assert "noise=" not in em.grade_filter("tension")


def test_grain_scale_tunes_every_tone(monkeypatch):
    monkeypatch.setenv("RUFUS_FILM_GRAIN_SCALE", "2.0")
    heavy = em.grain_filter("tension")
    monkeypatch.setenv("RUFUS_FILM_GRAIN_SCALE", "1.0")
    normal = em.grain_filter("tension")
    assert int(re.search(r"alls=(\d+)", heavy).group(1)) > \
           int(re.search(r"alls=(\d+)", normal).group(1))


def test_a_junk_grain_scale_does_not_crash_the_render(monkeypatch):
    monkeypatch.setenv("RUFUS_FILM_GRAIN_SCALE", "loud")
    assert "noise=" in em.grain_filter("neutral")


def test_grain_scale_zero_removes_the_filter(monkeypatch):
    monkeypatch.setenv("RUFUS_FILM_GRAIN_SCALE", "0")
    assert em.grain_filter("tension") == ""


# ------------------------------------------------------------ micro-pauses

def test_the_turn_gets_the_longest_hold():
    assert em.pause_after("revelation") == max(
        em.pause_after(t) for t in em.TONES)


def test_neutral_and_resolution_add_no_silence():
    assert em.pause_after("neutral") == 0.0
    assert em.pause_after("resolution") == 0.0


def test_no_pause_is_long_enough_to_read_as_a_broken_file():
    assert all(0.0 <= em.pause_after(t) <= 0.5 for t in em.TONES)


def test_tts_adds_the_tone_pause_on_top_of_punctuation():
    import tts_engine

    assert tts_engine._tone_pause("revelation") > 0
    assert tts_engine._tone_pause("neutral") == 0.0
    assert tts_engine._tone_pause("not a tone") == 0.0


def test_synthesize_accepts_tones_without_requiring_them():
    import inspect

    import tts_engine

    sig = inspect.signature(tts_engine.synthesize)
    assert sig.parameters["tones"].default is None


# ------------------------------------------------------- the pacing QC check

def test_a_long_static_stretch_is_flagged():
    import qc_check

    warns = qc_check._pacing_warnings([3.0, 6.0], 20.0)
    assert warns and "14.0s" in warns[0]


def test_normal_pacing_is_silent():
    import qc_check

    assert qc_check._pacing_warnings([3.0, 7.0, 11.0, 15.0], 19.0) == []


def test_pacing_is_a_warning_never_a_critical():
    """A long beat may be genuinely moving (Wan motion, a strong push-in) —
    this check cannot see that, so it must never hold an upload."""
    import qc_check

    info = {"width": 1080, "height": 1920, "fps": 30.0, "duration": 40.0,
            "has_video": True, "has_audio": True, "vcodec": "h264", "acodec": "aac"}
    critical, warnings = qc_check._evaluate(info, 20_000_000, -14.0, [2.0])
    assert critical == []
    assert any("without a cut" in w for w in warnings)


def test_no_cuts_means_no_pacing_opinion():
    """A caller that cannot supply cuts gets no warning, not a false one."""
    import qc_check

    assert qc_check._pacing_warnings(None, 40.0) == []
    assert qc_check._pacing_warnings([], 40.0) == []


def test_cuts_outside_the_duration_are_ignored():
    import qc_check

    assert qc_check._pacing_warnings([3.0, 999.0], 6.0) == []


def test_run_qc_still_works_without_cuts():
    import inspect

    import qc_check

    assert inspect.signature(qc_check.run_qc).parameters["cuts"].default is None


# ── the tone finally reaching the voice ──────────────────────────────────────
#
# The tone already reached the picture, the music and the silence between
# beats. It never reached the VOICE, and the reason was one sentence in
# emotional_map that was true when it was written: "this is the only prosody
# control a free local voice has". True of Kokoro, which has no SSML — and it
# quietly governed the DEFAULT backend, which is Edge, and which takes a rate,
# a pitch and a volume on every call.
#
# Six tones were being computed per run and the narration read all six the
# same way. That is what "the voice sounds flat" was.

def test_every_tone_has_a_voice_setting():
    """A tone the grader knows and the voice does not is a tone that reads as
    neutral in half the render — the exact split this change closes."""
    for tone in em.TONES:
        v = em.voice(tone)
        assert set(v) == {"rate", "pitch", "volume"}, tone


def test_neutral_is_exactly_no_change():
    """This is what makes the change safe to ship. tones_from_plan fails open
    to all-neutral — no plan, a short plan, junk in the plan — so a neutral
    tone MUST synthesize what the pipeline already produces."""
    assert em.voice("neutral") == {
        "rate": "+0%", "pitch": "+0Hz", "volume": "+0%"}


def test_the_base_rate_is_carried_not_replaced():
    """RUFUS_EDGE_RATE is an owner setting. A tone is a delta ON it, not a
    substitute for it — otherwise setting the voice faster would be silently
    undone by every non-neutral beat."""
    assert em.voice("neutral", 6)["rate"] == "+6%"
    assert em.voice("weight", 6)["rate"] == "-2%"


def test_the_units_are_the_ones_edge_wants():
    """Pitch is Hz for Edge; rate and volume are percent. Formatting them here
    rather than at each call site is the point — one place to get the unit
    wrong instead of several."""
    v = em.voice("revelation")
    assert v["pitch"].endswith("Hz")
    assert v["rate"].endswith("%") and v["volume"].endswith("%")
    assert v["rate"][0] in "+-" and v["pitch"][0] in "+-"


def test_weight_is_the_slowest_and_lowest_thing_in_the_video():
    """The vocabulary has to mean something audible or it is decoration.
    Consequence should not sound like an open question."""
    weight = em.voice("weight")
    curiosity = em.voice("curiosity")
    assert int(weight["rate"].rstrip("%")) < int(curiosity["rate"].rstrip("%"))
    assert int(weight["pitch"].rstrip("Hz")) < int(curiosity["pitch"].rstrip("Hz"))


def test_revelation_is_louder_and_resolution_quieter():
    assert int(em.voice("revelation")["volume"].rstrip("%")) > 0
    assert int(em.voice("resolution")["volume"].rstrip("%")) < 0


def test_an_unknown_tone_is_the_voice_we_already_ship():
    for junk in ("dramatic", "", None, 7, ["weight"]):
        assert em.voice(junk) == em.voice("neutral"), junk


def test_a_wild_base_rate_cannot_produce_an_unusable_voice():
    """RUFUS_EDGE_RATE is a free-text env var. A tone delta landing on top of
    "+200%" must still return something a person can listen to."""
    v = em.voice("curiosity", 200)
    assert int(v["rate"].rstrip("%")) <= 40


def test_an_all_neutral_run_takes_the_single_request_path():
    """Per-beat costs one network round trip per beat. Paying that to apply a
    delta of zero is a cost with nothing to show for it."""
    assert em.voice_is_neutral(None) is True
    assert em.voice_is_neutral([]) is True
    assert em.voice_is_neutral(["neutral", "neutral"]) is True
    assert em.voice_is_neutral(["neutral", "weight"]) is False


# ── the tone was live in the grade and silent in the voice ──────────────────
#
# A tone reached the Kokoro voice through exactly one mechanism: extra silence
# between chunks. That is inaudible on any text short enough to be a single
# chunk — a hook, a take, a one-line beat — which is why three "different"
# reads of an opening line came back the same. Kokoro exposes one knob, speed,
# so the rate half of a tone can reach it.

def test_a_tone_changes_the_kokoro_speed():
    import emotional_map as em
    assert em.kokoro_speed("weight") < em.kokoro_speed("neutral")
    assert em.kokoro_speed("curiosity") > em.kokoro_speed("neutral")


def test_every_tone_is_distinguishable_by_pace():
    """Three takes a person cannot tell apart is not a choice."""
    import emotional_map as em
    speeds = {em.kokoro_speed(t) for t in em.TONES}
    assert len(speeds) == len(em.TONES)


def test_neutral_is_exactly_the_base_speed():
    """Neutral is a zero delta — the voice the pipeline already ships."""
    import emotional_map as em
    assert em.kokoro_speed("neutral", 1.0) == 1.0
    assert em.kokoro_speed("neutral", 1.15) == 1.15


def test_an_unknown_tone_does_not_change_the_pace():
    import emotional_map as em
    assert em.kokoro_speed("shouty", 1.0) == 1.0


def test_a_fast_base_speed_cannot_be_pushed_out_of_range():
    """RUFUS_KOKORO_SPEED is an owner setting and stacking a tone on top of an
    already-fast base is how a voice ends up unlistenable."""
    import emotional_map as em
    assert 0.7 <= em.kokoro_speed("curiosity", 1.6) <= 1.3
    assert 0.7 <= em.kokoro_speed("weight", 0.5) <= 1.3


def test_the_backend_says_what_it_can_do_with_a_tone(monkeypatch):
    """A choice between three reads is theatre if the backend renders them the
    same, and which backend is live is a runtime fact."""
    import emotional_map as em
    monkeypatch.setenv("RUFUS_TTS", "edge")
    assert "pitch" in em.speaks_tone()
    monkeypatch.setenv("RUFUS_TTS", "kokoro")
    assert "pace only" in em.speaks_tone()
    monkeypatch.setenv("RUFUS_TTS", "elevenlabs")
    assert "pace only" in em.speaks_tone(), (
        "elevenlabs is never handed tones and falls back to Kokoro anyway")
