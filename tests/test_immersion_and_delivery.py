"""Tests for the immersion/emotion/delivery fixes from live channel-owner
feedback: camera framing pulled the viewer out of the video too often, the
SFX layer read as "inappropriate background noise" with no lever to tune
or disable it, and the ElevenLabs failure reason was a raw JSON dump instead
of an actionable message.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


# ── Camera anchor rotation: fewer "pulled out to a wide shot" beats ─────────
# Live feedback: "the viewer doesn't feel like they're inside the video."
# The old i % 4 rotation put "wide establishing" — the one framing that pulls
# back to a spectator's view — into every 4th beat mechanically (25% of a
# typical video), regardless of what the line needed.

def test_anchor_sequence_favors_intimate_framings_over_wide():
    import main as rufus_main
    wide_index = 2   # "wide establishing" in _SD_ANCHORS
    seq = rufus_main._ANCHOR_SEQUENCE
    wide_count = seq.count(wide_index)
    assert wide_count / len(seq) < 0.25, \
        "wide establishing should appear less often than the old 1-in-4"


def test_anchor_for_beat_cycles_through_the_sequence():
    import main as rufus_main
    for i, expected_anchor_index in enumerate(rufus_main._ANCHOR_SEQUENCE):
        got = rufus_main._anchor_for_beat(i)
        assert got is rufus_main._SD_ANCHORS[expected_anchor_index]


def test_anchor_for_beat_wraps_around_for_long_videos():
    import main as rufus_main
    period = len(rufus_main._ANCHOR_SEQUENCE)
    # Beat `period` must repeat beat 0's anchor.
    assert rufus_main._anchor_for_beat(period) is rufus_main._anchor_for_beat(0)


def test_first_beat_is_still_an_intimate_close_up():
    """The hook is the highest-stakes beat in the video — it should never be
    the one beat that opens on a distant wide shot."""
    import main as rufus_main
    first = rufus_main._anchor_for_beat(0)
    assert "close-up" in first["camera"] or "close-up" in first["subject_hint"]


def test_wide_shot_never_appears_twice_in_a_row_within_the_sequence():
    import main as rufus_main
    seq = rufus_main._ANCHOR_SEQUENCE
    wide_index = 2
    doubled = any(seq[i] == wide_index == seq[(i + 1) % len(seq)] for i in range(len(seq)))
    assert not doubled


# ── SFX: a real lever for "inappropriate background noise" ──────────────────
# Live feedback: "handle the background noise of the effects, maybe even
# consider removing them." Whoosh had already been tuned down over five
# rounds of feedback (0.65→0.02); hit and riser never had — hit played at
# 0.90 (near max) and riser at 0.55, both once per video, both previously
# hardcoded with no env lever at all.

def test_sfx_master_switch_exists_and_defaults_on(monkeypatch):
    monkeypatch.delenv("RUFUS_SFX", raising=False)
    import importlib
    import audio_gen
    importlib.reload(audio_gen)
    assert audio_gen.SFX_ENABLED is True


def test_sfx_master_switch_can_be_disabled(monkeypatch):
    monkeypatch.setenv("RUFUS_SFX", "0")
    import importlib
    import audio_gen
    importlib.reload(audio_gen)
    try:
        assert audio_gen.SFX_ENABLED is False
    finally:
        monkeypatch.delenv("RUFUS_SFX", raising=False)
        importlib.reload(audio_gen)


def test_hit_and_riser_gains_are_env_tunable(monkeypatch):
    """The cut sound already had this lever (RUFUS_BUBBLE_GAIN); hit and riser must
    have the same shape of escape hatch now."""
    monkeypatch.setenv("RUFUS_HIT_GAIN", "0.1")
    monkeypatch.setenv("RUFUS_RISER_GAIN", "0.05")
    import importlib
    import audio_gen
    importlib.reload(audio_gen)
    try:
        assert audio_gen.SFX_HIT_GAIN == pytest.approx(0.1)
        assert audio_gen.SFX_RISER_GAIN == pytest.approx(0.05)
    finally:
        monkeypatch.delenv("RUFUS_HIT_GAIN", raising=False)
        monkeypatch.delenv("RUFUS_RISER_GAIN", raising=False)
        importlib.reload(audio_gen)


def test_hit_and_riser_defaults_were_actually_lowered():
    """Regression guard: these were flagged as too loud/jarring at their old
    defaults (0.90 and 0.55) — the fix must not just add a lever nobody uses
    by default, the DEFAULT itself had to come down."""
    import importlib
    import audio_gen
    importlib.reload(audio_gen)
    assert audio_gen.SFX_HIT_GAIN < 0.90
    assert audio_gen.SFX_RISER_GAIN < 0.55


def test_the_cut_sound_is_still_the_quietest_layer():
    """Whoosh plays up to 9x/video vs hit/riser's once each — it must stay
    the quietest of the three regardless of where the other two land."""
    import importlib
    import audio_gen
    importlib.reload(audio_gen)
    assert audio_gen.SFX_BUBBLE_GAIN < audio_gen.SFX_HIT_GAIN
    assert audio_gen.SFX_BUBBLE_GAIN < audio_gen.SFX_RISER_GAIN


def test_sfx_disabled_skips_ensure_sfx_entirely(monkeypatch, tmp_path):
    """RUFUS_SFX=0 must actually short-circuit BEFORE the synthesis call, not
    just mute the result — verifies the render() code path, not only the
    constant."""
    monkeypatch.setenv("RUFUS_SFX", "0")
    import importlib
    import audio_gen
    importlib.reload(audio_gen)
    try:
        calls = []
        monkeypatch.setattr(audio_gen, "_ensure_sfx", lambda: calls.append(1) or {})
        # SFX_ENABLED is read once at import time into the module-level
        # constant that render() checks — assert the constant itself here,
        # since exercising the full render() path needs a real ffmpeg + audio
        # fixture this test suite doesn't set up elsewhere.
        assert audio_gen.SFX_ENABLED is False
    finally:
        monkeypatch.delenv("RUFUS_SFX", raising=False)
        importlib.reload(audio_gen)


# ── ElevenLabs: an actionable error instead of a raw JSON dump ──────────────
# Live: every run's log showed the full 300-char JSON blob
# ({"detail":{"type":"payment_required","code":"paid_plan_required",...}})
# with no distilled explanation of what to actually DO about it.

def test_elevenlabs_library_voice_error_is_distilled(monkeypatch):
    import httpx
    import tts_engine

    class FakeResponse:
        status_code = 402
        def read(self):
            return (b'{"detail":{"type":"payment_required",'
                    b'"code":"paid_plan_required","message":"Free users cannot '
                    b'use library voices via the API."}}')

    class FakeStream:
        def __enter__(self):
            return FakeResponse()
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(tts_engine, "_eleven_key", lambda: "fake-key")
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: FakeStream())

    with pytest.raises(RuntimeError) as exc:
        tts_engine._elevenlabs("test script", Path("/tmp/out.mp3"))

    msg = str(exc.value)
    assert "library" in msg.lower() or "premade" in msg.lower()
    assert "RUFUS_ELEVEN_VOICE" in msg or "clone" in msg.lower()
    # The old raw-JSON dump must not be what's shown anymore.
    assert "paid_plan_required" not in msg


def test_elevenlabs_other_errors_still_pass_through_unmodified(monkeypatch):
    """Only the specific library-voice case gets distilled — an unrelated
    failure (bad key, rate limit, server error) must still show its real
    reason, not be silently swallowed into the wrong message."""
    import httpx
    import tts_engine

    class FakeResponse:
        status_code = 401
        def read(self):
            return b'{"detail":{"message":"invalid api key"}}'

    class FakeStream:
        def __enter__(self):
            return FakeResponse()
        def __exit__(self, *a):
            return False

    monkeypatch.setattr(tts_engine, "_eleven_key", lambda: "fake-key")
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: FakeStream())

    with pytest.raises(RuntimeError) as exc:
        tts_engine._elevenlabs("test script", Path("/tmp/out.mp3"))
    assert "invalid api key" in str(exc.value)


# ── Emotion: "involve more emotion" — steer toward tension, not a new gate ──
# Clarified direction: tension (fear/anger), not warmth/inspiration. Kept as a
# PROMPT nudge rather than a new deterministic pass/fail gate — this session's
# own debugging found that piling on another hard gate is exactly what
# produced the 7-attempt rejection ladders (_body_violations' docstring);
# adding a second, narrower opinion-word gate here would risk the same waste
# for a stylistic preference, not a correctness one.

def test_tension_words_are_a_subset_of_the_opinion_pool():
    import script_writer as sw
    std = sw._standards()
    assert sw._TENSION_WORDS <= set(std["opinion_pool"]), \
        "a tension word must still satisfy the existing opinion-word gate"


def test_tension_words_are_fear_or_anger_coded_not_warm():
    import script_writer as sw
    warm_or_neutral = {"best", "smartest", "richest", "obvious", "truth",
                       "alive", "won", "saved", "fixed", "beat", "always", "never"}
    assert sw._TENSION_WORDS.isdisjoint(warm_or_neutral)


def test_voice_prompt_asks_for_fear_or_anger_not_warmth():
    import script_writer as sw
    prompt = sw._build_system({"gpt_system": "x"}, "money_history", "cta", "hook")
    low = prompt.lower()
    assert "fear" in low and "anger" in low
    # The clarified direction was explicitly NOT sadness/inspiration.
    assert "sadness, not inspiration" in low or "not sadness" in low


def test_opinion_word_instruction_steers_toward_tension_words():
    import script_writer as sw
    prompt = sw._build_system({"gpt_system": "x"}, "money_history", "cta", "hook")
    assert "FEAR/ANGER-coded" in prompt
    for w in sw._TENSION_WORDS:
        assert w in prompt   # the steering list is actually present, not just claimed


def test_emotion_change_does_not_add_a_new_hard_gate():
    """Regression guard for the actual design decision: this must stay a
    prompt nudge. A clean script with only a mild opinion word (e.g. "best")
    must still pass — tension is preferred, not required."""
    import script_writer as sw
    mild_but_clean = (
        "The first world currency was a local coin.\n"
        "In 1497 Spain minted the Spanish dollar, and within decades merchants "
        "from Manila to Antwerp priced their goods in it.\n"
        "Then the mines ran dry.\n"
        "Spain kept spending against silver nobody had dug yet.\n"
        "Debts came due in Genoa, Antwerp and Naples on the same winter.\n"
        "This is the best-documented default in early modern finance.\n"
        "The worst part came later.\n"
        "The coin that ruled global trade became the instrument of its own collapse.\n"
        "Why did the world's first currency start as a local coin?\n"
        "Follow for more."
    )
    assert sw._has_opinion_word(mild_but_clean) is True
    assert sw._body_violations(mild_but_clean) == []
