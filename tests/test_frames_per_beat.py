"""How many pictures one beat becomes, and why the number is announced.

The owner asked for forty pictures in a video and got ten, twice. The levers
that produce them are here: how many beats the script is cut into, and how
many stills each beat is drawn as. Both were capped somewhere quiet.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import comfy_client as cc  # noqa: E402


def test_four_frames_a_beat_is_forty_pictures_on_ten_beats():
    """The owner's actual request. Ten beats at four stills each."""
    assert len(cc._progression_modifiers(4)) == 4


def test_the_peak_is_always_in_the_arc():
    """The empty modifier IS the prompt as written, and it is the frame that
    was already rendered. An arc that drops it renders the beat as two lead-ins
    and no moment."""
    for n in range(2, 8):
        assert "" in cc._progression_modifiers(n), n


def test_the_arc_is_centred_on_the_peak():
    """Three frames is earlier / the moment / later — not two run-ups."""
    mods = cc._progression_modifiers(3)
    assert mods.index("") == 1


def test_asking_for_more_frames_than_exist_says_so(capsys):
    """It used to return four and print nothing, so RUFUS_FRAMES_PER_BEAT=6
    looked like it had no effect at all."""
    out = cc._progression_modifiers(20)
    assert len(out) == len(cc._PROGRESSION_STEPS)
    said = capsys.readouterr().out
    assert "more than the" in said
    assert "SD_CLIPS" in said, "the message has to name the lever that DOES work"


def test_one_frame_is_the_prompt_as_written():
    assert cc._progression_modifiers(1) == [""]


def test_stills_only_cuts_between_stills_rather_than_holding_one():
    """With every motion engine off, one still per beat is a photograph held
    for four to six seconds — which QC flags on its own."""
    src = Path(cc.__file__).read_text(encoding="utf-8")
    assert "STILLS-ONLY MEANS CUT, NOT HOLD" in src
    assert 'beat_mode = "cut"' in src
    assert "RUFUS_BEAT_MOTION=kenburns" in src, "the way back must be named"


# ── an explicit 1 is not the same integer as no answer ───────────────────────
#
# THE SETTINGS FILE THAT PROVED IT. config/dashboard_settings.json held neither
# SD_CLIPS nor RUFUS_FRAMES_PER_BEAT, and run_dashboard.bat sets
# RUFUS_STILLS_ONLY=1 — which the dashboard's child process inherits, because
# saved settings LAYER ON TOP of the environment rather than replacing it. So
# every dashboard-launched run took the auto-`cut` branch and rewrote 1 to 3,
# and a gallery of sixty stills came back in near-identical threes.
#
# The fix the owner would reach for first is the dashboard's own "Stills per
# beat" field, set to 1. That wrote RUFUS_FRAMES_PER_BEAT=1 — indistinguishable
# from unset at the branch that reads it — so the run still auto-selected
# `cut`, still rewrote it to 3, and still printed that it was cutting between
# stills. The comment three lines above that branch had promised for months
# that "an explicit RUFUS_BEAT_MOTION or RUFUS_FRAMES_PER_BEAT still wins".
#
# A default and a decision cannot be the same value. That is the whole bug.

def test_setting_one_still_a_beat_is_not_read_as_having_set_nothing(monkeypatch):
    monkeypatch.delenv("RUFUS_FRAMES_PER_BEAT", raising=False)
    assert cc._frames_per_beat_was_asked_for() is False
    monkeypatch.setenv("RUFUS_FRAMES_PER_BEAT", "1")
    assert cc._frames_per_beat_was_asked_for() is True
    assert cc._frames_per_beat() == 1, "the number itself must not change"


def test_an_empty_setting_is_not_a_decision(monkeypatch):
    """The dashboard writes "" for a field the owner cleared, and clearing a
    field is how you ask for the default back — not how you pin it to 1."""
    for blank in ("", "   "):
        monkeypatch.setenv("RUFUS_FRAMES_PER_BEAT", blank)
        assert cc._frames_per_beat_was_asked_for() is False, repr(blank)


def test_the_auto_cut_branch_defers_to_an_explicit_number():
    """Auto-selection may fill a gap; it may not overrule an answer. The
    branch has to consult the "was this asked for" question, or the two cases
    collapse back into one integer."""
    src = Path(cc.__file__).read_text(encoding="utf-8")
    branch = src.split("STILLS-ONLY MEANS CUT, NOT HOLD", 1)[1][:1600]
    assert "_frames_per_beat_was_asked_for()" in branch, (
        "the stills-only auto-cut can still overwrite an explicit "
        "RUFUS_FRAMES_PER_BEAT=1")
