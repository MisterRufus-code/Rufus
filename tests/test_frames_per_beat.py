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
