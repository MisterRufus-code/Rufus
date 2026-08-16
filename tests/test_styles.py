"""The look presets, and the two things a real gallery of them got wrong.

A style block is appended to every image prompt byte for byte, so anything it
forbids is forbidden in every frame of every video on that channel. That makes
it the highest-leverage text in the repo and the easiest place to be quietly,
consistently wrong — which is what happened: the owner ran a full stickman
sequence, opened the gallery, and every picture was a figure floating in white
with the same faint smile.

Both causes were single clauses in this file. These tests keep them out.
"""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

STYLES = json.loads((ROOT / "config" / "styles.json").read_text(encoding="utf-8"))


def _looks() -> dict:
    return {k: v for k, v in STYLES.items() if not k.startswith("_")}


def test_every_preset_is_one_block_of_text():
    """The suffix is pasted verbatim into every prompt. A list or a dict here
    would render as its Python repr."""
    for name, text in _looks().items():
        assert isinstance(text, str) and len(text) > 200, name


def test_stickman_asks_for_a_background_that_places_the_scene():
    """THE WHITE-VOID BUG. The preset said "on a pure white background" and
    "Everything not deliberately filled with colour is pure white", so a
    storyboard that carefully built a medieval hall with a beam of light from
    a high window rendered as two stick figures and a table in empty space.
    The style was overriding the whole scene description, every frame."""
    s = STYLES["stickman"]
    assert "pure white background" not in s
    assert "not deliberately filled with colour is pure white" not in s
    assert "BUILD THE WHOLE PLACE" in s
    assert "horizon line" in s


def test_stickman_backgrounds_stay_out_of_the_subjects_way():
    """The fix must not swing into the other failure — a busy background on a
    frame that is on screen for four seconds at thumbnail size reads as
    noise.

    NOT BY FADING IT, which is how this was worded before and what the second
    gallery showed: "soft muted flat colours, drawn thinner and paler than the
    foreground" produced sixty beige stills where only the figure looked
    finished. A background is quieter because it is simpler and further away,
    not because it is washed out."""
    s = STYLES["stickman"]
    assert "quieter than the subject" in s
    assert "reads instantly at thumbnail size" in s
    assert "paler than the foreground" not in s


def test_stickman_faces_carry_the_emotion():
    """The preset pinned every mouth to "a single thin curved line with a
    slight upturn" and banned eyebrows outright, so ten shots of a country
    losing its money came back with ten mild smiles. Eyebrows and a mouth
    curve are the entire emotional vocabulary of this art style."""
    s = STYLES["stickman"]
    assert "no eyebrows" not in s
    assert "eyebrow strokes" in s
    assert "THE FACE CARRIES THE EMOTION OF THE MOMENT" in s
    assert "must not be the same on every figure" in s


@pytest.mark.parametrize("feeling", ["anger", "shock", "delight"])
def test_stickman_spells_out_how_to_draw_a_feeling(feeling):
    """Naming the feeling is not enough — the model needs the geometry, since
    "sad" on a face with two dot eyes has no obvious drawing."""
    assert feeling in STYLES["stickman"]


def test_stickman_keeps_the_line_art_that_was_working():
    """The character consistency across the owner's gallery came from these
    exact constraints. The fix is additive or it trades one problem for a
    worse one."""
    s = STYLES["stickman"]
    for kept in ("thin, clean black line art of uniform weight",
                 "no nose", "no gradients", "no shading", "no film grain",
                 "flat unshaded colour fills", "reads instantly at thumbnail size"):
        assert kept.lower() in s.lower(), kept


def test_the_presets_are_reachable_by_name():
    import comfy_client
    presets = comfy_client.style_presets()
    for name in _looks():
        assert name in presets
