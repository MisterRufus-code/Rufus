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


# ── the two gallery bugs, for every preset and not just the one that had them ──
#
# Both were found by the owner opening a folder of sixty stills. They were
# fixed in `stickman` because that is the preset that was running, and the
# tests above pin them there — but the clauses that caused them are the kind
# any new preset would reach for, and the next gallery costs another night of
# the 3090.

@pytest.mark.parametrize("name", sorted(_looks()))
def test_no_preset_paints_the_scene_out(name):
    """THE WHITE-VOID BUG. "on a pure white background" overrode every
    storyboard that had carefully built a room, in every frame."""
    s = _looks()[name].lower()
    assert "pure white background" not in s
    assert "blank background" not in s
    assert "white void" not in s


@pytest.mark.parametrize("name", sorted(_looks()))
def test_no_preset_makes_the_background_fainter_instead_of_simpler(name):
    """THE BEIGE GALLERY. "drawn thinner and paler than the foreground" gave
    sixty stills where only the figure looked finished. A background is
    quieter because it is simpler and further away, not because it is washed
    out — and every preset now says which."""
    s = _looks()[name].lower()
    assert "paler than the foreground" not in s
    assert "because it is simpler" in s


@pytest.mark.parametrize("name", sorted(_looks()))
def test_every_preset_builds_a_place_and_keeps_it(name):
    """A style is appended to every prompt byte for byte, so a preset that
    says nothing about the background lets the model decide — and the model
    decides blank paper. ink_woodcut was the one preset with no scene clause
    at all, because the shared one is written for flat colour and an engraving
    has none."""
    s = _looks()[name]
    assert "BUILD THE WHOLE" in s, "no scene instruction at all"
    assert "same horizon height" in s, "the place has to persist across shots"
    assert "reads instantly at thumbnail size" in s.lower()


@pytest.mark.parametrize("name", sorted(_looks()))
def test_every_preset_forbids_the_photographic_tells(name):
    """The stills model's default is a photograph. Whatever the medium, the
    same four words are what make a drawing stop looking drawn."""
    s = _looks()[name].lower()
    for banned in ("no gradients", "no depth of field", "no film grain"):
        assert banned in s, banned


# ── the ink explainer look ───────────────────────────────────────────────────

def test_the_ink_explainer_is_not_the_woodcut():
    """Both are ink and they are different channels. The woodcut is an 1890s
    newspaper engraving — monochrome, dense, anatomical. The explainer is a
    notebook page: a fine nib, hatched tone, and one or two muted washes on
    the thing the shot is about."""
    ink = STYLES["ink_explainer"]
    wood = STYLES["ink_woodcut"]
    # Stated as what each one IS, not as words the other must avoid: the
    # explainer's own text says "not a printed engraving", and a test that
    # searched for the word would fail on the sentence drawing the very
    # distinction it is checking.
    assert "off-white paper" in ink and "muted washes" in ink
    assert "19th-century" in wood and "no colour" in wood.lower()
    assert "no grey fills, no gradients" in ink.lower()


def test_the_ink_explainer_draws_people_rather_than_stick_figures():
    """The difference from `stickman`, which is the question this preset was
    added to answer. Same channel, same scripts, a different hand."""
    ink = STYLES["ink_explainer"]
    assert "PEOPLE ARE DRAWN PROPERLY rather than as stick figures" in ink
    assert "stick-figure" not in ink.lower().replace("stick figures", "")


def test_the_ink_explainer_carries_the_face_vocabulary():
    """The lesson from the ten mild smiles: naming a feeling is not enough,
    the model needs the geometry."""
    ink = STYLES["ink_explainer"]
    assert "must not be the same on every figure" in ink
    for feeling in ("anger", "shock", "delight", "worry"):
        assert feeling in ink, feeling
