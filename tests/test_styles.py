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
    newspaper engraving — monochrome, dense, anatomical, printed. The explainer
    is a sketchbook page that gets fully painted: marker colour over a wobbling
    ink line, and the weather is half the picture."""
    ink = STYLES["ink_explainer"]
    wood = STYLES["ink_woodcut"]
    assert "19th-century" in wood and "no colour" in wood.lower()
    assert "FULLY PAINTED" in ink and "CARRIES THE WEATHER" in ink


def test_the_ink_explainer_keeps_the_stick_figures():
    """WRITTEN FROM THE WRONG GUESS FIRST. This preset was added with people
    "drawn properly rather than as stick figures", which is what an ink
    explainer sounds like. The reference frames say otherwise: the people ARE
    stick figures — white oval head, dot eyes, drawn brows, a scribble of hair
    — and the animals beside them are drawn in full, spots and proportions and
    all. That contrast is the style, and getting it backwards would have
    produced a different channel."""
    ink = STYLES["ink_explainer"]
    assert "PEOPLE ARE STICK FIGURES and stay that way" in ink
    assert "ANIMALS AND OBJECTS ARE DRAWN PROPERLY" in ink
    # The RULE, not the leopard that used to illustrate it — see
    # test_no_preset_names_a_specific_thing_to_draw for why the leopard went.
    assert "keeps its true shape, its real proportions" in ink


def test_the_ink_explainer_is_not_the_stickman_either():
    """The two share their figures and differ in the hand. stickman is thin,
    clean line art of uniform weight with flat unshaded fills — a poster.
    This wobbles, varies its weight, scribbles its shading and paints the
    weather across the frame."""
    ink = STYLES["ink_explainer"]
    stick = STYLES["stickman"]
    assert "VARYING weight" in ink and "wobbles" in ink
    assert "SHADING IS SCRIBBLED" in ink
    assert "uniform weight" in stick and "uniform weight" not in ink
    assert "flat unshaded colour fills" in stick
    assert "scribbled texture rather than with flat colour" in ink


def test_the_ink_explainer_gives_the_frame_one_temperature():
    """The reference frames carry their mood in the colour of the whole
    picture, not in the subject: a storm is grey-blue edge to edge and a fire
    is warm ochre on the cave wall behind it."""
    ink = STYLES["ink_explainer"]
    assert "THE WHOLE FRAME CARRIES ONE TEMPERATURE" in ink
    assert "same temperature" in ink, "and it persists across a sequence"


def test_the_ink_explainer_carries_the_face_vocabulary():
    """The lesson from the ten mild smiles: naming a feeling is not enough,
    the model needs the geometry."""
    ink = STYLES["ink_explainer"]
    assert "must not be the same on every figure" in ink
    for feeling in ("anger", "shock", "delight", "worry"):
        assert feeling in ink, feeling


@pytest.mark.parametrize("name", sorted(_looks()))
def test_every_preset_bans_lettering(name):
    """THE STRONGEST POSITION WAS EMPTY. The storyboard is told not to NAME
    printed words, and the negative conditioning leads with "text, letters,
    words" — and a real gallery still came back with "whole rouble payments
    from each village" written across three panels and "Proof" on four
    documents.

    The style block is the only text that is appended to EVERY prompt byte for
    byte, and it said nothing about lettering at all. A ban that is in two of
    the three places is a ban that depends on which of them the workflow
    actually applies."""
    s = _looks()[name]
    assert "NO LETTERING ANYWHERE IN THE FRAME" in s
    assert "drawn BLANK" in s


def test_the_dashboard_offers_every_preset_in_the_file():
    """A look the picker cannot offer is a look nobody uses. The select was
    written when there were three presets and stayed that way through five
    more — including the one added because the owner asked for it."""
    import re
    src = (ROOT / "scripts" / "dashboard.py").read_text(encoding="utf-8")
    block = src.split('("RUFUS_STYLE", "Style preset",', 1)[1][:400]
    m = re.search(r'"select:([a-z_,\s"]+?)"?,\s*\n?\s*"Named look', block)
    assert m, "could not find the RUFUS_STYLE picker"
    offered = {x for x in re.split(r'[,\s"]+', m.group(1)) if x}
    assert offered == set(_looks()), (
        f"picker and styles.json disagree: "
        f"only in file {set(_looks()) - offered}, "
        f"only in picker {offered - set(_looks())}")


# ── an example inside a style block is not an example ────────────────────────
#
# THE LION. A gallery of sixty stills for a video about Bear Stearns and the
# 2008 crisis had a lion in it, repeatedly, and every frame was set in a grassy
# riverside village with bones scattered on the ground. None of that came from
# the script or the storyboard. It came from here:
#
#   "a zebra has its stripes and mane, A LION its mane and tail tuft"
#   "...rooftops, ship masts, machinery, crates, BONES AND STONES scattered
#    in the dirt"
#
# A style block is appended to every prompt byte for byte. An image model does
# not read "a lion" as an illustration of a rule about animals — it reads it as
# a noun in the prompt, and it draws it. This repo has now had the same bug
# three times: the hook example "2,000 years ago" leaking into generated hooks,
# the storyboard's own examples, and this.

_NOUNS_THAT_GOT_DRAWN = [
    "zebra", "lion", "leopard", "deer", "boulder", "bones",
    "ship masts", "market stalls", "river bank", "cave mouth",
]


@pytest.mark.parametrize("name", sorted(_looks()))
def test_no_preset_names_a_specific_thing_to_draw(name):
    """The rule may be stated; the things may not be listed. "An animal keeps
    its real markings" is a rule. "A lion its mane and tail tuft" is a lion in
    every frame of every video on this channel."""
    s = _looks()[name].lower()
    named = [n for n in _NOUNS_THAT_GOT_DRAWN if n in s]
    assert not named, (
        f"{name} names {named} — appended to every prompt, that is not an "
        f"example, it is a subject")


@pytest.mark.parametrize("name", sorted(_looks()))
def test_the_rule_the_examples_were_illustrating_survives(name):
    """The fix must not throw out the instruction with the nouns. Every preset
    still has to say to build the place — deleting the list and leaving nothing
    would bring back the white-void bug the clause exists to stop."""
    s = _looks()[name]
    assert "BUILD THE WHOLE" in s
    assert "four to eight things" in s


def test_the_scene_comes_from_the_shot_and_not_from_the_style():
    """Four presets already said only "the four to eight things that say where
    this is" and stopped, and their galleries were not full of somebody else's
    scenery. The three that listed nouns now say where the nouns come from."""
    for name in ("stickman", "ink_woodcut", "ink_explainer"):
        assert "from the shot's own description and from nothing else" in \
            STYLES[name], name


def test_stickman_still_says_animals_are_drawn_properly():
    """The contrast IS the style — stick people, real animals — and it has to
    survive losing the zebra and the lion that illustrated it."""
    s = STYLES["stickman"]
    assert "ANIMALS AND OBJECTS ARE DRAWN PROPERLY" in s
    assert "true shape, proportions and markings" in s
    assert "stay simple stick figures" in s
