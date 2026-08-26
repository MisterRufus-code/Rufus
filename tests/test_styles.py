"""The look presets, and the two things a real gallery of them got wrong.

A style block is appended to every image prompt byte for byte, so anything it
forbids is forbidden in every frame of every video on that channel. That makes
it the highest-leverage text in the repo and the easiest place to be quietly,
consistently wrong — which is what happened: the owner ran a full stickman
sequence, opened the gallery, and every picture was a figure floating in white
with the same faint smile.

Both causes were single clauses in this file. These tests keep them out.
"""

import os
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

STYLES = json.loads((ROOT / "config" / "styles.json").read_text(encoding="utf-8"))


def _looks() -> dict:
    return {k: v for k, v in STYLES.items() if not k.startswith("_")}


# TWO DIFFERENT JOBS, AND ONLY ONE OF THEM IS A SEQUENCE.
#
# Eight of these presets render the ten beats of one video, so "the place
# persists across shots" and "four to eight things that say where this is" are
# load-bearing: they are what stop ten drawings of the same story looking like
# ten different worlds, and what stopped the white-void bug.
#
# `thumbnail` renders ONE picture, looked at 168 pixels wide in a phone feed.
# There are no consecutive shots for a horizon to stay level across, and four
# to eight background objects at that size are mush. It is held to its own
# version of the same protections below — specify the background, never leave
# it bare, read at thumbnail size — not exempted from them.
_SEQUENCE_ONLY = {"thumbnail"}

# THE ONE PRESET THAT DELIBERATELY DOES NOT LEGISLATE THE PLACE.
#
# The owner ran a control the pipeline could not: the same model, the same
# workflow, the same cfg, and a hand-written sixty-word prompt from Gemini —
#
#   "A 2D flat minimalist cartoon illustration featuring three iconic starter
#    Pokemon ... standing together side by side on a simple grassy field ...
#    Squirtle is smiling. Simple facial expressions, clean thick black
#    outlines, flat solid colors, bright sunny day, light blue sky ..."
#
# It came back excellent. The same workflow fed 859 words of stickman_lean
# came back poor, and the measurement says why: 859 words is fifteen chunks of
# 77-token CLIP conditioning, and the shot itself is one of them.
#
# Note what carries the place in the prompt that worked: "a simple grassy
# field", "bright sunny day, light blue sky" — the SHOT names it. Nothing in
# that prompt legislates backgrounds in general.
#
# stickman_micro is that bet, kept as a separate preset so it can be A/B'd
# against stickman_lean on the same topic without touching a single rule that
# was won the hard way. It keeps every cheap non-negotiable — the lettering
# ban, the photographic tells, the background-is-simpler-not-fainter clause —
# and drops only the twenty-five words that build a place, because the shot is
# supposed to do that. If the galleries say otherwise, delete the preset; the
# tests below are what it promises in the meantime.
_MICRO = {"stickman_micro"}


# EVERY PRESET THAT DRAWS THIS CHANNEL'S STICK-FIGURE LOOK.
#
# `stickman_lean` is the same rules with the prose taken out — 4,816
# characters down to 3,048 — written because a probe showed the block's tail
# was not being read at all. The claim being made about it is precisely "the
# same rules, fewer words", and the only way to know a 37% cut did not quietly
# drop one of them is to run every scar test below over both presets. Each of
# those tests is a bug that reached a gallery; a leaner block that loses one
# is not leaner, it is a regression with a smaller file size.
_STICK = ["stickman", "stickman_lean"]


def _story_looks() -> dict:
    return {k: v for k, v in _looks().items()
            if k not in _SEQUENCE_ONLY and k not in _MICRO}


def test_every_preset_is_one_block_of_text():
    """The suffix is pasted verbatim into every prompt. A list or a dict here
    would render as its Python repr."""
    for name, text in _looks().items():
        assert isinstance(text, str) and len(text) > 200, name


@pytest.mark.parametrize("stick", _STICK)
def test_stickman_asks_for_a_background_that_places_the_scene(stick):
    """THE WHITE-VOID BUG. The preset said "on a pure white background" and
    "Everything not deliberately filled with colour is pure white", so a
    storyboard that carefully built a medieval hall with a beam of light from
    a high window rendered as two stick figures and a table in empty space.
    The style was overriding the whole scene description, every frame."""
    s = _looks()[stick]
    assert "pure white background" not in s
    assert "not deliberately filled with colour is pure white" not in s
    assert "BUILD THE WHOLE PLACE" in s
    # THE RULE, NOT THE SCENERY. It used to pin "horizon line", and a horizon
    # is a thing you can draw — appended to every prompt it put an open
    # landscape behind an alleyway. The place instruction has to survive; the
    # nouns it used to name must not. See test_the_place_rule_names_no_scenery.
    assert "surface for the subject to stand on" in s
    assert "four to eight things" in s


@pytest.mark.parametrize("stick", _STICK)
def test_stickman_backgrounds_stay_out_of_the_subjects_way(stick):
    """The fix must not swing into the other failure — a busy background on a
    frame that is on screen for four seconds at thumbnail size reads as
    noise.

    NOT BY FADING IT, which is how this was worded before and what the second
    gallery showed: "soft muted flat colours, drawn thinner and paler than the
    foreground" produced sixty beige stills where only the figure looked
    finished. A background is quieter because it is simpler and further away,
    not because it is washed out."""
    s = _looks()[stick]
    assert "quieter than the subject" in s
    assert "reads instantly at thumbnail size" in s
    assert "paler than the foreground" not in s


@pytest.mark.parametrize("stick", _STICK)
def test_stickman_faces_carry_the_emotion(stick):
    """The preset pinned every mouth to "a single thin curved line with a
    slight upturn" and banned eyebrows outright, so ten shots of a country
    losing its money came back with ten mild smiles. Eyebrows and a mouth
    curve are the entire emotional vocabulary of this art style."""
    s = _looks()[stick]
    assert "no eyebrows" not in s
    assert "eyebrow strokes" in s
    assert "THE FACE CARRIES THE EMOTION OF THE MOMENT" in s
    assert "must not be the same on every figure" in s


@pytest.mark.parametrize("stick", _STICK)
@pytest.mark.parametrize("feeling", ["anger", "shock", "delight"])
def test_stickman_spells_out_how_to_draw_a_feeling(feeling, stick):
    """Naming the feeling is not enough — the model needs the geometry, since
    "sad" on a face with two dot eyes has no obvious drawing."""
    assert feeling in _looks()[stick]


@pytest.mark.parametrize("stick", _STICK)
def test_stickman_keeps_the_line_art_that_was_working(stick):
    """The character consistency across the owner's gallery came from these
    exact constraints. The fix is additive or it trades one problem for a
    worse one."""
    s = _looks()[stick]
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


@pytest.mark.parametrize("name", sorted(_story_looks()))
def test_every_preset_builds_a_place_and_keeps_it(name):
    """A style is appended to every prompt byte for byte, so a preset that
    says nothing about the background lets the model decide — and the model
    decides blank paper. ink_woodcut was the one preset with no scene clause
    at all, because the shared one is written for flat colour and an engraving
    has none."""
    s = _looks()[name]
    assert "BUILD THE WHOLE" in s, "no scene instruction at all"
    assert ("same horizon height" in s or "same eye level" in s), \
        "the place has to persist across shots"
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
# three times, and the two earlier ones are pinned by their own tests:
#
#   tests/test_script_writer.py       the hook example "2,000 years ago"
#                                     coming back as generated hooks
#   tests/test_preanalysis_examples.py  the pre-analysis examples — "Still
#                                     scared to retire", the leaky faucet,
#                                     the envelope from the IRS
#
# and this is the third.

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


@pytest.mark.parametrize("name", sorted(_story_looks()))
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
    for name in ("stickman", "stickman_lean", "ink_woodcut", "ink_explainer"):
        assert "from the shot's own description and from nothing else" in \
            STYLES[name], name


@pytest.mark.parametrize("stick", _STICK)
def test_stickman_still_says_animals_are_drawn_properly(stick):
    """The contrast IS the style — stick people, real animals — and it has to
    survive losing the zebra and the lion that illustrated it."""
    s = _looks()[stick]
    # LOWERCASE NOW, AND THAT IS THE POINT. The block was split so an object
    # beat is never sent the figure rules, which put this sentence into the top
    # half of the string — where the banknote rule applies and a shouted
    # drawable noun is a candidate for being painted. The rule survives; only
    # its capitals were a casualty of moving it somewhere they were unsafe.
    assert "animals and objects are drawn properly" in s.lower()
    assert "true shape, proportions and markings" in s
    assert "stay simple stick figures" in s


# ── where in the block a rule sits ───────────────────────────────────────────
#
# THE COMPLAINT, TWICE: every background comes back pale beige, and stickman
# forbids exactly that IN WORDS — "never wash the background out, never draw it
# paler or thinner than the foreground, and never leave it as bare paper".
#
# The first explanation was the checkpoint: z_image_turbo runs at CFG 1, so the
# negative prompt has no effect and suppression has to come from the positive
# one. That is true and workflow_bench.advisories says so. But it does not
# explain why the FIGURES obey their instructions while the BACKGROUND ignores
# its own — both are in the same positive prompt.
#
# What separates them is position. The block is ~3,500 characters and is
# appended AFTER the shot description, and the colour rules used to start at
# character 2,291 — the last third of the last thing in a prompt of roughly a
# thousand tokens. The rules that are obeyed were the ones at the top.
#
# So this is a positional experiment, not a fix that is known to work: the
# scene-and-colour paragraphs moved to just after the opening sentence.
# stickman ONLY, deliberately — ink_explainer and the rest keep the old order
# as the control, because two changes at once produce a result you cannot
# explain. If the next gallery has coloured backgrounds, the same move applies
# to the others and this comment becomes the reason. If it does not, the
# checkpoint was the whole story after all and this should be reverted rather
# than left as folklore.

@pytest.mark.parametrize("stick", _STICK)
def test_the_colour_rule_is_near_the_top_of_the_stickman_block(stick):
    s = _looks()[stick]
    at = s.index("COLOUR IS FLAT")
    assert at < len(s) // 2, (
        f"the colour rule is at character {at} of {len(s)} — back in the tail "
        f"of the block, where the pale-background bug lived")


@pytest.mark.parametrize("stick", _STICK)
def test_the_block_does_not_open_on_a_shouty_heading(stick):
    """THE BANKNOTE THAT SAID "BEHIND THE FIGURES".

    The first attempt moved the whole scene-and-colour section to the front,
    heading and all, so the block opened "...uniform weight. BEHIND THE
    FIGURES, BUILD THE WHOLE PLACE." The next gallery had colour in every
    background — the experiment worked — and one frame was a banknote with
    BEHIND THE FIGURES printed across it in serif capitals.

    Of course it was. An all-caps phrase at the head of a prompt is the
    strongest emphasis the prompt has, "FIGURES" is a drawable noun, and the
    model had just been handed a title. This repo keeps relearning that a
    style block has no meta level: every word in it is a word in the prompt.

    So the colour paragraph — the half that fixed the backgrounds — stays at
    the top, and the heading goes back into the body where it reads as an
    instruction rather than as a caption.
    """
    s = _looks()[stick]
    opening = s[:200].upper()
    for heading in ("BEHIND THE FIGURES", "THE FACE CARRIES", "NO LETTERING",
                    "THE PLACE PERSISTS", "ANIMALS AND OBJECTS"):
        assert heading not in opening, (
            f"the block opens on {heading!r} — an all-caps heading in the "
            f"first breath of a prompt gets drawn, not obeyed")


@pytest.mark.parametrize("stick", _STICK)
def test_the_experiment_did_not_lose_a_single_rule(stick):
    """Reordering must be a move, not an edit. Every sentence that was in the
    block before has to still be in it — a rule quietly dropped during a
    rearrangement is indistinguishable from the bug being 'fixed'."""
    s = _looks()[stick]
    for rule in ("never leave it as bare paper",
                 "Never wash the background out",
                 "BUILD THE WHOLE PLACE",
                 "THE PLACE PERSISTS",
                 "NO LETTERING ANYWHERE IN THE FRAME",
                 "THE FACE CARRIES THE EMOTION",
                 "No gradients"):
        assert rule in s, rule


# ── how big the people are ───────────────────────────────────────────────────
#
# THE FRAME OF EIGHTEEN. With Z-Image-Base at CFG 4 the colour rule and the
# lettering rule are both obeyed — blue sky, green ground, blank paper. What
# arrived with that obedience was a landscape: frame 01 is a vast green field
# with one small figure in the bottom corner, and 13 and 15 put their people
# in a quarter of the frame with three quarters of empty ground under them.
#
# Nothing was broken. The preset says "a ground plane filling the bottom third
# to half of the frame, a horizon line across it, and open sky above" — a
# landscape instruction with no subject in it — and a model that now follows
# instructions literally drew exactly that. The block had no rule about how
# big a person should be, so the only honest fix is to write one rather than
# to weaken the scene rule that took three galleries to get right.
#
# It leads on SCALE and not on PEOPLE or FIGURES on purpose. "COLOUR" survived
# being promoted to the head of the block because nobody can draw a colour;
# "BEHIND THE FIGURES" came back printed across a banknote. A capitalised
# drawable noun near the front of a prompt is a subject.

@pytest.mark.parametrize("stick", _STICK)
def test_the_preset_says_how_big_a_person_is_in_frame(stick):
    s = _looks()[stick]
    assert "SCALE:" in s
    assert "half and three quarters of the frame" in s


@pytest.mark.parametrize("stick", _STICK)
def test_the_scale_rule_is_in_the_half_of_the_block_that_gets_obeyed(stick):
    s = _looks()[stick]
    assert s.index("SCALE:") < len(s) // 2


# ── SCALE was overruling the camera the storyboard had already chosen ────────
#
# "Close on" never once produced a close-up. Not on either preset, not with
# the shot first or last, not in any probe run.
#
# storyboard.py already solves this and solves it well: it asks the model for
# a `framing` on every shot — wide|mid|close|detail — varies them so no
# distance runs three times, maps each to a phrase the image model
# understands (_FRAMINGS), and _apply_framing puts that phrase FIRST, with a
# docstring explaining that this is "the one instruction that decides what the
# picture IS rather than what is in it".
#
# And then SCALE, appended to every prompt byte for byte, said the figures
# stand between half and three quarters of the frame's height. That is a
# mid-shot, hardcoded onto every beat. On a `close` beat — "head and
# shoulders, filling most of the frame" — it contradicts it outright. On a
# `wide` beat — "the figures are small within it" — it contradicts it the
# other way.
#
# Neither half was wrong alone. SCALE stopped a real bug (a vast green field
# with one small figure in the corner) by making every shot a mid-shot; the
# framing vocabulary was added later to stop a sequence reading as a
# slideshow. Nothing told either about the other, which is this repo's
# recurring failure and not a new one: a constant quietly beating a variable.

# ── AND QUALIFYING IT WAS NOT ENOUGH ─────────────────────────────────────────
#
# The first fix wrote the conflict away in words: "unless the shot names its
# own distance ... the shot's distance always wins". Six probes later, `face`
# still came back full-body against a prompt whose FIRST words were "Close on
# one figure's face" — the acceptance test, failed.
#
# It could not have worked. A text encoder has no meta level, which is the
# lesson _detail_for_shot already carries in its own docstring. "half and three
# quarters of the frame's height" is concrete and drawable; "unless the shot
# names its own distance" is abstract and is not. Both reach the latent
# together and the drawable one wins.
#
# So the default is now DELIMITED and DELETED — the same move the block
# already makes at FIGURE ONLY, for the same stated reason: the only reliable
# way not to get a thing is not to mention it.

@pytest.mark.parametrize("stick", _STICK)
def test_the_default_distance_is_delimited_so_it_can_be_removed(stick):
    import comfy_client
    s = _looks()[stick]
    assert comfy_client.STYLE_SCALE_OPEN in s
    assert comfy_client.STYLE_SCALE_CLOSE in s
    head, _, rest = s.partition(comfy_client.STYLE_SCALE_OPEN)
    default, _, _ = rest.partition(comfy_client.STYLE_SCALE_CLOSE)
    # The contradicting sentence, and ONLY it, sits between the markers.
    assert "half and three quarters of the frame" in default
    assert "half and three quarters of the frame" not in head
    # The clause the white-void bug needs stays OUTSIDE, always shipped.
    assert "never instead of them" in head


# ── AND DELETING THE DISTANCE WAS NOT ENOUGH EITHER ──────────────────────────
#
# The bench of 2026-08-23 ran with the default distance gone and `face` came
# back full-body anyway — head to feet, against a prompt whose first words were
# "Close on one figure's face".
#
# The absence of a contradiction is not the presence of an instruction. Twelve
# other phrases in the block still required a whole body and a landscape, and
# not one of them can be drawn inside a head-and-shoulders frame:
#
#     a ground plane across the bottom third, a horizon line, open sky
#     two legs; each leg bends once at the knee
#     both arms are visible in every figure
#     PROPORTION IS FIXED: the head is one third of the whole figure's height
#
# The FIGURE half was written, every sentence of it, for a full-body mid-shot.
# So the style did not need a better sentence about distance — it needed the
# rules that ASSUME a distance to stop shipping when the shot names a closer
# one. Same mechanism as DEFAULT DISTANCE, second pair of markers.

# What only a camera far enough back can show. Kept as one list because the
# whole failure was that nobody had ever counted them.
# Phrased so both presets match: `stickman` says "a ground plane filling the
# bottom third to half of the frame" and "two thin legs" where `stickman_lean`
# says "across the bottom third" and "two legs".
_NEEDS_ROOM = [
    "surface for the subject to stand on", "closes off the distance",
    "BUILD THE WHOLE PLACE", "FIVE SEPARATE PARTS",
    "each leg bends once at the knee",
    "both arms are visible in every figure", "PROPORTION IS FIXED",
    "the legs the remaining third",
]

# True at any distance. A close-up that lost these would be a different style.
_ALWAYS = [
    "perfect oval head", "THE FACE CARRIES THE EMOTION", "COLOUR IS FLAT",
    "NO LETTERING", "no rounded 3D volume",
    "two different constructions in one picture",
]


def _composed(stick, shot):
    """The prompt the pipeline would actually send for this shot."""
    import comfy_client
    prev_style = os.environ.get("RUFUS_STYLE")
    prev_detail = os.environ.pop("RUFUS_STILLS_DETAIL", None)
    os.environ["RUFUS_STYLE"] = stick
    try:
        return comfy_client._with_detail(shot)
    finally:
        if prev_style is None:
            os.environ.pop("RUFUS_STYLE", None)
        else:
            os.environ["RUFUS_STYLE"] = prev_style
        if prev_detail is not None:
            os.environ["RUFUS_STILLS_DETAIL"] = prev_detail


@pytest.mark.parametrize("stick", _STICK)
def test_a_close_up_is_not_told_to_draw_legs_and_a_horizon(stick):
    """THE ACCEPTANCE TEST, AS A TEST. `face` asked for a close-up in six probe
    runs and two bench columns and never once got one."""
    got = _composed(stick, "Close on one figure's face at the moment the news "
                           "lands: brows raised high, mouth a small open oval.")
    for phrase in _NEEDS_ROOM:
        assert phrase not in got, phrase


@pytest.mark.parametrize("stick", _STICK)
def test_a_close_up_still_gets_the_face_and_the_look(stick):
    """The other direction, and the more dangerous one: dropping too much would
    buy a close-up by losing the style it is drawn in."""
    got = _composed(stick, "Close shot: head and shoulders, filling most of the "
                           "frame. One figure hears the news.")
    for phrase in _ALWAYS:
        assert phrase in got, phrase


@pytest.mark.parametrize("stick", _STICK)
def test_a_shot_that_names_no_distance_still_gets_every_rule(stick):
    """The white-void bug and the figure-in-the-corner bug are both still real.
    This mechanism must only fire when a shot has asked it to."""
    got = _composed(stick, "A clerk pushes a ledger across a counter.")
    for phrase in _NEEDS_ROOM:
        assert phrase in got, phrase
    assert "half and three quarters of the frame" in got


@pytest.mark.parametrize("stick", _STICK)
def test_wide_and_mid_keep_the_whole_body(stick):
    """A wide shot needs the ground plane MORE than a mid shot does — it is the
    framing the place rule was written for."""
    import storyboard
    for name in ("wide", "mid"):
        got = _composed(stick, f"{storyboard._FRAMINGS[name]}. A clerk reacts.")
        for phrase in _NEEDS_ROOM:
            assert phrase in got, f"{name}: {phrase}"


@pytest.mark.parametrize("stick", _STICK)
def test_no_marker_ever_reaches_the_image_model(stick):
    """A separator rendered as a literal line of text is a defect this file has
    already shipped once — see tests/test_style_probe.py."""
    import comfy_client, storyboard
    shots = ["A clerk pushes a ledger across a counter."] + [
        f"{p}. A clerk reacts." for p in storyboard._FRAMINGS.values()]
    for shot in shots:
        got = _composed(stick, shot)
        for marker in (comfy_client.STYLE_SCALE_OPEN, comfy_client.STYLE_SCALE_CLOSE,
                       comfy_client.STYLE_FARSHOT_OPEN, comfy_client.STYLE_FARSHOT_CLOSE,
                       comfy_client.STYLE_FIGURE_MARKER):
            assert marker not in got, f"{marker} in {shot[:30]}"


@pytest.mark.parametrize("stick", _STICK)
def test_every_close_framing_the_storyboard_can_pick_is_recognised(stick):
    """Ties the two files together the same way the DEFAULT DISTANCE test does:
    storyboard may add a fifth framing, and if it names a near one this fails
    rather than a gallery."""
    import comfy_client, storyboard
    assert comfy_client.is_close_shot(storyboard._FRAMINGS["close"])
    assert comfy_client.is_close_shot(storyboard._FRAMINGS["detail"])
    assert not comfy_client.is_close_shot(storyboard._FRAMINGS["wide"])
    assert not comfy_client.is_close_shot(storyboard._FRAMINGS["mid"])


@pytest.mark.parametrize("stick", _STICK)
def test_the_conditional_wording_that_did_not_work_is_gone(stick):
    """Left in alongside the mechanism it would be two rules for one job, and
    the losing one still ships in every prompt."""
    s = _looks()[stick]
    assert "unless the shot names its own distance" not in s
    assert "The shot's distance always wins" not in s


@pytest.mark.parametrize("stick", _STICK)
def test_the_style_leaves_room_for_every_framing_the_storyboard_can_choose(stick):
    """THE TEST THAT TIES THE TWO FILES TOGETHER. storyboard._FRAMINGS is the
    list of distances a shot may ask for; this block is appended to every one
    of them. An edit to either file now has to face the other, which is
    exactly what neither of them did the first time.

    It asserts BEHAVIOUR now, not vocabulary: every phrase storyboard can put
    in front of a shot must be recognised as naming a distance, and must
    actually remove the default from the block that follows it."""
    import comfy_client, storyboard
    assert set(storyboard._FRAMINGS) == {"wide", "mid", "close", "detail"}
    monkey = os.environ.get("RUFUS_STYLE")
    prev_detail = os.environ.pop("RUFUS_STILLS_DETAIL", None)
    os.environ["RUFUS_STYLE"] = stick
    try:
        for name, phrase in storyboard._FRAMINGS.items():
            assert comfy_client.names_own_distance(phrase), name
            got = comfy_client._with_detail(f"{phrase}. A clerk reacts.")
            assert "half and three quarters of the frame" not in got, name
            assert comfy_client.STYLE_SCALE_OPEN not in got, name
        # A beat that names no distance still gets the default, because
        # deleting it outright brings back the field with the figure in the
        # corner that SCALE was written for.
        plain = comfy_client._with_detail("A clerk reacts.")
        assert "half and three quarters of the frame" in plain
        assert comfy_client.STYLE_SCALE_OPEN not in plain
    finally:
        if monkey is None:
            os.environ.pop("RUFUS_STYLE", None)
        else:
            os.environ["RUFUS_STYLE"] = monkey
        if prev_detail is not None:
            os.environ["RUFUS_STILLS_DETAIL"] = prev_detail


@pytest.mark.parametrize("stick", _STICK)
def test_the_scale_rule_does_not_undo_the_scene_rule(stick):
    """The opposite failure is the white-void bug, which cost two galleries:
    a figure with nothing behind it. Filling the frame with the subject must
    not mean deleting the place."""
    s = _looks()[stick]
    assert "built BEHIND them, never instead of them" in s
    assert "BUILD THE WHOLE PLACE" in s
    assert "four to eight things" in s


@pytest.mark.parametrize("stick", _STICK)
def test_the_scale_rule_names_no_drawable_noun_in_capitals(stick):
    """Same rule the banknote taught. Anything shouted near the front of the
    block is a candidate for being painted."""
    import re
    # THE SHARED SECTION, NOT THE FIRST HALF OF THE STRING. The block is split
    # at FIGURE ONLY so a beat whose subject is an object is never sent the
    # figure rules at all. Everything before the marker ships with EVERY
    # prompt, so all of it is "the front" as far as this rule is concerned —
    # and everything after it is allowed to shout FIGURE, because it only
    # arrives when there is a figure to draw.
    s = _looks()[stick].split("--- FIGURE ONLY ---")[0]
    head = s
    for shout in re.findall(r"\b[A-Z]{3,}(?:\s+[A-Z]{2,})*\b", head):
        for noun in ("FIGURE", "PEOPLE", "PERSON", "MAN", "WOMAN", "ANIMAL",
                     "COIN", "PAPER", "SIGN"):
            assert noun not in shout, f"{shout!r} contains a drawable noun"


# ── how big the head is compared with the body ───────────────────────────────
#
# The eighteen-frame sheet had figures that were individually fine and
# collectively not one character: 05 and 09 carry heads roughly half the
# figure's height, 01 and 13 put tiny heads on spidery bodies. The owner read
# it as the figures being "a bit strange", which is exactly what inconsistent
# proportion looks like when every single frame is defensible on its own.
#
# The word "proportions" IS in this preset — applied to animals and objects,
# "a real illustration with its true shape, proportions and markings". For the
# PERSON there is a shape rule ("a perfect oval head slightly taller than
# wide") and a parts list ("one straight line for the torso, thin straight
# arms angled at the elbow") and no size relationship between any of them. The
# model was free to choose, and chose again every frame.
#
# Same class of gap as SCALE, and the same lesson the face fix taught: naming
# the thing is not enough, the model needs the geometry.

@pytest.mark.parametrize("stick", _STICK)
def test_the_figure_has_a_stated_head_to_body_ratio(stick):
    s = _looks()[stick]
    assert "PROPORTION IS FIXED" in s
    assert "one third of the whole figure's height" in s


@pytest.mark.parametrize("stick", _STICK)
def test_the_proportion_rule_covers_all_three_parts(stick):
    """A ratio for the head alone leaves the torso and legs free, which is
    most of the drift."""
    s = _looks()[stick]
    for part in ("head is about", "torso line about", "legs the remaining"):
        assert part in s, part


@pytest.mark.parametrize("stick", _STICK)
def test_the_proportion_rule_says_it_holds_across_shots(stick):
    """Per-frame correctness was never the problem — every frame in the sheet
    was defensible alone. Consistency between them is the thing being asked
    for, so the block has to ask for it in those words."""
    s = _looks()[stick]
    assert "DOES NOT DRIFT BETWEEN SHOTS" in s


@pytest.mark.parametrize("stick", _STICK)
def test_the_proportion_rule_did_not_displace_the_body_parts_list(stick):
    s = _looks()[stick]
    for kept in ("bends once at the elbow", "bends once at the knee",
                 "Small simple hands", "no muscle lines, no shading"):
        assert kept.lower() in s.lower(), kept


# ── the clause against writing was commissioning the things that carry it ────
#
# THE FOURTH TIME. Every preset ended with:
#
#   "NO LETTERING ANYWHERE IN THE FRAME: no words, no captions, no labels, no
#    signs, no titles, no numbers on a page or A COIN FACE. A surface that
#    would carry writing — A BOOK, A LEDGER, A DOCUMENT, A BANNER — is drawn
#    BLANK..."
#
# Five drawable objects, named, in a block appended byte for byte to every
# prompt. The owner asked why gold coins keep appearing. They keep appearing
# because every prompt on this channel asks for a coin. The eighteen-frame
# sheet also carried a book lying in an empty field, a document being signed,
# a loose sheet of paper and a chart on a stand — the whole list, drawn.
#
# The mechanism is the one this repo has now met four times: a style block has
# no meta level, and CLIP has no "not". "No numbers on a coin face" is read as
# numbers, and a coin face. The prior three were the hook example, the
# pre-analysis examples, and the lion — and this one hid behind the word NO
# for longer than any of them.
#
# The rule survives; the inventory does not. "Whatever THIS shot happens to
# contain that would normally carry writing is drawn BLANK" says the same
# thing and names nothing.

_OBJECTS_THE_NO_LETTERING_CLAUSE_ORDERED = [
    "a coin", "coin face", "a book", "a ledger", "a document", "a banner",
    "on a page",
]


@pytest.mark.parametrize("name", sorted(_looks()))
def test_no_preset_lists_the_things_that_carry_writing(name):
    s = _looks()[name].lower()
    named = [n for n in _OBJECTS_THE_NO_LETTERING_CLAUSE_ORDERED if n in s]
    assert not named, (
        f"{name} names {named} inside its no-lettering rule — appended to "
        f"every prompt, a forbidden object is still an object in the prompt")


@pytest.mark.parametrize("name", sorted(_looks()))
def test_the_no_lettering_rule_itself_survives(name):
    """Deleting the inventory must not delete the instruction. The lettering
    bug it exists to stop is real and cost two galleries."""
    s = _looks()[name]
    assert "NO LETTERING ANYWHERE IN THE FRAME" in s
    assert "is drawn BLANK" in s
    assert "wordless ruled lines" in s


def test_a_medium_named_after_a_book_is_not_an_object(name=None):
    """storybook illustration, picture-book, sketchbook page — these describe
    what the drawing IS, not something in it, and the check above must not
    sweep them away with the inventory it is aimed at."""
    joined = " ".join(_looks().values()).lower()
    assert "storybook illustration" in joined or "picture-book" in joined


# ── the preset was telling the model to draw two different bodies ────────────
#
# THE SEVENTEEN-FRAME SHEET. SCALE landed — the figures fill the frame now —
# and what was left looked to the owner like the anatomy being wrong. It is
# not anatomy. Two sentences in this preset asked for incompatible things:
#
#   "The body is stick-figure: ONE STRAIGHT LINE FOR THE TORSO ... no filled
#    body mass, no volume."
#   "CLOTHING is drawn as simple line-art garments with FLAT UNSHADED COLOUR
#    FILLS INSIDE THE OUTLINES."
#
# A garment with a colour fill needs a shape to fill. A torso that is one line
# has none. Frames 02 and 08 obey the first sentence and come back as bare
# scribbled strokes; 01, 05 and 15 obey the second and come back as solid
# white bodies wearing something. The model picked a different sentence each
# frame, which is exactly what "the figures are a bit strange" looks like when
# no single frame is wrong.
#
# The filled construction is the one the owner's own reference frames use, so
# that is the half that stays. "No volume" survives as "no shading, no rounded
# 3D volume" — the clause was protecting against rendering, and only got
# entangled with body mass by being in the same list.

@pytest.mark.parametrize("stick", _STICK)
def test_the_body_is_built_one_way_and_only_one_way(stick):
    s = _looks()[stick]
    assert "white fill inside the same clean black outline" in s
    assert "no filled body mass" not in s, "the contradiction is back"
    assert "one straight line for the torso" not in s


@pytest.mark.parametrize("stick", _STICK)
def test_the_body_and_the_clothing_rule_can_both_be_obeyed(stick):
    """The test that would have caught this: a garment needs something to sit
    on. If the block ever again says the torso has no fill AND that clothing
    fills it, one of them loses at random, per frame."""
    s = _looks()[stick]
    fills_clothing = "flat unshaded colour fills inside the outlines" in s
    denies_body = ("no filled body mass" in s or "no volume." in s
                   or "one straight line for the torso" in s)
    assert not (fills_clothing and denies_body)


@pytest.mark.parametrize("stick", _STICK)
def test_the_anti_rendering_rules_survived_the_rewrite(stick):
    """Dropping "no volume" wholesale would invite soft 3D shading back, which
    is the tell that stops a drawing looking drawn."""
    s = _looks()[stick].lower()
    for kept in ("no muscle lines", "no shading", "no rounded 3d volume",
                 "no gradients", "no film grain"):
        assert kept in s, kept


@pytest.mark.parametrize("stick", _STICK)
def test_the_preset_says_not_to_mix_constructions_within_one_picture(stick):
    """Per-frame consistency was never the complaint either — 08 has three
    figures built two different ways in the same drawing."""
    assert "two different constructions in one picture" in _looks()[stick]


# ── the overcorrection: from two bodies to no arms ───────────────────────────
#
# Fixing the two-constructions contradiction introduced a worse one. The
# replacement opened "The body is a stick figure built from FLAT FILLED
# SHAPES" and closed "Every part is white fill inside an outline", and the
# model read the pair as licence to draw ONE filled shape. The next gallery is
# a row of white pill-shaped figures with heads and legs and NO ARMS — the
# arms absorbed into the torso silhouette.
#
# Both halves of that sentence were mine and both were true individually. The
# missing word was SEPARATE. "Filled" answers what a limb is made of; nothing
# answered whether it is its own shape.
#
# So the rule now counts the parts, says they are drawn apart with daylight
# between them, and names the failure it is guarding against — one filled blob
# with a head on top. The fill survives, because bare pen strokes were the
# original bug and are not the fix for this one.

@pytest.mark.parametrize("stick", _STICK)
def test_the_body_is_described_as_separate_parts(stick):
    s = _looks()[stick]
    assert "FIVE SEPARATE PARTS" in s
    assert "never merged into a single silhouette" in s


# ROUND FOUR, AND THIS ONE OVERSHOT.
#
# The three rounds above each pushed harder on SEPARATION, and the block ended
# up saying it four times in one sentence: separate parts, drawn apart, daylight
# between them, clear of the torso. It never once said the arm was ATTACHED.
#
# The channel owner reported figures with a hole in the shoulder. A text encoder
# has no meta level — this block has been caught believing otherwise five times
# — so "daylight between the torso and the arms", rendered literally at the
# shoulder, IS a hole in the shoulder. The model was obeying the only
# instruction it had.
#
# So the shoulder now states the join and the separation moves to where it
# actually earns its keep: the flared-outline clause, which still says the arms
# are clear of a garment outline. That is the case separation was introduced
# for. Diagnosed from the owner's description, not yet from a frame.

@pytest.mark.parametrize("stick", _STICK)
def test_the_arms_are_required_to_be_visible(stick):
    """A gallery of armless figures is what happens when the block says what
    an arm is made of and never says there has to be one."""
    s = _looks()[stick]
    assert "both arms are visible in every figure" in s


@pytest.mark.parametrize("stick", _STICK)
def test_the_arm_is_told_where_it_joins_the_body(stick):
    """Four ways of saying the arm is separate and none saying it is attached
    is how a shoulder ends up with a hole in it."""
    s = _looks()[stick]
    assert "GROWS OUT OF the torso at the shoulder" in s
    assert "closing against it" in s
    assert "daylight between them" not in s


@pytest.mark.parametrize("stick", _STICK)
def test_separation_still_survives_where_it_was_won(stick):
    """It is removed from the shoulder, not from the block: the garment outline
    that swallows the arms is the failure separation was introduced against, and
    that clause still states it."""
    s = _looks()[stick]
    assert "arms clear of it" in s
    assert "never merged into a single silhouette" in s


# ── two rules the 37% cut lost, and no test noticed ──────────────────────────
#
# stickman_lean passed all twenty-five scar tests above and still shipped two
# regressions, both visible in the first six-probe gallery it rendered:
#
#   crowd   five figures, FOUR OF THEM WITH NO FACE AT ALL — no eyebrows, no
#           mouth, two dots and nothing else. The full block says "Every
#           figure has a perfect oval head ..."; the cut turned that into a
#           heading, "Head:", which reads as one head, and the model drew the
#           nearest one.
#   crowd   the hammer floats in mid-air and all five sets of arms hang dead
#           straight at their sides. The full block carries the geometry —
#           "shoulders thrown up, arms reaching, back hunched, head dropped,
#           one figure leaning away from another" — and the cut reduced it to
#           "The pose carries it too."
#
# Both are the SAME lesson the face vocabulary already taught this file and
# that its own test docstring states: naming the thing is not enough, the
# model needs the geometry. "The pose carries it too" is a label for a rule,
# not a rule. Neither had a test, so both cuts were green.

@pytest.mark.parametrize("stick", _STICK)
def test_the_head_rule_applies_to_every_figure_not_just_the_nearest(stick):
    """A crowd shot is five faces, and four of them came back blank."""
    s = _looks()[stick]
    assert "Every figure" in s, "the head rule reads as one head"


@pytest.mark.parametrize("stick", _STICK)
def test_the_pose_vocabulary_is_geometry_and_not_a_label(stick):
    """"both arms are visible in every figure, DOING SOMETHING" is the rule
    this channel keeps failing — thirteen of sixteen frames with nobody doing
    anything is why the `action` probe exists. A sentence that says the pose
    matters, without saying what a pose looks like, buys nothing."""
    s = _looks()[stick]
    for shape in ("shoulders thrown up", "arms reaching", "back hunched",
                  "head dropped"):
        assert shape in s, shape


@pytest.mark.parametrize("stick", _STICK)
def test_the_pose_vocabulary_names_no_number_of_figures(stick):
    """"one figure leaning away from another" was in that list, and it is a
    FIGURE COUNT in a block appended to every prompt on this channel — the
    same class as the lion, the banknote, the coin face and the no-lettering
    inventory, every one of which is pinned above by a test whose rule is that
    the rule may be stated and the things may not be listed.

    The four shapes that stay are single-figure geometry and are what made the
    hammer in the crowd probe get picked up instead of floating. This one
    describes a second person, and the shots that asked for ONE figure came
    back with two and with four.

    It was in the block for a long time and it was in the list I wrote to pin
    the list — which is the mistake one level up: a phrase does not become
    safe by being old, or by having a test."""
    s = _looks()[stick]
    assert "leaning away from another" not in s
    assert "one figure leaning" not in s


@pytest.mark.parametrize("stick", _STICK)
def test_the_blob_is_named_as_the_thing_to_avoid(stick):
    """Naming the failure is what made the other rules stick — "a wide
    landscape with a small figure off to one side is a picture of a
    landscape" did more than any amount of describing the right answer."""
    assert "never one filled blob with a head on top" in _looks()[stick]


@pytest.mark.parametrize("stick", _STICK)
def test_the_fill_survived_the_correction(stick):
    """Bare pen strokes were the ORIGINAL bug. Swinging back to them to cure
    the blob would just restore the first failure — this is the third pass
    over these two sentences and each one has to keep the last one's win."""
    s = _looks()[stick]
    assert "white fill inside the same clean black outline" in s
    assert "not a bare pen stroke" in s
    assert "Never two different constructions in one picture" in s


# ── the dress, which turned out to be the blob ───────────────────────────────
#
# The owner's words: "the clothing looks like a dress". He is right, and it is
# the same defect as the merged silhouette that survived every other fix.
#
#   "Clothing is drawn as simple line-art garments with flat unshaded colour
#    fills inside the outlines only."
#
# Unconditional, in a block appended to every prompt, so EVERY figure gets a
# garment whether or not the shot mentions one. A garment drawn around a stick
# torso — which has no waist — is a single outline widening from the shoulders
# to the ground. That is the flared shape he is seeing. It is also the outline
# that swallows the arms, which is why FIVE SEPARATE PARTS kept losing no
# matter how loudly it was stated: the block was asking for the limbs to be
# separate and, one sentence later, for a shape drawn around all of them.
#
# The clause survives because the contradiction it settled is real (see
# test_the_body_and_the_clothing_rule_can_both_be_obeyed). It becomes
# conditional, which is what it always meant.

@pytest.mark.parametrize("stick", _STICK)
def test_clothing_is_drawn_only_when_the_shot_asks_for_it(stick):
    s = _looks()[stick]
    assert "Clothing ONLY when the shot says what someone is wearing" in s
    assert "When the shot says nothing about clothes, draw none" in s


@pytest.mark.parametrize("stick", _STICK)
def test_the_flared_outline_is_named_as_the_thing_to_avoid(stick):
    """Named as GEOMETRY, not as a garment. "never a dress" would put the word
    dress in every prompt on this channel, and this block has been caught four
    times over believing it has a meta level — the lion, the banknote, the coin
    face, the figure count. "One outline that widens from the shoulders down"
    describes the same failure and names nothing drawable."""
    s = _looks()[stick]
    assert "widens from the shoulders down" in s
    assert "hides the limbs inside it" in s
    for garment in ("dress", "gown", "robe", "skirt"):
        assert garment not in s.lower(), garment


# ── the shot's own face, against four recipes that ship with every prompt ────
#
# The face probe asks for "brows raised high, mouth a small open oval" — the
# block's own shock recipe, almost word for word — and came back drawn as
# grief. All four recipes are in every prompt, in full, so the shot's line is
# one of five competing descriptions of the same face and it is the shortest.
#
# Same disease as SCALE overruling the camera, and the same cure: the list is
# the default for a shot that says nothing, and the shot wins when it speaks.

@pytest.mark.parametrize("stick", _STICK)
def test_the_shot_s_own_face_beats_the_style_s_recipes(stick):
    s = _looks()[stick]
    assert "When the shot describes a face, draw exactly what it describes" in s
    assert "the shot's face always wins" in s
    # The recipes stay — they are what a beat that describes no face needs,
    # and deleting them brings back ten shots of a country losing its money
    # rendered as ten mild smiles.
    for feeling in ("anger", "shock", "grief", "delight"):
        assert feeling in s, feeling


@pytest.mark.parametrize("stick", _STICK)
def test_the_style_yields_to_the_shot_in_all_three_places(stick):
    """THE PATTERN, ASSERTED AS A PATTERN. Distance, face and clothing were
    each written as a channel-wide constant, and each one silently beat the
    shot that had asked for something else — the same failure three times, and
    the same failure as the port outage and as the 2.6% prompt.

    Anything added to this block that describes what a picture CONTAINS rather
    than how it is DRAWN belongs on this list, with a clause saying the shot
    wins."""
    import comfy_client
    s = _looks()[stick]
    # Distance yields by DELETION — see the DEFAULT DISTANCE tests above. The
    # other two still yield by a clause, because neither has a concrete
    # drawable default that contradicts the shot the way SCALE's did.
    assert comfy_client.STYLE_SCALE_OPEN in s
    assert "the shot's face always wins" in s
    assert "Clothing ONLY when the shot says" in s


@pytest.mark.parametrize("stick", _STICK)
def test_the_phrase_that_caused_the_blob_is_gone(stick):
    s = _looks()[stick]
    assert "FLAT FILLED SHAPES" not in s
    assert "Every part is white fill inside an outline" not in s


# ── the thumbnail look ───────────────────────────────────────────────────────
#
# Exempt from the two sequence rules (see _SEQUENCE_ONLY) and from nothing
# else: it still has to ban lettering and the photographic tells, which the
# shared parametrized tests above cover. These are its own versions of the
# protections those two rules were giving the story presets.

def test_the_thumbnail_preset_still_specifies_its_background():
    """THE WHITE-VOID BUG DOES NOT CARE THAT THIS IS ONE PICTURE. A preset
    that says nothing about the background lets the model decide, and the model
    decides blank paper. This one says what to build; it just builds less of
    it."""
    s = STYLES["thumbnail"]
    assert "BUILD THE WHOLE PLACE" in s
    assert "never leave it\nas bare paper" in s or "never leave it as bare paper" in s
    assert "two or three large shapes at most" in s


def test_the_thumbnail_preset_is_not_faded_either():
    """The correction that cost a gallery: quieter because simpler, not
    because washed out."""
    s = STYLES["thumbnail"]
    assert "because it is simpler and further away, not because it\nis faded" in s \
        or "because it is simpler and further away, not because it is faded" in s
    assert "paler than the foreground" not in s


def test_the_thumbnail_preset_leaves_room_for_the_headline():
    """thumbnail_gen.compose draws the words across the lower third. A picture
    with the subject centred there produces a headline over a face."""
    s = STYLES["thumbnail"]
    assert "LEAVE THE LOWER THIRD OPEN" in s


def test_the_thumbnail_preset_is_built_for_the_size_it_is_seen_at():
    s = STYLES["thumbnail"].lower()
    assert "postage stamp" in s
    assert "reads instantly at thumbnail size" in s


def test_the_thumbnail_preset_is_not_the_stickman():
    """It exists because stickman is thin uniform line art on white — built to
    carry a story across ten beats, and at 168x94 a white rectangle."""
    assert "ONE SUBJECT, ENORMOUS IN FRAME" in STYLES["thumbnail"]
    assert "ONE SUBJECT, ENORMOUS IN FRAME" not in STYLES["stickman"]


# ── not every beat has a person in it ────────────────────────────────────────
#
# THE OWNER'S OBSERVATION, AND IT WAS RIGHT: "if the beat is about a banana,
# you can just put a picture of a banana." The FLUX instruction already allowed
# that — its own example is "the first coins of Lydia -> a macro shot of
# ancient electrum Lydian stater coins", no figure anywhere. What overrode it
# was the style block, appended byte for byte to EVERY prompt, roughly half of
# which describes how to draw a body. A paragraph about arms and legs sitting
# directly after "a macro shot of a banana" produces a figure, because a style
# block has no meta level: every word in it is a word in the prompt.

MARKER = "--- FIGURE ONLY ---"


def test_the_stickman_block_separates_the_figure_rules():
    assert MARKER in STYLES["stickman"]


def test_an_object_shot_is_never_told_how_to_draw_a_body(monkeypatch):
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    monkeypatch.setenv("RUFUS_STYLE", "stickman")
    import comfy_client
    obj = comfy_client._detail_for_shot("object")
    for limb_rule in ("FIVE SEPARATE PARTS", "oval head", "elbow",
                      "mid-thigh", "PROPORTION IS FIXED"):
        assert limb_rule not in obj, limb_rule


def test_a_figure_shot_still_gets_every_rule_it_always_had(monkeypatch):
    """The partition must not lose anything — these were each written after a
    real gallery came back wrong."""
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    monkeypatch.setenv("RUFUS_STYLE", "stickman")
    import comfy_client
    fig = comfy_client._detail_for_shot("figure")
    for rule in ("FIVE SEPARATE PARTS", "never one filled blob with a head on top",
                 "two different constructions in one picture",
                 "COLOUR IS FLAT", "NO LETTERING ANYWHERE IN THE FRAME",
                 "BUILD THE WHOLE PLACE"):
        assert rule in fig, rule


def test_an_object_shot_keeps_the_rules_that_are_not_about_bodies(monkeypatch):
    """Colour, place and lettering are as true of a coin as of a person — and
    an object floating on white is the same white-void bug the place rule was
    written for."""
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    monkeypatch.setenv("RUFUS_STYLE", "stickman")
    import comfy_client
    obj = comfy_client._detail_for_shot("object")
    for rule in ("COLOUR IS FLAT", "BUILD THE WHOLE PLACE",
                 "NO LETTERING ANYWHERE IN THE FRAME",
                 "never leave it as bare paper"):
        assert rule in obj, rule


def test_the_medium_line_does_not_name_a_figure(monkeypatch):
    """It is the first sentence every prompt sees. It used to open "Minimalist
    stick-figure cartoon illustration", which asks for a stick figure on a beat
    about a coin."""
    # The FIRST sentence specifically. Later in the shared half, "the people
    # stay simple stick figures, while anything that is not a person is a real
    # illustration" has to name a stick figure — that contrast IS the style,
    # and it is stated as a comparison rather than as the thing to draw.
    first = STYLES["stickman"].split(".")[0].lower()
    assert "stick figure" not in first and "stick-figure" not in first
    assert "line art" in first, "the opening line no longer names the medium"


def test_a_preset_without_the_marker_behaves_exactly_as_before(monkeypatch):
    """Seven of the eight presets have no figure section. They must be handed
    the whole block for both kinds, not silently truncated."""
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    monkeypatch.setenv("RUFUS_STYLE", "flat_vector")
    import comfy_client
    assert MARKER not in STYLES["flat_vector"]
    assert comfy_client._detail_for_shot("object") == comfy_client._detail_for_shot("figure")


def test_an_untagged_prompt_is_treated_as_a_figure():
    """Every prompt written before this existed, every hand-typed one in the
    dashboard's regen box, and every prompt from a model that ignored the
    instruction arrives untagged — and for all of them the old behaviour is
    right. Defaulting to object would silently strip the figure rules from a
    run that never asked for that."""
    import comfy_client
    assert comfy_client.shot_kind("A man counting coins.") == "figure"
    assert comfy_client.shot_kind("") == "figure"


def test_the_tag_never_reaches_the_image_model(monkeypatch):
    monkeypatch.delenv("RUFUS_STILLS_DETAIL", raising=False)
    monkeypatch.setenv("RUFUS_STYLE", "stickman")
    import comfy_client
    out = comfy_client._with_detail("[SHOT=object] A macro shot of a banana.")
    assert "[SHOT=" not in out
    assert out.startswith("A macro shot of a banana")


def test_the_prompt_writer_is_asked_to_choose_a_kind():
    """A tag nothing produces is a default that never changes."""
    src = (ROOT / "scripts" / "main.py").read_text(encoding="utf-8")
    block = src.split("_FLUX_INSTRUCTION = (", 1)[1].split("client = OpenAI")[0]
    assert "[SHOT=figure]" in block and "[SHOT=object]" in block
    assert "CHOOSE HONESTLY" in block


# ── the colour rule taught its point with a landscape ────────────────────────
#
# "COLOUR IS FLAT, SATURATED AND REAL: grass is green, sky is a clear light
# blue with a few simple white clouds, earth and sand are warm brown, water is
# blue, stone is grey" — seven drawable nouns, appended to every prompt.
#
# A shot reading "an alleyway, empty except for shadows stretching across the
# cracked asphalt" therefore also asked for grass, sky, sand and water, and
# every probe of the evening rendered the same coastline whatever the shot
# said. The place rule did it too: "a ground plane across the bottom third, a
# horizon line, open sky".
#
# This is the rule the repo already states — the rule may be given, the things
# may not be listed — applied to the two clauses that were never audited
# against it, because the audit list was built from the animal rule's nouns.

_SCENERY = ("grass", "sky", "clouds", "cloud", "earth", "sand", "water",
            "stone", "horizon")


@pytest.mark.parametrize("stick", _STICK)
def test_the_colour_rule_names_no_scenery(stick):
    s = _looks()[stick].lower()
    named = [n for n in _SCENERY if n in s]
    assert not named, (
        f"{stick} names {named} — appended to every prompt that is not an "
        f"example of a colour, it is a place")


@pytest.mark.parametrize("stick", _STICK)
def test_the_rule_the_scenery_was_illustrating_survives(stick):
    """Deleting the list must not delete the instruction: a colour rule with
    no rule left brings back the washed-out beige gallery, and a place rule
    with no rule left brings back the white void."""
    s = _looks()[stick]
    assert "SATURATED" in s
    assert "takes its own true colour at full strength" in s
    assert "BUILD THE WHOLE PLACE" in s
    assert "four to eight things" in s


@pytest.mark.parametrize("stick", _STICK)
def test_a_shot_with_its_own_place_is_not_also_sent_a_beach(stick):
    """The acceptance test, as a test. An alleyway is not a coastline."""
    got = _composed(stick, "Wide shot: the whole place is visible and the "
                           "figures are small within it. An alleyway, empty "
                           "except for shadows stretching across the cracked "
                           "asphalt.")
    low = got.lower()
    for noun in _SCENERY:
        assert noun not in low, noun
    # What the shot actually asked for still arrives.
    for noun in ("alleyway", "asphalt", "shadow"):
        assert noun in low, noun


# ── the hands, and the uniform that only half the frame got ──────────────────
#
# The owner's gallery: "יד לשישית" — a hand with six fingers — and a frame of
# four police officers where two wore a uniform and two were bare white
# figures. The second is the two-constructions failure the block already
# forbids, arriving through a rule that had nothing to say about roles.
#
# The style said only "Small simple hands" — no shape, no finger count — while
# shots asked for "a hand making a tally motion with fingers" and "hands
# clutching a phone". A diffusion model handed the word fingers draws fingers,
# and gets the number wrong.

@pytest.mark.parametrize("stick", _STICK)
def test_a_hand_is_one_shape_and_not_a_count_of_fingers(stick):
    s = _looks()[stick]
    assert "Small simple hands" in s, "the pinned phrase survives"
    assert "no separated fingers" in s
    assert "whatever the shot says the hand is doing" in s, \
        "a shot naming fingers must not win this one"


@pytest.mark.parametrize("stick", _STICK)
def test_a_role_is_worn_by_every_figure_that_has_it(stick):
    s = _looks()[stick]
    assert "A ROLE THE STORY NEEDS IS WORN" in s
    assert "EVERY figure of that role in the frame gets it" in s
    assert "two-constructions failure" in s


@pytest.mark.parametrize("stick", _STICK)
def test_the_role_rule_names_no_garment(stick):
    """THE FIFTH TIME. The lion, the banknote, the coin face, the figure count
    — and this block was one edit away from adding a tunic, a cap, a helmet, an
    apron and a crown to every prompt on the channel. The rule may be stated;
    the things may not be listed, and "taken from the shot's own words" is
    where the garment comes from."""
    s = _looks()[stick].lower()
    for garment in ("dress", "gown", "robe", "skirt", "tunic", "apron",
                    "helmet", "crown", "badge", "trousers", "coat"):
        assert garment not in s, garment
    assert "taken from the shot's own words" in _looks()[stick]


# THE BLOCK DESCRIBES PARTS AND NEVER THE JOINTS BETWEEN THEM.
#
# Three frames from one live gallery, one failure:
#
#   · a head drawn alone, with no body, resting on the corner of a white shape
#     lying on a wooden floor — the owner called it "a face appearing out of
#     the floor", and that is exactly what it is;
#   · a shoulder with a hole in it (see round four above);
#   · a background figure whose head is an empty oval with no eyes, no brows
#     and no mouth, standing among figures that have all three.
#
# The head had a whole sentence of detail and nothing anywhere saying it sits
# on a body. The arm had four ways of saying it was separate and none saying it
# was attached. "Every figure in frame" turned out not to reach the figures at
# the back. A text encoder draws what it is told and nothing it is not.
#
# Worth recording as a suspect for the empty oval specifically: NO LETTERING
# says a surface that would carry writing is "drawn BLANK", and a head in this
# style is a white oval. That contradiction is not settled here.

@pytest.mark.parametrize("stick", _STICK)
def test_the_head_is_told_it_sits_on_a_body(stick):
    """A head described in full and never attached to anything gets drawn on
    whatever white shape is nearest."""
    s = _looks()[stick]
    assert "THE HEAD SITS ON THE BODY" in s
    assert "never drawn on its own" in s


@pytest.mark.parametrize("stick", _STICK)
def test_the_face_marks_are_required_at_the_back_of_the_frame_too(stick):
    """"Every figure in frame" did not reach the figures at the back: one
    gallery has a suited figure in the middle distance with an empty oval for a
    head, among figures that all have eyes."""
    s = _looks()[stick]
    assert "THE THREE MARKS ARE ALWAYS DRAWN" in s
    assert "at the back of the frame exactly as on the ones at the front" in s


@pytest.mark.parametrize("clause", [
    "NO LETTERING ANYWHERE IN THE FRAME",
    "drawn BLANK",
    "no gradients",
    "no depth of field",
    "no film grain",
    "because it is simpler",
])
def test_the_micro_preset_keeps_every_cheap_non_negotiable(clause):
    """Short is the point, but these cost eight words between them and each
    one is a gallery this channel has already shipped."""
    assert clause in _looks()["stickman_micro"]


def test_the_micro_preset_still_splits_at_the_figure_marker():
    """[SHOT=object] drops the figure half. A preset without the marker sends
    the stick-figure rules on a beat whose subject is a ledger."""
    s = _looks()["stickman_micro"]
    assert "--- FIGURE ONLY ---" in s
    shared, _, figure = s.partition("--- FIGURE ONLY ---")
    assert "stick figures" in figure and "stick figures" not in shared


def test_the_micro_preset_is_actually_micro():
    """It exists to be short. A hundred and fifty words is not an experiment,
    it is stickman_lean with a haircut."""
    assert len(_looks()["stickman_micro"].split()) < 110
    assert len(_looks()["stickman_lean"].split()) > 700, "the control moved"


# ── an object shot has no house style for a person, so it must not draw one ──
#
# The [SHOT=object] tag drops the whole figure half, which is right: a beat
# about a ledger does not need six hundred words on limbs. But it leaves the
# prompt saying NOTHING about how this channel draws a person — and one live
# frame from the first stickman_micro gallery is two flat-vector men with hair
# and jackets standing in front of a stack of gold bars, on a shot that read
# "Gold bars being stacked neatly in a vault". The model added them and had no
# house style to add them in.

def test_an_object_shot_suppresses_people_in_the_negative():
    import comfy_client
    neg = comfy_client._stills_negative("object")
    for term in ("human figure", "hair", "detailed clothing"):
        assert term in neg, term


def test_a_figure_shot_does_not_suppress_people():
    """Obvious, and worth pinning: the same clause on a figure shot would take
    the channel's figures out of its own videos."""
    import comfy_client
    assert "human figure" not in comfy_client._stills_negative("figure")


def test_the_suppression_goes_in_the_negative_not_the_style():
    """This file has been caught five times believing a style block has a meta
    level. The only reliable way not to get a figure is not to mention one, and
    the negative is the one side where naming a thing pushes it away."""
    import comfy_client
    for name, text in _looks().items():
        assert "human figure" not in text, name
