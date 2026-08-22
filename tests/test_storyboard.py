"""Planning the pictures WITH the script, as one continuous sequence.

Why: the two halves were strangers. script_writer finishes a script,
_split_beats chops it into sentences, and a separate model reads those
sentences COLD and illustrates each one alone. It has never seen the story, so
it decorates each line independently and ten unrelated pictures come back.

The live proof, from the denarius run. The script's beat 2 was about the coin
holding 4.5 grams of silver. The image planned for it:

    "A medium portrait of a family gathered around a modest dinner table,
     sharing a simple meal of bread and vegetables..."

Not wrong. Not connected. Beat 8 became "a concerned modern-day person at a
kitchen table with financial documents" — the stock photo of an idea rather
than a moment in this story.

A storyboard fixes it by construction: one pass over the WHOLE script, shots
planned as a sequence that can carry something forward.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import storyboard  # noqa: E402

_LONG = "A worn silver coin lies alone on a bare wooden counter, its face rubbed smooth."


def _reply(n=3, visual=_LONG, **overrides):
    plan = {"through_line": "one coin, thinning",
            "shots": [{"n": i + 1, "visual": f"{visual} Shot {i + 1}.",
                       "carries_over": None} for i in range(n)]}
    plan.update(overrides)
    return plan


# ── Validation: shot i must be beat i ────────────────────────────────────────

def test_a_good_plan_returns_one_visual_per_beat():
    out = storyboard._clean(_reply(4), 4)
    assert out is not None and len(out) == 4
    assert all(_LONG.split(",")[0] in v for v in out)


def test_the_wrong_shot_count_is_rejected():
    """The renderer cuts on clip[i] belonging to beat[i]. A short list would
    narrate every later picture against the wrong sentence."""
    assert storyboard._clean(_reply(3), 5) is None
    assert storyboard._clean(_reply(7), 5) is None


def test_a_stub_visual_is_rejected():
    """"a coin" is not a shot — it would render as anything at all."""
    assert storyboard._clean(_reply(2, visual="a coin"), 2) is None


def test_carried_continuity_reaches_the_image_model():
    """The thread has to be IN the prompt, not only in the storyboard's notes
    — the renderer never sees the JSON."""
    plan = _reply(2)
    # A thread the shot does NOT already name — see _already_shows for why the
    # other case is deliberately silent now.
    plan["shots"][1]["carries_over"] = "the merchant's leather ledger"
    out = storyboard._clean(plan, 2)
    assert "Continuing from the previous shot: the merchant's leather ledger" in out[1]
    assert "Continuing from" not in out[0], "shot 1 has nothing to continue from"


def test_a_thread_the_shot_already_names_is_not_restated():
    """THE COIN BUG. A live run appended `Continuing from the previous shot:
    the lemonade stand.` to nine of ten shots, including one that opened "A
    child stands behind the wooden lemonade stand" — and the owner's gallery
    came back with the same object in every single picture. Duplication in a
    prompt reads as emphasis."""
    plan = _reply(2)
    plan["shots"][1]["carries_over"] = "the same coin from shot 1, now thinner"
    out = storyboard._clean(plan, 2)
    assert "Continuing from" not in out[1], "the shot already says coin"


def test_the_thread_cannot_be_restated_in_every_shot():
    """A thread named in every shot stops connecting them and starts making
    them the same picture — the identical failure as carrying a mood."""
    plan = _reply(10)
    for shot in plan["shots"]:
        shot["carries_over"] = "the merchant's leather ledger"
    out = storyboard._clean(plan, 10)
    carried = sum("Continuing from" in v for v in out)
    assert 0 < carried <= 4, carried


def test_junk_is_rejected():
    for junk in (None, [], "", {}, {"shots": "nope"}, {"shots": [1, 2]}):
        assert storyboard._clean(junk, 2) is None


# ── The instruction ──────────────────────────────────────────────────────────

def _prompt(beats=("The coin held four and a half grams of silver.",
                   "By the end it was mostly bronze.")):
    return storyboard._prompt("full script here", list(beats),
                              ["211 BC", "present day"])


def test_prompt_shows_the_whole_script_not_just_the_beats():
    """Seeing only the sentences is what produced ten unrelated pictures."""
    p = storyboard._prompt("THE ENTIRE NARRATION", ["a", "b"], [])
    assert "THE ENTIRE NARRATION" in p
    assert "read it all before you draw anything" in p


def test_prompt_demands_the_literal_subject_of_the_line():
    """The exact live failure, named as the worked example."""
    p = _prompt()
    assert "four and a half grams of silver, the shot is THAT COIN" in p
    assert "not a family at dinner" in p


def test_prompt_asks_for_continuity_between_shots():
    p = _prompt()
    assert "CARRY A PHYSICAL OBJECT FORWARD" in p
    assert "carries_over" in p
    assert "Use null when nothing physical genuinely carries over" in p


def test_prompt_puts_feeling_in_the_frame_not_in_an_adjective():
    """"revealing the anguish of misplaced trust" shipped in a real run."""
    p = _prompt()
    assert "revealing the anguish of misplaced trust" in p
    assert "A feeling named as a feeling gives the image model nothing to draw" in p


def test_prompt_asks_for_the_face_as_a_physical_thing():
    """The ban on named emotions was reading as a ban on FACES, and the
    gallery came back with ten identical blank expressions. A face is a
    physical thing; "brows pulled down" is drawable and "looking determined"
    is not.

    ASKING FOR GEOMETRY WAS NOT ENOUGH — the second gallery, sixty stills, had
    the same worried face on nearly every figure, because the shots were still
    written as "their face showing resignation" and the renderer answers an
    emotion word with its default. The rule now hands over the five faces the
    style block can actually draw, in its own words, so there is nothing left
    to paraphrase."""
    p = _prompt()
    assert "DRAW THE FACE AS GEOMETRY" in p
    assert "brows slanted up at their inner ends" in p
    assert "Never 'looking determined'" in p


def test_prompt_carries_the_per_beat_era_tag():
    p = _prompt()
    assert "[211 BC]" in p and "[present day]" in p
    assert "OBEY THE ERA TAG" in p


def test_prompt_forbids_naming_printed_words():
    p = _prompt()
    assert "NEVER NAME WORDS THAT WOULD BE PRINTED" in p


def test_prompt_leaves_style_to_the_renderer():
    """A style described here would collide with the house suffix
    comfy_client appends — the mixed-look failure, by another route."""
    p = _prompt()
    assert "no camera bodies, no lens specs, no style words" in p.lower()


def test_character_clause_is_carried_when_there_is_one():
    p = storyboard._prompt("s", ["a"], [], character_clause="- THE CHRONICLER: hooded.\n")
    assert "THE CHRONICLER: hooded." in p


# ── Fail-open ────────────────────────────────────────────────────────────────

def test_disabled_by_env(monkeypatch):
    monkeypatch.setenv("RUFUS_STORYBOARD", "0")
    assert storyboard.plan("s", ["a", "b"]) is None


def test_no_beats_means_no_plan():
    assert storyboard.plan("s", []) is None


def test_a_missing_key_is_survivable(monkeypatch, tmp_path):
    monkeypatch.setattr(storyboard, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("RUFUS_STORYBOARD", raising=False)
    assert storyboard.plan("s", ["a", "b"]) is None


def test_an_api_failure_is_survivable(monkeypatch, tmp_path):
    (tmp_path / "keys.json").write_text(json.dumps({"openai": "sk-real"}))
    monkeypatch.setattr(storyboard, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("RUFUS_STORYBOARD", raising=False)
    import openai
    monkeypatch.setattr(openai, "OpenAI",
                        lambda **kw: (_ for _ in ()).throw(RuntimeError("down")))
    assert storyboard.plan("s", ["a", "b"]) is None


def test_a_valid_reply_comes_back_in_beat_order(monkeypatch, tmp_path):
    (tmp_path / "keys.json").write_text(json.dumps({"openai": "sk-real"}))
    monkeypatch.setattr(storyboard, "CONFIG_DIR", tmp_path)
    monkeypatch.delenv("RUFUS_STORYBOARD", raising=False)

    body = json.dumps({"through_line": "one coin", "shots": [
        {"n": 1, "visual": "A bright new coin fills the frame on a stone counter.",
         "carries_over": None},
        {"n": 2, "visual": "The same counter, bare now, dust settled in the grain.",
         "carries_over": "the coin from shot 1"},
    ]})

    class Resp:
        class C:
            class M:
                content = body
            message = M()
        choices = [C()]

    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kw):
                    return Resp()

    import openai
    monkeypatch.setattr(openai, "OpenAI", lambda **kw: Client())
    out = storyboard.plan("script", ["one", "two"])
    assert len(out) == 2
    assert "bright new coin" in out[0]
    assert "the coin from shot 1" in out[1]


# ── main uses it, and can still live without it ──────────────────────────────

def test_main_tries_the_storyboard_before_the_per_beat_writer():
    src = (Path(__file__).parent.parent / "scripts" / "main.py").read_text()
    body = src.split("def _build_sd_prompts")[1]
    assert "storyboard.plan(" in body
    assert body.index("storyboard.plan(") < body.index("beat_lines = "), \
        "the storyboard must be tried BEFORE the per-beat prompt writer"


def test_main_defuses_printed_text_on_storyboard_shots_too():
    """The blank-surfaces clause is applied on the per-beat path; the
    storyboard path must not quietly skip it."""
    src = (Path(__file__).parent.parent / "scripts" / "main.py").read_text()
    body = src.split("def _build_sd_prompts")[1]
    sb = body[body.index("storyboard.plan("):]
    assert "_defuse_readable_text(s) for s in shots" in sb


def test_the_character_clause_exists_before_the_storyboard_reads_it():
    """Live: "storyboard skipped (non-fatal): cannot access local variable
    'char_clause' where it is not associated with a value" — the storyboard was
    moved ahead of the per-beat writer but the clause it passes was still built
    below it. Fail-open hid it as a one-line warning while the whole feature
    never ran once."""
    src = (Path(__file__).parent.parent / "scripts" / "main.py").read_text()
    body = src.split("def _build_sd_prompts")[1]
    assert body.index("char_clause = character_engine.character_clause") \
        < body.index("storyboard.plan(")


# ── A thread has to be a THING ───────────────────────────────────────────────
# The Great Depression run planned ten shots and every carries_over was a mood:
# "emptiness and desolation", "emptiness and chaos", "sense of despair and
# loss", "unresolved financial burden", "ongoing neglect", "threat of repeating
# past mistakes". Handing an image model "carry emptiness forward" four beats
# running renders four empty rooms — the instruction to CONNECT the shots
# became the instruction to REPEAT them.

def test_a_mood_is_not_a_thread():
    for mood in ("emptiness and desolation", "emptiness and chaos",
                 "sense of despair and loss", "unresolved financial burden",
                 "ongoing neglect", "threat of repeating past mistakes",
                 "disregard of issues"):
        assert not storyboard._is_a_thing(mood), mood


def test_an_object_is_a_thread():
    for thing in ("the same coin from shot 1, now thinner",
                  "the bronze lantern", "the same wooden table, emptier",
                  "his coat", "the queue outside the same door"):
        assert storyboard._is_a_thing(thing), thing


def test_a_mood_thread_is_dropped_and_the_shot_survives():
    """Dropping the clause is the right failure: a shot with no stated thread
    still renders fine, while a mood thread actively causes the repetition."""
    plan = _reply(2)
    plan["shots"][1]["carries_over"] = "emptiness and desolation"
    out = storyboard._clean(plan, 2)
    assert out is not None and len(out) == 2
    assert "Continuing from" not in out[1]
    assert _LONG.split(",")[0] in out[1], "the shot itself is kept"


def test_an_object_thread_still_reaches_the_prompt():
    plan = _reply(2)
    plan["shots"][1]["carries_over"] = "the same bronze lantern"
    assert "Continuing from the previous shot: the same bronze lantern" \
        in storyboard._clean(plan, 2)[1]


def test_the_prompt_names_the_mood_failure_with_its_real_answers():
    p = _prompt()
    assert "NEVER A MOOD" in p
    assert "emptiness and desolation" in p
    assert "Emptiness\" four beats running renders four empty rooms" in p


def test_the_prompt_demands_people_in_at_least_half_the_shots():
    """Eight of ten shots had nobody in them, for a script about one in four
    people losing their job."""
    p = _prompt()
    assert "PUT PEOPLE IN IT" in p
    assert "At least half the shots must show a person DOING something" in p


def test_the_through_line_must_be_an_object_too():
    p = _prompt()
    assert "one coin, thinning" in p
    assert "is an essay title and is" in p


# ── The Chronicler's cloak changed colour between shots ─────────────────────
# From the Great Depression run's own images: beat 1 rendered a tan-gold cloak,
# beat 5 a BLACK one, beat 10 brown. The prompts explain it exactly — beats 1
# and 10 described the cloak, beat 5 said only "The hooded figure, the
# Chronicler, appears again". The image model renders each beat from noise with
# no memory of the others, so naming him is not describing him.

_SHORT = ("a hooded figure in a weathered sepia-and-antique-gold cloak, "
          "calm face beneath the hood, carrying a small bronze lantern")

_SHOT_1 = ("A hooded figure in a weathered sepia-and-antique-gold cloak stands "
           "calmly. In one hand, he holds a small bronze lantern, casting a warm "
           "glow. The background is blurred, suggesting a historical setting.")
_SHOT_5 = ("The hooded figure, the Chronicler, appears again, standing amidst "
           "the desolation of a ruined town. His calm demeanor contrasts with "
           "the devastation around him, his bronze lantern glowing softly.")
_SHOT_10 = ("The hooded figure, the Chronicler, stands on a busy modern-day "
            "street. His bronze lantern glows subtly in daylight, serving as a "
            "reminder of history amidst the hustle and bustle.")


def test_the_shot_that_described_him_is_left_alone():
    """Beat 1 carried the full look and rendered correctly — don't pad it."""
    assert storyboard._pin_character(_SHOT_1, "the Chronicler", _SHORT) == _SHOT_1


def test_the_shots_that_only_named_him_get_his_look_back():
    for shot in (_SHOT_5, _SHOT_10):
        out = storyboard._pin_character(shot, "the Chronicler", _SHORT)
        assert out != shot
        assert "sepia-and-antique-gold" in out
        assert "identical in every appearance" in out


def test_a_shot_without_the_character_is_untouched():
    other = "An abandoned factory with rusting machinery and broken windows."
    assert storyboard._pin_character(other, "the Chronicler", _SHORT) == other


def test_no_character_configured_is_a_no_op():
    assert storyboard._pin_character(_SHOT_5, "", "") == _SHOT_5


def test_the_threshold_separates_the_real_shots():
    """0.67 for the one that worked, 0.33 and 0.27 for the two that didn't —
    the calibration is the live data, not a guess."""
    assert storyboard._restates_the_look(_SHOT_1, _SHORT)
    assert not storyboard._restates_the_look(_SHOT_5, _SHORT)
    assert not storyboard._restates_the_look(_SHOT_10, _SHORT)


def test_main_passes_the_niche_so_the_character_can_be_looked_up():
    src = (Path(__file__).parent.parent / "scripts" / "main.py").read_text(
        encoding="utf-8")
    body = src.split("def _build_sd_prompts")[1]
    call = body[body.index("storyboard.plan("):body.index("storyboard.plan(") + 260]
    assert "niche=niche" in call


# ── The setting: one place, decided before any image is generated ────────────

def _sb():
    import storyboard
    return storyboard


def test_a_room_with_surface_and_light_is_a_place():
    sb = _sb()
    assert sb._is_a_place(
        "A low stone counting-house: oak counter, lime-washed walls, one high "
        "window throwing hard light from the left")


def test_a_period_name_is_not_a_place():
    """'Renaissance Europe' makes every shot invent its own building — the same
    mistake _is_a_thing catches for the object thread, one level up."""
    sb = _sb()
    assert not sb._is_a_place("Renaissance Europe")
    assert not sb._is_a_place("a place of financial power")
    assert not sb._is_a_place("16th century Germany")


def test_a_room_with_no_light_is_rejected():
    """Light direction is what keeps two shots of one room looking like one
    room, so a setting without it has nothing to keep consistent."""
    sb = _sb()
    assert not sb._is_a_place("an oak counter against plaster walls with shelves")


def test_a_light_with_no_structure_is_rejected():
    sb = _sb()
    assert not sb._is_a_place("dim golden light falling softly from somewhere")


def test_empty_and_junk_settings_are_rejected():
    sb = _sb()
    for junk in ("", "   ", None):
        assert not sb._is_a_place(junk)


def test_a_shot_that_already_names_the_room_is_left_alone():
    """Verbatim from the 1893 run: this shot builds the place in its own words
    and still had the entire 40-word setting appended — five of ten shots did.
    Any structural anchor means the shot is already placed."""
    sb = _sb()
    setting = ("A cobblestone street in Philadelphia, flanked by brick buildings. "
               "The Philadelphia and Reading Railroad office is prominent, with "
               "tall windows and wooden doors. Afternoon light casts long shadows "
               "across the street.")
    shot = ("Groups of workers huddle outside the Philadelphia and Reading "
            "Railroad office, anxiously looking at the closed wooden doors.")
    assert sb._pin_setting(shot, setting) == shot


def test_a_shot_with_no_room_gets_a_COMPACT_pin():
    """Pinned, but in a few words — not the whole paragraph. ~40 words x 5
    shots competed with each beat's own description and the style suffix."""
    sb = _sb()
    setting = ("A cobblestone street in Philadelphia, flanked by brick buildings. "
               "The Philadelphia and Reading Railroad office is prominent, with "
               "tall windows and wooden doors. Afternoon light casts long shadows "
               "across the street.")
    shot = "A family is seen leaving a modest home, carrying their belongings."
    out = sb._pin_setting(shot, setting)
    assert out != shot
    assert len(out.split()) - len(shot.split()) <= 12
    assert "Philadelphia and Reading" not in out


def test_the_pin_does_not_splice_a_capital_mid_sentence():
    sb = _sb()
    out = sb._pin_setting(
        "A close-up of a coin.",
        "A stone counting-house with an oak counter and one high window, hard light.")
    assert " is A " not in out


# ── which nouns the pictures never showed ────────────────────────────────────

def test_the_case_this_check_exists_for():
    """The Fugger script said the family "controlled Europe's copper" and not
    one of nine shots contained copper."""
    sb = _sb()
    assert sb._unshown_nouns(
        ["A large medieval castle on a hill.", "A stained glass window.",
         "An old map of the continent."],
        ["He financed the Habsburgs, controlled Europe's copper, and "
         "overshadowed the Medicis."]) == ["copper"]


def test_a_sequence_that_shows_what_it_names_is_silent():
    sb = _sb()
    assert sb._unshown_nouns(
        ["A merchant's hands stack copper ingots on an oak counter."],
        ["He controlled Europe's copper."]) == []


def test_a_neighbouring_shot_counts():
    """The per-beat version fired on 7 of 10 shots and was wrong about most:

        beat 3: Their jobs vanished overnight. Then, panic spread.
        shot 3: The same single bronze coin is held tightly in a worker's hand.

    That is an excellent shot for that line. The sequence is planned as a
    whole, so a noun answered anywhere in it is answered."""
    sb = _sb()
    assert "coin" not in sb._unshown_nouns(
        ["A wide street at dawn.", "A bronze coin in a worker's hand."],
        ["The coin was worthless.", "Nobody would take it."])


def test_verbs_and_qualities_are_not_reported():
    """"shocked", "vanished", "failed", "overnight" and "then" were all
    reported as things the pictures failed to show — the same noise problem in
    a new shape."""
    sb = _sb()
    out = sb._unshown_nouns(
        ["A quiet street."],
        ["Unemployment shocked America. Their jobs vanished overnight. "
         "Then 500 banks failed."])
    for w in ("shocked", "vanished", "overnight", "then", "unemployment", "failed"):
        assert w not in out


def test_proper_nouns_are_not_reported():
    """You cannot point a camera at Europe."""
    sb = _sb()
    out = sb._unshown_nouns(
        ["A quiet street."],
        ["The Habsburgs ruled Europe from Philadelphia to America."])
    for w in ("europe", "america", "habsburgs", "philadelphia"):
        assert w not in out


def test_it_warns_and_never_rejects(capsys):
    sb = _sb()
    plan = {"shots": [
        {"n": 1, "visual": "A large medieval castle on a hill at dusk, doors "
                           "shut.", "carries_over": None}]}
    out = sb._clean(plan, 1, ["He controlled Europe's copper."])
    assert out is not None and len(out) == 1        # nothing rejected
    assert "copper" in capsys.readouterr().out


def test_clean_still_works_without_beats():
    """Every existing caller passed two arguments; the third must be optional."""
    sb = _sb()
    plan = {"shots": [{"n": 1, "visual": "A coin resting on a worn oak counter "
                                         "in hard light.", "carries_over": None}]}
    assert sb._clean(plan, 1) is not None


def test_the_scene_anchors_the_prompt():
    sb = _sb()
    scene = "In 1523 Jakob Fugger sent Charles V a letter demanding repayment."
    p = sb._prompt("script", ["b1", "b2"], [], "", scene)
    assert scene in p
    assert "THE MOMENT THIS SCRIPT WAS BUILT ON" in p


def test_no_scene_leaves_the_prompt_as_it_was():
    sb = _sb()
    assert "THE MOMENT THIS SCRIPT WAS BUILT ON" not in sb._prompt(
        "script", ["b1"], [], "", "")


def test_the_noun_appositive_abstraction_tail_is_cut():
    """From the 1893 run — the participial list missed this grammar:
    "The bronze coin is half-buried in the dirt, a symbol of forgotten
    prosperity." """
    sb = _sb()
    out = sb._strip_abstraction(
        "The bronze coin is half-buried in the dirt, a symbol of forgotten "
        "prosperity.")
    assert "symbol" not in out
    assert "half-buried in the dirt" in out


@pytest.mark.parametrize("tail", [
    ", a reminder of what was lost", ", an emblem of the old order",
    ", a testament to their power", ", a metaphor for the collapse",
])
def test_other_appositive_forms_are_cut(tail):
    sb = _sb()
    base = "A single bronze coin rests on the cobblestones"
    assert sb._strip_abstraction(base + tail + ".") .startswith(base)


def test_a_real_apposition_that_describes_a_thing_survives():
    """"a coin worn smooth" is a description, not an interpretation — cutting
    it would lose the only detail in the shot."""
    sb = _sb()
    text = "A bronze coin on the counter, worn smooth at the edges."
    assert sb._strip_abstraction(text) == text


def test_a_list_of_real_objects_is_not_eaten_by_the_appositive_rule():
    """"a picture of the founder" is a list ITEM here, not a trailing comment,
    and `[^.]*` deleted everything after it — the shot lost its brass lamp.
    An abstraction always trails, so the appositive branch has to end the
    sentence to count."""
    sb = _sb()
    text = ("A cluttered desk holds a ledger, a picture of the founder, and a "
            "brass lamp.")
    assert sb._strip_abstraction(text) == text


def test_concrete_nouns_the_suffix_filter_would_have_eaten_are_kept():
    """The suffix rule drops verbs and concepts by their endings, and took
    ship/city/fence/monument/building/ceiling/bread with them. Those are the
    exact things a shot should contain, so the Fugger case — a script naming
    something no picture shows — would have gone unreported had the noun been
    "ship" instead of "copper"."""
    sb = _sb()
    words = sb._content_words(
        "the ship reached the city past the fence and the monument into the "
        "building under a low ceiling with bread")
    for w in ("ship", "city", "fence", "monument", "building", "ceiling",
              "bread"):
        assert w in words, w


def test_the_fugger_case_still_works_when_the_noun_ends_in_a_verb_suffix():
    sb = _sb()
    assert sb._unshown_nouns(
        ["A stained glass window.", "An old map of the continent."],
        ["His ships carried it."]) == ["ships"]


# ── in_setting: which shots the place is pinned onto ─────────────────────────

def test_a_missing_in_setting_block_places_every_shot():
    """Defaulting to True is the safe direction: restating a place a shot was
    already in costs a few words, omitting it is the failure the setting
    exists to stop."""
    sb = _sb()
    assert sb._in_setting_flags({}, 3) == [True, True, True]
    assert sb._in_setting_flags({"shots": "not a list"}, 2) == [True, True]


def test_only_an_explicit_false_takes_a_shot_out_of_the_room():
    sb = _sb()
    raw = {"shots": [{"in_setting": False}, {"in_setting": True}, {},
                     {"in_setting": "no"}, "junk"]}
    assert sb._in_setting_flags(raw, 5) == [False, True, True, True, True]


def test_a_short_shots_list_still_returns_one_flag_per_beat():
    sb = _sb()
    assert sb._in_setting_flags({"shots": [{"in_setting": False}]}, 4) == [
        False, True, True, True]


def test_the_prompt_still_asks_for_the_setting_and_the_flag():
    """Pinning is driven entirely by these two keys. Rename or drop either and
    every shot silently loses its room while the suite stays green."""
    sb = _sb()
    p = sb._prompt("script", ["b1", "b2"], [], "", "")
    assert '"setting"' in p
    assert '"in_setting"' in p


# ── An outdoor place is still a place ────────────────────────────────────────

def test_a_riverside_is_a_room_a_camera_can_stand_in():
    """Thrown away on a live gold-rush run for containing no wall. Ground,
    rocks, a river, trees and dappled light is a place by every standard this
    check exists to enforce — the vocabulary was simply indoors-only, on a
    channel that covers gold rushes, ports, mines and fields."""
    assert storyboard._is_a_place(
        "A rugged riverside area with dusty ground, scattered rocks, and a "
        "slow-moving section of the American River. Tall trees line the "
        "background, and sunlight filters through the branches, casting "
        "dappled light across the scene.")


def test_a_label_is_still_rejected():
    """The outdoor vocabulary must not open the door to a period name."""
    for label in ("Renaissance Europe",
                  "a place of great financial importance in the modern world",
                  "the global economy during a time of change and upheaval"):
        assert not storyboard._is_a_place(label), label


def test_an_interior_still_passes():
    assert storyboard._is_a_place(
        "A dimly lit medieval hall with rough stone walls, a wooden table in "
        "the centre, and a narrow high window casting a beam across the room.")


# ── one object in every frame: "why all the images with coin" ────────────────
#
# The complaint that has outlived every other one, and the finding that was
# still at 52% of measured runs after the thread and setting clauses were
# capped. Those caps fixed the text this pipeline APPENDS; this is the
# storyboard's own choice of subject, and it comes from rules 1 and 3a
# disagreeing. Rule 1 says draw the literal thing the line names. A money
# script names money in nine of ten lines. Obeying rule 1 shot by shot is
# obeying it wrongly for the sequence, and nothing in a single prompt sees
# that, because the conflict only exists once all the shots are on the table.


class _FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class _FakeClient:
    """Records the prompt it was given and replies with canned JSON."""

    def __init__(self, reply: str):
        self._reply = reply
        self.prompts: list[str] = []
        outer = self

        class _Completions:
            def create(self, **kw):
                outer.prompts.append(kw["messages"][0]["content"])
                return type("R", (), {"choices": [_FakeChoice(outer._reply)]})()

        self.chat = type("C", (), {"completions": _Completions()})()


_COINS = [
    "A hand pushes a single coin across the oak counter.",
    "The same coin lies on the wooden table, alone.",
    "A coin is bitten between a merchant's teeth.",
    "The coin sits at the bottom of an open purse.",
    "A coin spins to a stop on the stone floor.",
    "A woman turns the coin over in her fingers.",
]
_BEATS = ["one", "two", "three", "four", "five", "six"]


def test_the_dominant_subject_is_counted_the_way_run_review_counts_it():
    """A pipeline that repairs by one definition while the analyzer reports by
    another produces a run that fixes itself and is then told it did not. That
    has happened in this repo more than once, so the two share the count."""
    import run_review
    d = storyboard.dominant_subject(_COINS)
    assert d["word"] == "coin"
    assert d["shots"] == 6
    assert d == {"word": run_review._dominant_subject(_COINS)["word"],
                 "shots": run_review._dominant_subject(_COINS)["prompts"],
                 "share": run_review._dominant_subject(_COINS)["share"]}


def test_the_threshold_is_run_reviews_threshold():
    import run_review
    assert storyboard._dominant_share_limit() == run_review.DOMINANT_SHARE


def test_a_varied_sequence_costs_no_second_call():
    """The repair must be rare. A second model call on every run is a tax on
    the runs that did nothing wrong."""
    varied = ["A hand pushes a coin across the counter.",
              "The queue outside has not moved since dawn.",
              "A clerk bolts the heavy door from inside.",
              "A child watches from the top of the stairs.",
              "Two men argue across an empty counter.",
              "A woman folds an apron and sets it down."]
    client = _FakeClient('{"shots": []}')
    out = storyboard._revary(client, "gpt-4o", "script", _BEATS, varied)
    assert out == varied
    assert not client.prompts, "no call should have been made"


def test_the_surplus_shots_are_re_planned(monkeypatch, capsys):
    monkeypatch.delenv("RUFUS_STORYBOARD_REPAIR", raising=False)
    client = _FakeClient(json.dumps({"shots": [
        {"n": 3, "visual": "A merchant's jaw clenches as he shakes his head at "
                           "the man across the counter."},
        {"n": 4, "visual": "A woman's hands close an empty purse and push it "
                           "into her coat."},
        {"n": 5, "visual": "A queue of people stands at a bolted door, nobody "
                           "moving."},
        {"n": 6, "visual": "A child on the stairs watches the adults argue "
                           "below, gripping the railing."},
    ]}))
    out = storyboard._revary(client, "gpt-4o", "script", _BEATS, _COINS)
    assert storyboard.dominant_subject(out)["share"] < 0.55
    # The introduction survives: the object is still in the sequence.
    assert out[0] == _COINS[0]
    assert "re-planned 4 shot(s)" in capsys.readouterr().out


def test_a_replacement_that_names_the_same_object_is_refused():
    """Otherwise the pass reports success on a reply that changed the words and
    not the picture."""
    client = _FakeClient(json.dumps({"shots": [
        {"n": 3, "visual": "The very same coin, now resting on a different "
                           "counter in better light."},
    ]}))
    out = storyboard._revary(client, "gpt-4o", "script", _BEATS, _COINS)
    assert out == _COINS


def test_the_revision_asks_for_a_verb_and_pins_each_shot_to_its_own_line():
    """A replacement that drifts to another part of the story is a worse
    defect than the repetition it fixes — it breaks rule 1, which is the one
    that makes the picture match the words."""
    client = _FakeClient('{"shots": []}')
    storyboard._revary(client, "gpt-4o", "the full narration", _BEATS, _COINS)
    p = client.prompts[0]
    assert "VERB" in p
    assert "ITS OWN LINE" in p
    assert "the full narration" in p
    # It must show the model the lines it is re-planning against.
    assert "LINE:" in p and "CURRENT SHOT:" in p


def test_the_repair_can_be_switched_off(monkeypatch):
    monkeypatch.setenv("RUFUS_STORYBOARD_REPAIR", "0")
    client = _FakeClient('{"shots": []}')
    assert storyboard._revary(client, "gpt-4o", "s", _BEATS, _COINS) == _COINS
    assert not client.prompts


def test_a_short_sequence_is_left_alone():
    """Three shots of one object is a sequence about that object, which is
    fine. Dominance needs enough frames to be dominance."""
    short = _COINS[:3]
    client = _FakeClient('{"shots": []}')
    assert storyboard._revary(client, "gpt-4o", "s", _BEATS, short) == short
    assert not client.prompts


def test_a_broken_reply_keeps_the_original_plan():
    """Fail-open, like every other model call in this file."""
    class _Boom(_FakeClient):
        def __init__(self):
            super().__init__("")
            outer = self

            class _C:
                def create(self, **kw):
                    raise RuntimeError("upstream is down")
            self.chat = type("C", (), {"completions": _C()})()

    assert storyboard._revary(_Boom(), "gpt-4o", "s", _BEATS, _COINS) == _COINS


def test_the_prompt_tells_the_model_to_vary_the_subject():
    p = storyboard._prompt("script", ["a", "b"], [])
    assert "MUST CHANGE ACROSS THE SEQUENCE" in p
    assert "a THIRD of the shots" in p
    assert "SHOW A DIFFERENT VERB" in p


def test_the_numbered_rules_are_still_unique():
    """Two rules sharing a number is how one of them gets ignored."""
    import re
    p = storyboard._prompt("script", ["a", "b"], [])
    rules = p.split("WHAT MAKES THIS A STORYBOARD", 1)[1]
    labels = re.findall(r"^(\d+[a-z]?)\. ", rules, flags=re.M)
    assert labels, "the rules section moved"
    assert len(labels) == len(set(labels)), labels


# ── the same face on every figure ────────────────────────────────────────────
#
# From a gallery of sixty stills: nearly every stick figure wore the same
# worried face — brows up at the inner ends, small frown — whatever the line
# was about. The style block defines five faces in GEOMETRY, and the shots were
# asking for emotions instead:
#
#     "their face showing resignation"
#     "a look of loss on their face"
#     "Roosevelt's face, determined and focused"
#
# An image model cannot draw resignation. Handed a word that matches none of
# the five, it falls back to its default, which is why the default was on every
# frame. Rule 4 already forbade naming feelings; it was being obeyed everywhere
# except the one place it mattered most.

def test_a_named_emotion_becomes_the_geometry_that_draws_it():
    got = storyboard._pin_expression(
        "A person hands a pile of coins to a teller, their face showing "
        "resignation.")
    assert "resignation" not in got
    assert "brows slanted up at their inner ends" in got


def test_the_look_of_x_form_is_caught_too():
    got = storyboard._pin_expression(
        "A hand clutches a few paper dollars, a look of loss on their face.")
    assert "look of loss" not in got
    assert "downward mouth curve" in got


def test_the_adverb_form_is_caught():
    got = storyboard._pin_expression("Roosevelt stands at the podium, "
                                     "looking determined.")
    assert "looking determined" not in got
    assert "brows angled down and inward" in got


def test_an_ordinary_list_is_not_mangled():
    """The abstraction stripper had exactly this bug once — a real object in a
    real list eaten because it matched a pattern. A picture of the founder is
    a picture, not a feeling."""
    s = "A cluttered desk holds a ledger, a picture of the founder, and a lamp."
    assert storyboard._pin_expression(s) == s


def test_a_word_outside_the_five_keeps_its_own_phrasing():
    """Forcing every unknown word into the nearest of five faces would be
    inventing an emotion the shot never asked for."""
    s = "A clerk stands at the counter, looking upward."
    assert storyboard._pin_expression(s) == s


def test_the_geometry_matches_the_style_block_word_for_word():
    """Two descriptions of a face in one prompt is two instructions competing.
    The renderer's own words are in config/styles.json — if they are edited
    there and not here, the prompt starts arguing with the style block."""
    import json
    style = json.loads((Path(storyboard.__file__).parent.parent / "config" /
                        "styles.json").read_text(encoding="utf-8"))["stickman"]
    for geom in storyboard._FACE_GEOMETRY.values():
        head = geom.split(" over ")[0].split(" with ")[0]
        assert head in style, head


def test_the_prompt_hands_over_the_five_faces():
    p = storyboard._prompt("script", ["a", "b"], [])
    assert "brows raised high over a small open oval mouth" in p
    assert "at least three of" in p
    assert "MUST DIFFER ACROSS THE SEQUENCE" in p


def test_one_face_across_a_whole_sequence_is_reported(capsys):
    """Nothing in the pipeline could have noticed this: the shots said
    "resignation", the renderer drew its default, and every stage downstream
    saw text that looked fine."""
    grief = storyboard._FACE_GEOMETRY["grief"]
    shots = [{"n": i + 1, "visual": f"A clerk stands at the counter, {grief}, "
                                    f"as the queue waits behind him.",
              "carries_over": None} for i in range(4)]
    storyboard._clean({"shots": shots}, 4)
    out = capsys.readouterr().out
    assert "faces: grief×4" in out
    assert "every face in this sequence is the same one" in out


def test_a_varied_sequence_draws_no_warning(capsys):
    geoms = [storyboard._FACE_GEOMETRY[k] for k in ("grief", "shock", "anger")]
    shots = [{"n": i + 1, "visual": f"A clerk at the counter, {g}, as the "
                                    f"queue waits behind him.",
              "carries_over": None} for i, g in enumerate(geoms)]
    storyboard._clean({"shots": shots}, 3)
    out = capsys.readouterr().out
    assert "every face in this sequence is the same one" not in out


# ── one world, moved through ─────────────────────────────────────────────────
#
# From five frames of a channel doing this well: a zebra at a waterhole, then
# the same waterhole with the zebra bolting, then the same field with a lion
# lying in it. Same horizon, same colours, same landmarks — the sequence reads
# as one place a moment later. Then, once, a figure shrugging on plain white:
# a bare frame that works precisely because everything around it is a full
# scene.

def test_the_prompt_asks_the_place_to_advance_rather_than_repeat():
    p = storyboard._prompt("script", ["a", "b"], [])
    assert "STAY IN THE PLACE, AND ADVANCE IT" in p
    assert "Say what CHANGED" in p


def test_a_bare_frame_is_allowed_but_rationed():
    """It works BECAUSE the rest is a full scene. Left unmentioned the model
    never uses it; left unqualified it becomes the default, which is the
    figure-floating-on-paper look this style is trying to leave behind."""
    p = storyboard._prompt("script", ["a", "b"], [])
    assert "DELIBERATE PUNCHLINE" in p
    assert "sparingly" in p


def test_the_style_no_longer_fades_the_background():
    """"soft muted flat colours, drawn thinner and paler than the foreground"
    was the instruction, and the whole gallery came back beige. The background
    is quieter because it is simpler and further away, not because it is
    faded."""
    import json
    style = json.loads((Path(storyboard.__file__).parent.parent / "config" /
                        "styles.json").read_text(encoding="utf-8"))["stickman"]
    assert "paler than the foreground" not in style
    assert "soft muted flat colours" not in style
    assert "SATURATED" in style
    assert "grass is green" in style


def test_the_style_keeps_people_simple_and_draws_the_animals():
    """The contrast is the charm: simple stick humans against a properly drawn
    zebra. The old text flattened both to one level of detail."""
    import json
    style = json.loads((Path(storyboard.__file__).parent.parent / "config" /
                        "styles.json").read_text(encoding="utf-8"))["stickman"]
    # Lowercase since the block was split: this sentence moved into the shared
    # half, which is the top of the string, and a shouted drawable noun there
    # is a candidate for being painted — the banknote rule.
    assert "animals and objects are drawn properly" in style.lower()
    # "stick-figure" used to be the block's opening word, describing the medium.
    # It now lives only in the FIGURE ONLY half — the medium line is shot-neutral
    # so an object beat is not told to draw a stick figure.
    assert "stick figure" in style.lower()


# ── a hundred and fifty shots do not fit in one reply ────────────────────────
#
# At Shorts length the whole sequence fits in one call and that is the POINT —
# the model sees every shot at once, which is what makes it a storyboard
# rather than fourteen illustrations. A nine-minute script has ten times the
# beats; asking for them in one reply hits the token ceiling and returns a
# truncated list, which _clean correctly rejects as the wrong shot count. The
# storyboard would then fall back to per-beat prompts — the exact failure this
# module was written to end, returning through the back door on the longer
# format.

def test_a_short_is_still_planned_in_one_pass():
    """Nothing about the existing channel may move."""
    assert storyboard._chunks(list(range(14)), 24) == [(0, list(range(14)))]


def test_a_long_script_is_split():
    windows = storyboard._chunks(list(range(150)), 24)
    assert len(windows) > 1
    assert sum(len(w) for _, w in windows) == 150
    offsets = [o for o, _ in windows]
    assert offsets == sorted(offsets)


def test_no_window_is_a_runt():
    """A final chunk of two shots gets a call that cannot see enough of the
    story to place them, so a short tail folds into the window before it."""
    for n in range(25, 130):
        windows = storyboard._chunks(list(range(n)), 24)
        assert all(len(w) >= 8 for _, w in windows), (n, [len(w) for _, w in windows])


def test_every_beat_is_planned_exactly_once():
    for n in (14, 25, 60, 150):
        windows = storyboard._chunks(list(range(n)), 24)
        seen = [b for _, w in windows for b in w]
        assert seen == list(range(n)), n


# ── the seam ─────────────────────────────────────────────────────────────────

def test_a_later_window_is_told_what_already_exists():
    """A window that cannot see the room the last one built, the object it is
    carrying, or the shot it continues from would start the world again every
    twenty-four pictures."""
    p = storyboard._prompt("s", ["a", "b"], [], carry="A hand on the counter.",
                           setting="A low stone hall, one high window",
                           through="one coin, thinning",
                           shot_offset=24, shot_total=150)
    assert "SHOTS 25–26 OF 150" in p
    assert "A low stone hall" in p
    assert "one coin, thinning" in p
    assert "A hand on the counter." in p
    assert "continues from that one" in p


def test_a_short_carries_no_seam_text():
    """It is planned in one pass and needs none of it; a Short whose prompt
    said "you are planning shots 1-14 of 14" would be answering a question
    nobody asked."""
    p = storyboard._prompt("s", ["a", "b"], [])
    assert "OF 0" not in p
    assert "already drawn" not in p
    assert "THE SHOT IMMEDIATELY BEFORE YOURS" not in p


def test_the_brief_follows_the_format(monkeypatch):
    """Hard-coded as "a 40-second vertical documentary Short" until long-form
    existed, and that line does real work — it is why shots are composed for a
    phone held upright. Handed a nine-minute landscape script it would have
    been quietly, confidently wrong."""
    import importlib
    import video_format
    assert "40-second vertical" in storyboard._brief()
    monkeypatch.setenv("RUFUS_FORMAT", "long")
    importlib.reload(video_format)
    assert "landscape" in storyboard._brief()
    assert "nine minutes" in storyboard._brief()


# ── a shot is also a DISTANCE ────────────────────────────────────────────────
#
# The brief asked for a subject and a place and never for a distance, so every
# sequence came back at the same comfortable middle one — which is not a look,
# it is the absence of one, and it reads as a slideshow however good the
# drawings are. The reference frames work precisely because the distance
# moves: a wide plain with a figure small in it, then a close on a face, then
# one object filling the frame.

def _shots(framings):
    return [{"n": i + 1, "carries_over": None,
             "visual": "A clerk stands at the oak counter as the queue waits "
                       "behind him, hands flat on the wood.",
             "framing": f}
            for i, f in enumerate(framings)]


def test_the_framing_reaches_the_prompt():
    out = storyboard._clean({"shots": _shots(["wide", "close", "detail"])}, 3)
    assert out[0].startswith("Wide shot:")
    assert out[1].startswith("Close shot:")
    assert out[2].startswith("Close detail:")


def test_it_goes_FIRST_where_the_renderer_weights_it():
    """An image model weights the opening of a prompt most heavily, and this
    is the one instruction that decides what the picture IS rather than what
    is in it."""
    out = storyboard._clean({"shots": _shots(["detail"])}, 1)
    assert out[0].index("Close detail") < out[0].index("clerk")


def test_an_unknown_framing_is_ignored_rather_than_rendered():
    out = storyboard._clean({"shots": _shots(["dutch angle drone orbit"])}, 1)
    assert out[0].startswith("A clerk")


def test_a_shot_with_no_framing_still_renders():
    """Framing is an improvement, never a requirement — a plan without it must
    produce the same pictures it always did."""
    shots = _shots(["wide"])
    del shots[0]["framing"]
    out = storyboard._clean({"shots": shots}, 1)
    assert out and out[0].startswith("A clerk")


# ── the backstop ─────────────────────────────────────────────────────────────

def test_a_run_of_four_wides_is_broken():
    """The prompt asks for variety and mostly gets it. This is for the run
    that comes back wide-wide-wide-wide."""
    got = storyboard._vary_framings(["wide"] * 6)
    assert got[:2] == ["wide", "wide"], "a deliberate pair survives"
    assert len(set(got)) > 1


def test_a_deliberate_pair_survives():
    """A wide establishing followed by a second wide is a real choice. A
    fourth is not."""
    assert storyboard._vary_framings(["wide", "wide", "mid", "close"]) == \
        ["wide", "wide", "mid", "close"]


def test_the_break_jumps_the_distance_rather_than_nudging_it():
    """The felt beat is the SIZE of the change, so a wide becomes a close,
    not a mid."""
    assert storyboard._vary_framings(["wide"] * 3)[2] == "close"
    assert storyboard._vary_framings(["close"] * 3)[2] == "wide"


def test_nothing_to_vary_is_left_alone():
    assert storyboard._vary_framings([]) == []
    assert storyboard._vary_framings(["", "", ""]) == ["", "", ""]


# ── it has to be said out loud ───────────────────────────────────────────────

def test_the_distances_used_are_printed(capsys):
    storyboard._clean({"shots": _shots(["wide", "close", "mid", "detail"])}, 4)
    assert "framing: " in capsys.readouterr().out


def test_a_sequence_that_never_moves_says_so(capsys):
    """Stuck at one distance is invisible in the prompts and obvious on
    screen — which is the whole reason to ask for framing at all."""
    storyboard._clean({"shots": _shots(["mid", "mid", "mid", "mid", "mid"])}, 5)
    out = capsys.readouterr().out
    assert "every shot is the same distance" in out or "broke" in out


# ── the brief ────────────────────────────────────────────────────────────────

def test_the_prompt_teaches_the_four_words_and_the_jump():
    p = storyboard._prompt("script", ["a", "b"], [])
    for word in ("wide", "mid", "close", "detail"):
        assert f"    {word}" in p, word
    assert "Never three of the same framing" in p
    assert "SCALE IS AN ARGUMENT" in p
    assert "cropped by the edges" in p


def test_the_json_contract_asks_for_it():
    p = storyboard._prompt("script", ["a", "b"], [])
    assert '"framing": "wide|mid|close|detail"' in p


# ── the gallery of sixteen ───────────────────────────────────────────────────
#
# A real run, opened as a folder: thirteen of the sixteen shots were one to
# three stick figures standing upright, facing the viewer, doing nothing, in
# the same pale street. Two more were six-panel contact sheets of the same
# figure with captions under each panel. Every one of those shots obeyed every
# rule the brief had at the time — which is the point.

def test_the_brief_demands_that_something_is_happening():
    """"Three villagers look worried" satisfies "show the literal thing", "one
    subject" and "the subject must change", and is a picture of nobody doing
    anything. The brief had no rule that a shot contains an EVENT."""
    p = _prompt()
    assert "SOMETHING IS HAPPENING IN THE FRAME" in p
    assert "Write the verb first" in p
    assert "standing upright, facing the viewer, doing nothing" in p


def test_the_brief_asks_for_a_verb_not_a_feeling():
    """The failure mode is a person who FEELS something rather than an action
    caught mid-movement, so the examples have to be verbs."""
    p = _prompt()
    for verb in ("goes over", "kicked in", "pours past", "slams", "dragged"):
        assert verb in p, verb


def test_the_brief_forbids_a_contact_sheet():
    """Two shots came back as six-panel grids of the same stick figure,
    because the line named several people and the shot was written as a list
    of them. A list of people renders as a sheet of people."""
    p = _prompt()
    assert "ONE SCENE, ONE CAMERA, ONE MOMENT" in p
    assert "never a grid" in p
    assert "character sheet" in p
