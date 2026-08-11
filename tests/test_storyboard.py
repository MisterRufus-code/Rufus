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
    plan["shots"][1]["carries_over"] = "the same coin from shot 1, now thinner"
    out = storyboard._clean(plan, 2)
    assert "Continuing from the previous shot: the same coin from shot 1" in out[1]
    assert "Continuing from" not in out[0], "shot 1 has nothing to continue from"


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
    """"his expression one of despair" and "revealing the anguish of misplaced
    trust" both shipped in the last run."""
    p = _prompt()
    assert "his expression one of despair" in p
    assert "What is IN the frame, never what someone feels" in p


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
        {"n": 2, "visual": "The same counter, the coin now thin and grey and alone.",
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
