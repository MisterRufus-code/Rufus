"""The pipeline now asks for one moment a camera could have filmed.

Nothing ever did before. The story architect produced SPINE FACT / THE TURN /
STAKES GAP / WHY NOW — four abstractions — and no stage was required to name a
date, a place, and a person doing something. That single gap produced all three
symptoms the owner reported:

  - scripts with no emotion and no personal story: "Jakob Fugger held wealth
    equivalent to two percent of Europe's GDP" — nobody wants anything, nobody
    loses anything, no moment happens;
  - repeated `MIND-READ` fact-gate caps (8/10 capped to 4/10, twice in one
    run) — with no event to carry feeling, the writer reached for motive, and
    the gate is right to refuse it;
  - pictures that illustrate the topic: you cannot storyboard "two percent of
    Europe's GDP", so the storyboard invented a castle "embodying the control
    and power within Europe".

The gate's own text names the standard this restores: "could a camera have
filmed it? Then it is an EVENT, and events are what this channel is made of."
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import script_writer as sw


PLAN = """THE SCENE: In 1523 Jakob Fugger sent Charles V a letter demanding repayment.
SPINE FACT: Fugger financed the imperial election.
THE TURN: The loan was called in.
STAKES GAP: x
WHY NOW: y"""


def test_the_scene_is_extracted():
    assert sw.scene_from_plan(PLAN) == (
        "In 1523 Jakob Fugger sent Charles V a letter demanding repayment.")


def test_markdown_bold_is_tolerated():
    """The architect returns the labels bolded about half the time."""
    assert sw.scene_from_plan(
        "**THE SCENE:** In 1560 Elizabeth ordered the recoinage.\n"
        "**SPINE FACT:** x") == "In 1560 Elizabeth ordered the recoinage."


def test_an_honest_none_yields_no_anchor():
    """A source with no filmable moment must say so rather than invent one — a
    fabricated scene fails the fact-check and holds the whole video."""
    assert sw.scene_from_plan("THE SCENE: NONE\nSPINE FACT: x") == ""


def test_a_plan_without_the_field_is_not_an_error():
    """Plans predate this field; the storyboard just gets no anchor."""
    assert sw.scene_from_plan("SPINE FACT: x\nTHE TURN: y") == ""


def test_empty_and_junk_are_safe():
    assert sw.scene_from_plan("") == ""
    assert sw.scene_from_plan("   ") == ""


def test_the_architect_asks_for_a_filmable_moment():
    src = Path(sw.__file__).read_text(encoding="utf-8")
    assert "THE SCENE:" in src
    assert "a camera could have filmed" in src
    # the worked contrast that teaches the distinction
    assert "two percent of Europe" in src


def test_the_architect_forbids_inventing_a_scene():
    src = Path(sw.__file__).read_text(encoding="utf-8")
    assert "write NONE rather than inventing" in src


def test_the_architect_now_asks_for_five_lines():
    src = Path(sw.__file__).read_text(encoding="utf-8")
    assert "exactly 5 short labeled lines" in src


def test_the_body_prompt_routes_feeling_through_the_event():
    """The fact gate passes emotional writing about what HAPPENED and refuses
    claims about what anyone felt. The instruction has to say which one."""
    src = Path(sw.__file__).read_text(encoding="utf-8")
    assert "USE THE SCENE" in src
    assert "Feeling comes from the EVENT" in src


def test_the_scene_is_published_for_the_storyboard():
    assert hasattr(sw, "LAST_SCENE")
    assert isinstance(sw.LAST_SCENE, str)


def test_storyboard_plan_accepts_the_scene():
    import inspect

    import storyboard

    assert inspect.signature(storyboard.plan).parameters["scene"].default == ""


def test_main_passes_the_scene_through():
    src = (Path(__file__).parent.parent / "scripts" / "main.py").read_text(
        encoding="utf-8")
    assert "LAST_SCENE" in src
    assert "scene=_scene" in src


# ── is THE SCENE actually filmable? ──────────────────────────────────────────
#
# Both scenes below are verbatim from real runs one day apart. The first
# produced a script that opened "February 20, 1893, Philadelphia — workers
# huddled outside the Philadelphia and Reading Railroad office as receivers
# took over." The second produced "The secret? Its historical resilience and
# trust." after three full script cycles. The difference is visible in the
# plan, before a cent is spent on prose.

_STRONG = ("February 20, 1893, in Philadelphia — workers gather outside the "
           "Philadelphia and Reading Railroad office, anxiously watching as "
           "receivers are appointed.")
_WEAK = ("In 2022, traders exchanged billions of pounds in currency markets, "
         "showcasing sterling's trading activity.")


def test_a_real_moment_is_not_flagged():
    assert sw.scene_weakness(_STRONG) is None


def test_the_scene_that_produced_a_topic_shaped_script_is_flagged():
    reason = sw.scene_weakness(_WEAK)
    assert reason
    assert "named place or person" in reason
    assert "MEANS" in reason


def test_a_place_alone_is_enough_to_pass():
    """Not every good scene names a person. "A trader at a London exchange"
    is somewhere, and somewhere is what the storyboard needs."""
    assert sw.scene_weakness(
        "In 2022, a trader at a bustling London foreign exchange market "
        "exchanges pounds for euros.") is None


def test_a_month_is_a_date_not_a_place():
    """"February" is capitalised and is not somewhere a camera can stand —
    counting it would let every dated non-scene through."""
    assert sw.scene_weakness("In February 2022, traders swapped currencies.")


def test_an_honest_none_is_not_weak():
    """The architect is told to write NONE when the source has no moment in
    it. Punishing that honesty pushes it to invent one, which fails the fact
    gate and holds the whole video."""
    assert sw.scene_weakness("") is None


def test_the_comment_tail_is_caught_even_with_a_place():
    assert sw.scene_weakness(
        "In 1893 crowds filled Wall Street, reflecting the fragility of the "
        "economy.")


def test_a_weak_scene_is_a_warning_and_never_a_rejection(capsys):
    """AGENTS.md: hard gates are for factual correctness. A vague moment is a
    quality problem — the run must still finish with the plan it has."""
    import inspect
    src = inspect.getsource(sw._story_architect)
    assert "scene_weakness" in src
    # The grounded plan is returned, never dropped, on every path out.
    assert "best_grounded or last_plan" in src


def test_a_grounded_but_weak_plan_outranks_an_ungrounded_retry():
    """The re-ask can come back with an invented motive. A dull true plan
    beats a vivid false one — the fact gate holds the video either way."""
    import inspect
    src = inspect.getsource(sw._story_architect)
    assert "best_grounded = best_grounded or plan" in src


# ── The parable failure ──────────────────────────────────────────────────────
# A money_history run seeded on Wikipedia's "Social cost" — a concept article
# with no event in it — produced this plan, and a whole video from it:
#
#     THE SCENE: A lemonade stand on a summer day where a child mixes lemons,
#                sugar, and water to sell lemonade.
#
# The owner's verdict: the clip talks about a concept with nothing historical
# or financial attached to it, on a channel whose entire promise is that the
# thing happened. Every gate on the path saw it and none of them acted.

_PARABLE = ("A lemonade stand on a summer day where a child mixes lemons, "
            "sugar, and water to sell lemonade.")


def test_the_lemonade_parable_is_flagged():
    reason = sw.scene_weakness(_PARABLE, "money_history")
    assert reason
    assert "no year or date" in reason


def test_an_explicit_hypothetical_is_flagged():
    """"Imagine" is the tell. A model reaches for it exactly when the source
    has no moment in it, which is the moment it must say NONE instead."""
    reason = sw.scene_weakness(
        "Imagine a farmer in a Bavarian village weighing his grain in 1840.",
        "money_history")
    assert reason and "hypothetical" in reason


def test_a_history_scene_must_be_pinned_to_a_time():
    """A moment without a time is an illustration. This is the check that
    separates "workers gathered outside the office" from "workers gather
    outside offices"."""
    assert sw.scene_weakness("Workers gather outside the Philadelphia and "
                             "Reading Railroad office.", "money_history")
    assert sw.scene_weakness(_STRONG, "money_history") is None


def test_old_dates_and_eras_count_as_a_time():
    for scene in ("In 991, Aethelred pays silver to Viking forces at Maldon.",
                  "In 44 BC, Brutus mints a coin in a Greek camp.",
                  "In the 1930s, crowds queue outside a Berlin bank."):
        assert sw.scene_weakness(scene, "money_history") is None, scene


def test_niches_without_a_history_promise_do_not_need_a_date():
    """A scene on a motivation channel is a person at a desk at 6am. Demanding
    a year there would reject every correct answer — so the rule asks the
    niche instead of applying one shape to all of them."""
    scene = "A runner ties her shoes on an empty track at the Vancouver oval."
    assert sw.scene_weakness(scene, "motivation") is None
    assert sw.scene_weakness(scene) is None


def test_the_architect_is_told_not_to_answer_with_a_parable():
    import inspect
    src = inspect.getsource(sw._story_architect)
    assert "NEVER ANSWER WITH A PARABLE" in src
    assert "lemonade stand" in src


def test_an_unfilmable_scene_holds_the_upload():
    """The warning existed and printed twice; nothing acted on it. A held
    video is still rendered and still reviewable — this is not a rejection,
    it is the difference between the pipeline noticing and the pipeline
    caring."""
    import inspect
    src = inspect.getsource(sw._story_architect)
    assert "LAST_SCENE_WEAKNESS" in src
    assert "will be HELD" in src

    main_src = Path(sw.__file__).parent.joinpath("main.py").read_text(encoding="utf-8")
    assert 'result or {}).get("scene_weak")' in main_src
    assert "no real moment in the plan" in main_src


def test_a_rejected_seed_is_not_used_anyway():
    """The old code allowed one retry and then printed "retry seed used
    anyway". A rejected seed is the cheapest thing here to throw away and the
    most expensive to keep — it is upstream of every other gate."""
    main_src = Path(sw.__file__).parent.joinpath("main.py").read_text(encoding="utf-8")
    # The old line, verbatim. It survives above only as a comment recording
    # what went wrong, which is why this looks for the code and not the words.
    # Scoped to the SEED block: a sibling footage supervisor still uses the
    # one-retry shape, and re-rolling a picture prompt is a different bet from
    # re-rolling the source the whole video is built on.
    seed_block = main_src.split("from supervisor import judge_seed")[1] \
                         .split("Pre-analysis runs here")[0]
    assert "used anyway" not in seed_block
    assert "RUFUS_SEED_TRIES" in main_src
    assert "no source the supervisor would accept" in main_src
