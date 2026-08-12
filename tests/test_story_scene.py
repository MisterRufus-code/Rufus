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
