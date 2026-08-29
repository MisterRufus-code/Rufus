"""What Rufus is made of, and what each part lets you do with it.

A pipeline assembles other people's work: a text model writes the script, an
image model draws every shot, a voice model reads it, a font sets the captions.
Some of those licences permit use and forbid COMMERCIAL use — which is exactly
what a monetised channel is — and nothing in this tree knew that. tts_engine's
docstring sells XTTS on quality and VRAM and says nothing about what its
licence permits, and that is true of every engine switch in the product.

The tests below are mostly about honesty rather than about licences: the danger
of a compliance tool is that it reads like an authority. Every one of these
pins some form of "does not claim more than it knows".
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import licensing  # noqa: E402


@pytest.fixture(autouse=True)
def answers_file(tmp_path, monkeypatch):
    monkeypatch.setattr(licensing, "ANSWERS_FILE", tmp_path / "licences.json")
    return tmp_path / "licences.json"


def test_nothing_is_cleared_until_somebody_read_the_page():
    """THE FAILURE MODE THIS FILE EXISTS TO AVOID. A manifest that ships with
    confident verdicts baked in is a table that was right the day it was
    written and goes stale silently — the same shape as every other constant
    in this tree that outlived the fact it recorded. A default of "nobody has
    looked" is worth more than a checked-in guess that reads like a fact."""
    r = licensing.report({})
    assert r["open"], "a fresh install must report open questions, not clearance"
    assert all(row["verdict"] == licensing.UNKNOWN for row in r["open"])


def test_the_report_answers_for_the_setup_in_front_of_you():
    """Listing every model the repository has ever mentioned is a reference
    document. What an owner needs is which ones are switched on right now."""
    xtts = licensing.report({"RUFUS_TTS": "xtts"})
    edge = licensing.report({"RUFUS_TTS": "edge"})
    assert "xtts-v2" in {r["key"] for r in xtts["active"]}
    assert "xtts-v2" not in {r["key"] for r in edge["active"]}
    assert "edge-tts" in {r["key"] for r in edge["active"]}


def test_an_engine_nobody_selected_is_still_listed_somewhere():
    """Inactive is not deleted. Switching engines is one environment variable,
    and the question it opens should be findable BEFORE the switch rather than
    discovered after a hundred videos."""
    r = licensing.report({"RUFUS_TTS": "edge"})
    assert "xtts-v2" in {row["key"] for row in r["inactive"]}


def test_selling_the_software_and_selling_the_videos_are_separate_questions():
    """A permissively licensed program can still produce output you may not
    sell, and a non-commercial MODEL says nothing about redistributing the
    program that calls it. Collapsing the two produces confident answers to
    the wrong question."""
    gates = {c.key: c.question for c in licensing.COMPONENTS}
    assert gates["ffmpeg"] == licensing.SELL
    assert gates["xtts-v2"] == licensing.MONETISE
    assert {licensing.SELL, licensing.MONETISE} == set(gates.values())


def test_every_question_carries_the_page_that_answers_it():
    """An open question with no link is a worry rather than a task."""
    for c in licensing.COMPONENTS:
        assert c.source.startswith("https://"), c.key
        assert c.what, c.key


def test_dependency_licences_are_read_and_not_asserted():
    """The one part of the picture the machine can answer for itself.
    requirements.txt holds FLOORS — `openai>=1.40` is not a version — so a
    hand-written table answers for whatever pip resolved on the day somebody
    typed it."""
    rows = {r["package"]: r for r in licensing.package_licences()}
    assert rows["requests"]["version"], "version read from the installed dist"
    assert rows["requests"]["family"] == "permissive"


def test_the_copyleft_dependency_is_surfaced_rather_than_buried():
    """edge-tts is LGPL and is the DEFAULT voice — the one dependency in the
    core set with obligations attached, and the one nothing mentioned."""
    rows = {r["package"]: r for r in licensing.package_licences()}
    edge = rows["edge-tts"]
    if edge["family"] == "not installed":
        pytest.skip("edge-tts not installed in this environment")
    assert edge["family"] == "copyleft", edge


def test_a_recorded_answer_says_who_read_what_and_when():
    """A verdict with no provenance cannot be re-checked, and licences change.
    Who, which page, what date — or it is just another opinion in a file."""
    licensing.record("ffmpeg", licensing.YES, by="daniel",
                     note="LGPL build, no libx264")
    r = licensing.report({})
    row = next(x for x in r["active"] if x["key"] == "ffmpeg")
    assert row["verdict"] == licensing.YES
    assert row["recorded"]["by"] == "daniel"
    assert row["recorded"]["on"]
    assert row["recorded"]["source_read"].startswith("https://")


def test_a_recorded_no_blocks_and_is_reported_apart_from_the_unknowns():
    """Two different actions. A NO is a decision to change the configuration;
    an UNKNOWN is twenty minutes of reading. A single list of "problems" makes
    the reader work out which is which."""
    licensing.record("z-image", licensing.NO, by="daniel",
                     note="research use only")
    r = licensing.report({"RUFUS_VIDEO_SOURCE": "comfy"})
    assert [x["key"] for x in r["blocking"]] == ["z-image"]
    assert "z-image" not in {x["key"] for x in r["open"]}


def test_a_verdict_nobody_could_type_is_refused():
    with pytest.raises(ValueError):
        licensing.record("ffmpeg", "probably fine")


def test_recording_against_a_component_that_does_not_exist_is_refused():
    """A typo that writes an answer nothing reads is a question that stays
    open while the file says it was handled."""
    with pytest.raises(ValueError):
        licensing.record("ffmpg", licensing.YES)


def test_an_unreadable_answers_file_is_not_read_as_clearance(answers_file,
                                                             capsys):
    """ABSENT AND CORRUPT ARE NOT THE SAME ANSWER — the same rule the user
    store learned the hard way, and here the failure would be worse: a file
    that lost a brace must not silently report everything cleared."""
    answers_file.write_text("{ not json", encoding="utf-8")
    r = licensing.report({})
    assert r["open"], "a broken file must fall back to unanswered"
    assert "could not be read" in capsys.readouterr().out


def test_the_cli_exits_non_zero_only_when_something_is_actually_blocked(
        monkeypatch, capsys):
    """Open questions are the normal state of a fresh install and must not
    fail a build; a component recorded as forbidding what this channel does
    is a different matter."""
    monkeypatch.setattr(sys, "argv", ["licensing.py"])
    assert licensing._cli() == 0
    licensing.record("z-image", licensing.NO, by="t")
    monkeypatch.setenv("RUFUS_VIDEO_SOURCE", "comfy")
    assert licensing._cli() == 1
    capsys.readouterr()


# ── and it reaches the person who has to decide ──────────────────────────────
#
# The manifest existing is half a feature. This repository's recurring bug is
# the other half: built, wired, tested and never actually put anywhere a person
# would look. Health check and the dashboard are the two places somebody is
# already standing when this matters.

def test_the_page_names_the_engines_this_setup_actually_uses(monkeypatch):
    import dashboard
    monkeypatch.setenv("RUFUS_TTS", "xtts")
    page = dashboard.app.test_client().get("/licence").get_data(as_text=True)
    assert "Coqui XTTS v2 weights" in page
    assert "coqui.ai/cpml" in page, "the page that answers it has to be one click"


def test_recording_from_the_page_writes_a_dated_attributed_answer(
        answers_file, monkeypatch):
    """A button whose answer never leaves the page is this repository's oldest
    bug wearing a compliance hat."""
    import dashboard
    c = dashboard.app.test_client()
    c.post("/licence/ffmpeg", data={"verdict": licensing.YES})
    recorded = json.loads(answers_file.read_text(encoding="utf-8"))
    row = recorded["components"]["ffmpeg"]
    assert row["verdict"] == licensing.YES
    assert row["on"] and row["source_read"].startswith("https://")


def test_a_hand_typed_verdict_is_refused_by_the_route(answers_file):
    import dashboard
    r = dashboard.app.test_client().post("/licence/ffmpeg",
                                         data={"verdict": "sure why not"})
    assert r.status_code in (301, 302) and "error" in r.headers["Location"]
    assert not answers_file.exists()


def test_the_preflight_says_so_before_a_first_render(capsys, monkeypatch):
    """health_check is the script somebody runs while deciding whether this
    setup is the one. "The model drawing every picture may not permit what you
    are about to do with them" belongs there rather than after a hundred
    videos — as a WARNING, because an unchecked question is the normal state of
    a fresh install and must not block a first run."""
    import health_check
    monkeypatch.setattr(sys, "argv", ["health_check.py"])
    try:
        health_check.run()
    except SystemExit:
        pass
    out = capsys.readouterr().out
    assert "Licensing" in out
