"""A nine-minute script is not a forty-second one with the cap raised.

script_writer is built around one hook, one 105-word body and one set of
checks over the whole thing, and every part of that is load-bearing at forty
seconds. Ask it for 1,300 words and you get a padded Short: one idea
stretched, the same figure restated to fill the space, and a cadence check
averaging over thirteen paragraphs it cannot see separately.

Length is not the difficulty. Structure is.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import longform_writer as lf  # noqa: E402


@pytest.fixture(autouse=True)
def _long(monkeypatch):
    monkeypatch.setenv("RUFUS_FORMAT", "long")


def _outline(n=6, **over):
    o = {
        "cold_open": "Imagine you have not eaten in two days.",
        "turn_line": "And then it starts raining.",
        "thesis": "Rain was not an inconvenience. It was a crisis.",
        "promise": "It could kill you in three different ways.",
        "sections": [{"title": f"Section {i}", "pays": f"piece {i}",
                      "fact": f"In 189{i} the mint struck {i}00 coins.",
                      "hinge": f"hinge {i}", "is_turn": i == 2}
                     for i in range(n)],
        "close": "and that is why",
        "note": "",
    }
    o.update(over)
    return o


# ── the outline is checked before any prose is paid for ─────────────────────

def test_a_plan_with_too_few_sections_is_refused():
    """Below four there is no arc, only a list."""
    assert lf._clean_outline(_outline(2)) is None


def test_a_section_with_no_fact_is_dropped_as_filler():
    o = _outline(6)
    o["sections"][0]["fact"] = ""
    got = lf._clean_outline(o)
    assert len(got["sections"]) == 5


def test_exactly_one_turn_survives():
    """Two reframings cancel each other out; none leaves a list of facts,
    which is the shape of every explainer nobody finishes."""
    o = _outline(6)
    for s in o["sections"]:
        s["is_turn"] = True
    got = lf._clean_outline(o)
    assert sum(1 for s in got["sections"] if s["is_turn"]) == 1


def test_a_plan_with_no_turn_gets_one():
    o = _outline(6)
    for s in o["sections"]:
        s["is_turn"] = False
    got = lf._clean_outline(o)
    assert sum(1 for s in got["sections"] if s["is_turn"]) == 1


def test_junk_is_refused():
    for junk in (None, [], "", {}, {"sections": "nope"}, _outline(6, cold_open="")):
        assert lf._clean_outline(junk) is None


# ── the cheapest place to catch an invention ────────────────────────────────

def test_an_outline_figure_absent_from_the_source_is_named():
    """The fact gate finds these too — after 1,300 words have been written
    and scored, which at long-form length is minutes of generation and the
    largest single cost in the run."""
    o = lf._clean_outline(_outline(5))
    source = "In 1890 the mint struck 000 coins. Also 1891, 1892, 1893, 1894."
    bad = lf.ungrounded_facts(o, source)
    assert bad, "invented figures must be named"
    assert any("'" in b for b in bad)


def test_a_grounded_outline_reports_nothing():
    o = lf._clean_outline(_outline(4))
    source = " ".join(f"In 189{i} the mint struck {i}00 coins." for i in range(4))
    assert lf.ungrounded_facts(o, source) == []


# ── the budget ───────────────────────────────────────────────────────────────

def test_the_sections_do_not_eat_the_whole_budget():
    """The cold open, thesis and close take roughly a fifth between them.
    total/n would overshoot the format ceiling by exactly that fifth, and a
    script over its ceiling is a render over its length."""
    per = lf._section_words(1250, 6)
    assert per * 6 < 1250
    assert per >= lf.MIN_SECTION_WORDS


def test_a_tiny_budget_still_gives_a_real_section():
    assert lf._section_words(200, 8) == lf.MIN_SECTION_WORDS


# ── what a model adds when asked for one section of something ───────────────

def test_headings_are_stripped():
    """"**Section 3: The Fire**" is not narration, and a voice engine reads it
    out loud. The prompt asks for none, which works most of the time — exactly
    the reliability that needs a deterministic backstop."""
    got = lf._strip_headings("**Section 2: The Fire**\n\nThey kept it going.\n"
                             "### What came next\nAll night.\n[b-roll: fire]")
    assert got == "They kept it going.\nAll night."


def test_ordinary_prose_is_untouched():
    text = "They kept it going. All night, and the next day."
    assert lf._strip_headings(text) == text


# ── the long-form shape of "why is everything coins" ────────────────────────

def test_one_subject_carrying_every_section_is_reported():
    """At forty seconds a repeated noun is the storyboard's problem. At 1,300
    words it is the script's, and it is what makes a long video feel like a
    short one said four times."""
    sections = ["The coin was struck in Venice that year." for _ in range(6)]
    got = lf.repeated_across_sections(sections)
    # WHICH word it names is a tie-break between the ones repeated equally
    # often ("coin" and "struck" both carry all six). The finding is that one
    # subject carries the script; asserting a particular winner would be
    # testing Counter's ordering.
    assert got and "6 of 6 sections" in got


def test_a_varied_script_reports_nothing():
    sections = ["The mint opened at dawn.", "Sailors queued on the quay.",
                "A clerk weighed the silver.", "The bread price doubled.",
                "Ships left the harbour empty.", "Nobody returned that spring."]
    assert lf.repeated_across_sections(sections) is None


def test_too_few_sections_to_conclude_anything():
    assert lf.repeated_across_sections(["a coin", "a coin"]) is None


# ── it is the format that decides, not a second flag ────────────────────────

def test_shorts_does_not_reach_this_writer(monkeypatch):
    """A second switch would let the two disagree — RUFUS_FORMAT=long with
    the long writer off is a nine-minute request answered by a forty-second
    writer."""
    monkeypatch.setenv("RUFUS_FORMAT", "short")
    assert lf.enabled() is False
    assert lf.write(None, "", "money_history") is None


def test_no_model_falls_back_rather_than_failing(monkeypatch, capsys):
    import llm
    monkeypatch.setattr(llm, "usable", lambda: False)
    assert lf.write(None, "", "money_history") is None
    assert "falling back" in capsys.readouterr().out


# ── the opening is assembled, not re-generated ──────────────────────────────

def test_the_opening_keeps_the_planned_sentences():
    """Asking a model to rewrite four sentences it just wrote is how the
    counted promise stops matching the sections that pay it."""
    o = lf._clean_outline(_outline(5))
    opening = lf._opening(o)
    for part in ("Imagine you have not eaten", "starts raining",
                 "It was a crisis", "three different ways"):
        assert part in opening


# ── the wiring ───────────────────────────────────────────────────────────────

def test_main_tries_long_form_first_and_falls_back():
    """The format switch has to REACH a different writer, and can never leave
    a run with no script at all."""
    import re
    src = (Path(lf.__file__).parent / "main.py").read_text(encoding="utf-8")
    i = src.index("import longform_writer")
    block = src[i:i + 1200]
    assert "longform_writer.enabled()" in block
    assert "longform_writer.write(" in block
    assert "if result is None:" in block, "the Shorts writer must still answer"
    assert "write_script_until_good(" in block


def test_it_passes_the_niche_name_not_a_config_dict():
    """`active` is the niche NAME everywhere else in run() — it is passed to
    judge_script_facts as niche_name. Calling .get() on it would raise at
    runtime on the first long-form run and nowhere earlier."""
    import re
    src = (Path(lf.__file__).parent / "main.py").read_text(encoding="utf-8")
    i = src.index("longform_writer.write(")
    call = src[i:i + 200]
    assert "active.get(" not in call
