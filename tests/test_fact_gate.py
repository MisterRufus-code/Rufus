"""Tests for the fact gate — grounding + misinformation check on final scripts.

Motivated by a live incident: a history.SE question about Benjamin Freedman's
1961 speech (a known conspiracy source) became an 8/10 script about a
"$50 billion deception" that would have auto-uploaded to an education channel.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import script_writer
from script_writer import _fact_gate


def _client_answering(text: str):
    client = MagicMock()
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = text
    resp.usage.prompt_tokens = 100
    resp.usage.completion_tokens = 10
    client.chat.completions.create.return_value = resp
    return client


_SEED = {"type": "wisdom", "content": "The US printed greenbacks in 1862.",
         "source": "United States, 1862"}


def test_fact_gate_pass():
    ok, reason, cost = _fact_gate(_client_answering("PASS"), _SEED, "some script")
    assert ok is True
    assert reason == ""
    assert cost >= 0


def test_fact_gate_fail_extracts_reason():
    ok, reason, _ = _fact_gate(
        _client_answering("FAIL: presents a conspiracy speech as established history"),
        _SEED, "some script")
    assert ok is False
    assert "conspiracy" in reason


def test_fact_gate_fail_open_on_api_error():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("rate limited")
    ok, reason, cost = _fact_gate(client, _SEED, "some script")
    assert ok is True          # a failed CHECK must never break a render
    assert cost == 0.0


def test_fact_gate_prompt_contains_seed_and_script():
    client = _client_answering("PASS")
    _fact_gate(client, _SEED, "THE SCRIPT BODY HERE")
    prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert "greenbacks in 1862" in prompt        # seed grounding included
    assert "THE SCRIPT BODY HERE" in prompt      # script under test included
    assert "conspiracy" in prompt.lower()        # misinformation clause present


def test_fact_gate_prompt_carves_out_ordinary_editorializing():
    """Live pattern: good scripts (8/10, 10/10) kept getting capped to 5/10
    because the checker model flagged ordinary 'why it happened' narration —
    'simpler for trade', 'redefined modern finance' — as an unsupported-motive
    violation. Rule 3 must be narrowed to actual secret/covert motive claims,
    with an explicit carve-out for normal editorial explanation, or this keeps
    capping good scripts for no real accuracy problem.

    The carve-out survives a later rewrite of this prompt into named
    categories; only its wording moved (to "ordinary cause-and-effect
    narration"), so this asserts the intent rather than the old phrasing."""
    client = _client_answering("PASS")
    _fact_gate(client, _SEED, "some script")
    prompt = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    low = prompt.lower()
    # Vocabulary moved from "secret/covert motive" to the named categories
    # MIND-READ ("hidden motive") and CONSPIRACY ("hidden cabals"); the target
    # is the same narrow one.
    assert "hidden motive" in low or "hidden cabals" in low \
        or "secret" in low or "covert" in low
    assert "cause-and-effect" in low or "ordinary editorial" in low \
        or "normal explanatory" in low
    assert "simpler for trade" in low, "the worked example must survive"


def test_write_script_wires_the_gate_and_caps_score():
    import inspect
    src = inspect.getsource(script_writer.write_script)
    assert "_fact_gate" in src
    assert "score_min - 3" in src   # cap lands below the auto-upload threshold


# ── The Gresham's-law run: three cycles burned, 8/10 capped to 4/10 ─────────
# Every cycle failed on MIND-READ for "people noticed and stashed the good ones
# away" — which is not an inference about anyone's inner life, it is what
# Gresham's law SAYS. The gate was failing the excerpt for restating the
# excerpt. CLAUDE.md's warning about the "wasted-generation rejection ladder"
# names this exact shape, so the fix narrows the category rather than adding one.

def _prompt_text():
    client = _client_answering("PASS")
    _fact_gate(client, _SEED, "some script")
    return client.chat.completions.create.call_args.kwargs["messages"][0]["content"]


def test_an_unnamed_aggregate_is_not_a_named_actor():
    """"People", "traders", "the public" claim nobody's private mind, because
    nobody in particular is being described."""
    p = _prompt_text()
    assert "UNNAMED aggregate" in p
    assert "not named actors" in p


def test_restating_the_sources_own_mechanism_passes():
    """The worked example must survive — it is the one that actually fired."""
    p = _prompt_text()
    assert "RESTATING THE SOURCE'S OWN MECHANISM" in p
    assert "Gresham's law" in p
    assert "people keep the good coin and spend the bad" in p


def test_contradicted_requires_an_actual_disagreement():
    """The same run's third cycle called it CONTRADICTED that the script said
    bad money circulates while the source said good money is retained and bad
    money circulates. Those agree."""
    p = _prompt_text()
    assert "side by side and check they actually DISAGREE" in p
    assert "Agreement restated in different words is a PASS" in p
