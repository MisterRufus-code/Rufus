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


def test_write_script_wires_the_gate_and_caps_score():
    import inspect
    src = inspect.getsource(script_writer.write_script)
    assert "_fact_gate" in src
    assert "score_min - 3" in src   # cap lands below the auto-upload threshold
