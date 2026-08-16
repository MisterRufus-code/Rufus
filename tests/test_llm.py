"""Twelve call sites, ten modules, and nowhere to put a base_url.

Every LLM caller did the same two lines: read config/keys.json, construct
`OpenAI(api_key=key)`. Nothing was wrong with any of them, and together they
made one thing impossible — pointing the pipeline at a different endpoint. A
local model on the owner's own 3090 is not a feature request, it is a
base_url, and there was nowhere to put it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import llm  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for v in ("RUFUS_LLM_BASE", "RUFUS_LLM_KEY", "RUFUS_LLM_MODEL",
              "RUFUS_LLM_MODEL_HOOK_GEN"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(llm, "_said", False)


def test_cloud_is_the_default_and_unchanged():
    assert llm.base_url() == ""
    assert llm.is_local() is False


def test_a_local_base_is_picked_up(monkeypatch):
    monkeypatch.setenv("RUFUS_LLM_BASE", "http://localhost:11434/v1/")
    assert llm.is_local()
    assert llm.base_url() == "http://localhost:11434/v1"  # trailing slash gone


def test_a_local_server_needs_no_real_key(monkeypatch, tmp_path):
    """Demanding one would mean keeping a fake OpenAI key in config just to
    run offline — a confusing thing to find in a file called keys.json six
    months later."""
    monkeypatch.setattr(llm, "KEYS_FILE", tmp_path / "nope.json")
    monkeypatch.setenv("RUFUS_LLM_BASE", "http://localhost:8000/v1")
    assert llm.api_key() == "local"
    assert llm.usable() is True


def test_no_key_and_no_local_base_is_still_unusable(monkeypatch, tmp_path):
    """The callers already treat that as "skip this stage and carry on", and
    they must keep doing so."""
    monkeypatch.setattr(llm, "KEYS_FILE", tmp_path / "nope.json")
    assert llm.api_key() == ""
    assert llm.usable() is False


def test_a_placeholder_key_does_not_count(monkeypatch, tmp_path):
    f = tmp_path / "keys.json"
    f.write_text('{"openai": "YOUR_KEY_HERE"}', encoding="utf-8")
    monkeypatch.setattr(llm, "KEYS_FILE", f)
    assert llm.api_key() == ""


def test_a_broken_keys_file_is_survivable(monkeypatch, tmp_path):
    f = tmp_path / "keys.json"
    f.write_text("{ not json", encoding="utf-8")
    monkeypatch.setattr(llm, "KEYS_FILE", f)
    assert llm.api_key() == ""


# ── per-role models ──────────────────────────────────────────────────────────

def test_with_nothing_set_the_caller_keeps_its_own_model():
    """A pipeline with none of these set must behave exactly as it did."""
    assert llm.model_for("hook_gen", "gpt-4o-mini") == "gpt-4o-mini"


def test_a_general_override_applies_everywhere(monkeypatch):
    monkeypatch.setenv("RUFUS_LLM_MODEL", "qwen3:14b")
    assert llm.model_for("storyboard", "gpt-4o") == "qwen3:14b"


def test_a_role_override_beats_the_general_one(monkeypatch):
    """They are not one job: eight hook candidates at temperature 1.0 wants
    speed, the storyboard has to hold fourteen shots in its head at once."""
    monkeypatch.setenv("RUFUS_LLM_MODEL", "qwen3:14b")
    monkeypatch.setenv("RUFUS_LLM_MODEL_HOOK_GEN", "llama3.2:3b")
    assert llm.model_for("hook_gen", "gpt-4o-mini") == "llama3.2:3b"
    assert llm.model_for("storyboard", "gpt-4o") == "qwen3:14b"


# ── the log has to say which brain answered ──────────────────────────────────

def test_a_local_run_says_so_once(monkeypatch, capsys):
    """A run that came out differently because a 14B model at home served it
    is a run whose log has to say so."""
    monkeypatch.setenv("RUFUS_LLM_BASE", "http://localhost:11434/v1")
    llm.announce()
    llm.announce()
    llm.announce()
    assert capsys.readouterr().out.count("[llm] local endpoint") == 1


def test_a_cloud_run_says_nothing(capsys):
    llm.announce()
    assert capsys.readouterr().out == ""


def test_the_writers_ask_the_factory_rather_than_building_a_client():
    """The point of the exercise. A module that still constructs its own
    client is a module a local endpoint cannot reach."""
    import inspect
    import script_writer, storyboard, supervisor
    for mod in (script_writer, storyboard, supervisor):
        src = inspect.getsource(mod)
        assert "llm.client(" in src, mod.__name__
