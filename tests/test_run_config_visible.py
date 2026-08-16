"""Saying what a run is doing, before it spends half an hour doing it.

THE REPORT: "the last run was bullshit — it added images on top of images and
didn't use the stickman." Both were true, both were DEFAULTS, and both were
invisible.

The style was in the log — inside all twenty-eight prompts, as the tail of a
250-word block, where nobody reads it. The insert layer announced itself once,
eight hundred lines below where the decision was made. The information existed
and was unreadable, which is the same as not having it.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import comfy_client  # noqa: E402
import insert_director  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for var in ("RUFUS_STYLE", "RUFUS_STILLS_DETAIL", "RUFUS_INSERTS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(comfy_client, "_LOOK_SAID", "")


def test_the_look_is_named_even_when_it_is_the_default(capsys):
    """The silent fall-through is what rendered a run in flat vector while the
    owner expected stickman."""
    comfy_client._detail_suffix()
    out = capsys.readouterr().out
    assert "look:" in out
    assert "built-in default" in out
    assert "RUFUS_STYLE" in out, "it must say how to choose another"


def test_the_default_line_lists_the_looks_available(capsys):
    comfy_client._detail_suffix()
    out = capsys.readouterr().out
    assert "stickman" in out


def test_a_named_style_says_which_one(monkeypatch, capsys):
    monkeypatch.setenv("RUFUS_STYLE", "stickman")
    comfy_client._detail_suffix()
    assert "look: stickman" in capsys.readouterr().out


def test_a_literal_override_says_so(monkeypatch, capsys):
    monkeypatch.setenv("RUFUS_STILLS_DETAIL", "pencil sketch on grey paper")
    comfy_client._detail_suffix()
    assert "literal override" in capsys.readouterr().out


def test_the_look_is_announced_once_not_per_prompt(monkeypatch, capsys):
    """Twenty-eight prompts must not become twenty-eight identical lines."""
    monkeypatch.setenv("RUFUS_STYLE", "stickman")
    for _ in range(5):
        comfy_client._detail_suffix()
    assert capsys.readouterr().out.count("look: stickman") == 1


def test_a_typo_style_is_still_loud(monkeypatch, capsys):
    monkeypatch.setenv("RUFUS_STYLE", "stikman")
    comfy_client._detail_suffix()
    out = capsys.readouterr().out
    assert "not a known style" in out
    assert "stickman" in out, "it must show the real names"


# ── the header ──────────────────────────────────────────────────────────────

def test_the_run_states_its_configuration_up_front():
    """Three lines at step 2.5 beat a style string buried in prompt 14."""
    src = (Path(comfy_client.__file__).parent / "main.py").read_text(encoding="utf-8")
    block = src.split("Generating clips from script content")[1][:1400]
    assert "look:" in block
    assert "inserts:" in block
    assert "beat(s)" in block


def test_the_header_cannot_break_the_run():
    src = (Path(comfy_client.__file__).parent / "main.py").read_text(encoding="utf-8")
    block = src.split("Generating clips from script content")[1][:1600]
    assert "except Exception" in block
