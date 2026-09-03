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

def _header_block() -> str:
    """The step-2.5 header, sliced by its own boundaries rather than by a
    character count. Both of these tests used a fixed window (`[:1400]`) and
    both went green on a header that could not run at all: the print
    referenced `max_scenes`, which is assigned in the NEXT block, so every
    live run printed "(config summary unavailable: cannot access local
    variable 'max_scenes')" while the tests read the source and saw the words
    they were looking for. Then a comment grew and the window slid off the
    end, and they failed for a reason that had nothing to do with the bug."""
    src = (Path(comfy_client.__file__).parent / "main.py").read_text(encoding="utf-8")
    after = src.split("Generating clips from script content")[1]
    block = after.split("max_scenes = _target_beats")[0]
    # Comments dropped: prose ABOUT the bug must not read as the bug, the same
    # reason test_launchers._body strips REM lines.
    return "\n".join(l for l in block.splitlines()
                     if not l.strip().startswith("#"))


def test_the_run_states_its_configuration_up_front():
    """Three lines at step 2.5 beat a style string buried in prompt 14."""
    block = _header_block()
    assert "look:" in block
    assert "inserts:" in block
    assert "beat(s)" in block


def test_the_header_cannot_break_the_run():
    assert "except Exception" in _header_block()


def test_the_header_uses_no_name_bound_after_it():
    """The bug itself, as a test. `max_scenes` is assigned in the block below
    the header, so naming it there is an UnboundLocalError every time — caught
    by the header's own try/except and printed as a shrug. A try/except that
    exists so a cosmetic line cannot kill a render also let that line never
    work once."""
    assert "max_scenes" not in _header_block()


def test_the_header_says_where_the_beat_count_came_from():
    """"pictures: 24" is a fact; "pictures: 24 (SD_CLIPS=24)" is the same fact
    with its cause. A saved SD_CLIPS silently overriding the adaptive count is
    exactly the surprise this header exists to prevent, and the header could
    not report it while it named the number without its source."""
    block = _header_block()
    assert "SD_CLIPS" in block
    assert "from the script" in block
