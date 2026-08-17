"""AGENTS.md is the canonical guidance file, and it has to stay true.

Two guidance files drift apart, and the one that drifts is the one nobody
notices is wrong — so CLAUDE.md is a pointer, not a copy, and these tests fail
if that inverts or if the documented commands stop existing.
"""

import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
AGENTS = ROOT / "AGENTS.md"
CLAUDE = ROOT / "CLAUDE.md"


def test_agents_md_exists():
    assert AGENTS.exists(), "AGENTS.md is the file every non-Claude agent looks for"


def test_claude_md_points_at_agents_md_instead_of_copying_it():
    text = CLAUDE.read_text(encoding="utf-8")
    assert "AGENTS.md" in text
    # A pointer, not a second copy that can rot independently.
    assert len(text.splitlines()) < 20


def test_every_script_named_in_agents_md_exists():
    text = AGENTS.read_text(encoding="utf-8")
    named = set(re.findall(r"`(scripts/[a-z_]+\.py)`", text))
    named |= {f"scripts/{m}" for m in re.findall(r"`([a-z_]+\.py)`", text)}
    missing = [p for p in sorted(named) if not (ROOT / p).exists()]
    assert not missing, f"AGENTS.md names scripts that do not exist: {missing}"


def test_every_config_named_in_agents_md_is_real_or_documented_as_absent():
    """Template files are legitimately absent until the owner exports them —
    but a config path that is neither present nor described as a template the
    owner creates is a documentation bug."""
    text = AGENTS.read_text(encoding="utf-8")
    named = set(re.findall(r"`(config/[a-z_]+\.json)`", text))
    for path in sorted(named):
        if (ROOT / path).exists():
            continue
        assert "template" in text.lower() or "export" in text.lower(), (
            f"{path} is missing and AGENTS.md does not explain that it is "
            f"owner-exported")


def test_the_encoding_rule_is_documented_with_its_evidence():
    """The cp1255 round trip is the whole reason the rule exists — a rule
    without its evidence gets 'simplified' away by the next agent."""
    text = AGENTS.read_text(encoding="utf-8")
    assert "utf-8" in text
    assert "cp1255" in text
    assert "ג€”" in text


def test_the_which_rule_is_documented():
    text = AGENTS.read_text(encoding="utf-8")
    assert "shutil.which" in text
    assert "WinError 2" in text


def test_the_no_new_hard_gates_rule_survives():
    """Load-bearing: this repo has real wasted-generation bugs from stacking
    deterministic gates for stylistic preferences."""
    text = AGENTS.read_text(encoding="utf-8")
    assert "rejection ladder" in text


def test_the_approve_boundary_is_documented_as_immovable():
    # Whitespace-normalised: the file is hard-wrapped, so a phrase that spans a
    # line break is still present to a reader and must still count here.
    text = re.sub(r"\s+", " ", AGENTS.read_text(encoding="utf-8"))
    assert "Only `owner` may approve" in text
    assert "does not move without explicit, unambiguous instruction" in text


def test_the_test_command_is_the_one_that_actually_runs():
    text = AGENTS.read_text(encoding="utf-8")
    assert "python -m pytest -q" in text


def test_the_instruction_surfaces_are_documented():
    """"How do I instruct this thing" had no answer in the repo. All four
    places must be named, in leverage order, with the one that overrides the
    others called out."""
    text = re.sub(r"\s+", " ", AGENTS.read_text(encoding="utf-8"))
    for surface in ("config/gold_examples.json", "DIRECTION.md",
                    "config/direction/<channel>.md", "gpt_system",
                    "config/script_standards.json"):
        assert surface in text, surface
    assert "the model mimics these more than any instruction" in text
    assert "overrides all prose" in text


def test_direction_md_exists_and_carries_its_marker():
    dm = ROOT / "DIRECTION.md"
    assert dm.exists()
    assert "## The direction" in dm.read_text(encoding="utf-8")


def test_adding_another_llm_stage_is_documented_as_the_wrong_reflex():
    text = re.sub(r"\s+", " ", AGENTS.read_text(encoding="utf-8"))
    assert "Adding another LLM stage is almost never the answer" in text


def test_the_format_rule_is_documented_with_its_evidence():
    """Today's largest class of bug, and the one most likely to be repeated by
    an agent who only reads one module. A rule without its evidence gets
    'simplified' away by the next agent, so the three shapes it took have to
    survive in the file."""
    text = AGENTS.read_text(encoding="utf-8")
    assert "video_format.py" in text
    assert "1080, 1920" in text, "the literal that was in seven modules"
    for evidence in ("measurement contradicting the feature",
                     "loop line",
                     "SPREAD rather than truncate"):
        assert evidence.lower() in text.lower(), evidence


def test_the_two_renderers_rule_is_documented():
    """A caption rule written into one renderer is a video that looks
    different depending on which engine drew it, and nobody watches both."""
    text = AGENTS.read_text(encoding="utf-8")
    assert "remotion_renderer" in text and "audio_gen" in text
    assert "both call" in text.lower() or "one function" in text.lower()
