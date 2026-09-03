"""The owner's own instructions, read by the content agents.

AGENTS.md instructs the coding agents. Nothing instructed the CONTENT agents —
every creative-direction document the owner wrote reached this pipeline only
because somebody read it and hand-translated it into prompt text, which meant it
was never theirs to change. DIRECTION.md is.

The guardrails are the design, not decoration. Each one is a bug this repo
already shipped:

  - unbounded prose: the body prompt once said "split it into short sentences"
    while the cadence gate demanded a 15-word sentence, and scripts were
    rejected for obeying the instruction;
  - prose vs numeric gates: script_standards.json is enforced in code, so
    "keep it to 60 words" produces a run of scripts rejected for being under
    min_words rather than shorter scripts;
  - invisible loading: "I edited the file and nothing changed" must not be
    indistinguishable from having edited the wrong file.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import script_writer as sw

ROOT = Path(__file__).parent.parent


@pytest.fixture
def direction(tmp_path, monkeypatch):
    """Redirect both layers into a temp dir."""
    monkeypatch.setattr(sw, "DIRECTION_FILE", tmp_path / "DIRECTION.md")
    monkeypatch.setattr(sw, "DIRECTION_DIR", tmp_path / "direction")
    monkeypatch.setenv("RUFUS_CHANNEL", "main_en")
    (tmp_path / "direction").mkdir()
    return tmp_path


# ── fail-open ────────────────────────────────────────────────────────────────

def test_no_files_changes_nothing(direction):
    text, note = sw.load_direction()
    assert text == ""
    assert "prompts unchanged" in note


def test_an_empty_file_is_the_same_as_no_file(direction):
    (direction / "DIRECTION.md").write_text("   \n\n", encoding="utf-8")
    assert sw.load_direction()[0] == ""


def test_an_unreadable_file_does_not_raise(direction, monkeypatch):
    (direction / "DIRECTION.md").write_text("Open on a moment.", encoding="utf-8")

    def boom(*a, **k):
        raise OSError("permission denied")

    monkeypatch.setattr(Path, "read_text", boom)
    assert sw.load_direction()[0] == ""


# ── layering ─────────────────────────────────────────────────────────────────

def test_the_channel_file_is_appended_not_substituted(direction):
    """channel_config layers niche_overrides on top of the base rather than
    replacing it; direction follows the same shape."""
    (direction / "DIRECTION.md").write_text("SHARED RULE", encoding="utf-8")
    (direction / "direction" / "main_en.md").write_text("CHANNEL RULE",
                                                        encoding="utf-8")
    text, note = sw.load_direction()
    assert "SHARED RULE" in text and "CHANNEL RULE" in text
    assert text.index("SHARED RULE") < text.index("CHANNEL RULE")
    assert "DIRECTION.md + config/direction/main_en.md" in note


def test_a_channel_without_its_own_file_gets_the_shared_one(direction):
    (direction / "DIRECTION.md").write_text("SHARED RULE", encoding="utf-8")
    text, note = sw.load_direction()
    assert "SHARED RULE" in text
    assert "no config/direction/main_en.md" in note


def test_the_channel_comes_from_the_env(direction, monkeypatch):
    monkeypatch.setenv("RUFUS_CHANNEL", "main_he")
    (direction / "DIRECTION.md").write_text("SHARED", encoding="utf-8")
    (direction / "direction" / "main_he.md").write_text("HEBREW", encoding="utf-8")
    text, _ = sw.load_direction()
    assert "HEBREW" in text


def test_a_channel_file_alone_still_loads(direction):
    (direction / "direction" / "main_en.md").write_text("ONLY CHANNEL",
                                                        encoding="utf-8")
    assert "ONLY CHANNEL" in sw.load_direction()[0]


# ── the owner's notes never reach the model ──────────────────────────────────

def test_only_the_direction_section_is_sent(direction):
    """Notes above the marker teach the OWNER how the file works. Sending them
    would spend prompt budget on a file the model cannot edit."""
    (direction / "DIRECTION.md").write_text(
        "# DIRECTION.md\n\nKeep it under 400 words. Numbers here lose.\n\n"
        "## The direction\n\nOpen on a moment.\n", encoding="utf-8")
    text, _ = sw.load_direction()
    assert "Open on a moment." in text
    assert "Numbers here lose" not in text


def test_a_file_without_the_marker_is_sent_whole(direction):
    """Someone who writes three plain lines must not get silence."""
    (direction / "DIRECTION.md").write_text("Open on a moment.", encoding="utf-8")
    assert sw.load_direction()[0] == "Open on a moment."


# ── the size cap ─────────────────────────────────────────────────────────────

def test_an_oversized_file_is_truncated_and_says_so(direction):
    (direction / "DIRECTION.md").write_text("word " * 900, encoding="utf-8")
    text, note = sw.load_direction()
    assert len(text.split()) == sw.DIRECTION_MAX_WORDS
    assert "TRUNCATED" in note
    assert "500 dropped" in note


def test_the_cap_measures_both_layers_together(direction):
    """Two files is exactly how a prompt grows without anyone noticing."""
    (direction / "DIRECTION.md").write_text("word " * 300, encoding="utf-8")
    (direction / "direction" / "main_en.md").write_text("word " * 300,
                                                        encoding="utf-8")
    _, note = sw.load_direction()
    assert "TRUNCATED" in note


# ── prose cannot beat a numeric gate ─────────────────────────────────────────

def test_a_length_instruction_is_flagged(direction):
    """script_standards.json is enforced in code. 'Keep it to 60 words' does
    not shorten anything — it produces scripts rejected for being under
    min_words."""
    (direction / "DIRECTION.md").write_text("Keep it to 60 words.",
                                            encoding="utf-8")
    _, note = sw.load_direction()
    assert "script_standards.json" in note
    assert "WINS" in note


@pytest.mark.parametrize("line", [
    "Aim for 40 seconds.", "No sentence over 20 words.", "Use 3-5 sentences.",
])
def test_length_conflicts_are_caught_in_several_shapes(direction, line):
    (direction / "DIRECTION.md").write_text(line, encoding="utf-8")
    assert "script_standards.json" in sw.load_direction()[1]


def test_direction_without_numbers_is_not_flagged(direction):
    (direction / "DIRECTION.md").write_text(
        "Open on a moment, not a summary. Feeling comes from the event.",
        encoding="utf-8")
    assert "script_standards.json" not in sw.load_direction()[1]


# ── wiring ───────────────────────────────────────────────────────────────────

def test_the_block_is_announced_every_run(direction, capsys):
    (direction / "DIRECTION.md").write_text("Open on a moment.", encoding="utf-8")
    sw._direction_block()
    assert "[gpt] direction:" in capsys.readouterr().out


def test_an_absent_file_is_announced_too(direction, capsys):
    """Silence is the failure mode: an instruction file nobody can confirm
    loaded is the fail-silent shape this repo keeps hitting."""
    sw._direction_block()
    assert "[gpt] direction:" in capsys.readouterr().out


def test_the_block_is_empty_without_files(direction):
    assert sw._direction_block() == ""


def test_the_system_prompt_places_direction_after_the_niche():
    """The niche says WHAT the channel is about; direction says HOW the owner
    wants it made, and only makes sense on top of that."""
    src = Path(sw.__file__).read_text(encoding="utf-8")
    assert "NICHE:\n{niche_context}\n\n{direction_blk}" in src


def test_the_storyboard_reads_the_same_files():
    """Half of what the owner writes is about pictures — scripts-only would
    silently drop that half."""
    import storyboard

    src = Path(storyboard.__file__).read_text(encoding="utf-8")
    assert "_direction_clause()" in src
    assert "script_writer.load_direction()" in src


def test_the_storyboard_survives_a_broken_loader(monkeypatch):
    import storyboard

    monkeypatch.setattr(sw, "load_direction",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    assert storyboard._direction_clause() == ""


# ── the shipped template ─────────────────────────────────────────────────────

def test_the_shipped_file_fits_under_its_own_cap():
    """It is the example everything else is written against."""
    text, note = sw.load_direction()
    assert "TRUNCATED" not in note
    assert len(text.split()) <= sw.DIRECTION_MAX_WORDS


def test_the_shipped_file_does_not_trip_its_own_conflict_check():
    assert "script_standards.json" not in sw.load_direction()[1]


def test_the_shipped_file_keeps_its_notes_out_of_the_prompt():
    raw = (ROOT / "DIRECTION.md").read_text(encoding="utf-8")
    assert "Examples beat rules" in raw          # the owner is told
    assert "Examples beat rules" not in sw.load_direction()[0]   # the model is not
