"""Tests for character_engine.py — recurring-character config/text-clause
support (the "fixed character across scenes and topics" feature)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import character_engine as ce


def _write_niches(tmp_path, character=None, style_suffix="sepia, gold tones"):
    cfg = {"niches": {"money_history": {"style_suffix": style_suffix}}}
    if character is not None:
        cfg["niches"]["money_history"]["character"] = character
    p = tmp_path / "niches.json"
    p.write_text(json.dumps(cfg))
    return p


_CHAR = {
    "enabled": True,
    "name": "the Chronicler",
    "description": "grey hair, round spectacles, brown leather satchel",
}


# ── niche_character / enabled ────────────────────────────────────────────────

def test_niche_character_none_when_no_niche(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=_CHAR))
    assert ce.niche_character(None) is None
    assert ce.niche_character("") is None


def test_niche_character_none_when_not_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=None))
    assert ce.niche_character("money_history") is None


def test_niche_character_none_when_niches_file_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "NICHES_FILE", tmp_path / "missing.json")
    assert ce.niche_character("money_history") is None


def test_niche_character_none_when_description_missing(monkeypatch, tmp_path):
    bad = {"enabled": True, "name": "X"}
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=bad))
    assert ce.niche_character("money_history") is None


def test_niche_character_returns_configured_block(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=_CHAR))
    got = ce.niche_character("money_history")
    assert got["name"] == "the Chronicler"


def test_enabled_true_when_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=_CHAR))
    monkeypatch.delenv("RUFUS_CHARACTER_MODE", raising=False)
    assert ce.enabled("money_history") is True


def test_enabled_false_without_character_config(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=None))
    assert ce.enabled("money_history") is False


def test_enabled_false_when_global_kill_switch_set(monkeypatch, tmp_path):
    # RUFUS_CHARACTER_MODE=0 must win even with a valid character configured —
    # same convention as every other Rufus feature toggle.
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=_CHAR))
    monkeypatch.setenv("RUFUS_CHARACTER_MODE", "0")
    assert ce.enabled("money_history") is False


def test_enabled_false_when_niche_disables_its_own_character(monkeypatch, tmp_path):
    # The per-niche "enabled" field is the switch money_history ships OFF by
    # default in config/niches.json (needs the owner's real description /
    # reference art first) — niche_character() still returns the block (so a
    # UI could show/edit it), but enabled() must say no.
    disabled = dict(_CHAR, enabled=False)
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=disabled))
    monkeypatch.delenv("RUFUS_CHARACTER_MODE", raising=False)
    assert ce.niche_character("money_history") is not None
    assert ce.enabled("money_history") is False
    assert ce.character_clause("money_history") == ""


# ── character_clause ─────────────────────────────────────────────────────────

def test_character_clause_empty_when_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=None))
    assert ce.character_clause("money_history") == ""


def test_character_clause_empty_when_kill_switch_set(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=_CHAR))
    monkeypatch.setenv("RUFUS_CHARACTER_MODE", "0")
    assert ce.character_clause("money_history") == ""


def test_character_clause_names_the_character_and_locks_wardrobe(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=_CHAR))
    monkeypatch.delenv("RUFUS_CHARACTER_MODE", raising=False)
    clause = ce.character_clause("money_history")
    assert "the Chronicler" in clause
    assert "grey hair, round spectacles, brown leather satchel" in clause
    assert "SAME person" in clause
    assert "never their face, hair, build, or wardrobe" in clause


def test_character_clause_falls_back_to_generic_name(monkeypatch, tmp_path):
    no_name = {"enabled": True, "description": "a masked figure"}
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=no_name))
    monkeypatch.delenv("RUFUS_CHARACTER_MODE", raising=False)
    clause = ce.character_clause("money_history")
    assert "the recurring character" in clause
    assert "a masked figure" in clause


# ── reference_image_path ─────────────────────────────────────────────────────

def test_reference_image_path_none_when_not_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=None))
    assert ce.reference_image_path("money_history") is None


def test_reference_image_path_uses_configured_relative_path(monkeypatch, tmp_path):
    cfg = dict(_CHAR, reference_image="config/character_reference_money_history.png")
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=cfg))
    p = ce.reference_image_path("money_history")
    assert p == ce.REPO_ROOT / "config" / "character_reference_money_history.png"


def test_reference_image_path_default_when_not_specified(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=_CHAR))
    p = ce.reference_image_path("money_history")
    assert p.name == "character_reference_money_history.png"


# ── character_sheet_prompt ───────────────────────────────────────────────────

def test_character_sheet_prompt_none_when_not_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=None))
    assert ce.character_sheet_prompt("money_history") is None


def test_character_sheet_prompt_includes_description_and_style(monkeypatch, tmp_path):
    monkeypatch.setattr(
        ce, "NICHES_FILE",
        _write_niches(tmp_path, character=_CHAR, style_suffix="sepia, gold tones"))
    prompt = ce.character_sheet_prompt("money_history")
    assert "grey hair, round spectacles, brown leather satchel" in prompt
    assert "reference sheet" in prompt.lower()
    assert "sepia, gold tones" in prompt
