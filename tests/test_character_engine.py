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
    assert "SAME figure" in clause
    assert "vary ONLY pose, action, and framing" in clause


# ── short_ref: the compact per-beat form ─────────────────────────────────────
# The full description and the per-beat prompt budget ("2 to 4 sentences",
# ~180 tokens each) are arithmetically incompatible. Live, demanding a
# ~100-word description in every prompt made the rule unsatisfiable and the
# model dropped the character from all 10 prompts. short_ref is the fix.

def test_short_ref_prefers_short_description(monkeypatch, tmp_path):
    cfg = dict(_CHAR, short_description="a hooded figure with a bronze lantern")
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=cfg))
    assert ce.short_ref("money_history") == "a hooded figure with a bronze lantern"


def test_short_ref_falls_back_to_first_clause_of_description(monkeypatch, tmp_path):
    """A niche that never defined a short form must still work, and must still
    produce something far shorter than the full description."""
    long_desc = ("a timeless guide, NOT tied to any era: a deep hooded cloak in "
                 "weathered sepia. Reads as a narrator outside time. Never "
                 "redesign the cloak to match a scene's period.")
    cfg = {"enabled": True, "name": "the Chronicler", "description": long_desc}
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=cfg))
    got = ce.short_ref("money_history")
    assert got == "a deep hooded cloak in weathered sepia"
    assert len(got) < len(long_desc) / 2


def test_short_ref_empty_without_character(monkeypatch, tmp_path):
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=None))
    assert ce.short_ref("money_history") == ""


def test_character_clause_uses_short_form_not_full_description(monkeypatch, tmp_path):
    """The whole point of the split — the clause must carry the compact token,
    never the full character sheet."""
    cfg = dict(_CHAR, short_description="a hooded figure with a bronze lantern")
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=cfg))
    monkeypatch.delenv("RUFUS_CHARACTER_MODE", raising=False)
    clause = ce.character_clause("money_history")
    assert "a hooded figure with a bronze lantern" in clause
    assert "grey hair, round spectacles, brown leather satchel" not in clause


def test_character_sheet_prompt_still_uses_the_full_description(monkeypatch, tmp_path):
    """The one-time reference portrait is exactly where the full detail belongs
    — short_ref must NOT have leaked into it."""
    cfg = dict(_CHAR, short_description="a hooded figure with a bronze lantern")
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=cfg))
    prompt = ce.character_sheet_prompt("money_history")
    assert "grey hair, round spectacles, brown leather satchel" in prompt


def test_character_clause_stays_compact_enough_for_the_prompt_budget(monkeypatch, tmp_path):
    """Regression guard for the arithmetic conflict that killed the feature:
    the clause competes with a '2 to 4 sentences per prompt' budget, so it must
    stay far below the ~1,300 chars it was when the model gave up on it.

    Ceiling raised 700 -> 850 when the all-or-nothing rule was added, after a
    live beat put the lantern alone on a table with no Chronicler in the frame.
    What the budget actually constrains is the description the model must
    REPEAT in every prompt — short_ref, pinned separately below — not the rule
    text around it, which the model reads once. The 1,300-char figure that
    broke the feature was 100 words of DESCRIPTION, and short_ref is still
    ~120 chars."""
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=_CHAR))
    monkeypatch.delenv("RUFUS_CHARACTER_MODE", raising=False)
    assert len(ce.character_clause("money_history")) < 850


def test_the_repeated_part_stays_tiny(monkeypatch, tmp_path):
    """The real budget constraint: what goes into EVERY prompt is short_ref,
    and that is what must stay small however the surrounding rules grow."""
    monkeypatch.setattr(ce, "NICHES_FILE", _write_niches(tmp_path, character=_CHAR))
    assert len(ce.short_ref("money_history")) < 200


def test_money_history_ships_no_character_at_all():
    """REMOVED BY THE OWNER. money_history ran with "the Chronicler" — a
    hooded narrator pinned into the first, middle and last shot of every
    sequence — and the decision was to take it out completely: three of ten
    frames went to a mascot instead of the story, and every one of them also
    dragged in the restatement clause that keeps his cloak consistent.

    The MECHANISM stays (character_engine.py is generic and the five SD niches
    still ship starter mascots, all disabled). What must not come back is a
    character on this niche — with no `character` key, every call site falls
    through and nothing about the character path executes."""
    import json as _json
    from pathlib import Path as _Path
    real = _json.loads((_Path(__file__).parent.parent / "config" / "niches.json")
                       .read_text(encoding="utf-8"))
    assert "character" not in real["niches"]["money_history"]
    assert "chronicler" not in _json.dumps(real).lower()


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
