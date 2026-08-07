"""Tests for _build_sd_prompts' FLUX instruction — verifies the beat-to-image
anchoring rule that addresses "the scene doesn't match what's being said"."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import main


def _fake_openai_response(lines: list[str]):
    resp = MagicMock()
    resp.choices = [MagicMock()]
    resp.choices[0].message.content = "\n".join(lines)
    return resp


def test_flux_instruction_requires_locking_to_beats_specific_anchor(tmp_path, monkeypatch):
    """The prompt sent to GPT must instruct it to anchor on the single most
    specific noun/number in each beat, not drift to a generic related scene —
    this is the fix for 'images don't match what's being said'."""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"openai": "sk-test-key-1234567890"}))
    monkeypatch.setattr(main, "CONFIG_DIR", tmp_path)

    niches_file = tmp_path / "niches.json"
    niches_file.write_text(json.dumps({"niches": {"money_history": {"video_source": "comfy"}}}))
    monkeypatch.setattr(main, "NICHES_FILE", niches_file)
    monkeypatch.setenv("RUFUS_VIDEO_SOURCE", "comfy")

    captured = {}

    class FakeClient:
        def __init__(self, api_key=None):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(model, messages, **kw):
                    captured["prompt"] = messages[0]["content"]
                    return _fake_openai_response(
            [f"A photoreal scene depicting beat number {i} in vivid detail." for i in range(2)])

    monkeypatch.setattr("openai.OpenAI", FakeClient)

    main._build_sd_prompts("Rome debased the denarius. Weimar printed too much money.",
                          "money_history", max_scenes=2)

    assert "LOCK TO THE BEAT'S ANCHOR" in captured["prompt"]
    assert "Never drift to a generic or merely thematically-related scene" in captured["prompt"]


def test_flux_instruction_steers_away_from_hard_to_render_faces(tmp_path, monkeypatch):
    """Diffusion models render tight front-facing emotional close-ups of faces
    worst (uncanny/distorted) — the instruction must tell GPT to avoid them and
    keep people mid-distance / three-quarter. Fix for 'the faces are scary'."""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"openai": "sk-test-key-1234567890"}))
    monkeypatch.setattr(main, "CONFIG_DIR", tmp_path)

    niches_file = tmp_path / "niches.json"
    niches_file.write_text(json.dumps({"niches": {"money_history": {"video_source": "comfy"}}}))
    monkeypatch.setattr(main, "NICHES_FILE", niches_file)
    monkeypatch.setenv("RUFUS_VIDEO_SOURCE", "comfy")

    captured = {}

    class FakeClient:
        def __init__(self, api_key=None):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(model, messages, **kw):
                    captured["prompt"] = messages[0]["content"]
                    return _fake_openai_response(
                        [f"A photoreal scene depicting beat number {i}." for i in range(2)])

    monkeypatch.setattr("openai.OpenAI", FakeClient)

    main._build_sd_prompts("Nixon spoke on TV. A worried family watched.",
                          "money_history", max_scenes=2)

    prompt = captured["prompt"]
    assert "PEOPLE & FACES" in prompt
    assert "extreme close-up" in prompt.lower()
    assert "three-quarter" in prompt.lower()


def test_flux_instruction_pushes_illustration_over_photorealism(tmp_path, monkeypatch):
    """Per channel-owner direction (money_history's visual identity moved to
    flat 2D illustration, matched by comfy_client.DEFAULT_DETAIL_SUFFIX and
    config/niches.json's style_suffix — changed together): the instruction
    must explicitly demand flat illustration technique and ban the
    photographic/photorealistic alternative it used to require."""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"openai": "sk-test-key-1234567890"}))
    monkeypatch.setattr(main, "CONFIG_DIR", tmp_path)

    niches_file = tmp_path / "niches.json"
    niches_file.write_text(json.dumps({"niches": {"money_history": {"video_source": "comfy"}}}))
    monkeypatch.setattr(main, "NICHES_FILE", niches_file)
    monkeypatch.setenv("RUFUS_VIDEO_SOURCE", "comfy")

    captured = {}

    class FakeClient:
        def __init__(self, api_key=None):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(model, messages, **kw):
                    captured["prompt"] = messages[0]["content"]
                    return _fake_openai_response(
                        [f"A photoreal scene depicting beat number {i}." for i in range(2)])

    monkeypatch.setattr("openai.OpenAI", FakeClient)

    main._build_sd_prompts("Rome debased the denarius.", "money_history", max_scenes=2)

    prompt = captured["prompt"]
    assert "FLAT 2D ILLUSTRATION" in prompt
    assert "not a photograph" in prompt.lower()
    assert "flat" in prompt.lower() and "color fills" in prompt.lower()
    # The old photorealistic requirement must be genuinely gone, not just
    # supplemented — a leftover "shot on a real camera, film grain" demand
    # would still push FLUX toward photorealism regardless of the new banner.
    assert "shot on a Leica" not in prompt
    assert "visible film grain" not in prompt.lower()


def test_flux_instruction_includes_character_clause_when_niche_has_one(tmp_path, monkeypatch):
    """When a niche's character block is enabled (character_engine.py), the
    FLUX instruction sent to GPT must carry the recurring-character clause so
    every beat's prompt describes the same person — the text-level layer of
    the "fixed character across scenes" feature."""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"openai": "sk-test-key-1234567890"}))
    monkeypatch.setattr(main, "CONFIG_DIR", tmp_path)

    niches_file = tmp_path / "niches.json"
    niches_file.write_text(json.dumps({"niches": {"money_history": {
        "video_source": "comfy",
        "character": {
            "enabled": True,
            "name": "the Chronicler",
            "description": "grey hair, round spectacles, brown leather satchel",
        },
    }}}))
    monkeypatch.setattr(main, "NICHES_FILE", niches_file)
    monkeypatch.setenv("RUFUS_VIDEO_SOURCE", "comfy")
    monkeypatch.delenv("RUFUS_CHARACTER_MODE", raising=False)

    import character_engine
    monkeypatch.setattr(character_engine, "NICHES_FILE", niches_file)

    captured = {}

    class FakeClient:
        def __init__(self, api_key=None):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(model, messages, **kw):
                    captured["prompt"] = messages[0]["content"]
                    return _fake_openai_response(
                        [f"A scene depicting beat number {i}." for i in range(2)])

    monkeypatch.setattr("openai.OpenAI", FakeClient)

    main._build_sd_prompts("Rome debased the denarius.", "money_history", max_scenes=2)

    prompt = captured["prompt"]
    assert "the Chronicler" in prompt
    assert "grey hair, round spectacles, brown leather satchel" in prompt
    assert "SAME person" in prompt


def test_flux_instruction_omits_character_clause_without_niche_config(tmp_path, monkeypatch):
    """A niche with no "character" block in niches.json (every niche today)
    must get an instruction identical in shape to before this feature —
    no stray clause, no crash."""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"openai": "sk-test-key-1234567890"}))
    monkeypatch.setattr(main, "CONFIG_DIR", tmp_path)

    niches_file = tmp_path / "niches.json"
    niches_file.write_text(json.dumps({"niches": {"money_history": {"video_source": "comfy"}}}))
    monkeypatch.setattr(main, "NICHES_FILE", niches_file)
    monkeypatch.setenv("RUFUS_VIDEO_SOURCE", "comfy")

    import character_engine
    monkeypatch.setattr(character_engine, "NICHES_FILE", niches_file)

    captured = {}

    class FakeClient:
        def __init__(self, api_key=None):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(model, messages, **kw):
                    captured["prompt"] = messages[0]["content"]
                    return _fake_openai_response(
                        [f"A scene depicting beat number {i}." for i in range(2)])

    monkeypatch.setattr("openai.OpenAI", FakeClient)

    main._build_sd_prompts("Rome debased the denarius.", "money_history", max_scenes=2)

    assert "RECURRING CHARACTER" not in captured["prompt"]


# ── SD (Realistic Vision) prompt branch also gets the character clause ───────
# character_engine.py is generic per-niche, not FLUX-only — finance,
# motivation, mindset, business, and personal_development each ship their
# own starter character (disabled by default) alongside money_history's
# Chronicler, so the SD-token prompt branch must carry the clause too.

def test_sd_instruction_includes_character_clause_when_niche_has_one(tmp_path, monkeypatch):
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"openai": "sk-test-key-1234567890"}))
    monkeypatch.setattr(main, "CONFIG_DIR", tmp_path)

    niches_file = tmp_path / "niches.json"
    niches_file.write_text(json.dumps({"niches": {"finance": {
        "video_source": "sd",
        "character": {
            "enabled": True,
            "name": "the Strategist",
            "description": "sharp asymmetric bob, charcoal blazer",
        },
    }}}))
    monkeypatch.setattr(main, "NICHES_FILE", niches_file)
    monkeypatch.delenv("RUFUS_VIDEO_SOURCE", raising=False)
    monkeypatch.delenv("RUFUS_CHARACTER_MODE", raising=False)

    import character_engine
    monkeypatch.setattr(character_engine, "NICHES_FILE", niches_file)

    captured = {}

    class FakeClient:
        def __init__(self, api_key=None):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(model, messages, **kw):
                    captured["prompt"] = messages[0]["content"]
                    return _fake_openai_response(
                        [f"RAW photo, (a scene:1.35), doing thing {i}, a "
                         f"setting, soft light, 50mm f/1.4, cinematic grade"
                         for i in range(2)])

    monkeypatch.setattr("openai.OpenAI", FakeClient)

    main._build_sd_prompts("People overspend on rent.", "finance", max_scenes=2)

    prompt = captured["prompt"]
    assert "the Strategist" in prompt
    assert "sharp asymmetric bob, charcoal blazer" in prompt


def test_sd_instruction_omits_character_clause_by_default(tmp_path, monkeypatch):
    """Production default: every SD niche ships its character disabled, so
    a real run's prompt must carry no trace of it until the owner opts in."""
    keys_file = tmp_path / "keys.json"
    keys_file.write_text(json.dumps({"openai": "sk-test-key-1234567890"}))
    monkeypatch.setattr(main, "CONFIG_DIR", tmp_path)

    niches_file = tmp_path / "niches.json"
    niches_file.write_text(json.dumps({"niches": {"finance": {
        "video_source": "sd",
        "character": {
            "enabled": False,
            "name": "the Strategist",
            "description": "sharp asymmetric bob, charcoal blazer",
        },
    }}}))
    monkeypatch.setattr(main, "NICHES_FILE", niches_file)
    monkeypatch.delenv("RUFUS_VIDEO_SOURCE", raising=False)

    import character_engine
    monkeypatch.setattr(character_engine, "NICHES_FILE", niches_file)

    captured = {}

    class FakeClient:
        def __init__(self, api_key=None):
            pass

        class chat:
            class completions:
                @staticmethod
                def create(model, messages, **kw):
                    captured["prompt"] = messages[0]["content"]
                    return _fake_openai_response(
                        [f"RAW photo, (a scene:1.35), doing thing {i}, a "
                         f"setting, soft light, 50mm f/1.4, cinematic grade"
                         for i in range(2)])

    monkeypatch.setattr("openai.OpenAI", FakeClient)

    main._build_sd_prompts("People overspend on rent.", "finance", max_scenes=2)

    assert "the Strategist" not in captured["prompt"]
    assert "RECURRING CHARACTER" not in captured["prompt"]


# ── No-readable-text net (_defuse_readable_text) ──────────────────────────────
# Seen live: "calendar page turning to December 31, 2022", "newspaper
# headlines about the crisis", "'Follow' button with Bitcoin graphics" —
# diffusion models render these as garbled AI gibberish, the single most
# recognizable giveaway in the finished videos.

def test_defuse_appends_clause_for_text_props():
    import main
    p = main._defuse_readable_text(
        "An extreme close-up of a newspaper headline about the crisis")
    assert "no" in p.lower() and "readable text" in p.lower()


def test_defuse_covers_screens_and_buttons():
    import main
    for risky in ("a smartphone screen showing a Follow button",
                  "a calendar page turning to December 31",
                  "a weathered ledger open on a desk",
                  "a stock ticker board with numbers"):
        out = main._defuse_readable_text(risky)
        assert "readable text" in out.lower(), risky


def test_defuse_leaves_clean_prompts_untouched():
    import main
    clean = "A wide establishing shot of a misty mountain valley at dawn"
    assert main._defuse_readable_text(clean) == clean


def test_defuse_is_idempotent():
    import main
    once  = main._defuse_readable_text("a newspaper on a desk")
    twice = main._defuse_readable_text(once)
    assert once == twice
