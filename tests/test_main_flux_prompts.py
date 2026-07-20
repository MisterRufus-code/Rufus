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


def test_flux_instruction_pushes_photorealism_over_illustration(tmp_path, monkeypatch):
    """FLUX can drift toward a painterly/CGI/illustration look — the instruction
    must explicitly demand photographic technique (real camera/lens, film
    grain, natural skin texture) and ban the synthetic-looking alternative."""
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
    assert "PHOTOREALISM" in prompt
    assert "not a painting" in prompt.lower() or "never a painting" in prompt.lower()
    assert "film grain" in prompt.lower()
    assert "lens" in prompt.lower()


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
