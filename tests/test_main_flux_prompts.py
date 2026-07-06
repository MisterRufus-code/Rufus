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
