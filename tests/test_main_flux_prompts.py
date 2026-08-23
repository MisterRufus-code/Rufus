"""Tests for _build_sd_prompts' FLUX instruction — verifies the beat-to-image
anchoring rule that addresses "the scene doesn't match what's being said"."""

import json
import pytest
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
    assert "SAME figure" in prompt

    # POSITION, not just presence. The clause was previously rule 8 of 15,
    # ~41% into an 8KB instruction, with 3.5KB of further rules after it and
    # no restatement — and a live run produced 10/10 prompts with no character
    # in them. It must now sit ahead of the diluting rule list, and be restated
    # after the freshness block (which otherwise reads as an order to stop
    # repeating the one element that must repeat).
    assert prompt.index("RECURRING CHARACTER") < prompt.index("RULES:"), \
        "character clause must precede the rule list, not be buried inside it"
    assert prompt.count("RECURRING CHARACTER") == 1, "clause duplicated"
    reminder = prompt.index("REMINDER — the recurring character")
    assert reminder > prompt.index("RECURRING CHARACTER")
    assert reminder > len(prompt) * 0.8, "closing reminder must sit at the tail"


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
    # The closing reminder is character-only — it must not appear either, or a
    # no-character niche gets a dangling reference to nothing.
    assert "REMINDER — the recurring character" not in captured["prompt"]


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


# ── Blank-surfaces net (_defuse_readable_text) ────────────────────────────────
# Seen live: "calendar page turning to December 31, 2022", "newspaper
# headlines about the crisis", "'Follow' button with Bitcoin graphics" —
# diffusion models render these as garbled AI gibberish, the single most
# recognizable giveaway in the finished videos.

def test_defuse_appends_clause_for_text_props():
    import main
    p = main._defuse_readable_text(
        "An extreme close-up of a newspaper headline about the crisis")
    assert main._DETEXT_SENTINEL in p.lower()


def test_defuse_covers_screens_and_buttons():
    import main
    for risky in ("a smartphone screen showing a Follow button",
                  "a calendar page turning to December 31",
                  "a weathered ledger open on a desk",
                  "a stock ticker board with numbers"):
        out = main._defuse_readable_text(risky)
        assert main._DETEXT_SENTINEL in out.lower(), risky


def test_defuse_clause_never_names_text_in_the_positive_prompt():
    """The whole point of the rewrite. CLIP has no "not" operator: the old
    clause said "absolutely no readable text, numbers, or interface elements"
    and the sampler painted exactly those. Suppression words belong in the
    negative conditioning (comfy_client.DEFAULT_STILLS_NEGATIVE) — this clause
    may only describe the surface affirmatively."""
    import main
    clause = main._DETEXT_CLAUSE.lower()
    for banned in ("readable", "text", "lettering", "writing", "numeral",
                   " no ", "words"):
        assert banned not in clause, f"positive clause must not contain {banned!r}"


def test_defuse_leaves_clean_prompts_untouched():
    import main
    clean = "A wide establishing shot of a misty mountain valley at dawn"
    assert main._defuse_readable_text(clean) == clean


def test_defuse_is_idempotent():
    import main
    once  = main._defuse_readable_text("a newspaper on a desk")
    twice = main._defuse_readable_text(once)
    assert once == twice


# ── Merged-reply recovery (_split_merged_prompts) ────────────────────────────
# Observed live: the model ignored "one prompt per line" and returned all ten
# as one paragraph. The newline split then yielded ONE enormous prompt, main.py
# called generate_clips with n=1, and a 40-second video was rendered from a
# SINGLE image — with nothing but a log warning to show for it.

def _merged_blob(count):
    return " ".join(
        f"A medium portrait of historical scene number {i}, with people mid-action "
        f"and period-accurate detail throughout the frame." for i in range(count))


def test_split_merged_recovers_each_prompt():
    got = main._split_merged_prompts(_merged_blob(10), 10)
    assert len(got) == 10
    assert all(len(g) > 20 for g in got)
    assert "number 0" in got[0] and "number 9" in got[-1]


def test_split_merged_preserves_beat_order():
    got = main._split_merged_prompts(_merged_blob(6), 6)
    assert [g.split("number ")[1][0] for g in got] == list("012345")


def test_split_merged_leaves_a_genuinely_single_prompt_alone():
    """A real single prompt is 2-4 sentences; against n=10 it must NOT be
    chopped into ten fragments — that would be worse than the bug."""
    single = ("A wide establishing shot of a Roman forum. Traders argue over a "
              "scale. Coins spill across the stone. Dust hangs in the light.")
    assert main._split_merged_prompts(single, 10) == []


def test_split_merged_noop_for_single_beat():
    assert main._split_merged_prompts(_merged_blob(10), 1) == []


def test_split_merged_rejects_chunks_that_would_be_too_short():
    assert main._split_merged_prompts("a. b. c. d. e. f. g. h. i. j.", 10) == []


# ── Scenes made of lettering, not just objects that bear it ─────────────────
# Live: shot 7 of the Great Depression run was "A group of well-dressed
# individuals ignoring a nearby protest of unemployed workers". It named no
# text-bearing OBJECT, so the blank-surfaces clause never fired — and the
# rendered image came back with protest signs reading "ISSUES" in garbled type,
# the single clearest tell that a machine made the picture.

import importlib, sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).parent.parent / "scripts"))
_main = importlib.import_module("main")


def test_a_protest_triggers_the_blank_surfaces_clause():
    shot = ("A group of well-dressed individuals ignoring a nearby protest of "
            "unemployed workers.")
    assert _main._DETEXT_SENTINEL in _main._defuse_readable_text(shot)


def test_other_lettering_scenes_trigger_it_too():
    for scene in ("A busy storefront on a wet morning.",
                  "The trading floor after closing.",
                  "A classroom of empty chairs.",
                  "A quiet memorial at the edge of the square.",
                  "A rally filling the street.",
                  "A digital display above the desks."):
        assert _main._DETEXT_SENTINEL in _main._defuse_readable_text(scene), scene


def test_a_scene_with_no_lettering_is_still_left_alone():
    """The clause costs tokens — a clean prompt must stay clean."""
    plain = "A worn silver coin lies alone on a bare wooden counter."
    assert _main._defuse_readable_text(plain) == plain


def test_the_clause_is_never_applied_twice():
    once = _main._defuse_readable_text("A protest in the square.")
    assert _main._defuse_readable_text(once) == once


# ── the words a shot asked to have printed ───────────────────────────────────
#
# storyboard rule 6 already says "NEVER NAME WORDS THAT WOULD BE PRINTED IN
# FRAME ... Write the object as a blank physical thing instead." The model
# ignores it, and a live gallery came back with two frames of legible text:
#
#     "A protest sign reading 'Stop Police Brutality' held in a crowd"
#     "A newspaper being folded with a headline about police reform visible"
#
# Both prompts ALSO carried _DETEXT_CLAUSE, which says every sign is blank —
# appended after the quoted string, which is the shape that loses every time
# here. The clause describes the surface we want and stays; the words no
# longer reach the encoder to be painted.

@pytest.mark.parametrize("shot,expect_gone,expect_kept", [
    ("A protest sign reading 'Stop Police Brutality' held in a crowd, faces unseen.",
     "Stop Police Brutality", "protest sign"),
    ("A newspaper being folded with a headline about police reform visible.",
     "police reform", "newspaper"),
    ('A book titled "How Money Really Began" on a table.',
     "How Money Really Began", "book"),
    ("A banner that says 'Part of a series' above the door.",
     "Part of a series", "banner"),
])
def test_the_named_words_are_removed_and_the_object_stays(shot, expect_gone, expect_kept):
    out = main._strip_named_words(shot)
    assert expect_gone not in out
    assert expect_kept in out


@pytest.mark.parametrize("shot", [
    "A clerk pushes an open ledger across a counter, its pages turned toward us.",
    "Five people crowd against the shutters of a closed bank, one hammering on the wood.",
    "The table goes over and coins scatter across the floorboards.",
])
def test_a_shot_that_names_no_words_is_untouched(shot):
    assert main._strip_named_words(shot) == shot


def test_the_blank_surface_clause_still_follows():
    """Deleting the words must not delete the description of what we DO want:
    a prompt with a sign in it still says the sign is blank."""
    out = main._defuse_readable_text(
        "A protest sign reading 'Stop Police Brutality' held in a crowd.")
    assert "Stop Police Brutality" not in out
    assert main._DETEXT_SENTINEL in out.lower()
