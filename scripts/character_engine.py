#!/usr/bin/env python3
"""
character_engine.py — recurring-character support for Rufus' image generation.

Grew out of the channel-owner's explicit direction: "I want the fixed/
consistent-character option, building scenes like the [animated-character]
videos I've seen on YouTube — we prepare the scene and generate each scene in
detail, creating continuity for the video AND for the topics" (i.e. one
recurring character carried across beats within a video, and across
different videos/topics over time — not per-video regeneration).

Two independent layers, deliberately decoupled so this is useful with ZERO
ComfyUI changes and gets strictly better once the one-time image-conditioning
setup is done:

1. TEXT-LEVEL consistency (active immediately, no ComfyUI changes needed):
   every beat's image prompt is told to describe the SAME character — same
   face, hair, build, wardrobe, spelled out identically — varying only pose/
   action/framing per beat. FLUX reads full sentences, so identical wording
   pulls the model toward a visually similar (not pixel-identical) result
   beat to beat. See character_clause(), consumed by main._build_sd_prompts.

2. IMAGE-LEVEL consistency (opt-in, needs a one-time ComfyUI export): a
   persistent reference portrait is fed into an image-conditioning node chain
   (IPAdapter Plus / IPAdapter FaceID / PuLID / InstantID — whichever your
   ComfyUI has installed) the SAME way every other Rufus engine works: a
   proven, user-exported API template. See comfy_template.py's module
   docstring for why hand-wiring one from documentation is the wrong move —
   the Wan 2.2 integration proved that gets subtly wrong wiring/settings that
   only an export of a verified real run catches.

   One-time ComfyUI-side setup:
     a. Build a character-consistency workflow: your normal stills checkpoint
        + an IPAdapter/FaceID/PuLID node (ComfyUI Manager → search
        "IPAdapter" or "PuLID") feeding a LoadImage of a reference portrait
        into the sampler alongside the positive prompt.
     b. Set the positive-prompt text field to the literal placeholder
        RUFUS_PROMPT (same convention as config/stills_api.json).
     c. Generate a few times across different prompts/seeds and confirm the
        face/outfit actually holds — that's the whole point of the export-a-
        proven-run rule.
     d. Export (API) → save as config/character_stills_api.json.

   Until that file exists, comfy_client's character path always returns None
   and Rufus silently uses the ordinary (non-character) stills pipeline —
   this module must never block or degrade a run that hasn't set it up.

The reference portrait is auto-bootstrapped the first time character mode is
active for a niche: rendered once via the PLAIN stills template from a
character-sheet prompt built out of the niche's character description (see
character_sheet_prompt), saved to disk, and reused for every beat and every
future video for that niche — that persistence, not per-video regeneration,
is what gives the cross-topic continuity the owner asked for.
"""

import json
import os
from pathlib import Path

NICHES_FILE = Path(__file__).parent.parent / "config" / "niches.json"
REPO_ROOT = Path(__file__).parent.parent


def _off(var: str) -> bool:
    return os.environ.get(var, "1").strip().lower() in ("0", "false", "no", "off")


def niche_character(niche: str | None) -> dict | None:
    """The niche's "character" config block from niches.json (name,
    description, optional reference_image path), or None if the niche has
    none configured / niches.json can't be read. Fail-open — a broken or
    absent config must fall back to the ordinary pipeline, never crash it."""
    if not niche:
        return None
    try:
        data = json.loads(NICHES_FILE.read_text(encoding="utf-8"))
        cfg = data["niches"].get(niche, {}).get("character")
    except Exception:
        return None
    if not isinstance(cfg, dict) or not cfg.get("description"):
        return None
    return cfg


def enabled(niche: str | None) -> bool:
    """Whether recurring-character mode should be attempted for this niche.
    RUFUS_CHARACTER_MODE=0 is a global kill switch that wins even if the
    niche has a character block configured — same on/off convention as every
    other Rufus feature toggle (RUFUS_STILLS_ONLY, RUFUS_SFX, ...). The
    niche's own "enabled" field (default True once a character block exists)
    is the per-niche switch — money_history's "the Chronicler" is a
    deliberately timeless/archetypal design (a Calliope-Labs-style hooded
    storyteller, not pinned to any era's fashion) specifically so it can be
    on by default without fighting the PERIOD ACCURACY rule the rest of
    every scene still has to follow."""
    if _off("RUFUS_CHARACTER_MODE"):
        return False
    cfg = niche_character(niche)
    if cfg is None:
        return False
    return cfg.get("enabled", True) is not False


def short_ref(niche: str | None) -> str:
    """The COMPACT per-beat form of the character — a short repeatable visual
    token, not the full character sheet.

    This split exists because the full `description` and the per-beat prompt
    budget are arithmetically incompatible. The prompt builder tells GPT to
    write "2 to 4 vivid natural-language sentences per prompt" and allots
    roughly 180 tokens each; money_history's `description` alone is ~100 words
    over 3 sentences. Asking for it verbatim in EVERY prompt consumed the whole
    budget, left no room for the beat's literal subject (which the surrounding
    rules demand in much stronger language), and the model resolved the
    impossible constraint by dropping the character entirely — observed live as
    10/10 scene prompts with no character in them at all.

    So: `short_description` (~15 words) goes in every beat prompt, and the full
    `description` is reserved for character_sheet_prompt(), where a one-time
    reference portrait genuinely wants every detail. Falls back to the first
    sentence of `description` when a niche hasn't defined a short form, so an
    existing character block keeps working."""
    cfg = niche_character(niche)
    if not cfg:
        return ""
    short = (cfg.get("short_description") or "").strip()
    if short:
        return short
    # Fall back to the leading clause of the long description — better than
    # shipping 100 words into a 2-sentence budget.
    return (cfg["description"].split(".")[0].split(":")[-1]).strip()


CHARACTER_BEAT_MODES = ("all", "anchor")


def beat_mode(niche: str | None = None) -> str:
    """Which beats carry the recurring character: "all" or "anchor".

    "all"    — every beat that shows a person (the original behaviour).
    "anchor" — only the opening, one middle, and the closing beat; every other
               beat is character-free. Fewer, better appearances.

    RUFUS_CHARACTER_BEATS wins, then the niche's own `character.beats`, then
    "all" so an existing config keeps behaving exactly as it did."""
    env = os.environ.get("RUFUS_CHARACTER_BEATS", "").strip().lower()
    if env in CHARACTER_BEAT_MODES:
        return env
    cfg = niche_character(niche) or {}
    cfgd = str(cfg.get("beats", "")).strip().lower()
    return cfgd if cfgd in CHARACTER_BEAT_MODES else "all"


def anchor_beats(n_beats: int) -> list[int]:
    """1-based beat numbers the character appears in under "anchor" mode:
    the first, the middle, and the last. Deduplicated and ordered, so a
    2-beat script yields [1, 2] rather than a repeat."""
    if n_beats <= 0:
        return []
    return sorted({1, (n_beats + 1) // 2, n_beats})


def character_clause(niche: str | None, n_beats: int | None = None) -> str:
    """Instruction line for _build_sd_prompts' prompt, telling GPT which beats
    depict the SAME recurring character. Returns "" when character mode isn't
    on for this niche, so callers can always safely append the result without
    an extra branch.

    Deliberately built from short_ref(), not `description` — see short_ref's
    docstring for why the long form makes this rule unsatisfiable.

    The ALL-OR-NOTHING sentence is not decoration. Live, a batch came back with
    a clean-shaven man in a MODERN BUSINESS SUIT holding the Chronicler's
    bronze lantern, and another in a lab coat holding it — the model had kept
    the character's most concrete prop and dropped everything else that makes
    him him. A stray lantern in a boardroom is worse than no character at all:
    it reads as a continuity error rather than a motif."""
    cfg = niche_character(niche) if enabled(niche) else None
    if not cfg:
        return ""
    name = cfg.get("name") or "the recurring character"
    mode = beat_mode(niche)
    if mode == "anchor" and n_beats:
        which = ", ".join(str(b) for b in anchor_beats(n_beats))
        scope = (f"ONLY beats {which} show {name}. Every other beat has NO "
                 f"{name} in it at all — not the figure, not the cloak, not "
                 f"the lantern, nothing of his. ")
    else:
        scope = f"Every beat showing a person MUST show {name}. "
    return (
        f"- RECURRING CHARACTER (non-negotiable): {scope}"
        f"{name} is the SAME figure every time: {short_ref(niche)}. Repeat "
        f"that description near-verbatim wherever he appears; vary ONLY pose, "
        f"action, and framing. He is a timeless narrator-guide, not an "
        f"inhabitant of the era, so PERIOD ACCURACY governs the rest of the "
        f"frame, never him.\n"
        f"- ALL OR NOTHING: he appears as the COMPLETE figure above, or not at "
        f"all. Never hand his cloak, hood, or lantern to anyone else. A modern "
        f"businessman holding his lantern is a continuity ERROR.\n"
    )


def reference_image_path(niche: str | None) -> Path | None:
    """Where the persistent reference portrait for this niche's character
    lives, resolved relative to the repo root. Once rendered this file is
    reused across every beat and every future video for the niche — that
    persistence (not per-video regeneration) is what gives cross-topic
    continuity, not just within-video consistency."""
    cfg = niche_character(niche)
    if not cfg:
        return None
    rel = cfg.get("reference_image") or f"config/character_reference_{niche}.png"
    return REPO_ROOT / rel


def character_sheet_prompt(niche: str | None) -> str | None:
    """The prompt used to bootstrap the reference portrait: a plain, neutral
    character sheet in the niche's own visual style, generated once and then
    reused as the image-conditioning reference for every beat afterward."""
    cfg = niche_character(niche)
    if not cfg:
        return None
    try:
        data = json.loads(NICHES_FILE.read_text(encoding="utf-8"))
        style = data["niches"].get(niche, {}).get("style_suffix", "")
    except Exception:
        style = ""
    name = cfg.get("name") or "the character"
    desc = cfg["description"]
    prompt = (
        f"Character reference sheet of {name}: {desc}. Neutral three-quarter "
        f"standing pose, plain uncluttered background, even flat lighting, "
        f"full figure clearly visible head to waist."
    )
    return f"{prompt} {style}".strip() if style else prompt
