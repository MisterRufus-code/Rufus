#!/usr/bin/env python3
"""
storyboard.py — plan the pictures WITH the script, as one continuous scene.

THE PROBLEM THIS SOLVES. Today the two halves are strangers. script_writer
finishes a script; main._split_beats chops it into sentences; then a separate
model reads those ten sentences cold and illustrates each one on its own. It
has never seen the story — only a list of lines — so it decorates each line
independently and the result is ten unrelated pictures that happen to share a
colour palette.

That is exactly what the owner reported, and the last run shows it plainly.
The script's beat 2 was about the denarius holding 4.5 grams of silver. The
image generated for it: "a family gathered around a modest dinner table,
sharing a simple meal of bread and vegetables". Not wrong, not connected. Beat
8 became "a concerned modern-day person at a kitchen table with financial
documents" — the stock-photo of an idea rather than a moment in this story.

A storyboard fixes it by construction. One pass sees the WHOLE script and
plans the pictures as a sequence, where each shot may carry something forward
from the one before it — the same coin, thinner; the same table, emptier; the
same hand, now closing on nothing. That continuity is what makes a Short feel
authored instead of assembled, and it cannot be produced one sentence at a
time by a model that has not seen the others.

CONTRACT: fail-open. No key, no reply, bad JSON, wrong shot count — all return
None, and main falls back to the per-beat prompt writer it uses today. A
storyboard is a better way to get the pictures, never a prerequisite for
getting them.
"""

import json
import os
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"

MODEL_DEFAULT = "gpt-4o"          # the arc is the point; a mini model loses it
MIN_VISUAL_CHARS = 40


def enabled() -> bool:
    return os.environ.get("RUFUS_STORYBOARD", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _model() -> str:
    return os.environ.get("RUFUS_STORYBOARD_MODEL", MODEL_DEFAULT).strip() \
        or MODEL_DEFAULT


def _prompt(script: str, beats: list[str], era_tags: list[str],
            character_clause: str = "") -> str:
    numbered = "\n".join(
        f"{i + 1}. [{era_tags[i] if i < len(era_tags) else 'present day'}] {b}"
        for i, b in enumerate(beats))
    return (
        "You are the storyboard artist for a 40-second vertical documentary "
        "Short. You are given the FINISHED narration. Your job is to plan the "
        "pictures as ONE CONTINUOUS SEQUENCE, not to illustrate each sentence "
        "on its own.\n\n"
        f"FULL SCRIPT (read it all before you draw anything):\n{script}\n\n"
        f"THE {len(beats)} SHOTS, with the era each one is set in:\n{numbered}\n\n"
        "WHAT MAKES THIS A STORYBOARD AND NOT A SLIDESHOW:\n"
        "1. SHOW THE LITERAL THING THE LINE NAMES. If the line says the coin "
        "held four and a half grams of silver, the shot is THAT COIN — not a "
        "family at dinner, not a person looking thoughtful. A picture of the "
        "general topic instead of the specific sentence is the single most "
        "common way this goes wrong.\n"
        "2. CARRY SOMETHING FORWARD. Wherever it is true to the script, a shot "
        "should continue the one before it: the same coin now thinner, the "
        "same market now empty, the same hand now closing on nothing. Name "
        "what carries over in `carries_over`. Not every shot can — but a "
        "sequence where NOTHING does is the failure this exists to prevent.\n"
        "3. ONE SUBJECT PER SHOT. A frame with a coin AND a merchant AND a "
        "market AND a ledger has no subject. Choose the one thing the line is "
        "actually about and fill the frame with it.\n"
        "4. LET THE FRAME CARRY THE FEELING, not a described emotion. Not 'his "
        "expression one of despair' — a stall with nothing left on it. Not "
        "'revealing the anguish of misplaced trust' — a fist of coins held out "
        "and no one reaching. What is IN the frame, never what someone feels.\n"
        "5. OBEY THE ERA TAG on each shot. [present day] means an ordinary "
        "scene of today; a year means every visible detail belongs to it.\n"
        "6. NEVER NAME WORDS THAT WOULD BE PRINTED IN FRAME. No headline text, "
        "no inscriptions, no dates on objects — the image model garbles "
        "lettering and it is the clearest sign a machine made this. Write the "
        "object as a blank physical thing instead.\n"
        f"{character_clause}"
        "\nEach `visual` is 2-3 plain sentences describing only what the camera "
        "sees: the subject, what it is doing or how it sits, and the setting. "
        "No camera bodies, no lens specs, no style words — the renderer adds "
        "the house style itself.\n\n"
        "Reply with ONLY this JSON:\n"
        '{"through_line": "<the one visual idea the sequence returns to>",\n'
        ' "shots": [{"n": 1, "visual": "...", "carries_over": null}, ...]}\n'
        f"Exactly {len(beats)} shots, n from 1 to {len(beats)}."
    )


def _clean(plan: dict, n_beats: int) -> list[str] | None:
    """The visuals in beat order, or None if the reply can't be trusted.

    Beat i must line up with shot i — the renderer cuts on the assumption that
    clip[i] belongs to beat[i], so a short or mis-numbered list would narrate
    every later picture against the wrong sentence."""
    if not isinstance(plan, dict):
        return None
    shots = plan.get("shots")
    if not isinstance(shots, list) or len(shots) != n_beats:
        return None
    out: list[str] = []
    for entry in shots:
        if not isinstance(entry, dict):
            return None
        visual = str(entry.get("visual", "")).strip()
        if len(visual) < MIN_VISUAL_CHARS:
            return None
        carries = entry.get("carries_over")
        if isinstance(carries, str) and carries.strip():
            # Stated as continuity so the image model has the thread too, not
            # only the storyboard's own notes.
            visual = f"{visual.rstrip('.')}. Continuing from the previous shot: {carries.strip()}."
        out.append(visual)
    return out


def plan(script: str, beats: list[str], era_tags: list[str] | None = None,
         character_clause: str = "") -> list[str] | None:
    """One visual per beat, planned as a sequence. None to use the old path."""
    if not enabled() or not beats:
        return None
    era_tags = era_tags or []
    try:
        from openai import OpenAI
        keys_file = CONFIG_DIR / "keys.json"
        key = ""
        if keys_file.exists():
            key = json.loads(keys_file.read_text()).get("openai", "")
        if not key or key.startswith("YOUR_") or key.startswith("FILL_"):
            return None
        resp = OpenAI(api_key=key).chat.completions.create(
            model=_model(),
            messages=[{"role": "user",
                       "content": _prompt(script, beats, era_tags, character_clause)}],
            temperature=0.8, max_tokens=2000, timeout=120,
            response_format={"type": "json_object"},
        )
        raw = json.loads(resp.choices[0].message.content or "{}")
        visuals = _clean(raw, len(beats))
    except Exception as e:
        print(f"[storyboard] skipped (non-fatal): {e}")
        return None

    if visuals is None:
        print("[storyboard] reply didn't validate — using per-beat prompts")
        return None
    through = str(raw.get("through_line", "")).strip()
    if through:
        print(f"[storyboard] through-line: {through}")
    print(f"[storyboard] {len(visuals)} shots planned as one sequence")
    return visuals
