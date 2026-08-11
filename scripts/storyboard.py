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
import re
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
        "2. CARRY A PHYSICAL OBJECT FORWARD — NEVER A MOOD. `carries_over` "
        "must name a THING you could point a camera at: the same coin now "
        "thinner, the same lantern, the same wooden table, the same coat. "
        "It must NOT be a feeling or an idea. These are all real answers a "
        "previous run gave, and every one of them is WRONG: \"emptiness and "
        "desolation\", \"emptiness and chaos\", \"sense of despair and loss\", "
        "\"unresolved financial burden\", \"ongoing neglect\", \"threat of "
        "repeating past mistakes\". Carrying a mood forward does not connect "
        "the shots — it REPEATS them. \"Emptiness\" four beats running renders "
        "four empty rooms, and the sequence looks like the same picture over "
        "and over. Objects can travel through a story; feelings can only be "
        "restated. Use null when nothing physical genuinely carries over — "
        "that is far better than naming a mood.\n"
        "3. ONE SUBJECT PER SHOT. A frame with a coin AND a merchant AND a "
        "market AND a ledger has no subject. Choose the one thing the line is "
        "actually about and fill the frame with it.\n"
        "3b. PUT PEOPLE IN IT. At least half the shots must show a person "
        "DOING something — hands counting coins, a man locking a door, a woman "
        "carrying a crate, a queue that does not move. The same previous run "
        "returned eight shots out of ten with no one in them (\"devoid of "
        "people\", \"absence of workers\", \"abandoned\", \"unused\", "
        "\"barren\") for a script about one in four people losing their job. "
        "Empty rooms are the cheapest way to say a thing is bad, and they make "
        "a viewer feel nothing. A face or a pair of hands is what makes a "
        "number land.\n"
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
        '{"through_line": "<the ONE physical object the sequence returns to — '
        'a thing, not a topic. \\"one coin, thinning\\" is right; \\"rising '
        'unemployment and its impact on society\\" is an essay title and is '
        'wrong>",\n'
        ' "shots": [{"n": 1, "visual": "...", "carries_over": null}, ...]}\n'
        f"Exactly {len(beats)} shots, n from 1 to {len(beats)}."
    )


# Words that name a FEELING or an IDEA rather than a thing you could point a
# camera at. A carries_over made only of these is the failure this list exists
# to catch — see _is_a_thing.
_ABSTRACT = {
    "emptiness", "empty", "desolation", "desolate", "despair", "hope",
    "hopelessness", "chaos", "silence", "loss", "neglect", "burden", "threat",
    "disregard", "fear", "anxiety", "anguish", "grief", "sorrow", "tension",
    "uncertainty", "instability", "collapse", "decline", "crisis", "ruin",
    "poverty", "wealth", "struggle", "suffering", "hardship", "mood",
    "atmosphere", "sense", "feeling", "tone", "theme", "idea", "impact",
    "consequence", "aftermath", "legacy", "history", "past", "future",
    "unresolved", "ongoing", "looming", "foreboding", "repeating", "mistakes",
    "issues", "problems", "lessons", "change", "shift", "power", "greed",
    "value", "trust", "belief", "doubt", "time", "society", "economy",
    # Domain adjectives: they qualify a thing but never name one, so
    # "unresolved financial burden" must not read as concrete.
    "financial", "economic", "monetary", "social", "political",
    "historical", "personal", "public", "general", "widespread",
}
_STOPWORDS = {
    "a", "an", "the", "of", "and", "or", "in", "on", "at", "to", "from",
    "with", "for", "same", "previous", "shot", "still", "now", "its", "his",
    "her", "their", "this", "that", "these", "those", "is", "are", "was",
    "were", "be", "been", "it", "as", "by", "more", "less", "one",
}


def _is_a_thing(carries: str) -> bool:
    """Whether `carries_over` names something a camera could point at.

    THE LIVE FAILURE, in full. Every carries_over in the Great Depression run
    was a mood: "emptiness and desolation", "emptiness and chaos", "sense of
    despair and loss", "unresolved financial burden", "ongoing neglect",
    "threat of repeating past mistakes". Not one named a physical object.

    That is worse than no continuity at all. An image model handed "carry
    emptiness forward" four beats running renders four empty rooms — the
    instruction to connect the shots becomes the instruction to repeat them,
    which is exactly the repetitiveness the owner reported. A thread has to be
    a THING: the same coin, the same table, the same lantern. Objects can move
    through a story; feelings can only be restated.

    Fail-open in spirit: a carries_over that fails this is dropped, and the
    shot itself is kept. A shot with no stated thread still renders fine."""
    words = [w for w in re.findall(r"[a-z]+", carries.lower())
             if w not in _STOPWORDS]
    concrete = [w for w in words if w not in _ABSTRACT]
    return bool(concrete)


# How much of the character's look a shot must already restate before we accept
# it as self-sufficient. Calibrated on the Great Depression run's three
# Chronicler shots: 0.67 for the one that described him, 0.33 and 0.27 for the
# two that only named him — and those two are the ones that came back wrong.
_CHARACTER_ECHO_MIN = 0.5


def _restates_the_look(visual: str, short_ref: str) -> bool:
    """Whether this shot repeats enough of the character's appearance."""
    words = {w for w in re.findall(r"[a-z]+", short_ref.lower())
             if w not in _STOPWORDS and len(w) > 2}
    if not words:
        return True
    seen = set(re.findall(r"[a-z]+", visual.lower()))
    return len(words & seen) / len(words) >= _CHARACTER_ECHO_MIN


def _pin_character(visual: str, name: str, short_ref: str) -> str:
    """Restate the character's LOOK in any shot that only names them.

    The image model renders each beat from noise with no memory of the others,
    so "The hooded figure, the Chronicler, appears again" gives it nothing to
    match and it invents an appearance.

    Live proof, from the Great Depression run's own images. Beat 1 said
    "weathered sepia-and-antique-gold cloak" and rendered a tan-gold cloak.
    Beat 10 said "sepia-and-gold cloak" and rendered brown. Beat 5 said only
    "The hooded figure, the Chronicler, appears again" — and rendered him in a
    BLACK cloak. One character, named three times, three different colours on
    screen, in a channel whose whole point is a recurring figure.

    The storyboard is the right place to fix it: it is the only stage that
    knows a later shot is the SAME person as an earlier one.
    """
    if not name or not short_ref:
        return visual
    bare = re.sub(r"(?i)^the\s+", "", name).strip()
    if not bare or bare.lower() not in visual.lower():
        return visual
    if _restates_the_look(visual, short_ref):
        return visual
    return (f"{visual.rstrip().rstrip('.')}. {name} is {short_ref} — the same "
            f"figure, identical in every appearance.")


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
            carries = carries.strip()
            if _is_a_thing(carries):
                # Stated as continuity so the image model has the thread too,
                # not only the storyboard's own notes.
                visual = (f"{visual.rstrip('.')}. Continuing from the previous "
                          f"shot: {carries}.")
            else:
                print(f"[storyboard] dropped mood-only thread: {carries!r}")
        out.append(visual)
    return out


def _character(niche: str | None) -> tuple[str, str]:
    """(name, short_ref) for the niche's recurring character, or ("", "")."""
    if not niche:
        return "", ""
    try:
        import character_engine
        if not character_engine.enabled(niche):
            return "", ""
        cfg = character_engine.niche_character(niche) or {}
        return str(cfg.get("name", "")).strip(), character_engine.short_ref(niche)
    except Exception as e:
        print(f"[storyboard] character look-up skipped (non-fatal): {e}")
        return "", ""


def plan(script: str, beats: list[str], era_tags: list[str] | None = None,
         character_clause: str = "", niche: str | None = None) -> list[str] | None:
    """One visual per beat, planned as a sequence. None to use the old path."""
    if not enabled() or not beats:
        return None
    era_tags = era_tags or []
    try:
        from openai import OpenAI
        keys_file = CONFIG_DIR / "keys.json"
        key = ""
        if keys_file.exists():
            key = json.loads(keys_file.read_text(encoding="utf-8")).get("openai", "")
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
        if visuals:
            name, short = _character(niche)
            pinned = [_pin_character(v, name, short) for v in visuals]
            n_fixed = sum(1 for a, b in zip(visuals, pinned) if a != b)
            if n_fixed:
                print(f"[storyboard] restated {name}'s look in {n_fixed} shot(s) "
                      f"that only named him")
            visuals = pinned
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
