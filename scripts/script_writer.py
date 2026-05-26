#!/usr/bin/env python3
"""
script_writer.py
Writes a viral Shorts script from a scene description.
- Post-generation banned-phrase filter (rejects + re-rolls instead of just prompting)
- Scorer uses a different model (gpt-4o) to reduce self-grading bias
- Max 3 attempts, keeps best
"""

import json
import os
import re
import sys
from pathlib import Path

from openai import OpenAI

CONFIG_DIR     = Path(__file__).parent.parent / "config"
NICHES_FILE    = CONFIG_DIR / "niches.json"
KEYS_FILE      = CONFIG_DIR / "keys.json"
BLACKLIST_FILE = CONFIG_DIR / "blacklist.json"
LEARNINGS_FILE = CONFIG_DIR / "learnings.json"

BANNED_PHRASES = [
    "buckle up", "let's dive in", "game-changer", "game changer", "unlock",
    "skyrocket", "leverage", "dive deep", "delve", "it's important to note",
    "in today's world", "the truth is", "here's the thing",
    "at the end of the day", "paradigm", "disrupt", "journey",
    "navigating", "landscape", "crucial", "vital", "actionable",
    "revolutionize", "groundbreaking", "cutting-edge", "seamlessly",
    "robust", "empower", "synergy",
]

SCORE_MIN    = 7
MAX_ATTEMPTS = 3


def _load_niche():
    data   = json.loads(NICHES_FILE.read_text())
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]
    return data["niches"][active], active


def _load_key() -> str:
    keys = json.loads(KEYS_FILE.read_text())
    key  = keys.get("openai", "")
    if not key or key.startswith("YOUR_") or key.startswith("FILL_"):
        raise ValueError("OpenAI key not set in config/keys.json")
    return key


def _load_learnings() -> dict:
    if LEARNINGS_FILE.exists():
        return json.loads(LEARNINGS_FILE.read_text())
    return {}


def _ideate_angle(client: OpenAI, niche_name: str, scene: str) -> str:
    """Generate one contrarian insight before writing the script."""
    prompt = (
        f"Background scene: {scene}\n"
        f"Niche: {niche_name}\n\n"
        "Give me ONE counterintuitive insight about this scene that would surprise someone "
        "in this niche. Challenge a common assumption. Create a curiosity gap.\n"
        "Reply with ONLY the insight in one sentence, max 15 words. No label. No intro."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.85,
            max_tokens=35,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def _generate(client: OpenAI, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.9,
        max_tokens=120,
    )
    return resp.choices[0].message.content.strip()


def _find_banned(script: str) -> str | None:
    """Return first banned phrase found (whole-word match), else None."""
    text = script.lower()
    for phrase in BANNED_PHRASES:
        pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
        if re.search(pattern, text):
            return phrase
    return None


def _score(client: OpenAI, script: str) -> int:
    """Score with gpt-4o (different model class than writer → less self-bias)."""
    prompt = (
        f"Score this YouTube Shorts script 1-10:\n\n\"{script}\"\n\n"
        "Criteria:\n"
        "- Hook (first line <5 words, creates instant curiosity): /3\n"
        "- Punch (no filler, every word earns its place, raw tone): /3\n"
        "- Loop (ending makes viewer want to replay from start): /2\n"
        "- Human (sounds like real person talking, zero AI clichés): /2\n\n"
        "Reply with ONLY a single integer 1-10."
    )
    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5,
        )
        return int(result.choices[0].message.content.strip().split()[0])
    except Exception:
        return 5


def write_script(scene_description: str) -> str:
    niche, active = _load_niche()
    client        = OpenAI(api_key=_load_key())
    learnings     = _load_learnings()

    angle = _ideate_angle(client, active, scene_description)
    if angle:
        print(f"[gpt] angle: {angle}")

    # Banned list NOT injected into prompt (listing forbidden words primes the model).
    # Filtered post-generation instead.
    system = (
        niche["gpt_system"] + "\n\n"
        "Write like a real person talking to a friend. Raw. Direct. "
        "No corporate speak. No AI tone.\n"
        "Every sentence must earn its place. If a word can be cut, cut it."
    )

    winning_hooks = learnings.get("winning_hooks", [])[:3]
    avoid_hooks   = learnings.get("losing_hooks",  [])[:3]
    hook_hint     = ""
    if winning_hooks:
        hook_hint += f"\nHooks that performed well: {winning_hooks}"
    if avoid_hooks:
        hook_hint += f"\nAvoid hooks like: {avoid_hooks}"

    cta        = niche["cta"]
    angle_line = f"Core insight to build the script around: {angle}\n" if angle else ""
    base_usr   = (
        f"Scene in the video: {scene_description}\n"
        f"{angle_line}\n"
        f"Write a 20-25 second Shorts script. MAX 35 words total.\n"
        f"First line: hook under 5 words that creates a curiosity gap.\n"
        f"Last line: \"{cta}\"\n"
        f"Design the ending to make the viewer want to replay from the start.\n"
        f"No intro. No filler. No setup.\n"
        f"Output ONLY the script text.{hook_hint}"
    )

    best_script = ""
    best_score  = 0
    last_script = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        variation   = "" if attempt == 1 else f" (attempt {attempt} – be bolder, fewer clichés)"
        script      = _generate(client, system, base_usr + variation)
        last_script = script

        banned = _find_banned(script)
        if banned:
            print(f"[gpt] attempt {attempt}/3 – rejected ('{banned}' detected)")
            continue

        score = _score(client, script)
        print(f"[gpt] attempt {attempt}/3 – score: {score}/10 – {len(script.split())} words")

        if score > best_score:
            best_score  = score
            best_script = script

        if score >= SCORE_MIN:
            break

    if not best_script:
        print(f"[gpt] ⚠ all {MAX_ATTEMPTS} attempts had banned phrases – using last")
        best_script = last_script

    if best_score < SCORE_MIN:
        print(f"[gpt] ⚠ best score {best_score}/10 – using best attempt anyway")

    return best_script


# ── Blacklist ───────────────────────────────────────────────────────────────────

def _blacklist_key(script: str) -> str:
    return " ".join(script.lower().split()[:10])


def check_blacklist(script: str) -> bool:
    if not BLACKLIST_FILE.exists():
        return False
    items = json.loads(BLACKLIST_FILE.read_text())
    key   = _blacklist_key(script)
    return key in items


def add_to_blacklist(script: str) -> None:
    BLACKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    items = json.loads(BLACKLIST_FILE.read_text()) if BLACKLIST_FILE.exists() else []
    key   = _blacklist_key(script)
    if key not in items:
        items.append(key)
    BLACKLIST_FILE.write_text(json.dumps(items[-1000:], indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_writer.py '<scene description>'")
        sys.exit(1)
    desc   = " ".join(sys.argv[1:])
    script = write_script(desc)
    print(f"\nSCRIPT={script}")
