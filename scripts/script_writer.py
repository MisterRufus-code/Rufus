#!/usr/bin/env python3
"""
script_writer.py
Writes a value-focused Shorts script from a seed (real source material) + scene description.

Key differences from previous version:
- Takes a `seed` dict with REAL source material (Reddit story or curated quote)
- Target 35-50s (80-130 words) — long enough for real value
- System prompt rewritten: tired expert tone, no viral-creator clichés
- Scorer weights "substance" highest (uses specifics from source)
- max_tokens raised to 250
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

# Phrases that flag AI-generated content. Filtered post-generation (not in prompt).
BANNED_PHRASES = [
    # AI-creator slop
    "buckle up", "let's dive in", "let me tell you", "imagine this", "picture this",
    "here's the thing", "here's why", "the truth is", "let's talk about",
    "in this video", "in today's video", "in today's world",
    # Generic motivational filler
    "game-changer", "game changer", "unlock", "skyrocket", "leverage",
    "dive deep", "delve", "revolutionize", "groundbreaking", "cutting-edge",
    "seamlessly", "robust", "empower", "synergy",
    # Corporate buzzwords
    "it's important to note", "at the end of the day", "paradigm", "disrupt",
    "journey", "navigating", "landscape", "crucial", "vital", "actionable",
    # Hedging that kills authority
    "in many ways", "in some sense", "you could argue", "some say",
    # AI essay structure
    "first and foremost", "moreover", "furthermore", "in conclusion",
    "to sum up", "all in all",
]

SCORE_MIN     = 7
MAX_ATTEMPTS  = 3

# Target script length (35-50s spoken @ ~150 wpm = 80-130 words)
MIN_WORDS     = 80
MAX_WORDS     = 130


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


def _seed_block(seed: dict) -> str:
    """Format the seed as a clearly-delimited block for the writer."""
    if not seed:
        return ""
    if seed.get("type") == "reddit":
        return (
            "REAL SOURCE MATERIAL (Reddit story):\n"
            f"Subreddit: {seed.get('source', '')}\n"
            f"Title:     {seed.get('title', '')}\n"
            f"Story:     {seed.get('content', '')}\n"
        )
    if seed.get("type") == "wisdom":
        return (
            "REAL SOURCE MATERIAL (quote):\n"
            f"Quote:  \"{seed.get('content', '')}\"\n"
            f"Author: {seed.get('source', 'Unknown')}\n"
        )
    return ""


def _generate(client: OpenAI, system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=0.85,
        max_tokens=250,
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


def _score(client: OpenAI, script: str, seed: dict) -> int:
    """Score with gpt-4o (different model class → less self-bias)."""
    seed_text = seed.get("content", "") if seed else ""
    prompt = (
        f"Score this YouTube Shorts script 1-10:\n\n"
        f"SCRIPT:\n\"{script}\"\n\n"
        f"SOURCE IT WAS BASED ON:\n\"{seed_text}\"\n\n"
        "Criteria:\n"
        "- Substance (uses specifics from source — names, numbers, story details): /3\n"
        "- Hook (first line is short, curiosity gap, no setup): /2\n"
        "- Punch (no filler, every word earns its place): /2\n"
        "- Loop (ending pulls viewer back to start): /2\n"
        "- Human (sounds like tired expert, zero AI clichés): /1\n\n"
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


def _build_system(niche_cfg: dict) -> str:
    return (
        "You are an experienced writer in this niche. You compose 35-50 second YouTube "
        "Shorts based on REAL source material provided by the user. Your job is to "
        "compress and sharpen, never to invent.\n\n"
        "VOICE:\n"
        "- Talk like a tired expert who has seen this 100 times. Not a viral creator.\n"
        "- Specific over generic. If the source has a name, number, or detail — USE IT.\n"
        "- Short, conversational sentences. The reader should hear a real human, not narration.\n"
        "- Never moralize. Never summarize. Never invent stats. Tell the actual story.\n\n"
        "STRUCTURE:\n"
        "- Line 1: Hook under 8 words. Concrete curiosity gap. No setup. No throat-clearing.\n"
        "- Body: Tell the story or unpack the quote using SPECIFICS from the source.\n"
        "- Final line: A loop line that makes the viewer want to replay from the start.\n\n"
        "RULES:\n"
        f"- {MIN_WORDS}-{MAX_WORDS} words total (35-50 seconds when spoken).\n"
        "- No phrases like 'here's why', 'the truth is', 'let me tell you', 'imagine this'.\n"
        "- No AI essay structure (first, moreover, in conclusion, etc.).\n"
        "- No corporate buzzwords (leverage, unlock, journey, paradigm, etc.).\n"
        "- If the source is a quote, attribute it briefly in the body ('Buffett said...').\n"
        "- Output ONLY the script text. No labels. No quotation marks around it."
    )


def write_script(scene_description: str, seed: dict | None = None) -> str:
    niche, active = _load_niche()
    client        = OpenAI(api_key=_load_key())
    learnings     = _load_learnings()

    if seed:
        if seed.get("type") == "reddit":
            print(f"[gpt] seed: Reddit story from {seed.get('source', '')}")
        else:
            print(f"[gpt] seed: quote — {seed.get('source', 'Unknown')}")

    system = _build_system(niche)

    winning_hooks = learnings.get("winning_hooks", [])[:3]
    avoid_hooks   = learnings.get("losing_hooks",  [])[:3]
    hook_hint     = ""
    if winning_hooks:
        hook_hint += f"\n\nHook patterns that previously worked: {winning_hooks}"
    if avoid_hooks:
        hook_hint += f"\nAvoid hook patterns like: {avoid_hooks}"

    cta       = niche["cta"]
    seed_blk  = _seed_block(seed) if seed else ""
    base_usr  = (
        f"{seed_blk}\n"
        f"Background scene in the video: {scene_description}\n\n"
        f"Write the Short script now. Compress the source into {MIN_WORDS}-{MAX_WORDS} words. "
        f"End with this exact CTA on its own line: \"{cta}\"\n"
        f"The line before the CTA should be a loop line that pulls the viewer back to the hook."
        f"{hook_hint}"
    )

    best_script = ""
    best_score  = 0
    last_script = ""

    for attempt in range(1, MAX_ATTEMPTS + 1):
        variation   = "" if attempt == 1 else f"\n\nAttempt {attempt}: be more specific, fewer clichés, more concrete details from the source."
        script      = _generate(client, system, base_usr + variation)
        last_script = script

        banned = _find_banned(script)
        if banned:
            print(f"[gpt] attempt {attempt}/3 – rejected ('{banned}' detected)")
            continue

        word_count = len(script.split())
        if word_count < MIN_WORDS:
            print(f"[gpt] attempt {attempt}/3 – too short ({word_count} words, need {MIN_WORDS}+)")
            continue

        score = _score(client, script, seed)
        print(f"[gpt] attempt {attempt}/3 – score: {score}/10 – {word_count} words")

        if score > best_score:
            best_score  = score
            best_script = script

        if score >= SCORE_MIN:
            break

    if not best_script:
        print(f"[gpt] ⚠ no attempt passed filters – using last")
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
