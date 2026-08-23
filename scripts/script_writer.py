#!/usr/bin/env python3
"""
script_writer.py — Hook-first Shorts script writer.

Three-phase architecture:
  A. Hook factory   — 8 hook candidates (gpt-4o-mini, temp 1.0)
  B. Hook scorer    — regex pre-filter + LLM scoring → pick winner
  C. Body generator — gpt-4o conditioned on the winning hook, up to 3 attempts

Standards live in config/script_standards.json (single source of truth).
Every attempt is logged: logs/scripts/YYYYMMDD.jsonl + script_attempts table.

Returns: (script, run_id, score, criterion_scores, attempts_used, final_temperature, reasoning)
"""

import json
import os
import random
import re
import sys
import time
from pathlib import Path

from openai import OpenAI

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from script_logger import new_run_id, log_attempt, estimate_cost
import db_manager
from db_manager    import save_attempt

CONFIG_DIR          = Path(__file__).parent.parent / "config"
NICHES_FILE         = CONFIG_DIR / "niches.json"
KEYS_FILE           = CONFIG_DIR / "keys.json"
BLACKLIST_FILE      = CONFIG_DIR / "blacklist.json"
LEARNINGS_FILE      = CONFIG_DIR / "learnings.json"
GOLD_EXAMPLES_FILE  = CONFIG_DIR / "gold_examples.json"
STANDARDS_FILE      = CONFIG_DIR / "script_standards.json"


# ── Standards loader ────────────────────────────────────────────────────────────

_standards_cache: dict | None = None

def _standards() -> dict:
    global _standards_cache
    if _standards_cache is None:
        _standards_cache = json.loads(STANDARDS_FILE.read_text(encoding="utf-8"))
    return _standards_cache


def _load_niche():
    data   = json.loads(NICHES_FILE.read_text(encoding="utf-8"))
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]
    return data["niches"][active], active


def _load_key() -> str:
    keys = json.loads(KEYS_FILE.read_text(encoding="utf-8"))
    key  = keys.get("openai", "")
    if not key or key.startswith("YOUR_") or key.startswith("FILL_"):
        raise ValueError("OpenAI key not set in config/keys.json")
    return key


def _load_learnings() -> dict:
    """Per-channel learnings (channel_config resolves the path; legacy installs
    keep reading config/learnings.json via the shim)."""
    try:
        from channel_config import load_channel
        path = load_channel().learnings_path
    except Exception:
        path = LEARNINGS_FILE
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _recent_video_rows(niche_name: str, limit: int = 12) -> list[tuple[str, str]]:
    """(script_hook, script_full) of the most recent videos in this niche for
    the ACTIVE CHANNEL (RUFUS_CHANNEL env, set by main.py).

    Read-only peek at rufus.db — returns [] on any failure so generation
    never depends on the DB being present.
    """
    try:
        import sqlite3
        db = Path(__file__).parent.parent / "rufus.db"
        if not db.exists():
            return []
        chan = os.environ.get("RUFUS_CHANNEL", "main_en")
        with sqlite3.connect(str(db)) as c:
            rows = c.execute(
                "SELECT script_hook, script_full FROM videos "
                "WHERE niche=? AND (channel=? OR channel IS NULL) "
                "ORDER BY id DESC LIMIT ?",
                (niche_name, chan, limit),
            ).fetchall()
        return [(r[0] or "", r[1] or "") for r in rows]
    except Exception:
        return []


def _overused_hook_openers(niche_name: str, lookback: int = 30, top_n: int = 5,
                           share_threshold: float = 0.15) -> list[str]:
    """Long-term semantic-decay guard: if a small set of hook OPENING WORDS
    dominates the last `lookback` shipped hooks in this niche, name the worst
    offenders so the factory is told to avoid them. Without this, hooks
    slowly converge on the model's favorite few shapes over hundreds of
    videos, and _novelty_block's "don't resemble the last 10" alone doesn't
    catch a slow drift across dozens/hundreds — nobody notices until someone
    actually reads the DB. Read-only peek at rufus.db (via db_manager, so
    tests can monkeypatch db_manager.DB_FILE the same way test_db_manager.py
    already does); [] on any failure or with too little history to draw a
    real conclusion (fail-open, same pattern as _recent_video_rows)."""
    try:
        chan = os.environ.get("RUFUS_CHANNEL", "main_en")
        with db_manager._conn() as c:
            rows = c.execute(
                "SELECT script_hook FROM videos WHERE niche=? "
                "AND (channel=? OR channel IS NULL) ORDER BY id DESC LIMIT ?",
                (niche_name, chan, lookback),
            ).fetchall()
    except Exception:
        return []

    hooks = [r[0].strip() for r in rows if r[0] and r[0].strip()]
    if len(hooks) < 10:
        return []

    from collections import Counter
    openers = Counter(
        h.split()[0].lower().strip(".,!?\"'’")
        for h in hooks if h.split()
    )
    total = sum(openers.values())
    if not total:
        return []
    return [w for w, n in openers.most_common(top_n) if n / total >= share_threshold]


def _novelty_block(niche_name: str) -> str:
    """Prompt block: recent hooks to avoid + analytics winners/losers to learn from.

    This is the anti-repetition memory — without it every video is a cold
    start and GPT converges on its favorite structures.
    """
    parts = []

    recent = [h for h, _ in _recent_video_rows(niche_name) if h]
    if recent:
        listed = "\n".join(f"- {h}" for h in recent[:10])
        parts.append(
            "ALREADY PUBLISHED — your hooks must NOT resemble these in topic, "
            f"structure, or rhythm (write something a returning viewer would see as NEW):\n{listed}"
        )

    overused = _overused_hook_openers(niche_name)
    if overused:
        parts.append(
            "OPENER RESET — your recent hooks have leaned too heavily on these "
            f"opening words; do NOT start any hook with: {', '.join(overused)}"
        )

    learnings = _load_learnings()
    winners = learnings.get("winning_hooks") or []
    losers  = learnings.get("losing_hooks") or []
    if winners:
        listed = "\n".join(f"- {h}" for h in winners[:3])
        parts.append(
            f"ANALYTICS WINNERS — these hook styles held viewers; channel that energy "
            f"(do NOT copy the wording):\n{listed}"
        )
    if losers:
        listed = "\n".join(f"- {h}" for h in losers[:3])
        parts.append(f"ANALYTICS LOSERS — these styles lost viewers; avoid their patterns:\n{listed}")

    return ("\n\n".join(parts) + "\n\n") if parts else ""


def _hook_shape(h: str) -> str:
    """Coarse structural signature of a hook, for de-duplicating same-shaped hooks."""
    h = h.strip()
    if re.match(r"^[\$\d]", h):
        kind = "number"
    elif h.endswith("?"):
        kind = "question"
    elif re.match(r"^(You|You're|Your)\b", h, re.IGNORECASE):
        kind = "identity"
    elif re.match(r"^I\b", h):
        kind = "confession"
    elif re.match(r"^[A-Z][a-z]+(?:'s)?\s", h):
        kind = "name"
    else:
        kind = "other"
    first_two = " ".join(re.sub(r"[^\w\s$]", "", h).lower().split()[:2])
    return f"{kind}:{first_two}"


def _dedupe_similar_hooks(hooks: list[str]) -> list[str]:
    """Drop hooks that share the same structural shape AND opening words —
    the scorer should choose between genuinely different angles, not 8
    variations of one idea."""
    seen: set[str] = set()
    out = []
    for h in hooks:
        sig = _hook_shape(h)
        if sig in seen:
            continue
        seen.add(sig)
        out.append(h)
    return out


def _load_gold_examples(niche_name: str) -> list[dict]:
    if not GOLD_EXAMPLES_FILE.exists():
        return []
    data = json.loads(GOLD_EXAMPLES_FILE.read_text(encoding="utf-8"))
    return data.get(niche_name, [])


def _gold_voice_note() -> str:
    """The `note` key of gold_examples.json.

    THE BEST DESCRIPTION OF THIS CHANNEL'S VOICE, AND THE MODEL HAD NEVER SEEN
    IT. It says: open on the VIEWER or a sharp take, use the famous name as
    PROOF mid-script rather than as the subject of a biography, present tense
    and second person, a loop that echoes the hook, an ending that earns a
    comment instead of begging for a follow.

    Every one of those is a rule about HOW a fact is said rather than which
    fact it is — which is the exact axis the scripts were weakest on — and it
    sat in a JSON key that _load_gold_examples never read. It was documentation
    for whoever opened the file. Two examples were carrying the entire weight
    of teaching a voice that had been written down in words all along.
    """
    if not GOLD_EXAMPLES_FILE.exists():
        return ""
    try:
        data = json.loads(GOLD_EXAMPLES_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    # `voice` is for the model; `note` is for whoever opens the file. Sending
    # the developer note as well would tell the model that it "mimics these
    # more than any instruction", which is true, useless to it, and a strange
    # thing to say to somebody you are about to instruct.
    return (data.get("voice") or "").strip()


def _pick_cta(niche_cfg: dict, niche_name: str = "") -> str:
    """Pick a CTA, avoiding the ones used in the last few videos of this niche
    (a CTA the viewer saw yesterday reads as a template, not a sign-off)."""
    pool = niche_cfg.get("cta_pool") or [niche_cfg.get("cta", "Follow for more.")]
    if niche_name and len(pool) > 1:
        recent_last_lines = {
            full.strip().splitlines()[-1].strip()
            for _, full in _recent_video_rows(niche_name, limit=4) if full.strip()
        }
        fresh = [c for c in pool if c not in recent_last_lines]
        if fresh:
            pool = fresh
    return random.choice(pool)


# ── Owner-written creative direction ─────────────────────────────────────────
#
# AGENTS.md instructs the coding agents; nothing instructed the CONTENT agents.
# The channel owner's direction only ever reached this pipeline when somebody
# read it and hand-translated it into prompt text, which means it was never
# theirs to change. These two files are.
#
# Layered, not replaced — the same shape channel_config.py already uses to put
# a channel's niche_overrides on top of the base rather than substituting for
# it, so a second channel costs one new file and changes nothing existing.
ROOT_DIR       = Path(__file__).parent.parent
DIRECTION_FILE = ROOT_DIR / "DIRECTION.md"
DIRECTION_DIR  = CONFIG_DIR / "direction"

# The system prompt is already ~2,000 tokens, and this repo has shipped two
# instructions that contradicted each other (DELIVERY said "split it into short
# sentences" while the cadence gate demanded a 15+ word sentence — the model was
# penalised for obeying one). An unbounded free-text file is how that happens
# again, so it is bounded and the truncation is announced rather than silent.
DIRECTION_MAX_WORDS = 400

# Prose that tries to set something the numeric gates already enforce. Whatever
# DIRECTION.md says, script_standards.json wins — so say so once, loudly, at
# the point of conflict, rather than letting "keep it to 60 words" produce a
# run of scripts all rejected for being under min_words.
_DIRECTION_CONFLICT_RE = re.compile(
    r"\b\d+\s*(?:-|to|–)?\s*\d*\s*(?:words?|sentences?|seconds?|secs?)\b",
    re.IGNORECASE,
)


# Everything from this heading onward is what the model sees. Anything above it
# is for the human editing the file — how the layering works, why numbers here
# lose to script_standards.json, the cap. Sending that to the model would spend
# prompt budget teaching it to use a file it cannot edit, and the examples in
# those notes ("keep it to 60 words") would trip the very conflict check they
# exist to explain. A file with no marker is sent whole, so a plain one still
# works.
DIRECTION_MARKER = "## The direction"


def _channel() -> str:
    return os.environ.get("RUFUS_CHANNEL", "main_en").strip() or "main_en"


def _direction_body(text: str) -> str:
    """The part of a direction file addressed to the model, not to the owner."""
    idx = text.find(DIRECTION_MARKER)
    return text if idx < 0 else text[idx + len(DIRECTION_MARKER):].strip()


def load_direction() -> tuple[str, str]:
    """(direction text, one-line description of where it came from).

    Fail-open like every other optional input here: no files, unreadable files,
    anything — returns ("", ...) and the prompts are byte-identical to a run
    without this feature.
    """
    parts: list[str] = []
    found: list[str] = []
    missing_channel = ""

    for path, label in ((DIRECTION_FILE, "DIRECTION.md"),
                        (DIRECTION_DIR / f"{_channel()}.md",
                         f"config/direction/{_channel()}.md")):
        try:
            if path.exists():
                text = _direction_body(path.read_text(encoding="utf-8").strip())
                if text:
                    parts.append(text)
                    found.append(label)
                    continue
        except OSError:
            pass
        if path is not DIRECTION_FILE:
            missing_channel = label

    if not parts:
        return "", "none (no DIRECTION.md — prompts unchanged)"

    text = "\n\n".join(parts)
    words = text.split()
    note = " + ".join(found)
    if missing_channel:
        # Named explicitly: "I edited the file and nothing changed" is otherwise
        # indistinguishable from having edited the wrong one.
        note += f" only (no {missing_channel})"
    if len(words) > DIRECTION_MAX_WORDS:
        over = len(words) - DIRECTION_MAX_WORDS
        text = " ".join(words[:DIRECTION_MAX_WORDS])
        note += (f", {len(words)} words — TRUNCATED to {DIRECTION_MAX_WORDS} "
                 f"({over} dropped; the system prompt is already long and "
                 f"competing instructions are what make a model ignore both)")
    else:
        note += f", {len(words)} words"

    if (hit := _DIRECTION_CONFLICT_RE.search(text)):
        note += (f"\n[gpt] direction: ⚠ mentions '{hit.group(0)}' — lengths are "
                 f"enforced numerically by config/script_standards.json, which "
                 f"WINS. Prose here cannot loosen a gate, only lose to it.")
    return text, note


def _direction_block() -> str:
    """The direction as a prompt block, printed once so it is never invisible."""
    text, note = load_direction()
    print(f"[gpt] direction: {note}")
    if not text:
        return ""
    return f"CHANNEL DIRECTION (the owner's standing instructions):\n{text}\n\n"


def _seed_block(seed: dict) -> str:
    if not seed:
        return ""
    if seed.get("type") == "reddit":
        return (
            "SOURCE MATERIAL (real Reddit story):\n"
            f"Subreddit: {seed.get('source', '')}\n"
            f"Title:     {seed.get('title', '')}\n"
            f"Story:     {seed.get('content', '')}\n"
        )
    if seed.get("type") == "hackernews":
        return (
            "SOURCE MATERIAL (real Hacker News post):\n"
            f"Title:     {seed.get('title', '')}\n"
            f"Body:      {seed.get('content', '')}\n"
        )
    if seed.get("type") == "stackexchange":
        return (
            "SOURCE MATERIAL (real StackExchange question/story):\n"
            f"Site:      {seed.get('source', '')}\n"
            f"Title:     {seed.get('title', '')}\n"
            f"Story:     {seed.get('content', '')}\n"
        )
    if seed.get("type") == "wikipedia":
        return (
            "SOURCE MATERIAL (Wikipedia article summary — real, sourced facts; "
            "use ONLY details stated here or universally established):\n"
            f"Article:   {seed.get('title', '')}\n"
            f"Summary:   {seed.get('content', '')}\n"
            f"URL:       {seed.get('url', '')}\n"
        )
    if seed.get("type") == "rss":
        return (
            "SOURCE MATERIAL (real news article):\n"
            f"Source:    {seed.get('source', '')}\n"
            f"Title:     {seed.get('title', '')}\n"
            f"Summary:   {seed.get('content', '')}\n"
            f"URL:       {seed.get('url', '')}\n"
        )
    if seed.get("type") == "wisdom":
        return (
            "SOURCE MATERIAL (real quote):\n"
            f"\"{seed.get('content', '')}\"\n"
            f"— {seed.get('source', 'Unknown')}\n"
        )
    return ""


def _build_gold_block(examples: list[dict]) -> str:
    if not examples:
        return ""
    lines = ["\n── GOLD STANDARD EXAMPLES ── Study these. This is the exact level required.\n"]
    note = _gold_voice_note()
    if note:
        lines.append(f"WHAT MAKES THEM WORK — this is the voice, not a suggestion:\n{note}\n")
    # FOUR, NOT TWO. The slice was written when every niche had exactly two,
    # and it silently discarded the examples added to fix this file's actual
    # problem: both money_history originals demonstrate the same move — deny
    # the obvious explanation — so the set taught one lesson twice, and a
    # writer shown one lesson writes one shape. The two new ones drop the
    # viewer into a moment and name what something cost. Roughly 600 tokens
    # for the highest-transfer part of the whole prompt.
    for i, ex in enumerate(examples[:4], 1):
        src = f"{ex.get('seed_type', 'source')}: \"{ex.get('seed_content', '')[:120]}\" — {ex.get('seed_source', '')}"
        lines.append(f"Example {i} | {src}\n")
        lines.append(ex.get("script", ""))
        lines.append("")
    lines.append("── END EXAMPLES ──")
    return "\n".join(lines)


# ── Regex pre-filters (deterministic, free) ─────────────────────────────────────

# Match a "specific": digit-run, dollar amount, year, or proper noun (capitalized
# word not at sentence start). We approximate by matching any CapWord — the LLM
# rubric still judges authenticity.
_SPECIFIC_RE = re.compile(r"\$[\d,]+|\d+|\b[A-Z][a-z]+")
_SENTENCE_RE = re.compile(r"[^.!?]+[.!?]")


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z']+", text.lower())


def _content_tokens(text: str) -> set[str]:
    stopwords = set(_standards()["stopwords"])
    return {w for w in _word_tokens(text) if w not in stopwords and len(w) > 2}


# Cliché FAMILIES, not fixed strings.
#
# The exact-phrase ban list is trivially routed around, and the model does it:
# a shipped Monte dei Paschi script opened a scene with "Picture the Medici
# family counting coins" ('picture this' is banned) and pivoted on "The truth?"
# ('the truth is' is banned). Both passed every gate and went out. Banning the
# construction instead of one spelling of it is what closes that.
#
# Scoped deliberately narrowly: only the imperative scene-setting opener and
# the reveal-pivot form. "The picture showed a queue outside the bank" and
# "investors could not imagine a default" are legitimate and must still pass.
_BANNED_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(?:^|(?<=[.!?…—]\s))\s*picture\s+(?:this|that|the|a|an|yourself)\b",
                re.IGNORECASE | re.MULTILINE), "picture <scene> opener"),
    (re.compile(r"(?:^|(?<=[.!?…—]\s))\s*imagine\s+(?:this|that|the|a|an|yourself|if)\b",
                re.IGNORECASE | re.MULTILINE), "imagine <scene> opener"),
    # No trailing \b on the punctuation branch: "The truth? Ignore history" has
    # a space after the "?", which is not a word boundary, so a shared \b at the
    # end silently failed to match the exact form that shipped.
    (re.compile(r"\bhere'?s\s+the\s+truth\b|\bthe\s+truth\s+is\b|\bthe\s+truth\s*[?:,]",
                re.IGNORECASE), "'the truth' reveal-pivot"),
    (re.compile(r"\bask\s+yourself\b", re.IGNORECASE), "'ask yourself'"),
]


def _find_banned(script: str) -> str | None:
    text = script.lower()
    for phrase in _standards()["banned_phrases"]:
        if re.search(r"\b" + re.escape(phrase.lower()) + r"\b", text):
            return phrase
    for pattern, label in _BANNED_PATTERNS:
        if pattern.search(script):
            return label
    return None


def _find_hedging(script: str) -> str | None:
    text = script.lower()
    for phrase in _standards()["hedging_words"]:
        if re.search(r"\b" + re.escape(phrase.lower()) + r"\b", text):
            return phrase
    return None


# Last-resort substitutions so a stray banned word never ships a 0/10 script.
# Single-word swaps are grammar-safe; unmapped banned phrases are simply removed.
_BANNED_SYNONYMS = {
    "crucial": "key", "vital": "key", "leverage": "use", "robust": "strong",
    "delve": "look", "actionable": "practical", "skyrocket": "soar",
    "unlock": "reveal", "empower": "enable", "revolutionize": "transform",
    "disrupt": "upend", "journey": "path", "navigating": "handling",
    "landscape": "field", "synergy": "teamwork", "paradigm": "model",
    "groundbreaking": "new", "seamlessly": "smoothly",
    "game-changer": "turning point", "game changer": "turning point",
    "cutting-edge": "advanced", "dive deep": "look closely",
}


def _repair_banned(script: str) -> str:
    """Replace/strip banned phrases so the final script always passes the ban check.

    Mapped words get a grammar-safe synonym; everything else is deleted. Used only
    as a salvage step when the model can't produce a clean script on its own.
    """
    out = script
    for phrase in _standards()["banned_phrases"]:
        pat = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
        if not pat.search(out):
            continue
        repl = _BANNED_SYNONYMS.get(phrase.lower(), "")
        out = pat.sub(repl, out)
    # tidy artefacts left by deletions: doubled spaces, space/leading-comma, empties.
    # Only collapse spaces/tabs — never newlines (that would merge separate lines).
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"[ \t]+([,.;:!?])", r"\1", out)
    out = re.sub(r"^[\s,;:]+", "", out, flags=re.MULTILINE)   # drop leading punct from stripped connectors
    out = "\n".join(ln.strip() for ln in out.splitlines() if ln.strip())
    return out


def _hook_pre_check(hook: str) -> str | None:
    """Return rejection reason or None. Deterministic gate before LLM scoring."""
    h     = hook.strip().strip('"').strip("'")
    h_low = h.lower()
    n     = len(h.split())
    hs    = _standards()["hook"]
    if n < hs["min_words"]:
        return f"hook too short ({n} words, need ≥{hs['min_words']})"
    if n > hs["hard_max_words"]:
        return f"hook too long ({n} words, hard cap {hs['hard_max_words']})"
    for bad in hs["forbidden_openers"]:
        if h_low.startswith(bad):
            return f"forbidden opener: '{bad}'"
    if (b := _find_banned(h)):
        return f"banned phrase in hook: '{b}'"
    return None


# Faceless channel: a first-person "confession" hook is by definition a
# fabricated personal story — nobody behind this channel escaped debt bondage
# or earned $1.2M. These sailed through scoring, then the fact gate rejected
# the script built on them, capping every run at 5/10 and holding every upload.
_HOOK_FIRST_PERSON_RE = re.compile(
    r"(?i)\b(i|i'm|i'd|i've|my|mine|we|we're|we've|our|ours)\b")

_HOOK_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")


_LIST_MARKER_RE = re.compile(r"^\s*\d+[.)]\s*", re.MULTILINE)


def _strip_list_markers(text: str) -> str:
    """Drop leading "1. " / "2) " enumeration from each line.

    The pre-analysis is returned as a numbered list (1. CONTRADICTION,
    2. HOOK ANGLE, … 8. SENSORY ANCHOR). Feeding that verbatim into the
    invented-number grounding corpus made the digits 1-8 read as legitimate
    figures "from the source", so a fabricated "$1 billion" hook sailed
    through the check and only got caught later by the fact gate — capping
    the whole video to 5/10 (observed live on the Swiss-banking run). The
    list markers are formatting, never source facts."""
    return _LIST_MARKER_RE.sub("", text or "")


# ── Spoken vs. written numbers ───────────────────────────────────────────────
# A Short is HEARD. "$4,210,500,000,000 for one dollar?" is a real figure from
# a real source, and a voice engine reads it as forty syllables of digits that
# nobody can follow and that instantly sounds machine-generated. A person says
# "four point two trillion marks".
#
# But the writer could not say that: the grounding check compares number TOKENS
# against the source, so "4.2 trillion" was rejected as "number '4.2' not in
# source (invented figure)" while the unreadable full form passed. The rule
# meant to stop invented figures was forcing unspeakable ones.
#
# So the check is magnitude-aware: a hook number is grounded when it is a
# correct ROUNDING of a source number, not only when its digits appear
# verbatim. 4.2 trillion is 4,210,500,000,000 to two significant figures — the
# same fact, said out loud.
_SCALE_WORDS = {
    "thousand": 1e3, "million": 1e6, "billion": 1e9, "trillion": 1e12,
    "quadrillion": 1e15,
}
_SCALED_NUM_RE = re.compile(
    r"(?i)(\d+(?:[.,]\d+)*)\s*(thousand|million|billion|trillion|quadrillion)?")


def _numeric_values(text: str) -> list[tuple[str, float, int]]:
    """Every number in `text` as (as-written, value, significant digits).

    "4.2 trillion" -> ("4.2", 4.2e12, 2)
    "4,210,500,000,000" -> ("4,210,500,000,000", 4.2105e12, 11)

    Reading the scale WORD as part of the number is the whole point: parsed
    without it, "4.2 trillion" is the number 4.2, which matches nothing."""
    out: list[tuple[str, float, int]] = []
    for raw, scale in _SCALED_NUM_RE.findall(text):
        digits = raw.replace(",", "")
        try:
            value = float(digits)
        except ValueError:
            continue
        sig = len(digits.replace(".", "").lstrip("0")) or 1
        if scale:
            value *= _SCALE_WORDS[scale.lower()]
        out.append((raw, value, sig))
    return out


def _rounds_to(source_value: float, hook_value: float, sig: int) -> bool:
    """Is `hook_value` what you get by rounding `source_value` to `sig` figures?"""
    if source_value == 0 or hook_value == 0:
        return source_value == hook_value
    import math
    exponent = math.floor(math.log10(abs(source_value)))
    factor = 10 ** (exponent - sig + 1)
    return abs(round(source_value / factor) * factor - hook_value) < abs(factor) / 2


def _ungrounded_number(hook: str, source_text: str) -> str | None:
    """The first hook number that is neither in the source verbatim nor a
    correct rounding of a source figure. None when every number checks out."""
    src_tokens = {n.replace(",", "") for n in _HOOK_NUMBER_RE.findall(source_text)}
    src_values = [v for _, v, _ in _numeric_values(source_text)]
    for raw, value, sig in _numeric_values(hook):
        if raw.replace(",", "") in src_tokens:
            continue                       # verbatim, as before
        if any(_rounds_to(s, value, sig) for s in src_values):
            continue                       # "4.2 trillion" of 4,210,500,000,000
        return raw
    return None


def _hook_grounding_check(hook: str, source_text: str) -> str | None:
    """Reject hooks the fact gate is guaranteed to kill later. Deterministic.

    Two failure modes seen live: (1) first-person fabricated experience
    ('I escaped $50K in debt bondage'), (2) numbers invented from thin air
    ('$3.3 billion lost in hours' — nowhere in the source). Catching them
    BEFORE scoring means the factory's other candidates get their shot,
    instead of the script being built on a doomed hook."""
    if _HOOK_FIRST_PERSON_RE.search(hook):
        return "first-person confession (faceless channel — fabricated persona)"
    # Whole-TOKEN match, not substring: the old `num in source_text` check let
    # a fabricated magnitude through whenever its bare digits appeared inside a
    # real source number — e.g. "$1 billion" passed because "1" is a substring
    # of the source year "1873". Tokenizing both sides and requiring set
    # membership closes that (a live 5/10-cap cause), while "1873" still
    # matches the real "1873" token.
    bad = _ungrounded_number(hook, source_text)
    if bad is not None:
        return f"number '{bad}' not in source/analysis (invented figure)"
    return None


# Comma-grouped digits are ONE number, not several. The old pattern was
# r"\b\d{3,}\b", which splits "10,000,000" on its commas into three separate
# "000" tokens and then reports the script as repeating "000" 3x — a script
# that in fact contains that figure exactly once. Observed live: three
# consecutive false rejections on the Venezuela hyperinflation script
# (2026-07-31 10:43), each one an entire wasted generation spent rewriting a
# script that was never actually wrong. Matching the whole comma-grouped run
# and stripping the separators is what makes the count mean what it claims.
def grounding_corpus(seed: dict | None, analysis: str) -> str:
    """Everything a hook's numbers may be drawn from.

    ONE CORPUS, TWO READERS. The gate built this string from the full seed
    content plus the title plus the analysis; the prompt's allowed-numbers
    list was built from the FORMATTED seed block and the analysis. Two
    definitions of "the source" means the list shown to the writer and the set
    enforced against it can differ, and the writer is the one that pays: a
    figure the gate would have accepted is missing from its list, so it never
    uses it, and a hook it could have grounded is never written.

    That is the same shape as every other defect in this file's history — the
    checker knowing something the generator was never told.
    """
    return " ".join(filter(None, [
        (seed.get("content") or "") if seed else "",
        (seed.get("title") or "") if seed else "",
        _strip_list_markers(analysis or ""),
    ]))


def _allowed_numbers_block(*sources: str) -> str:
    """The exact figures a hook may use, listed for the generator.

    _hook_grounding_check() rejects any number in a hook that isn't a token of
    the source — but the PROMPT only ever said "use the source's numbers"
    without saying WHICH, leaving the model to extract them from prose. It
    routinely guessed plausible neighbours instead: four different Ibn Battuta
    hooks invented 1331/1352/1354/1355 in one run, and invented figures were
    the single largest accuracy failure live (21.4% of all rejections).

    Listing the permitted tokens turns "don't invent numbers" from a rule the
    model must infer into a closed set it can copy from. Derived with the SAME
    regex the check uses, so the prompt and the gate can't disagree about what
    counts as grounded.
    """
    tokens: list[str] = []
    for src in sources:
        for n in _HOOK_NUMBER_RE.findall(_strip_list_markers(src or "")):
            if n not in tokens:
                tokens.append(n)
    if not tokens:
        # No figures in the source at all — say so, rather than printing an
        # empty list that reads as "no constraint".
        return ("- The source contains NO numbers. Do not put any digit in a hook — "
                "build it on a name, place, or documented event instead.\n")
    shown = tokens[:40]
    more = f" (+{len(tokens) - len(shown)} more in the source above)" if len(tokens) > len(shown) else ""
    return (f"- NUMBERS YOU MAY USE — copy from this list EXACTLY, character for "
            f"character. Any other digit is an automatic rejection, including a "
            f"nearby year or a rounded version of one of these:\n"
            f"  {', '.join(shown)}{more}\n")


_REPEATED_NUM_RE = re.compile(r"\b\d[\d,]*\d\b|\b\d+\b")


def _repeated_number(script: str) -> str | None:
    """A distinctive figure (year, dollar amount, count) restated verbatim
    reads as padding, not new information — the "1873 three times" pattern
    seen live: under grounding pressure the model reaches for its one solid,
    already-verified number again and again instead of finding fresh
    specifics elsewhere in the source. Deterministic and free, so it catches
    this before an expensive LLM score gets spent on a redundant script."""
    counts: dict[str, int] = {}
    for raw in _REPEATED_NUM_RE.findall(script):
        n = raw.replace(",", "")
        if len(n) < 3:
            continue          # 1-2 digit numbers repeat legitimately ("3 banks", "3 years")
        counts[n] = counts.get(n, 0) + 1
    # A YEAR is the story's SETTING, not one of its statistics. A video about
    # the 1973 oil crisis names 1973 in the hook and again in the body, and
    # that is orientation, not padding. Holding years to the same limit as
    # figures killed five of six body attempts in one live run — every cycle
    # rejected for "number '1973' repeated" on a script about 1973 — and one
    # of the salvaged drafts went on to score 9/10. Three mentions is still
    # padding; two is how you tell a story about a year.
    repeats = {n: c for n, c in counts.items()
               if c >= (3 if _looks_like_a_year(n) else 2)}
    if not repeats:
        return None
    worst = max(repeats, key=repeats.get)
    return (f"number '{worst}' repeated {repeats[worst]}x — restate with a "
            f"NEW specific each time, not the same figure")


def _looks_like_a_year(n: str) -> bool:
    """A bare 4-digit number in the range a history script uses as a date."""
    return len(n) == 4 and n.isdigit() and 1000 <= int(n) <= 2099


def _em_dash_overuse(script: str) -> str | None:
    """3+ em-dashes in an 80-115 word script is a real AI-cadence tell — one
    or two is normal, good writing (this very sentence uses one). Only fires
    at >=3 to keep false positives near zero, since the quality gate is
    already flagged as too aggressive elsewhere — this must not add to that."""
    count = script.count("—")
    if count >= 3:
        return (f"em-dash overuse ({count} in the script — vary punctuation; "
                f"use a period, comma, or colon for some of these instead)")
    return None


def _specificity_density(text: str) -> float:
    """Specifics per 25 words. ≥1.0 means the body is grounded."""
    words = len(_word_tokens(text))
    if words == 0:
        return 0.0
    specs = len(_SPECIFIC_RE.findall(text))
    return specs / max(words / 25.0, 1.0)


def _sentence_stats(text: str) -> tuple[float, int]:
    """Return (avg_words_per_sentence, num_sentences)."""
    sentences = [s.strip() for s in _SENTENCE_RE.findall(text) if s.strip()]
    if not sentences:
        return 0.0, 0
    word_counts = [len(s.split()) for s in sentences]
    return sum(word_counts) / len(sentences), len(sentences)


def _loop_echoes_hook(script: str) -> tuple[bool, set[str]]:
    """Return (echoes, shared_content_tokens). Loop = second-to-last non-empty line."""
    lines = [l.strip() for l in script.strip().split("\n") if l.strip()]
    if len(lines) < 3:
        return False, set()
    hook_tokens = _content_tokens(lines[0])
    loop_tokens = _content_tokens(lines[-2])  # second-to-last (last is CTA)
    shared = hook_tokens & loop_tokens
    needed = _standards()["loop"]["min_shared_content_tokens"]
    return len(shared) >= needed, shared


# The fear/anger-coded subset of opinion_pool — used to steer _build_system's
# prompt toward the tension register channel-owner feedback asked for
# ("scared" over "best", "crushed" over "real"), NOT as a new pass/fail gate.
# Deliberately a prompt nudge, not a deterministic check: this session's own
# debugging found that piling on another hard-fail gate is what produced the
# 7-attempt rejection ladders in the first place (see _body_violations) — the
# opinion-word requirement already passes at one word from the full pool, and
# adding a second, narrower gate here would risk the same waste for a
# stylistic preference, not a correctness one.
_TENSION_WORDS = {
    "worst", "scared", "ruined", "broken", "fake", "rigged", "lie",
    "dead", "lost", "destroyed", "crushed", "ignored", "afraid",
}


def _has_opinion_word(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(r"\b" + re.escape(w) + r"\b", text_lower)
               for w in _standards()["opinion_pool"])


def _hook_already_present(first_line: str, hook: str) -> bool:
    """Fuzzy, not exact: GPT often rephrases/repunctuates the hook it was
    given ("…retire." vs "…retire!"), and the old exact-match check then
    re-inserted the original ABOVE the paraphrase — so the voice read the
    hook twice back-to-back (a real produced-video bug; nothing downstream
    caught it because the script *looked* fine). ≥60% of the hook's words
    appearing in line 1 counts as 'the hook is there'."""
    def _tokens(s: str) -> set:
        return set(re.findall(r"[a-z0-9']+", s.lower()))
    hook_toks = _tokens(hook)
    if not hook_toks:
        return True
    return len(hook_toks & _tokens(first_line)) / len(hook_toks) >= 0.6


def _body_violations(script: str) -> list[str]:
    """EVERY rejection reason for this script, not just the first one.

    Why this exists: _body_pre_check() historically returned on the first
    failure, so a script that broke three rules was rejected three separate
    times — one whole LLM generation burned per rule, each retry told about
    only the single problem it had just been caught on. The live failure log
    shows exactly that ladder (2026-08-02 01:15: too long → no opinion word →
    hedging → banned phrase → no opinion word → loop echo, seven attempts on
    one script). Collecting all of them means one generation produces the
    complete correction list, so the next attempt can fix everything at once.

    Order still matters for the caller's headline reason: the checks are
    appended cheapest/most-structural first, so violations[0] stays the same
    string the old single-return version would have produced.
    """
    body = _standards()["body"]
    out: list[str] = []

    words = len(script.split())
    if words < body["min_words"]:
        out.append(f"too short ({words} words, need ≥{body['min_words']})")
    elif words > body["max_words"]:
        out.append(f"too long ({words} words, cap {body['max_words']})")

    density = _specificity_density(script)
    if density < body["specificity_per_25_words"]:
        out.append(f"low specificity ({density:.2f}/25w, need ≥{body['specificity_per_25_words']})")

    avg, n = _sentence_stats(script)
    if n < 3:
        out.append(f"too few sentences ({n})")
    elif avg > body["max_avg_sentence_words"]:
        out.append(f"sentences too long (avg {avg:.1f} words, cap {body['max_avg_sentence_words']})")
    elif avg < body["min_avg_sentence_words"]:
        out.append(f"sentences too short (avg {avg:.1f} words, floor {body['min_avg_sentence_words']})")

    echoes, _ = _loop_echoes_hook(script)
    if not echoes:
        out.append("loop no echo (second-to-last line shares no content tokens with hook)")

    if not _has_opinion_word(script):
        out.append("no opinion word (need ≥1 from opinion_pool)")

    if (h := _find_hedging(script)):
        out.append(f"hedging word: '{h}'")

    if (rep := _repeated_number(script)):
        out.append(rep)

    if (dash := _em_dash_overuse(script)):
        out.append(dash)

    if (cad := _cadence_violation(script)):
        out.append(cad)

    return out


def _body_pre_check(script: str) -> str | None:
    """First rejection reason for the full body, or None. Runs AFTER banned check.

    Thin wrapper over _body_violations() — kept because the headline reason is
    what gets logged per attempt and shown in the dashboard's rejection table.
    """
    violations = _body_violations(script)
    return violations[0] if violations else None


def _protected_lines(script: str) -> tuple[list[str], list[str], list[str]]:
    """Split a script into (head, middle, tail) where head and tail must not be
    edited by any repair.

    Line 1 is the hook — the one line the whole video is scored on. The last
    line is the CTA, which is dictated verbatim by the niche. The second-to-last
    is the loop line, which must keep sharing a content word with the hook. Every
    repair below works only on what is left.
    """
    lines = [l.strip() for l in script.split("\n") if l.strip()]
    if len(lines) < 4:
        return lines, [], []
    return lines[:1], lines[1:-2], lines[-2:]


def _repair_length(script: str, cap: int) -> str:
    """Drop trailing body sentences until the script is under `cap` words.

    A 120-word script against a 115-word cap used to cost a whole generation —
    the live log shows attempt 1 of 3 burned on a five-word overage. That is the
    banned-phrase lesson again: if the fix is mechanical and we would accept it
    anyway, spending an LLM round-trip to arrive at it is pure waste.

    Trims from the END of the middle block because a beat's closing sentence is
    the most expendable — the setup and the turn carry the structure. The caller
    re-validates everything afterwards, so a trim that breaks specificity or the
    loop is discarded exactly like a banned-phrase repair that guts a sentence.
    """
    head, middle, tail = _protected_lines(script)
    if not middle:
        return script

    def _words(hd, md, tl) -> int:
        return len(" ".join(hd + md + tl).split())

    middle = list(middle)
    # Sentence-level first: finer-grained than dropping whole lines, so the
    # script loses the least it can while still getting under the cap.
    while _words(head, middle, tail) > cap:
        for i in range(len(middle) - 1, -1, -1):
            sentences = [s.strip() for s in _SENTENCE_RE.findall(middle[i]) if s.strip()]
            if len(sentences) > 1:
                middle[i] = " ".join(sentences[:-1])
                break
        else:
            if len(middle) <= 1:
                break
            middle.pop()
    return "\n".join(head + middle + tail)


def _split_for_punch(script: str) -> str:
    """Cut one long body sentence at a comma so a short one exists.

    THE OTHER HALF OF THE CADENCE GATE, WHICH HAD NO REPAIR AT ALL. The gate
    rejects two things — "missing a longer, flowing sentence (≥15 words)" and
    "missing a short, punchy sentence (≤6 words)" — and only the first could be
    repaired. The second was 21 live rejections, every one of them buying with
    a whole generation an edit a human editor makes with one keystroke.

    Splitting at a comma is the safe direction: the clause boundary is already
    there, and promoting it to a full stop is what DELIVERY asks for anyway
    ("Never write a long comma-chained run-on when the moment deserves a hard
    stop"). The prefix must be a plausible sentence on its own, so 3-6 words,
    and the remainder must not be left as a fragment.
    """
    head, middle, tail = _protected_lines(script)
    if not middle:
        return script
    for i, line in enumerate(middle):
        sentences = [s.strip() for s in _SENTENCE_RE.findall(line) if s.strip()]
        for j, s in enumerate(sentences):
            if len(s.split()) < 12 or not s.rstrip().endswith("."):
                continue
            for m in re.finditer(r",\s+", s):
                left, right = s[:m.start()].strip(), s[m.end():].strip()
                if not (3 <= len(left.split()) <= 6) or len(right.split()) < 4:
                    continue
                piece = f"{left}. {right[0].upper()}{right[1:]}"
                new_middle = list(middle)
                new_middle[i] = " ".join(sentences[:j] + [piece] + sentences[j + 1:])
                return "\n".join(head + new_middle + tail)
    return script


def _repair_cadence(script: str) -> str:
    """Give the script the rhythm contrast the gate asks for, without a retry.

    The observed failure is "missing a longer, flowing sentence (≥15 words)",
    and it is the prompt's own fault: DELIVERY tells the model to "split it into
    short sentences instead", then this gate rejects the result for having only
    short sentences. The prompt now states the rhythm requirement too, but a
    model that still lands short should not cost a generation to fix — joining
    sentences with a comma is exactly the edit a human editor would make.

    A RUN, NOT A PAIR. Joining exactly two was measured declining on the very
    shape the prompt produces: six sentences of six words each pair to twelve,
    below the fifteen the gate wants, so the repair refused and the attempt was
    spent. Three of them make eighteen. The run grows until the total lands in
    range, which is the same edit, continued.

    And when what is missing is the SHORT sentence instead, the fix is the
    opposite cut — see _split_for_punch.
    """
    if "punchy" in (_cadence_violation(script) or ""):
        return _split_for_punch(script)

    head, middle, tail = _protected_lines(script)
    if not middle:
        return script

    def _joinable(parts: list[str]) -> str | None:
        """One flowing sentence from a run of short ones, or None."""
        if len(parts) < 2 or not all(parts):
            return None
        n = sum(len(p.split()) for p in parts)
        # Below 15 it does not satisfy the gate; above 26 the "fix" is a run-on
        # worse than the violation, and DELIVERY explicitly forbids those.
        if n < 15 or n > 26:
            return None
        # NEVER join across a question or an exclamation. Stripping that mark
        # destroys the sentence: a live run produced
        #   "But why does it still hold immense value, during global financial
        #    crises, investors turned to the stability of the pound sterling."
        # from a clean question followed by a clean statement, and that
        # ungrammatical line went into the narration. The mark is also the
        # pacing — DELIVERY says punctuation is how the voice is heard, and a
        # rhetorical question is the one beat whose punctuation is the point.
        if not all(p.rstrip().endswith(".") for p in parts[:-1]):
            return None
        # EVERY part but the last loses its full stop, not just the first. A
        # three-sentence run joined with the middle periods left in place reads
        # back as three sentences again — _SENTENCE_RE splits on them — so the
        # "long sentence" the gate is looking for is never there and the repair
        # spends itself for nothing.
        out = parts[0].rstrip(".")
        for k, p in enumerate(parts[1:], start=1):
            piece = p if k == len(parts) - 1 else p.rstrip(".")
            out = f"{out}, {piece[0].lower()}{piece[1:]}"
        return out

    def _join(a: str, b: str) -> str | None:
        return _joinable([a, b])
        # NEVER join across a question or an exclamation. Stripping that mark
        # destroys the sentence: a live run produced
        #   "But why does it still hold immense value, during global financial
        #    crises, investors turned to the stability of the pound sterling."
        # from a clean question followed by a clean statement, and that
        # ungrammatical line went into the narration. The mark is also the
        # pacing — DELIVERY says punctuation is how the voice is heard, and a
        # rhetorical question is the one beat whose punctuation is the point.
        if not a.rstrip().endswith("."):
            return None
        return f"{a.rstrip('.')}, {b[0].lower()}{b[1:]}"

    # Within a single line first — the least disruptive edit available. Shortest
    # run wins, so two sentences are joined before three are.
    for run in (2, 3, 4):
        for i, line in enumerate(middle):
            sentences = [s.strip() for s in _SENTENCE_RE.findall(line) if s.strip()]
            for j in range(len(sentences) - run + 1):
                merged = _joinable(sentences[j:j + run])
                if merged is None:
                    continue
                new_middle = list(middle)
                new_middle[i] = " ".join(sentences[:j] + [merged]
                                         + sentences[j + run:])
                return "\n".join(head + new_middle + tail)

    # Then across two adjacent body lines. These scripts are usually written one
    # beat per line, so the two short sentences that need joining are far more
    # often neighbours across a line break than inside one line.
    for span in (2, 3, 4):
        for i in range(len(middle) - span + 1):
            block = [[s.strip() for s in _SENTENCE_RE.findall(l) if s.strip()]
                     for l in middle[i:i + span]]
            if not all(block):
                continue
            # The tail of the first line, the whole of the ones between, and the
            # head of the last — the sentences that actually sit next to each
            # other across the breaks.
            parts = [block[0][-1]] + [s for b in block[1:-1] for s in b] + [block[-1][0]]
            merged = _joinable(parts)
            if merged is None:
                continue
            new_middle = list(middle)
            new_middle[i] = " ".join(block[0][:-1] + [merged] + block[-1][1:])
            del new_middle[i + 1:i + span]
            return "\n".join(head + new_middle + tail)
    return script


def _cadence_violation(script: str) -> str | None:
    """Pattern-interrupt check: a script whose sentences are all a similar
    length reads as monotone/algorithmic even with perfect content and a
    passing avg-sentence-length check (min/max avg says nothing about
    VARIETY — five 10-word sentences average the same as one 4-word and one
    16-word sentence). Requires at least one short, punchy sentence
    (≤6 words — the hook line itself usually already provides this) AND one
    longer, flowing one (≥15 words) somewhere in the body. Deliberately
    loose: not every sentence needs to hit these, just some contrast must
    exist somewhere in the script."""
    sentences = [s.strip() for s in _SENTENCE_RE.findall(script) if s.strip()]
    if len(sentences) < 3:
        return None   # too short a script to meaningfully judge rhythm
    lengths = [len(s.split()) for s in sentences]
    has_short = any(n <= 6 for n in lengths)
    has_long  = any(n >= 15 for n in lengths)
    if has_short and has_long:
        return None
    missing = []
    if not has_short:
        missing.append("a short, punchy sentence (≤6 words)")
    if not has_long:
        missing.append("a longer, flowing sentence (≥15 words)")
    return f"cadence: missing {' and '.join(missing)} — every sentence is a similar length"


# ── Pre-analysis ────────────────────────────────────────────────────────────────

def _pre_analyze(client: OpenAI, seed: dict, scene: str, run_id: str,
                 niche: str, trending_context: str = "") -> tuple[str, float]:
    """Cheap pre-pass: extract hook angle + structural cues. Returns (text, cost)."""
    is_wisdom = seed and seed.get("type") == "wisdom"
    is_rss    = seed and seed.get("type") == "rss"
    model     = _standards()["models"]["pre_analyze"]

    trending_note = (
        f"\nCURRENTLY TRENDING (Google Trends, past 7 days): {trending_context}\n"
        "Reference a trending term in the hook IF it fits naturally — don't force it.\n"
        if trending_context else ""
    )

    if is_wisdom:
        quote  = seed.get("content", "")
        author = seed.get("source", "Unknown")
        prompt = (
            f"QUOTE: \"{quote}\"\nAUTHOR: {author}\n{trending_note}\n"
            "You are a historical researcher finding hook material for a 40-second YouTube Short.\n\n"
            "1. BIOGRAPHICAL FACT: One real, verifiable fact about this person's life that PROVES the quote through their actions. "
            "Must be concrete — include a number, year, event, or documented outcome.\n"
            "2. BEHAVIOR CONDEMNED: What specific thing do most people do that this quote calls wrong? One sentence.\n"
            "3. PARADOX: 'Most people [X]. This quote reveals [Y instead].' Must be counterintuitive.\n"
            "4. EMOTIONAL STAKES: One sentence — what does someone lose, fear, or regret if they never understand this quote?\n"
            "5. HOOK ANGLE: One ≤8-word seed phrase that leads with the BIOGRAPHICAL FACT — not the quote text.\n"
            "6. LOOP ANGLE: One question for the second-to-last line that makes viewers want to replay from line 1.\n"
            "7. VIDEO QUERIES: 3 comma-separated stock footage search terms that visually match the hook angle "
            "(e.g. for a 2008 crisis hook: 'stock market crash, trading floor panic, financial chart red').\n"
            "8. SENSORY ANCHOR: One physical sensation, concrete image, or specific sound from this person's "
            "experience — something a viewer can feel in their body. "
            "Not 'it was stressful' → 'the 2am phone call, hands shaking, $2.4M in margin calls'.\n\n"
            "Reply ONLY with these 8 numbered items. No full script."
        )
        max_toks = 320
    elif is_rss:
        seed_blk = _seed_block(seed)
        prompt = (
            f"{seed_blk}\n{trending_note}"
            "You are a news analyst finding the most surprising hook for a 40-second YouTube Short.\n\n"
            "1. SURPRISE: The single most counterintuitive or shocking fact in this news story. One sentence.\n"
            "2. HOOK ANGLE: One ≤8-word seed phrase. Lead with the number/name/surprise — not the headline.\n"
            "3. HUMAN ANGLE: Who is affected and how? One sentence with a specific person, company, or group.\n"
            "4. EMOTIONAL STAKES: What does the average viewer lose, fear, or gain from this story? One sentence.\n"
            "5. CONCRETE DETAIL: The most specific, vivid detail (exact dollar amount, date, percentage, or name).\n"
            "6. LOOP ANGLE: One question for the second-to-last line that makes viewers want to rewatch from line 1.\n"
            "7. VIDEO QUERIES: 3 comma-separated stock footage search terms that visually match the hook angle.\n"
            "8. SENSORY ANCHOR: One concrete physical image or moment from this story a viewer can visualize instantly.\n\n"
            "Reply with ONLY these 8 numbered items. No full script."
        )
        max_toks = 280
    else:
        seed_blk = _seed_block(seed) if seed else f"Scene description: {scene}"
        prompt = (
            f"{seed_blk}\nBackground scene: {scene}\n{trending_note}\n"
            "Before writing the script, find these things in the source. Use REAL details — no invention.\n\n"
            "1. CONTRADICTION: One sentence — the surprising paradox in this source.\n"
            "2. HOOK ANGLE: One ≤8-word seed phrase. Lead with the number/name/contradiction. "
            "Wrong: 'Spain received a lot of silver.' "
            "Right: 'Spain's silver made Spain poorer.'\n"
            "3. CORE: One sentence — the insight this source proves.\n"
            "4. EMOTIONAL STAKES: One sentence — what does the average person lose or fear if they ignore this insight?\n"
            "5. CONCRETE DETAIL: The single most specific, vivid detail from the source (a number, name, date, or documented outcome).\n"
            "6. LOOP ANGLE: One question for the second-to-last line.\n"
            "7. VIDEO QUERIES: 3 comma-separated stock footage search terms that visually match the hook angle "
            "(e.g. for a coin-debasement story: 'silver coins, mint press, market stall').\n"
            "8. SENSORY ANCHOR: One physical sensation, concrete image, or specific moment from this source that "
            "a viewer can feel in their body. Not abstract emotions — a specific scene. "
            "Example: 'the shopkeeper biting the coin, then setting it aside'.\n\n"
            "THE EXAMPLES ABOVE ARE SHAPES. Their words, numbers and subjects must "
            "not appear in your answer — take only the form. This is not a "
            "stylistic nicety: the previous version of this prompt used a "
            "personal-finance example, \"$2.4M by 38, still scared to retire\", "
            "and the model returned that exact phrase as the HOOK ANGLE for a "
            "Hungarian hyperinflation article AND for a 1901 Iowa newspaper "
            "page, dragging both analyses to modern retirement anxiety — a "
            "subject neither source mentions and this channel is not about. "
            "Every one of the 8 items must come from THIS source.\n\n"
            "Reply with ONLY these 8 numbered items. No full script."
        )
        max_toks = 280

    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_toks,
            timeout=90,
        )
        ms = int((time.time() - t0) * 1000)
        usage = resp.usage
        cost = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        text = resp.choices[0].message.content.strip()
        log_attempt({
            "run_id": run_id, "niche": niche,
            "seed_type": seed.get("type") if seed else None,
            "phase": "pre_analyze", "attempt_n": 0,
            "model": model, "temperature": 0.3,
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "cost_usd": cost, "ms": ms, "body": text, "accepted": True,
        })
        return text, cost
    except Exception as e:
        print(f"[gpt] pre-analyze failed: {e}")
        return "", 0.0


# ── Phase A: Hook factory ───────────────────────────────────────────────────────

def _hook_styles_block(niche_cfg: dict) -> str:
    """The niche's declared hook_styles, finally sent somewhere.

    money_history has asked for counterintuitive / shocking_stat / warning
    since it was written, and a grep for "hook_styles" across every script in
    this repo returned nothing at all. A niche has been declaring how it wants
    to open its videos into a void.

    Descriptions rather than bare slugs: "warning" alone tells the model
    nothing it does not already imagine, and half of these names mean something
    slightly different to a language model than they do to whoever typed them.
    """
    styles = [str(x).strip() for x in (niche_cfg.get("hook_styles") or []) if str(x).strip()]
    if not styles:
        return ""
    known = {
        "counterintuitive": "the thing everyone believes, shown backwards by the source",
        "shocking_stat":    "one figure from the source that should not be possible",
        "warning":          "the pattern is running again now, and the viewer is inside it",
        "scene":            "drop the viewer into a real documented moment, second person",
        "cost":             "what it actually took from someone real, stated cold",
        "question":         "a question about the source's facts the viewer cannot not answer",
    }
    lines = ["THIS CHANNEL'S HOOK STYLES — bias the candidates toward these:"]
    for st in styles:
        lines.append(f"- {st}: {known.get(st, 'as the name suggests')}")
    lines.append("Cover more than one of them across the set; eight variations of "
                 "a single style is one hook with eight haircuts.\n\n")
    return "\n".join(lines)


def _hook_factory(client: OpenAI, seed: dict, analysis: str, niche_name: str,
                  niche_cfg: dict, run_id: str, temperature: float = None,
                  model: str = None) -> tuple[list[str], float]:
    """Generate N hook candidates as a numbered list. Returns (hooks, cost_usd)."""
    std       = _standards()
    n_hooks   = std["scoring"]["hook_candidates"]
    temperature = temperature if temperature is not None else std["scoring"]["hook_temperature"]
    model     = model or std["models"]["hook_gen"]
    hs        = std["hook"]
    seed_blk  = _seed_block(seed) if seed else ""

    # EVERY forbidden opener, not the first eight. The gate rejects all
    # twenty-one and the generator was shown eight, so 'what if' (index 8) and
    # 'stop' (index 14) were rules nothing had ever told it about — five
    # rejections in one rejection log, each one a wasted candidate, each one
    # the model obeying instructions it was given while breaking instructions
    # it was not. Twenty-one short strings cost nothing in a prompt; a hidden
    # rule costs a generation every time it fires.
    forbidden_str = ", ".join(f"'{x}'" for x in hs["forbidden_openers"])
    novelty_blk   = _novelty_block(niche_name)
    styles_blk    = _hook_styles_block(niche_cfg)
    # The SAME corpus the gate will check against — see grounding_corpus.
    numbers_blk   = _allowed_numbers_block(grounding_corpus(seed, analysis))

    prompt = (
        f"{seed_blk}\n"
        f"NICHE: {niche_name}\n"
        f"PRE-ANALYSIS:\n{analysis}\n\n"
        f"{novelty_blk}"
        f"{styles_blk}"
        f"Generate exactly {n_hooks} numbered HOOK LINES for a YouTube Short.\n\n"
        f"HOOK RULES — every line must obey ALL of these:\n"
        f"- Length: {hs['min_words']}–{hs['max_words']} words (HARD CAP {hs['hard_max_words']})\n"
        f"- Must contain at least one of: a number, a dollar amount, a proper noun (real person/place), or a year\n"
        f"- GROUNDING (NON-NEGOTIABLE): every number, name, date, and claim must come "
        f"straight from the SOURCE / PRE-ANALYSIS above. NEVER invent a statistic, "
        f"dollar amount, 'secret', or personal story — a fabricated hook fails the "
        f"downstream fact-check and kills the whole video. NEVER write in first "
        f"person ('I', 'my', 'we') — this is a faceless channel with no narrator persona.\n"
        f"{numbers_blk}"
        f"- Must create an itch, not file a report. Any ONE of these does it: it\n"
        f"  contradicts what the viewer believes, OR it drops them inside a real\n"
        f"  documented moment in second person ('In 1560 England, you'd spend the\n"
        f"  worst coin first'), OR it names what something actually cost someone\n"
        f"  real. These are equally good — see the eight angles below.\n"
        f"- Must NOT start with any of: {forbidden_str}\n"
        f"- Must NOT use vague generalities — every word earns its place\n\n"
        # THE EXAMPLES CARRY NO DIGITS ANY MORE. They used to, and the model
        # copied them: '2,000 years ago, Seneca described your anxiety
        # exactly.' produced '2,000 years ago, Rome faced inflation', '2,000
        # years later, inflation still haunts us', 'Inflation has shaped
        # economies for over 2,300 years' and 'Inflation has existed for over
        # 2,300 years' — four rejections across three different runs, every
        # one for a number that came from this prompt rather than from any
        # source. The same leak sent a Reddit-era '$2.4M by 38' into the hook
        # angle of an article about the pengő. An example is a SHAPE; the
        # instant it contains a concrete figure it becomes a suggestion, and
        # the grounding gate is downstream of the suggestion.
        f"Each of the {n_hooks} hooks should attack the source from a DIFFERENT angle. "
        f"The examples below show SHAPE ONLY — they deliberately contain no "
        f"figures, because any number you take from this prompt instead of "
        f"from the source above is an automatic rejection:\n"
        "1. Number-first  — lead with the source's most devastating number. "
        "(the tension is in the contradiction, not the number itself)\n"
        "2. Name-first    — the source's real person/place makes it instantly credible. "
        "(e.g. '<investor>'s worst trade made him a fortune.' — the reversal IS the hook)\n"
        "3. Time-contrast — the source's date reveals how long the pattern has existed. "
        "(e.g. '<ancient writer> described your anxiety exactly.' — the gap creates the itch)\n"
        "4. Identity hit  — names what the viewer is actually doing wrong, using the "
        "source's insight. (blame + specificity + fix)\n"
        "5. Counter-claim — the thing everyone believes that the source shows is backwards. "
        "(must be provable FROM the source)\n"
        "6. Scene-drop    — drop the viewer into a moment the source actually documents, "
        "with its real date/place. (mystery drives completion — but the moment must be real)\n"
        "7. Loaded question — a question about the source's facts the viewer cannot NOT answer. "
        "(specific, counterintuitive)\n"
        "8. Aftermath     — the source's documented consequence, stated cold. "
        "(the verified outcome IS the hook — no embellishment needed)\n\n"
        f"Output FORMAT — exactly one hook per line, numbered 1-{n_hooks}, no commentary:\n"
        f"1. <hook>\n2. <hook>\n...\n{n_hooks}. <hook>"
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=400,
        timeout=60,
    )
    ms    = int((time.time() - t0) * 1000)
    usage = resp.usage
    cost  = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
    raw   = resp.choices[0].message.content.strip()

    hooks: list[str] = []
    for line in raw.split("\n"):
        m = re.match(r"^\s*(\d+)[.):]\s*(.+)$", line.strip())
        if m:
            hook = m.group(2).strip().strip('"').strip("'")
            if hook:
                hooks.append(hook)

    # GPT often ignores the "different angle" instruction and writes 8 variants
    # of one idea — collapse same-shaped hooks so the scorer sees real variety.
    before = len(hooks)
    hooks  = _dedupe_similar_hooks(hooks)
    if len(hooks) < before:
        print(f"[hook] deduped {before - len(hooks)} same-shaped hook(s) → {len(hooks)} distinct")

    log_attempt({
        "run_id": run_id, "niche": niche_name,
        "seed_type": seed.get("type") if seed else None,
        "phase": "hook_gen", "attempt_n": 1,
        "model": model, "temperature": temperature,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cost_usd": cost, "ms": ms,
        "body": raw, "n_hooks_parsed": len(hooks),
    })
    return hooks, cost


# ── Phase B: Hook scorer ────────────────────────────────────────────────────────

def _hook_scorer(client: OpenAI, hooks: list[str], seed: dict, niche_name: str,
                 run_id: str, analysis: str = "") -> tuple[int, float, str, float]:
    """Score hooks (regex pre-filter then LLM). Returns (winner_idx, score, reason, cost)."""
    std       = _standards()
    model     = std["models"]["hook_score"]
    # 1200, not 300: the scorer is asked below whether a hook's contradiction is
    # SUPPORTED by the source, and it cannot answer that from a lead sentence.
    # Live (run #59, Comstock Lode) the deciding fact — "named after Canadian
    # miner Henry Comstock" — sat just past the 300-char cut, so the scorer gave
    # a top score to a hook the source contradicts. Same argument as the
    # architect's 600 → 2500 widening, and the cost is one prompt, once.
    seed_text = (seed.get("content") or "")[:1200] if seed else ""
    # Full grounding corpus for the invented-number check — the whole seed
    # (not the 300-char scoring excerpt) plus the pre-analysis, since a
    # legitimate hook may cite a figure the analysis surfaced from the source.
    grounding = grounding_corpus(seed, analysis)

    # 1. Regex pre-filter
    survivors: list[tuple[int, str]] = []
    for i, h in enumerate(hooks):
        reason = _hook_pre_check(h) or _hook_grounding_check(h, grounding)
        if reason:
            save_attempt(run_id=run_id, niche=niche_name,
                         seed_type=seed.get("type") if seed else None,
                         phase="hook_gen", attempt_n=i + 1,
                         hook=h, rejected_reason=reason, accepted=False)
            log_attempt({
                "run_id": run_id, "niche": niche_name, "phase": "hook_filter",
                "attempt_n": i + 1, "hook": h, "rejected_reason": reason,
                "accepted": False,
            })
        else:
            survivors.append((i, h))

    if len(survivors) < std["scoring"]["min_surviving_hooks"]:
        return -1, 0, f"only {len(survivors)} hook(s) passed regex pre-filter", 0.0

    # 2. LLM scoring (one call, JSON return)
    numbered = "\n".join(f"{i+1}. {h}" for i, (_, h) in enumerate(survivors))
    prompt = (
        f"You are a viral YouTube Shorts editor. Score these hook candidates 0-10 each.\n\n"
        f"SOURCE: \"{seed_text}\"\n\n"
        f"HOOKS:\n{numbered}\n\n"
        "SCORING — two steps, no exceptions:\n\n"
        "STEP 1 — BINARY GATE (if ANY gate fails → maximum score is 3, hard cap):\n"
        "  • Contains a specific number, dollar amount, year, or proper noun (real person/place)?\n"
        "  • Does it do at least ONE of these THREE? All three are equally good.\n"
        "    This gate asks whether the line WORKS, never which shape it picked —\n"
        "    a contradiction is not worth more than the other two, and scoring it\n"
        "    higher is how ten videos in a row all open 'X didn't do Y'. That is a\n"
        "    format, and a format stops being surprising by the third one.\n"
        "      A. CONTRADICTS — states or implies the opposite of common belief.\n"
        "         ('Rome didn't run out of silver.')\n"
        "      B. PUTS THE VIEWER INSIDE A REAL MOMENT — second person and/or\n"
        "         present tense, in a place and time the source documents.\n"
        "         ('In 1560 England, you'd spend the worst coin first.') This is\n"
        "         NOT a contradiction and does not need to be: the itch is 'why\n"
        "         would I do that?' rather than 'wait, is that true?'\n"
        "      C. NAMES WHAT IT COST — the price someone real paid, stated cold.\n"
        "         The stake is the hook. ('Henry the Eighth cut the silver by two\n"
        "         thirds and the country hid its coins in the walls.')\n"
        "  • Is whichever of A/B/C it did ACTUALLY SUPPORTED BY THE SOURCE ABOVE? Read the\n"
        "    source and answer honestly. A hook asserting something the source does\n"
        "    not state — or that the source contradicts — FAILS this gate no matter\n"
        "    how good it sounds — an invented scene and an invented cost fail this\n"
        "    gate exactly as an invented contradiction does. Example of a failure:\n"
        "    source says a silver lode was\n"
        "    \"named after Canadian miner Henry Comstock\" and the hook claims \"Henry\n"
        "    Comstock didn't discover the Comstock Lode\" — the source does not say\n"
        "    that, so the hook is a guess dressed as a revelation. This gate exists\n"
        "    because the gate above REWARDS contradiction, and an unsupported\n"
        "    contradiction is the single most expensive thing you can approve: the\n"
        "    hook cannot be changed later, so the whole script, all its images, its\n"
        "    voiceover and its render get built on it before a fact-check rejects it.\n"
        f"  • ≤{_standards()['hook']['hard_max_words']} words?\n"
        "If all four pass, proceed to Step 2. If any fail → score 1-3 and stop.\n\n"
        "STEP 2 — HOW BADLY DOES THE VIEWER NEED THE NEXT LINE? (all gates passed — score 4-10):\n"
        "  • LOW — mildly interesting; the viewer can put the phone down: 4-6\n"
        "  • MEDIUM — the viewer wants the answer: 7-8\n"
        "  • HIGH — the viewer cannot leave without it: 9-10\n\n"
        "A 9/10 hook leaves the viewer holding one of these and unable to drop it:\n"
        "  'wait, is that actually true?'  (A)\n"
        "  'why would I have done that?'   (B)\n"
        "  'what did that cost him?'       (C)\n"
        "Judge the PULL, not the shape. A 7/10 is solid. A 5/10 fails. A 3/10 gets cut.\n\n"
        "Reply ONLY with this JSON array, one object per hook, in order:\n"
        '[{"i": 1, "score": 0-10, "reason": "one-sentence citing which gate passed/failed or surprise level"}, ...]'
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=600,
        timeout=60,
    )
    ms    = int((time.time() - t0) * 1000)
    usage = resp.usage
    cost  = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
    raw   = resp.choices[0].message.content.strip()

    # Tolerant JSON parse (model sometimes wraps in code fences)
    raw_clean = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        scores = json.loads(raw_clean)
        if isinstance(scores, dict):  # tolerate {"results": [...]}
            scores = scores.get("results") or scores.get("hooks") or (list(scores.values())[0] if scores else [])
    except Exception as e:
        print(f"[hook_score] JSON parse failed: {e} — raw: {raw[:200]}")
        scores = []

    log_attempt({
        "run_id": run_id, "niche": niche_name,
        "seed_type": seed.get("type") if seed else None,
        "phase": "hook_score", "attempt_n": 1,
        "model": model, "temperature": 0.0,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "cost_usd": cost, "ms": ms,
        "body": raw, "n_scored": len(scores),
    })

    # Map back to original index
    best_score = -1
    best_idx   = -1
    best_reason = ""
    for entry in scores:
        try:
            s     = int(entry.get("score", 0))
            i_rel = int(entry.get("i", 0)) - 1
            if 0 <= i_rel < len(survivors):
                orig_i, hook_text = survivors[i_rel]
                save_attempt(run_id=run_id, niche=niche_name,
                             seed_type=seed.get("type") if seed else None,
                             phase="hook_score", attempt_n=orig_i + 1,
                             hook=hook_text, total_score=s,
                             rejected_reason=None,
                             accepted=False, cost_usd=cost / max(len(scores), 1))
                if s > best_score:
                    best_score = s
                    best_idx   = orig_i
                    best_reason = str(entry.get("reason", ""))[:200]
        except Exception:
            continue

    if best_idx == -1:
        if not survivors:
            return -1, 0, "no scoring entries and no survivors to fall back to", 0.0
        # Random survivor, not survivors[0] — a deterministic fallback would
        # systematically bias toward whichever hook shape passes regex first.
        best_idx   = random.choice(survivors)[0]
        best_score = 0
        best_reason = "all scoring entries malformed — fell back to random survivor"

    return best_idx, best_score, best_reason, cost


# ── Story architect: plan before prose ────────────────────────────────────────

# THE SCENE from the most recent architect plan — the one filmable moment the
# script was built to land on. A module global rather than a return value
# because _build_sd_prompts is several call sites away from here and this is
# diagnostic/advisory context, not a result anyone writes from. Same pattern as
# audio_gen.LAST_CUTS.
LAST_SCENE: str = ""


def scene_from_plan(plan: str) -> str:
    """The THE SCENE line out of an architect plan, or "" if there isn't one.

    "" covers both a plan from before this field existed and an honest NONE
    from a source with no filmable moment in it — the storyboard simply gets no
    anchor, which is exactly how it behaved before.
    """
    if not plan:
        return ""
    for raw in plan.splitlines():
        line = raw.strip().lstrip("*# ").strip()
        if not line.upper().startswith("THE SCENE"):
            continue
        _, _, value = line.partition(":")
        value = value.strip().strip("*").strip()
        return "" if value.upper().startswith("NONE") else value
    return ""


# A scene that ends by telling you what it MEANS. Exactly the defect
# storyboard._ABSTRACTION_TAIL strips out of shot descriptions, one stage
# earlier and doing more damage: the storyboard can drop a tail off a picture,
# but a scene whose last clause is a comment was never a moment to begin with.
_SCENE_COMMENT_TAIL = re.compile(
    r",\s*(?:showcasing|highlighting|reflecting|demonstrating|illustrating|"
    r"underscoring|symbolizing|symbolising|signifying|representing|"
    r"emphasizing|emphasising|revealing|marking|proving|showing)\b",
    re.IGNORECASE,
)

# Capitalised words that are dates or grammar, not somewhere a camera can stand.
_NOT_A_PROPER_PLACE = {
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December", "Monday", "Tuesday",
    "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "The", "This",
    "That", "These", "Those", "There", "Then", "When", "While", "During",
    "After", "Before", "None",
}


# A HYPOTHETICAL DRESSED AS A SCENE. The failure, verbatim, on a channel whose
# whole promise is financial HISTORY: the seed was Wikipedia's "Social cost" —
# a concept article with no event in it — and the architect answered
#
#     THE SCENE: A lemonade stand on a summer day where a child mixes lemons,
#                sugar, and water to sell lemonade.
#
# which became a whole video about an imaginary child. The owner's verdict was
# blunt and right: the clip talks about a concept with nothing historical or
# financial attached to it. A parable is what a model reaches for when the
# source has no moment in it, and it is worse than admitting NONE, because a
# fabricated scene reads fluent enough to survive every downstream gate.
_HYPOTHETICAL = re.compile(
    r"\b(imagine|suppose|picture (?:a|an|yourself)|think of|consider a|"
    r"for example|say you|let's say|hypothetical|thought experiment|"
    r"a typical|an average|a simple example)\b", re.IGNORECASE)

# A year. On a history channel a moment without one is not a moment — it is an
# illustration. Matches 1893, 991, 1016 AD, 44 BC, the 1930s.
_HAS_YEAR = re.compile(r"\b\d{3,4}s?\b|\b\d{1,4}\s*(?:BC|BCE|AD|CE)\b")


def _wants_a_date(niche: str) -> bool:
    """Whether this niche's scenes have to be pinned to a time.

    money_history sells one thing: something that actually happened, with a
    date on it. motivation and mindset do not — a scene there is a person at a
    desk at 6am, and demanding a year would reject every correct answer. So
    this asks the niche rather than applying one rule to all of them.
    """
    return "history" in (niche or "").lower()


def scene_weakness(scene: str, niche: str = "") -> str | None:
    """Why THE SCENE is not yet a filmable moment, or None if it is.

    ADVISORY, NEVER A REJECTION — see AGENTS.md on the rejection ladder. The
    architect's own prompt already asks for "a date or year, a place, and ONE
    named person doing ONE specific thing"; nothing checked that it listened,
    and the difference between a run that works and one that does not is
    visible in that single line before a cent is spent on prose:

        strong: "February 20, 1893, in Philadelphia — workers gather outside
                 the Philadelphia and Reading Railroad office, anxiously
                 watching as receivers are appointed."
        weak:   "In 2022, traders exchanged billions of pounds in currency
                 markets, showcasing sterling's trading activity."

    The weak one produced "The secret? Its historical resilience and trust."
    after three full script cycles. "Currency markets" is a category, not a
    room; nothing in it is anywhere, and the trailing "showcasing…" is the
    plan admitting it has a topic rather than an event.

    An empty scene is not weak — the architect is told to write NONE when the
    source genuinely has no moment in it, and that honesty is worth keeping.
    """
    if not scene:
        return None

    faults = []
    named = [m.group(0) for m in re.finditer(r"\b[A-Z][a-z]{2,}\b", scene)
             if m.start() > 0 and m.group(0) not in _NOT_A_PROPER_PLACE]
    if not named:
        faults.append("no named place or person — a category is not somewhere "
                      "a camera can stand")
    if _SCENE_COMMENT_TAIL.search(scene):
        faults.append("it ends on what the moment MEANS instead of what "
                      "happens in it")
    if _HYPOTHETICAL.search(scene):
        faults.append("it is a hypothetical, not something that happened")
    if _wants_a_date(niche) and not _HAS_YEAR.search(scene):
        faults.append("no year or date — on a history channel a moment "
                      "without a time is an illustration, not an event")
    return "; ".join(faults) or None
# One cheap pass BEFORE any drafting: pins down the single most compelling,
# source-grounded angle, the exact moment the reversal should hinge on, and
# why THIS telling matters right now — instead of Phase C writing blind from
# raw pre-analysis and hoping a good shape falls out. Feeds every attempt (not
# just the first), so retries have a real plan to hew to, not just corrections.
# RUFUS_SCRIPT_ARCHITECT=0 disables (fail-open — a plan-less run just writes
# exactly as before).

# THE LAST PLAN'S UNFIXED SCENE PROBLEM, for the caller to act on. A module
# global rather than a return value because _story_architect is three calls
# deep inside write_script and every caller in the tree shares that signature —
# this is a diagnosis to hold an upload on, not a result anyone writes from.
# Reflects the LAST plan built, which is the one the shipped script was written
# to.
LAST_SCENE_WEAKNESS: str = ""


def _architect_enabled() -> bool:
    return os.environ.get("RUFUS_SCRIPT_ARCHITECT", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _story_architect(client: OpenAI, seed: dict, analysis: str, hook: str,
                     run_id: str, niche_name: str) -> tuple[str, float]:
    """Returns (plan_text, cost_usd) — a plan already checked against the
    source, not just written with a grounding instruction and hoped for.

    Also sets LAST_SCENE_WEAKNESS, cleared on entry so a stale diagnosis from
    an earlier cycle can never hold a later, healthy run.

    WHY THIS CHECK EXISTS: the architect's own prompt already tells it not to
    invent motives, but nothing verified that it listened — its plan went
    straight into the body prompt as "STORY PLAN (write to this shape)", so an
    ungrounded THE TURN or STAKES GAP got faithfully dramatized into prose by
    the body writer, and only the FINAL fact gate caught it — after an entire
    body-generation cycle (up to 3 attempts, ~$0.03-0.04) had already been
    spent writing to a plan that could never pass. Three consecutive live
    runs hit exactly this: every one exhausted all 3 cycles and still shipped
    at the hard-capped 4/10, needing manual review every time.

    Checking the ~130-word plan through the same fact gate that would catch it
    anyway costs about $0.002 — roughly 1/15th of one wasted body cycle — and
    catches it BEFORE that spend, not after. Retries the plan itself (cheap)
    rather than accepting a bad plan and hoping the body writer's own
    grounding instructions save it (they didn't, three times running).
    """
    globals()["LAST_SCENE_WEAKNESS"] = ""
    if not _architect_enabled():
        return "", 0.0
    model = _standards()["models"].get("architect", "gpt-4o-mini")
    # 2500, not 600: the architect is the component that invents the "turn" and
    # the stakes, and a 600-char window (barely the lead paragraph) left it
    # nothing to build from but its own priors — so it wrote plausible-sounding
    # motives the fact gate then rejected, capping the video to 5/10. Now that
    # seeds carry real article bodies (research.WIKI_FULLTEXT_CHARS), give the
    # architect enough of one to find a REAL turn instead of inventing one.
    seed_text = (seed.get("content") or "")[:2500] if seed else ""
    prompt = (
        f"HOOK (already chosen, will not change): \"{hook}\"\n"
        f"SOURCE: \"{seed_text}\"\n"
        f"PRE-ANALYSIS:\n{analysis}\n\n"
        "You are planning a 35-50 second video BEFORE any prose is written. "
        "In under 150 words, reply in exactly 5 short labeled lines:\n"
        "THE SCENE: one moment from the SOURCE a camera could have filmed — a "
        "date or year, a place, and ONE named person doing ONE specific thing. "
        "Not a summary of what they were like, not what they controlled or were "
        "worth: the smallest concrete event the source actually supports. "
        "\"In 1523 Jakob Fugger sent Charles V a letter demanding repayment\" is "
        "a scene. \"Jakob Fugger held two percent of Europe's GDP\" is not — "
        "nobody is doing anything and no camera could point at it. If the source "
        "genuinely contains no such moment, write NONE rather than inventing "
        "one; a missing scene is recoverable, a fabricated one fails the "
        "fact-check and holds the whole video.\n"
        "NEVER ANSWER WITH A PARABLE. When the source is a concept rather than "
        "an event, the tempting answer is an illustration — \"a lemonade stand "
        "on a summer day where a child mixes lemons and sugar\", \"imagine a "
        "farmer weighing his grain\", \"a typical household budget\". That is "
        "the single worst answer available. This channel's whole promise is "
        "that the thing HAPPENED, to real people, on a date; an invented "
        "example delivers a lecture with no history in it and the video is "
        "held. A real event you had to dig for beats a clean hypothetical "
        "every time, and NONE beats both.\n"
        "SPINE FACT: the one specific, source-grounded detail everything else "
        "must hang on — not a theme, an actual fact.\n"
        "THE TURN: the exact moment or fact the reversal should hinge on — a "
        "MOMENT the viewer can picture, not a statistic restated. This must be "
        "a DIRECT CONSEQUENCE of the SPINE FACT, not a separate idea — if it "
        "doesn't follow logically from the spine fact, pick a different turn "
        "that does.\n"
        "STAKES GAP: what does the viewer specifically LOSE by not knowing "
        "this — a concrete cost of staying ignorant of it (a mistake they'd "
        "keep making, a decision they'd get wrong, a belief that's actually "
        "backwards), not a vague 'this matters'.\n"
        "WHY NOW: the single sharpest, most concrete reason a viewer should "
        "care about THIS today — not a generic 'this matters', the real stake.\n"
        "Be concrete. No fluff, no restating the hook.\n"
        "GROUNDING RULE (hard): every one of the 4 lines must be traceable to the "
        "SOURCE/PRE-ANALYSIS above. Do NOT invent motives, secret deals, dollar "
        "figures, or sweeping claims ('reshaped history', 'changed everything') the "
        "source does not actually state — the fact-check will reject the script and "
        "hold the whole video if you do.\n"
        "WHY PEOPLE ACTED IS THE #1 REJECTION CAUSE. Measured across recent runs, "
        "five of eight fact-check failures were an invented MOTIVE, not a wrong "
        "date or figure: 'merely took credit', 'policymakers were scared to act', "
        "'a secret motive for the disbandment', 'silenced by those who feared "
        "inflation'. Sources record what people DID; they almost never record what "
        "people FELT or INTENDED. THE TURN must therefore be an EVENT or an "
        "OUTCOME, never a state of mind. Write 'traders swapped the cheap coin for "
        "the good one until the good one vanished' (observable), not 'traders "
        "schemed to drain the treasury' (mind-reading). If the drama you want "
        "needs a motive the source does not give, you have the wrong turn — find "
        "a real one."
    )
    # 2 attempts, not 3+: this is a cheap pre-check, not the main quality gate
    # (the body still goes through the full fact gate regardless). Two shots
    # at a ~130-word plan is enough to shake loose an invented motive without
    # turning a cost-saving measure into its own expensive loop.
    MAX_PLAN_ATTEMPTS = 2
    total_cost = 0.0
    last_plan, last_reason = "", ""
    # A grounded plan is never thrown away for a weak scene. If the re-ask
    # comes back ungrounded, this is what we fall back to — a true plan with a
    # vague moment beats a vivid invented one every time.
    best_grounded, weak_retry = "", ""

    for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
        this_prompt = prompt
        if attempt > 1 and weak_retry:
            this_prompt += (
                f"\n\nYour previous THE SCENE was not yet a filmable moment: "
                f"{weak_retry}\nEverything else in the plan was fine. Write the "
                f"plan again with a THE SCENE that names WHERE it happens and "
                f"WHO is doing the one thing that happens, both taken from the "
                f"SOURCE. If the source names no place and no person, write "
                f"NONE — do not invent either."
            )
        elif attempt > 1 and last_reason:
            this_prompt += (
                f"\n\nYour previous plan was REJECTED by the fact-checker: "
                f"\"{last_reason}\"\nWrite a new plan that avoids this — stick "
                f"to what the SOURCE literally states, no interpretive leaps."
            )
        try:
            t0 = time.time()
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": this_prompt}],
                temperature=0.6, max_tokens=220, timeout=60)
            ms    = int((time.time() - t0) * 1000)
            usage = resp.usage
            cost  = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
            total_cost += cost
            plan  = resp.choices[0].message.content.strip()
        except Exception as e:
            print(f"[gpt] story architect skipped (non-fatal): {e}")
            return last_plan, total_cost   # fail-open: use whatever we have, even ""

        passed, reason, check_cost = _fact_gate(client, seed, plan)
        total_cost += check_cost
        log_attempt({
            "run_id": run_id, "niche": niche_name, "phase": "story_architect",
            "attempt_n": attempt, "model": model, "cost_usd": cost + check_cost,
            "ms": ms, "body": plan, "accepted": passed,
            "rejected_reason": None if passed else reason,
        })
        if passed:
            best_grounded = best_grounded or plan
            weak = scene_weakness(scene_from_plan(plan), niche_name)
            if not weak:
                globals()["LAST_SCENE_WEAKNESS"] = ""
                return plan, total_cost
            if attempt < MAX_PLAN_ATTEMPTS:
                print(f"[gpt] ⚠ THE SCENE is not filmable yet ({weak}) — "
                      f"re-asking once for a place and a person")
                weak_retry, last_reason = weak, ""
                continue
            # Out of attempts. Loud, because this line decides the run: a scene
            # with nowhere and nobody in it is where a script ends up saying
            # "The secret? Its historical resilience and trust."
            # HOLD THE UPLOAD, do not just warn. This printed twice on a
            # money_history run and the video published-path went ahead
            # anyway: a script about an imaginary child's lemonade stand, on a
            # channel that exists to tell you what actually happened. The
            # warning was right and nothing acted on it, which is this repo's
            # oldest failure mode wearing a new hat.
            globals()["LAST_SCENE_WEAKNESS"] = weak
            print(f"[gpt] ⚠ THE SCENE stayed unfilmable ({weak}) — the upload "
                  f"will be HELD. The seed itself is usually the cause; a "
                  f"source with no moment in it cannot produce one.")
            return plan, total_cost

        print(f"[gpt] story architect attempt {attempt}/{MAX_PLAN_ATTEMPTS} "
              f"ungrounded ({reason}) — {'retrying' if attempt < MAX_PLAN_ATTEMPTS else 'using anyway'}")
        last_plan, last_reason = plan, reason
        weak_retry = ""

    # Exhausted retries: use the last plan anyway rather than blocking the
    # render — the body's OWN fact gate still runs at the end regardless, so
    # this pre-check can only save cost, never be the sole line of defense.
    # A grounded-but-weak plan outranks an ungrounded one: the weak scene costs
    # us a dull video, the ungrounded one costs us a wrong claim.
    return best_grounded or last_plan, total_cost


# ── Phase C: Body generator ─────────────────────────────────────────────────────

def _fix_for_rejection(rejection: str, std: dict, hook_token_str: str,
                       opinion_all: str) -> str:
    """Map a pre-filter rejection reason → a one-line CRITICAL correction for
    the next attempt. A module-level function (not a write_script closure)
    specifically so it's directly unit-testable — extracted while fixing a
    real ordering bug below.

    Order matters: "sentences too long"/"sentences too short" must be
    checked BEFORE the plain "too long"/"too short" (whole-body word count)
    — both contain "too short"/"too long" as a substring, and the generic
    check used to win first, so a script with fine total length but
    overly-short individual sentences got told to "write more words total"
    instead of the actual fix (lengthen sentences)."""
    if rejection.startswith("banned"):
        bad = rejection.split("'")[1] if "'" in rejection else ""
        return f"CRITICAL: '{bad}' is a BANNED phrase — never use it or any variation."
    if "loop no echo" in rejection:
        return (f"CRITICAL: the second-to-last line must echo a word from the hook "
                f"({hook_token_str}).")
    if "opinion word" in rejection:
        return f"CRITICAL: include at least one opinion word: {opinion_all}."
    if "hedging" in rejection:
        bad = rejection.split("'")[1] if "'" in rejection else ""
        return f"CRITICAL: remove '{bad}' — no hedging language."
    if "sentences too long" in rejection:
        return f"CRITICAL: shorten sentences (avg ≤{std['body']['max_avg_sentence_words']} words)."
    if "sentences too short" in rejection:
        return (f"CRITICAL: lengthen sentences (avg ≥{std['body']['min_avg_sentence_words']} "
                f"words) — add detail, don't just add more short sentences.")
    if "cadence" in rejection:
        return ("CRITICAL: vary sentence rhythm — include at least one short, punchy "
                "sentence (4-6 words) AND one longer, flowing sentence (15+ words). "
                "Right now every sentence is roughly the same length, which reads as "
                "monotone even with good content.")
    if "too short" in rejection:
        return f"CRITICAL: write at least {std['body']['min_words']} words total (aim for ~100)."
    if "too long" in rejection:
        return f"CRITICAL: keep it under {std['body']['max_words']} words."
    if "specificity" in rejection:
        return "CRITICAL: add concrete specifics — a name, year, or dollar amount per sentence."
    if rejection.startswith("number"):
        bad = rejection.split("'")[1] if "'" in rejection else ""
        return (f"CRITICAL: you repeated the number '{bad}' more than once — every "
                f"sentence needs a DIFFERENT specific, never the same figure restated.")
    if rejection.startswith("em-dash"):
        return ("CRITICAL: use at most 2 em-dashes in the whole script — replace "
                "the rest with a period, comma, or colon.")
    return ""

def _build_system(niche_cfg: dict, niche_name: str, cta: str, hook: str) -> str:
    gold_examples = _load_gold_examples(niche_name)
    gold_block    = _build_gold_block(gold_examples)
    niche_context = niche_cfg.get("gpt_system", "")
    std           = _standards()
    body          = std["body"]
    banned_all    = ", ".join(f"'{p}'" for p in std["banned_phrases"])
    opinion_all   = ", ".join(std["opinion_pool"])
    hedging_all   = ", ".join(std["hedging_words"])
    tension_hint  = ", ".join(_TENSION_WORDS & set(std["opinion_pool"])) or opinion_all

    # Niche first, channel direction second: the niche says WHAT this channel is
    # about, the direction says HOW the owner wants it made, and the second only
    # makes sense on top of the first.
    direction_blk = _direction_block()

    return f"""You are the most exacting short-form script writer working today.
Your standard: if a line does not earn its place, cut it. If a word is vague, replace it with something specific.

NICHE:
{niche_context}

{direction_blk}
YOUR JOB:
You are given REAL source material and a HOOK that has already been chosen. Write the body of a 35-50 second YouTube Short that delivers on the hook.

VOICE:
- Sound like someone who has been in this field for 20 years and is slightly impatient with people who haven't figured this out.
- Make the viewer FEEL something specific — real fear or real anger at what happened, not just learn a fact. A viewer who finishes the video informed but unmoved is a script that failed, even if every number in it checks out.
- Every beat should carry a little dread or indignation: what SHOULD have scared the people in this story and didn't, what SHOULD outrage the viewer about how it played out. That's the emotional register — not sadness, not inspiration.
- Specific always beats vague. A name beats "someone". A number beats "many". A year beats "recently".
- Short sentences ({body['min_avg_sentence_words']}-{body['max_avg_sentence_words']} words avg). Vary rhythm deliberately.
- Never moralize. Never summarize. Trust the audience.

MOTIVE — THE ONE THING THAT KILLS A FINISHED VIDEO:
- The fact-check runs AFTER this script is written. When it rejects, the score
  is capped and the whole video is held — the images, the voiceover and the
  render are already paid for. Measured across recent runs, FIVE OF EIGHT
  rejections were an invented motive, not a wrong date or figure:
  "Comstock merely took credit", "policymakers were scared to act", "a secret
  motive for the disbandment", "silenced by those who feared inflation more
  than inequality", "implies a conspiracy or intentional suppression".
- The reason is simple: sources record what people DID. They almost never
  record what people FELT, FEARED, or INTENDED. Any sentence explaining WHY
  someone acted is a guess unless the source says so in as many words.
- So: attribute to the OUTCOME, never to the mind.
    ✗ "Traders schemed to drain the treasury."   (mind-reading)
    ✓ "Traders swapped the cheap coin for the good one until the good one
       was gone."                                 (observable, and better TV)
    ✗ "The government hid the collapse."          (motive)
    ✓ "The announcement came after the banks had already closed."  (event)
- DREAM LANGUAGE IS MIND-READING WEARING A COAT, and it is the form that
  actually keeps failing. Three full cycles of one run were burned on it,
  every one rejected for the same thing: "each dreaming of riches", "their
  dreams crushed", "chasing shiny dreams", "hopeful prospectors". Nobody
  filmed a dream. Same for "desperate to", "convinced that", "believing",
  "certain that", "in the hope of", "unaware that".
    ✗ "miners flooded in, each dreaming of riches"
    ✓ "miners flooded in — forty thousand of them in a year"
    ✗ "Marshall left empty-handed, his dreams crushed"
    ✓ "Marshall left with nothing; he died on a small pension"
  The second version of each is shorter, harder, and a camera could have
  filmed it. That is the whole test.
- This does NOT mean writing blandly. Indignation about what HAPPENED is
  wanted. Certainty about what someone was thinking is what gets rejected.
- NAMING THE LIMIT IS A THIRD OPTION, and the one that keeps getting missed.
  Between asserting something the source does not support and leaving it out
  entirely there is a move that is both honest and better television: say
  what is known, then say where the knowing stops.
    ✗ "They kept the fire alive because they knew they could not restart it."
    ✓ "We are not sure how reliably anyone could start a fire from scratch
       then. What we do know is that they carried embers wrapped in leaves."
    ✗ "The paintings were made on rainy days."
    ✓ "We cannot prove the paintings were made on rainy days. But they are
       hundreds of metres in, and nobody crawls that far when the hunting is
       good."
  A viewer trusts a narrator who says "we don't know" and distrusts one who
  never does.
- USE THESE EXACT FORMS, because the hedging gate below rejects the obvious
  alternatives and you would lose the attempt: "we are not sure", "we cannot
  prove", "no source records", "the evidence stops at", "almost certainly",
  "what we do know is". Those are precise about the EVIDENCE.
  Do NOT reach for any of these — every one is an automatic rejection:
  "maybe", "perhaps", "could be", "might be", "kind of", "sort of",
  "I think", "possibly", "probably", "somewhat". They soften the CLAIM
  without telling anyone why, which is the opposite move and the reason the
  ban exists. (The list above is config/script_standards.json's own
  hedging_words, in full — a rule the writer is judged by and never shown is
  a rule it breaks.)

THREE DEVICES THAT DO THE MOST WORK PER WORD. Read off a long-form history
script that holds an audience for eleven minutes; all three fit forty seconds.

- NEGATION THEN CORRECTION. State what a thing was NOT, then what it was. The
  contradiction the hook gate wants, in one sentence and no extra words.
    ✓ "Rain wasn't an inconvenience. It was a crisis."
    ✓ "The coin didn't lose value. It lost silver."
  Two short sentences beat one long one here — the full stop IS the turn.

- THE OBJECT AS PROOF. An abstract claim is an argument; an object somebody
  dug up is evidence, and it is also a picture the storyboard can draw.
    ✗ "Losing your fire was serious."
    ✓ "The 5,300-year-old man found frozen in the Alps was carrying tinder
       fungus, flint and iron pyrite in his belt pouch."
  When the source names a thing that survives — a pouch, a ledger, a wreck, a
  stamped coin — put the thing in the script. It grounds the claim, it passes
  the fact gate, and it gives the pictures something real to be about.

- THE CONSEQUENCE STACK. Two or three short clauses in the same shape,
  escalating, then a hard stop.
    ✓ "You can't hunt. You can't see tracks. You can't move."
    ✓ "The mint closed. The wages stayed. The bread doubled."
  This is the one place repetition is wanted: the repeated shape is what
  makes the last item land. Do not use it more than once in a script.

NUMBERS ARE SPOKEN, NOT PRINTED:
- A voice engine reads every digit you write. "$4,210,500,000,000 for one
  dollar?" comes out as forty syllables of numerals — unfollowable, and the
  single clearest signal to a viewer that a machine wrote this. That exact
  line shipped in a real video of this channel.
- Write the figure the way a person SAYS it: "four point two trillion marks
  for a single dollar". Rounding to two or three significant figures is not
  a loss of accuracy — nobody parses 4,210,500,000,000 while watching, and
  the grounding check accepts a correct rounding of a source number.
- Same rule for every other unspeakable form:
    ₹15.3 lakh crore  ->  "fifteen point three lakh crore" (or "almost all
      of it", when the exact figure is not the point)
    99.3%             ->  "ninety-nine point three percent", or better,
      "all but a fraction of it came back"
    1:15.5            ->  "fifteen and a half ounces of silver to one of gold"
- Give a number its UNIT and its MEANING in the same breath. "156 billion
  marks" is a quantity; "a debt the size of the entire prewar economy" is a
  fact the ear can hold.
- ONE big number per script, at most. A second one cancels the first — the
  viewer stops counting and starts skipping.

WHERE THE FEELING ACTUALLY COMES FROM:
- The instinct, when a script feels dry, is to reach for adjectives
  ("devastating", "shocking") or for someone's state of mind ("they were
  terrified"). Both are weak, and the second one gets the video held.
- Feeling comes from a PHYSICAL CONSEQUENCE LANDING ON ONE PERSON. Compare
  two lines from real scripts of this channel:
      "Policymakers were scared to act."
        — a claim about invisible minds. Rejected, video held.
      "People carted wheelbarrows of worthless notes to the shops — and
       still went home hungry."
        — nothing but observable fact, and it hits ten times harder.
  Same intent. One is guessing, the other is showing.
- So when you want the viewer to FEEL the collapse: what did it cost someone
  to carry, to queue for, to hand over, to go without? Money that stops
  working is a person walking home with nothing. Write that.
- Three levers, all of them factual:
    SCALE made physical — not "hyperinflation was extreme" but "a loaf cost
      more than a house had the year before".
    THE SMALL DETAIL — the wheelbarrow, the wall the coins were hidden in,
      the queue going round the block. One concrete object beats a paragraph.
    THE REVERSAL STATED FLATLY — "Same coin. Same face. Same name." Rhythm
      and restraint carry more weight than an adjective ever will.
- Understatement outperforms emphasis here. The facts of this niche are
  already extreme; your job is to place them, not to sell them.

SOUND — THIS IS HEARD, NOT READ:
- Every line gets spoken aloud by a voice engine. Write what a person SAYS, not
  what an encyclopedia prints. "Rome ran out of silver" is speech. "The
  debasement represented a transformation in monetary policy" is print.
- BANNED — abstract nouns built from verbs: transformation, evolution,
  implementation, development, consideration, significance, importance,
  relevance, emergence, adoption, expansion, transition, progression. Use the
  VERB instead: not "the shekel's transformation from weight to coin" but
  "the shekel stopped being a weight and became a coin". This single swap is
  the difference between a script that sounds alive and one that sounds like
  a textbook read aloud.
- Talk TO the viewer at least once. Use "you" or "your" somewhere in the body.
  Not "people ignored these gaps" — "you've held one of these and never
  thought about it". A script with no "you" in it is a lecture.
- NEVER narrate the video's own purpose. Lines like "ignoring this would mean
  missing how X shaped Y", "this is a truth still relevant today", or "this
  matters because" are commentary ABOUT the script, not the script. Show the
  thing and stop. The viewer draws the conclusion.
- Contractions and fragments are good: "It wasn't." "Not even close." "Then
  the money stopped working." They are how people actually talk.

STRUCTURE — 3-BEAT ARC, NON-NEGOTIABLE:
LINE 1 (HOOK): USE EXACTLY THIS LINE, DO NOT REWRITE OR REPHRASE IT:
"{hook}"

BEAT 1 — SETUP (lines 2-3): Establish the REAL situation with one specific fact (number, name, date). Make it feel LIVED — a concrete detail, not a description. The viewer should see the scene, not just hear a summary.

BEAT 2 — TURN (lines 4-5): The gut-punch reversal that changes everything the viewer just assumed.
  • Start with "But" or "Until" or "Then" — these words signal the turn to the viewer's brain.
  • Use SENSORY or SCENE language: what did it LOOK like, what did it FEEL like at that moment.
  • This is NOT a statistic. It is a MOMENT. "Then the bank called." "But the statement showed $11." "Until the day his wife found the account."
  • The tension here is what makes viewers replay the video. Make it land hard.

BEAT 3 — PAYOFF (lines 6-7): Reveal the mechanism — WHY the turn happened, what pattern it proves. Give the viewer the "click" moment where they understand something they've been living without knowing. No advice. Show the truth, then step back.

BODY — TARGET {body['min_words'] + (body['max_words'] - body['min_words']) * 2 // 3} WORDS total including hook and CTA.
{body['max_words']} is a HARD CAP that fails the script; do not write up to it, write to the target.
- Every sentence either adds evidence or builds tension. No filler.
- Use specific names, numbers, dates, dollar amounts. At least one specific per 25 words.
- OPINION WORD (required): body must contain at least one of these exact words: {opinion_all}
  Prefer a FEAR/ANGER-coded one where the moment genuinely earns it — {tension_hint} —
  over a milder one like "best" or "real"; the milder ones exist for beats that
  aren't emotionally charged, not as the default choice.

DELIVERY (this is read aloud by TTS, not just read on screen):
- Punctuation IS the pacing — the voice engine pauses longer after ".", "?", "!",
  and dramatic beats ("—" or "..."), and only briefly after ",". Choose punctuation
  for how the line should be HEARD, not just grammar.
- Put a dash or ellipsis right before the turn/reveal so the pause lands like a
  breath before the punch — not mid-sentence filler.
- Never write a long comma-chained run-on when the moment deserves a hard stop.
  Split it into short sentences instead.

RHYTHM (required — a script of same-length sentences reads as machine-written):
- At least ONE short, punchy sentence of 6 words or fewer. The hook usually is one.
- At least ONE longer, flowing sentence of 15 words or more, somewhere in the body.
- These two coexist with the rule above: hard stops at the MOMENTS, one sustained
  sentence where the explanation genuinely needs room to breathe. Contrast between
  sentence lengths is the point; every sentence being short is its own monotony.

SECOND-TO-LAST LINE (LOOP):
A question or restatement that SHARES AT LEAST ONE CONTENT WORD with the hook. This drives replays.

LAST LINE (CTA): Always exactly this, on its own line:
"{cta}"

ANTI-HALLUCINATION:
Never invent: a person's first name, dollar amount, percentage, date, or company event not in the source. For wisdom seeds, well-documented historical facts (S&P returns, named historical figures' biographies) ARE allowed — they illustrate the quote.

BANNED PHRASES — every one of these causes automatic rejection, no exceptions:
{banned_all}

HEDGING — never use any of: {hedging_all}

Output ONLY the script text. No labels. No "Here is the script:". No quotes around it.
{gold_block}"""


def _fixes_from_crits(crits: dict, std: dict, opinion_all: str,
                      reasoning: str = "") -> list[str]:
    """Turn a low LLM score into concrete corrections for the NEXT attempt.

    Real gap this closes: _fix_for() already converts a pre-filter rejection
    (banned phrase, no loop echo, etc.) into a specific instruction carried
    into every later attempt — but a LOW LLM SCORE previously only added a
    compact numeric summary ("spec=1, hook=1, loop=0") to the retry prompt,
    with no actual correction. So a weak attempt just retried "cold" at a
    different temperature instead of fixing the specific flaw the critic
    found — the likely reason scores swing hard between videos (10 on one
    seed, 5 on the next) even when the pre-filter never rejected anything.
    Mirrors _fix_for's style so both paths carry equal weight in the prompt."""
    fixes = []
    body = std["body"]
    if crits.get("specificity", 3) < 2:
        fixes.append("CRITICAL: ground EVERY claim in a real number, name, or date "
                     "from the source — the critic found this too vague/unsupported.")
    if crits.get("hook", 2) < 2:
        fixes.append("CRITICAL: the body must directly pay off the hook's specific "
                     "claim by the loop line — the critic found it unresolved.")
    if crits.get("compression", 2) < 2:
        fixes.append(f"CRITICAL: cut padding — every sentence must reveal a fact or "
                     f"build tension, avg sentence length under "
                     f"{body['max_avg_sentence_words']} words.")
    if crits.get("loop", 2) < 2:
        fixes.append("CRITICAL: the second-to-last line must structurally mirror the "
                     "hook (echo one of its concrete words), not just share a theme.")
    if crits.get("human", 2) < 2:
        fixes.append(
            "CRITICAL: the viewer has to be IN this and has to feel it. Put 'you' "
            "in the body doing something, in present tense. Give one real person "
            "something to lose and let it cost them. And put a TURN in the middle "
            "— one line where what the viewer assumed becomes what was never "
            f"true. Opinion words ({opinion_all}) help but do not replace those "
            "three; a narrator with strong adjectives is still a narrator. "
            "The turn and the cost must both come from the source: do NOT invent "
            "why anyone acted — no 'they were afraid', no 'it was buried', no "
            "attributed motive the source does not state. That is the single most "
            "expensive rejection in this pipeline, because it is caught after the "
            "whole video is rendered.")
    # The disqualifier list isn't parsed into structured criteria like
    # SPECIFICITY/HOOK/etc. — it's echoed in the raw reasoning text (the
    # scorer prompt asks for "DISQUALIFIERS: [list, or 'none']"), so this
    # checks the text directly rather than a crits dict key.
    if "sensory" in (reasoning or "").lower():
        fixes.append("CRITICAL: put a concrete sensory/physical detail — something "
                     "a viewer could see, hear, feel, smell, or taste — in the "
                     "FIRST THIRD of the body, right after the hook. The critic "
                     "found the setup too abstract this early, nothing to "
                     "picture before the viewer decides whether to keep watching.")
    return fixes


def _generate(client: OpenAI, system: str, user: str, model: str,
              temperature: float) -> tuple[str, float, int, int, int]:
    """Returns (text, cost, ms, prompt_toks, completion_toks)."""
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=500,
        timeout=90,
    )
    ms    = int((time.time() - t0) * 1000)
    usage = resp.usage
    cost  = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
    return resp.choices[0].message.content.strip(), cost, ms, usage.prompt_tokens, usage.completion_tokens


# ── Body scorer ─────────────────────────────────────────────────────────────────

def _is_long_form() -> bool:
    """Which shape of video is being scored. Never raises — a scorer that
    cannot import the profile scores a Short, which is the default format."""
    try:
        import video_format
        return video_format.is_long()
    except Exception:
        return False


# PAYOFF is the long-form name for the LOOP criterion and is stored under the
# same key. The axis really is one axis — did the ending close the circle the
# opening drew — but the DEVICE is different, and asking a nine-minute
# explainer for a Shorts loop line is asking it for something its own writer is
# told not to produce. One key, so db_manager's column and _fixes_from_crits
# keep working and the two formats stay comparable in the review queue.
_CRIT_RE = re.compile(
    r"^(SPECIFICITY|HOOK|COMPRESSION|LOOP|PAYOFF|HUMAN|TOTAL):\s*(\d+)",
    re.MULTILINE | re.IGNORECASE,
)


def _score(client: OpenAI, script: str, seed: dict, hook: str, run_id: str,
           niche: str) -> tuple[int, dict, str, float, int]:
    """LLM rubric scoring. Returns (total, crits_dict, reasoning, cost, ms)."""
    std       = _standards()
    model     = std["models"]["body_score"]
    seed_text = (seed.get("content", "") or "")[:500] if seed else ""
    seed_type = (seed.get("type") or "unknown") if seed else "unknown"
    is_wisdom = seed_type == "wisdom"

    invented_disqualifier = (
        "□ Script invents fictional characters or made-up people not verifiable as real historical figures\n"
        "   (For this quote-based seed, well-documented historical facts are EXPECTED.)\n"
        if is_wisdom else
        "□ Script invents a person, dollar amount, percentage, or date not present in the source material\n"
    )
    # SPECIFICITY GAVE UP A POINT, AND IT IS THE ONE AXIS THAT COULD AFFORD IT.
    # Nine of the ten points measured accuracy and mechanics; ONE measured
    # voice, and it did so by looking for the words "worst" and "wrong". The
    # scripts came back true, tight, correctly looped and completely flat —
    # exactly what a rubric shaped like that asks for.
    #
    # Specificity is the most heavily guarded thing in this pipeline even after
    # losing a point: a regex pre-filter enforces one specific per 25 words, the
    # seed gate refuses ungroundable sources, the hook scorer checks every claim
    # against the source, and the fact gate re-checks every figure afterwards.
    # Four guards. Voice had one point and no guard at all.
    specificity_criterion = (
        "SPECIFICITY 0-2: Does the script ground claims in real, verifiable history? "
        "Well-documented facts ARE the specifics. 0=vague, 1=some grounded, 2=every claim grounded.\n"
        if is_wisdom else
        "SPECIFICITY 0-2: Does the script use real details from source? "
        "0=invented/vague, 1=some grounded, 2=every claim grounded.\n"
    )

    # THE RUBRIC KNOWING WHICH VIDEO IT IS LOOKING AT.
    #
    # Four of the lines below describe a Short specifically, and three of those
    # are not merely irrelevant to a nine-minute explainer — they penalise it
    # for doing what longform_writer was told to do. The loop line is a
    # DISQUALIFIER (final ≤4), and long-form does not end on a loop: it ends by
    # paying a counted promise, which is the shape its outline plans. So every
    # long-form script would have come back capped at 4, held from publishing
    # forever, by a gate measuring a device its own generator is instructed not
    # to write. That is this repo's own named failure — the gate knowing
    # something the generator was never told — with the sign flipped.
    #
    # It stays ONE rubric out of ten with the same criteria names, because the
    # review queue sorts both formats by this number and a second scale would
    # make them incomparable. Only the sentences that name a Shorts device are
    # swapped for the long-form equivalent of the same axis.
    long_form = _is_long_form()
    editor_frame = (
        "You are a ruthless documentary editor. Score the SCRIPT BODY (the cold "
        "open is pre-vetted). This is a nine-minute narrated explainer, not a "
        "Short — length is the format here, not padding. Judge whether the "
        "promises it makes get paid.\n\n"
        if long_form else
        "You are a ruthless short-form editor. Score the SCRIPT BODY (the hook is pre-vetted).\n\n"
    )
    close_disqualifier = (
        "□ The close does not pay the counted promise the opening made ('three "
        "ways', 'two things that had to be true') — it stops rather than lands\n"
        if long_form else
        "□ Loop line (second-to-last) shares zero content words with the hook\n"
    )
    sensory_disqualifier = (
        "□ NO EARLY SENSORY DETAIL: zero concrete physical detail a viewer could "
        "see, hear, feel, smell, or taste appears in the OPENING SECTION — the "
        "viewer decides in the first thirty seconds whether this is worth nine "
        "minutes, and an abstraction cannot be pictured\n\n"
        if long_form else
        "□ NO EARLY SENSORY DETAIL: zero concrete physical detail a viewer could see, hear, "
        "feel, smell, or taste appears in the FIRST THIRD of the body (the setup, right after "
        "the hook) — a sensory detail buried near the end doesn't stop the swipe; it has to "
        "land while the viewer is still deciding whether to keep watching\n\n"
    )
    hook_criterion = (
        "HOOK 0-2: Does the body carry through the situation the cold open put "
        "the viewer inside, or is it abandoned once the facts start? "
        "0=abandoned, 1=partial, 2=carried to the close.\n"
        if long_form else
        "HOOK 0-2: Does the body deliver on the cognitive itch the hook opened? 0=unanswered, 1=partial, 2=paid off in loop.\n"
    )
    # A twelve-word average over 1,300 words is a machine gun, not compression,
    # and the real long-form padding failure is different: the same fact said
    # again in a later section because the writer had space to fill.
    compression_criterion = (
        "COMPRESSION 0-2: Every SECTION earns its place. Penalize a fact or "
        "figure restated in a later section, a section that adds no new claim, "
        "and hedging (maybe/perhaps/could/might). A long sentence is not "
        "padding; a repeated idea is. 0=repetitive, 1=mostly tight, 2=every "
        "section adds.\n"
        if long_form else
        "COMPRESSION 0-2: Every sentence earns its place. Penalize avg sentence >12 words and hedging (maybe/perhaps/could/might). 0=padded, 1=mostly tight, 2=every word counts.\n"
    )
    close_criterion = (
        "PAYOFF 0-2: Does the close pay the counted promise and land its last "
        "line against the viewer's own life? 0=it stops, 1=it summarizes, "
        "2=it pays the count and lands.\n"
        if long_form else
        "LOOP 0-2: Does the second-to-last line mirror the hook's structure or pose the question the hook answered? Token-echo required. 0=no echo, 1=thematic only, 2=structural mirror.\n"
    )
    close_reply = (
        "PAYOFF: [0-2]/2 — [quote the closing line, name the promise it pays]\n"
        if long_form else
        "LOOP: [0-2]/2 — [quote the loop line, explain echo]\n"
    )

    prompt = (
        f"SCRIPT:\n\"{script}\"\n\n"
        f"FIXED HOOK (line 1): \"{hook}\"\n"
        f"SOURCE ({seed_type} seed): \"{seed_text}\"\n\n"
        + editor_frame +
        "STEP 1 — DISQUALIFIERS (any one → final ≤4):\n"
        + invented_disqualifier +
        "□ Script uses placeholder names (John/Sarah/Mike/Alex) as if real\n"
        "□ Script adopts first-person voice of someone in the source\n"
        "□ Script has zero specifics (no number, name, date, or verbatim detail)\n"
        + close_disqualifier +
        "□ BORING: Body has no tension, contradiction, or turning point — reads like a neutral Wikipedia summary\n"
        + sensory_disqualifier +
        "STEP 2 — SCORE EACH (only if no disqualifiers):\n"
        + specificity_criterion
        + hook_criterion
        + compression_criterion
        + close_criterion +
        "HUMAN 0-2: Can the viewer FEEL this, and is the viewer IN it? Three things,\n"
        "score what is actually there:\n"
        "  • Is the viewer in the script — 'you', present tense, something they do\n"
        "    or already did? A script with no 'you' is a lecture.\n"
        "  • Is there one real person with something at stake, and does it cost\n"
        "    them? A fact nobody paid for is trivia.\n"
        "  • Is there a TURN — a point where the meaning changes and what the\n"
        "    viewer assumed becomes what was never true? Without one the body is a\n"
        "    list of true sentences.\n"
        "THE TURN AND THE COST MUST BE IN THE SOURCE. This criterion pays for\n"
        "tension, and the cheapest way to fake tension is to invent why somebody\n"
        "acted — 'policymakers were scared to act', 'the finding was buried by\n"
        "people who feared it'. Five of the last eight fact-check rejections were\n"
        "exactly that, and each one was caught only AFTER the images and the\n"
        "render had been paid for. A turn is a change in what the documented FACTS\n"
        "MEAN, never a claim about what someone privately wanted. Score 0 on this\n"
        "criterion for any attributed motive, fear, or intent the source does not\n"
        "state outright — an invented feeling is worth less than a flat script,\n"
        "because a flat script is at least cheap to reject.\n"
        "0=none of the three, an encyclopedia entry read aloud. 1=one of them.\n"
        "2=the viewer is in it, somebody pays, and the meaning turns.\n"
        "Opinion words (worst/wrong/smartest/scared) help but do not by themselves\n"
        "earn a 2 — 'the worst monetary decision in history' is still a narrator\n"
        "talking about strangers.\n\n"
        "STEP 3 — REPLY EXACTLY:\n"
        "DISQUALIFIERS: [list, or 'none']\n"
        "SPECIFICITY: [0-2]/2 — [explain]\n"
        "HOOK: [0-2]/2 — [explain]\n"
        "COMPRESSION: [0-2]/2 — [explain]\n"
        + close_reply +
        "HUMAN: [0-2]/2 — [name which of the three are present]\n"
        "TOTAL: [sum]/10"
    )

    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=450,
            timeout=90,
        )
        ms       = int((time.time() - t0) * 1000)
        usage    = resp.usage
        cost     = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        reasoning = resp.choices[0].message.content.strip()

        crits: dict = {}
        total = None
        for m in _CRIT_RE.finditer(reasoning):
            key = m.group(1).lower()
            # One axis, two device names — see _CRIT_RE. Stored under "loop"
            # so the column, the retry fixes and the queue ranking are the
            # same measurement in both formats.
            if key == "payoff":
                key = "loop"
            val = int(m.group(2))
            if key == "total":
                total = val
            else:
                crits[key] = val
        if total is None:
            total = sum(crits.values()) if crits else 5
        return total, crits, reasoning, cost, ms
    except Exception as e:
        return 5, {}, f"scorer error: {e}", 0.0, 0


# ── Public API ──────────────────────────────────────────────────────────────────

def preanalyze(seed: dict, scene: str = "") -> tuple[str, str, float]:
    """Run pre-analysis before video selection so the picker uses the hook angle.

    Returns (analysis_text, run_id, cost_usd).
    Call this from main.py after get_seed(), before pick_best_video().
    Pass the returned analysis and run_id into write_script() to skip the
    duplicate API call inside the script writer.
    """
    _, active = _load_niche()
    import llm
    llm.announce()
    client    = llm.client(_load_key())
    run_id    = new_run_id()
    if seed:
        print(f"[gpt] run_id={run_id} seed: {seed.get('type', '?')} from {seed.get('source', 'Unknown')}")
    trending_context = (seed or {}).get("trending_context", "")
    analysis, cost = _pre_analyze(client, seed, scene, run_id, active,
                                  trending_context=trending_context)
    if analysis:
        print(f"[gpt] analysis:\n{analysis}")
    return analysis, run_id, cost


def write_script_until_good(scene_description: str, seed: dict | None = None,
                            precomputed_analysis: str = None,
                            run_id: str = None) -> dict:
    """Run full script CYCLES until one is both high-scoring and factual.

    Why this exists (the retries were happening at the wrong level): inside
    write_script the hook is chosen ONCE, then up to 3 body attempts run
    against it. When the fact gate rejects the hook's own core claim — live:
    "Swiss banking secrecy protected Nazi assets", unsupportable from the
    source — redrafting the body under that same doomed hook cannot ever
    succeed. Every retry was spent on a premise that was already lost.

    So this escalates instead of repeating: each cycle is a COMPLETE fresh
    attempt (new hook factory → new angle → new body → fact gate), and the
    previous cycle's rejection is fed forward so the hook factory stops
    reaching for the same unsupportable claim.

    Stops as soon as a cycle is genuinely good (score >= target AND fact gate
    passed) — no wasted spend when the first try is fine. Otherwise it keeps
    the best-scoring attempt so far and returns that; a run is NEVER failed
    over this, matching the rest of the pipeline's fail-open policy.

    Bounded on purpose — "loop until perfect" with no ceiling is how you get a
    runaway bill on a topic whose source genuinely can't support an
    interesting claim:
      RUFUS_SCRIPT_CYCLES    3      full cycles (1 = old single-pass behavior)
      RUFUS_SCRIPT_MAX_COST  0.30   hard USD ceiling across all cycles
    """
    std        = _standards()
    target     = std["scoring"]["score_min"]
    max_cycles = max(1, int(os.environ.get("RUFUS_SCRIPT_CYCLES", "3")))
    max_cost   = float(os.environ.get("RUFUS_SCRIPT_MAX_COST", "0.30"))

    best: dict | None = None
    spent = 0.0
    scene = scene_description

    for cycle in range(1, max_cycles + 1):
        result = write_script(scene, seed=seed,
                              precomputed_analysis=precomputed_analysis,
                              run_id=run_id)
        spent += result.get("cost_usd", 0.0) or 0.0
        result["cost_usd"] = spent          # report cumulative, not per-cycle

        if best is None or result.get("score", 0) > best.get("score", 0):
            best = result

        good = result.get("score", 0) >= target and result.get("fact_ok", True)
        if good:
            if cycle > 1:
                print(f"[gpt] cycle {cycle}/{max_cycles}: "
                      f"{result['score']}/10 and factual — accepted")
            return result

        why = (result.get("fact_reason")
               or f"scored {result.get('score', 0)}/10, below the {target}/10 bar")
        if cycle >= max_cycles:
            print(f"[gpt] ⚠ {max_cycles} cycle(s) exhausted — keeping the best "
                  f"({best.get('score', 0)}/10). Last issue: {why}")
            break
        if spent >= max_cost:
            print(f"[gpt] ⚠ cost ceiling ${max_cost:.2f} reached after cycle "
                  f"{cycle} — keeping the best ({best.get('score', 0)}/10)")
            break

        print(f"[gpt] cycle {cycle}/{max_cycles} not good enough ({why}) — "
              f"retrying with a DIFFERENT angle")
        # Feed the failure forward. A fresh hook factory runs next cycle, and
        # this steers it off the angle that just failed instead of letting it
        # rediscover the same doomed claim.
        scene = (f"{scene_description}\n\nAVOID — a previous attempt on this "
                 f"exact source was rejected for: {why}. Choose a DIFFERENT "
                 f"angle that the source material actually supports; do not "
                 f"restate or lightly reword the rejected claim.")

    best["cost_usd"] = spent
    return best


def write_script(scene_description: str, seed: dict | None = None,
                 precomputed_analysis: str = None, run_id: str = None) -> dict:
    """Three-phase hook-first script writer. Returns dict with full metadata.

    Single pass — see write_script_until_good() for the escalating retry loop
    that calls this repeatedly with a fresh hook when a cycle isn't good enough.

    Pass precomputed_analysis + run_id (from preanalyze()) to skip the
    redundant pre-analysis API call when analysis was already run for video
    selection.

    Return shape:
        {
          "script": str,
          "run_id": str,
          "score": int,
          "criterion_scores": dict,
          "attempts_used": int,
          "final_temperature": float,
          "reasoning": str,
          "cost_usd": float,
        }
    """
    std            = _standards()
    niche, active  = _load_niche()
    import llm
    llm.announce()
    client         = llm.client(_load_key())
    run_id         = run_id or new_run_id()
    total_cost     = 0.0

    if seed and not precomputed_analysis:
        print(f"[gpt] run_id={run_id} seed: {seed.get('type', '?')} from {seed.get('source', 'Unknown')}")

    trending_context = (seed or {}).get("trending_context", "")

    # Pre-analysis — skip if already done before video selection
    if precomputed_analysis:
        analysis = precomputed_analysis
    else:
        analysis, cost = _pre_analyze(client, seed, scene_description, run_id, active,
                                      trending_context=trending_context)
        total_cost += cost
        if analysis:
            print(f"[gpt] analysis:\n{analysis}")

    # CTA
    cta = _pick_cta(niche, active)
    print(f"[gpt] cta: {cta}")

    # ── Phase A + B: get a winning hook (1 retry allowed) ─────────────────────
    winning_hook = None
    winning_hook_score = 0
    hook_score_min = std["scoring"]["hook_score_min"]
    for hook_attempt in range(1, 3):  # max 2 tries at hook factory
        temp = (std["scoring"]["hook_temperature"]
                if hook_attempt == 1
                else std["scoring"]["hook_retry_temperature"])
        hook_model = (std["models"]["hook_gen"]
                      if hook_attempt == 1
                      else std["models"]["hook_gen_escalation"])

        print(f"[gpt] Phase A: hook factory ({hook_model}, temp={temp})")
        hooks, c = _hook_factory(client, seed, analysis, active, niche,
                                 run_id, temperature=temp, model=hook_model)
        total_cost += c
        print(f"[gpt] generated {len(hooks)} hook candidates")

        if len(hooks) < std["scoring"]["min_surviving_hooks"]:
            print(f"[gpt] only {len(hooks)} parsed — retrying" if hook_attempt == 1 else "[gpt] ⚠ still too few hooks")
            continue

        idx, score, reason, c = _hook_scorer(client, hooks, seed, active, run_id,
                                             analysis=analysis)
        total_cost += c
        if idx < 0:
            print(f"[gpt] Phase B: {reason}")
            continue

        candidate_hook  = hooks[idx]
        candidate_score = score
        print(f"[gpt] Phase B candidate ({score}/10): {candidate_hook}")
        print(f"[gpt]   reason: {reason}")

        # Always keep the best candidate across attempts so we can fall back to it
        # if neither attempt hits the minimum threshold.
        if candidate_score > winning_hook_score:
            winning_hook       = candidate_hook
            winning_hook_score = candidate_score

        if candidate_score >= hook_score_min:
            print(f"[gpt] Phase B: hook accepted ({score}/10 ≥ {hook_score_min})")
            break
        else:
            print(f"[gpt] Phase B: {score}/10 < {hook_score_min} threshold — retrying hook factory"
                  if hook_attempt == 1 else
                  f"[gpt] Phase B: retry also {score}/10 — using best available hook")

    if not winning_hook:
        # Last-resort: take whatever hook we have, even if weak
        if hooks:
            winning_hook = hooks[0]
            winning_hook_score = 0
            print(f"[gpt] ⚠⚠⚠ shipping ZERO-SCORE hook (all {len(hooks)} candidates failed filter) — script quality unchecked")
            print(f"[gpt]     hook: {winning_hook}")
        else:
            raise RuntimeError("Hook factory produced zero parseable hooks across 2 attempts")

    # ── Story architect: plan before prose (see _story_architect docstring) ────
    story_plan, architect_cost = _story_architect(client, seed, analysis,
                                                  winning_hook, run_id, active)
    total_cost += architect_cost
    if story_plan:
        print(f"[gpt] story architect:\n{story_plan}")
    # Published for the storyboard, so the pictures anchor to the same moment
    # the words turn on. Empty when the source had no filmable moment in it —
    # said out loud, because a run with no scene is the run that produces an
    # encyclopedia entry, and that should be visible rather than inferred from
    # a flat-feeling video.
    global LAST_SCENE
    LAST_SCENE = scene_from_plan(story_plan)
    if story_plan:
        _scene_note = LAST_SCENE or ("(none in this source — the script will "
                                     "read as summary, not story)")
        print(f"[gpt] scene: {_scene_note}")

    # ── Phase C: body generation ──────────────────────────────────────────────
    system = _build_system(niche, active, cta, winning_hook)
    seed_blk       = _seed_block(seed) if seed else ""
    hook_tokens    = _content_tokens(winning_hook)
    hook_token_str = ", ".join(sorted(hook_tokens)) or "(none)"
    opinion_all    = ", ".join(std["opinion_pool"])
    plan_blk       = f"STORY PLAN (write to this shape):\n{story_plan}\n\n" if story_plan else ""
    # The plan's THE SCENE is the one line in it a camera could point at, and
    # it is what this channel has been missing. Scripts built only from the
    # aggregate lines read as encyclopedia entries with a rhythm ("held wealth
    # equivalent to two percent of Europe's GDP" — nobody wants anything,
    # nobody loses anything, no moment happens), and with no event to carry
    # feeling the writer reaches for MOTIVE instead, which is the #1 fact-gate
    # rejection. Naming the scene here routes the emotion through the event,
    # which is exactly the register the gate passes.
    if story_plan and "NONE" not in story_plan.upper().split("SPINE FACT")[0]:
        plan_blk += (
            "USE THE SCENE. Open on it, or turn on it — put the viewer in that "
            "moment with the person doing the thing. Aggregate facts (totals, "
            "shares, net worths) are EVIDENCE you cut to afterwards, never the "
            "way in. A script that never lands in a place with a person in it "
            "is an encyclopedia entry, however specific its numbers are.\n"
            "Feeling comes from the EVENT, not from stating what anyone felt "
            "or intended — that is both better writing and the only version "
            "the fact-check passes.\n\n"
        )
    base_usr = (
        f"{seed_blk}\n"
        f"Background scene: {scene_description}\n\n"
        f"PRE-ANALYSIS:\n{analysis}\n\n"
        f"{plan_blk}"
        f"Write the COMPLETE SCRIPT — all lines from hook through CTA.\n"
        f"Line 1 must be exactly: {winning_hook}\n"
        f"Total word count: {std['body']['min_words']}-{std['body']['max_words']} words.\n"
        f"Second-to-last line (LOOP) must contain at least one of these words from the hook: {hook_token_str}\n"
        f"Body must contain at least one of these opinion words: {opinion_all}\n"
        f"Last line must be exactly: {cta}"
    )

    temps = std["scoring"]["body_temperatures"]
    max_attempts  = std["scoring"]["max_body_attempts"]
    score_min     = std["scoring"]["score_min"]
    body_model    = std["models"]["body_gen"]
    # 500 tokens (≈375 words) gives room for the 80-115 word script + retry pressure
    # blocks that accumulate across attempts without ever cutting off mid-sentence.
    last_rejection = ""        # what failed in the previous attempt (for the crit note)
    accumulated_fixes: list[str] = []   # ALL corrections so far — carried forward each retry
    rejected_pool: list[tuple[int, str]] = []   # (word_count, script) for salvage fallback

    def _fix_for(rejection: str) -> str:
        return _fix_for_rejection(rejection, std, hook_token_str, opinion_all)

    best = {"script": "", "score": 0, "crits": {}, "reasoning": "",
            "temperature": temps[0], "attempt_n": 0}

    for attempt in range(1, max_attempts + 1):
        temp = temps[min(attempt - 1, len(temps) - 1)]

        # Retry pressure — carry ALL prior corrections forward so the model can't
        # fix one constraint while regressing on another (the oscillation bug).
        if attempt == 1 or not accumulated_fixes:
            push = "" if attempt == 1 else (
                f"\n\nAttempt {attempt}. Keep only sentences that REVEAL, NAME A SPECIFIC, "
                "or BUILD TENSION. The COMPLETE SCRIPT is required — hook on line 1, body, "
                "then CTA on the last line."
            )
        else:
            crit_note = (
                f" (prev score {best['score']}/10 — "
                f"spec={best['crits'].get('specificity','?')}, "
                f"hook={best['crits'].get('hook','?')}, "
                f"loop={best['crits'].get('loop','?')})"
                if best["score"] > 0 else ""
            )
            fixes_blk = "\n".join(f"- {f}" for f in accumulated_fixes)
            push = (
                f"\n\nAttempt {attempt}{crit_note}. You MUST satisfy ALL of these "
                f"(failures from earlier attempts — do not regress on any):\n{fixes_blk}\n"
                "Keep only sentences that REVEAL, NAME A SPECIFIC, or BUILD TENSION. "
                "The COMPLETE SCRIPT is required — hook on line 1, body, then CTA on the last line."
            )

        script, c, ms, p_toks, c_toks = _generate(client, system, base_usr + push,
                                                  model=body_model, temperature=temp)
        total_cost += c

        # Detect echo-only: model returned only the hook line (or nothing)
        # This happens when an imperative hook word ("Stop scrolling!") causes early stop
        # or the model misreads "write the body" as "write the non-hook portion only".
        hook_words_n = len(winning_hook.split())
        if len(script.split()) <= hook_words_n + 3:
            echo_push = (
                "\n\nCRITICAL: You must output AT LEAST 80 words. "
                "Write the COMPLETE SCRIPT — hook as line 1, then body sentences, then CTA. "
                "Do not output only the first line."
            )
            print(f"[gpt] attempt {attempt}/{max_attempts} – echo-only ({len(script.split())} words), forcing full script")
            log_attempt({
                "run_id": run_id, "niche": active,
                "seed_type": seed.get("type") if seed else None,
                "phase": "body_gen", "attempt_n": attempt,
                "model": body_model, "temperature": temp,
                "hook": winning_hook, "body": script,
                "rejected_reason": f"echo-only ({len(script.split())} words)",
                "accepted": False, "cost_usd": c, "ms": ms,
            })
            # Retry immediately with forced instruction (same attempt slot, higher temp)
            script, c2, ms2, p_toks, c_toks = _generate(
                client, system, base_usr + echo_push, model=body_model,
                temperature=min(temp + 0.2, 1.2))
            total_cost += c2
            ms += ms2
            c  += c2

        # Ensure hook is on line 1 — INSERT if missing (don't replace a body line).
        lines = [l.strip() for l in script.split("\n") if l.strip()]
        if lines and not _hook_already_present(lines[0], winning_hook):
            lines.insert(0, winning_hook)
            script = "\n".join(lines)

        # Pre-score regex rejections (cheap).
        #
        # A banned phrase is REPAIRED here rather than rejected outright. It was
        # already being repaired unconditionally as a safety net further down
        # (see _repair_banned at the end of this function), so rejecting first
        # meant spending an entire extra generation to arrive at a fix we were
        # willing to apply anyway. Banned phrases were 24.7% of all rejections
        # live, overwhelmingly single-word swaps ('journey'→'path',
        # 'crucial'→'key'). The model is still TOLD about it via accumulated_
        # fixes, so later attempts stop reaching for the word — the correction
        # survives, only the wasted round-trip goes away.
        violations: list[str] = []
        if (_banned := _find_banned(script)):
            repaired = _repair_banned(script)
            # Only accept the repair if it didn't wreck the script's structure —
            # unmapped phrases get DELETED, which can gut a sentence.
            if not _find_banned(repaired) and len(repaired.split()) >= _standards()["body"]["min_words"]:
                print(f"[gpt] repaired banned phrase '{_banned}' in place (no retry spent)")
                script = repaired
                fix = _fix_for(f"banned phrase: '{_banned}'")
                if fix and fix not in accumulated_fixes:
                    accumulated_fixes.append(fix)
            else:
                violations.append(f"banned phrase: '{_banned}'")

        # Same trade for the two mechanical violations that dominate the live
        # ladders. A five-word overage and a missing long sentence are edits we
        # would make by hand and accept; buying them with a whole generation is
        # the "wasted-generation rejection ladder" AGENTS.md warns about. The
        # model is still TOLD via accumulated_fixes, so later attempts stop
        # producing them — only the round-trip goes away.
        #
        # A repair is accepted ONLY if it strictly reduces the violation list.
        # That is what keeps a trim from silently trading "too long" for
        # "low specificity" or a broken loop.
        for _label, _repair in (
            ("too long", lambda s: _repair_length(s, _standards()["body"]["max_words"])),
            ("cadence",  _repair_cadence),
        ):
            _before = _body_violations(script)
            if not any(v.startswith(_label) for v in _before):
                continue
            _candidate = _repair(script)
            if _candidate == script:
                continue
            _after = _body_violations(_candidate)
            if len(_after) < len(_before) and not any(
                    v.startswith(_label) for v in _after):
                print(f"[gpt] repaired '{_label}' in place (no retry spent)")
                script = _candidate
                fix = _fix_for(next(v for v in _before if v.startswith(_label)))
                if fix and fix not in accumulated_fixes:
                    accumulated_fixes.append(fix)

        violations += _body_violations(script)
        rejection = violations[0] if violations else None

        if rejection:
            last_rejection = rejection  # carry forward to next attempt's push
            # EVERY violation becomes a correction, not just the headline one —
            # otherwise a script breaking three rules costs three generations to
            # learn about all three (the observed 7-attempt ladders).
            for v in violations:
                fix = _fix_for(v)
                if fix and fix not in accumulated_fixes:
                    accumulated_fixes.append(fix)   # remembered for ALL future attempts
            if len(violations) > 1:
                print(f"[gpt]   (+{len(violations) - 1} more: "
                      f"{'; '.join(v.split('(')[0].strip() for v in violations[1:])})")
            rejected_pool.append((len(script.split()), script))
            print(f"[gpt] attempt {attempt}/{max_attempts} – rejected ({rejection})")
            save_attempt(run_id=run_id, niche=active,
                         seed_type=seed.get("type") if seed else None,
                         phase="body_gen", attempt_n=attempt,
                         hook=winning_hook, body=script, temperature=temp,
                         rejected_reason=rejection, accepted=False,
                         cost_usd=c, ms=ms)
            log_attempt({
                "run_id": run_id, "niche": active,
                "seed_type": seed.get("type") if seed else None,
                "phase": "body_gen", "attempt_n": attempt,
                "model": body_model, "temperature": temp,
                "prompt_tokens": p_toks, "completion_tokens": c_toks,
                "cost_usd": c, "ms": ms,
                "hook": winning_hook, "body": script,
                "rejected_reason": rejection, "accepted": False,
            })
            continue

        last_rejection = ""  # passed pre-filter
        # LLM scoring
        total, crits, reasoning, sc_cost, sc_ms = _score(
            client, script, seed, winning_hook, run_id, active)
        total_cost += sc_cost

        word_count = len(script.split())
        print(f"[gpt] attempt {attempt}/{max_attempts} – score: {total}/10 – {word_count} words "
              f"(spec={crits.get('specificity', '?')}, "
              f"hook={crits.get('hook', '?')}, "
              f"comp={crits.get('compression', '?')}, "
              f"loop={crits.get('loop', '?')}, "
              f"human={crits.get('human', '?')})")

        save_attempt(run_id=run_id, niche=active,
                     seed_type=seed.get("type") if seed else None,
                     phase="body_gen", attempt_n=attempt,
                     hook=winning_hook, body=script, temperature=temp,
                     total_score=total, criterion_scores=crits,
                     accepted=(total >= score_min),
                     cost_usd=c + sc_cost, ms=ms + sc_ms)
        log_attempt({
            "run_id": run_id, "niche": active,
            "seed_type": seed.get("type") if seed else None,
            "phase": "body_gen", "attempt_n": attempt,
            "model": body_model, "temperature": temp,
            "prompt_tokens": p_toks, "completion_tokens": c_toks,
            "cost_usd": c + sc_cost, "ms": ms + sc_ms,
            "hook": winning_hook, "body": script,
            "total_score": total, "criterion_scores": crits,
            "reasoning": reasoning, "accepted": (total >= score_min),
        })

        if total > best["score"]:
            best.update(script=script, score=total, crits=crits,
                        reasoning=reasoning, temperature=temp, attempt_n=attempt)

        if total >= score_min:
            break

        # See _fixes_from_crits' docstring: convert THIS attempt's specific weak
        # criteria into corrections the next attempt must satisfy, same as the
        # pre-filter path already does — a low LLM score must not just retry cold.
        for fix in _fixes_from_crits(crits, std, opinion_all, reasoning=reasoning):
            if fix not in accumulated_fixes:
                accumulated_fixes.append(fix)

    if not best["script"]:
        # No attempt cleared the pre-filter. Salvage the rejected attempt closest to
        # passing (most words = nearest the length floor, usually the richest body)
        # rather than blindly shipping whatever the last loop iteration produced.
        if rejected_pool:
            salvage = max(rejected_pool, key=lambda x: x[0])[1]
        else:
            salvage = script
        print(f"[gpt] ⚠ no body attempt passed — salvaging closest rejected attempt")
        best["script"] = salvage
        best["attempt_n"] = max_attempts
        # Score the salvaged text so the reported number reflects the actual script
        # quality rather than the initial 0 placeholder.
        try:
            s_total, s_crits, s_reason, s_cost, _ = _score(
                client, salvage, seed, winning_hook, run_id, active)
            total_cost += s_cost
            best.update(score=s_total, crits=s_crits, reasoning=s_reason)
            print(f"[gpt] salvage scored {s_total}/10")
        except Exception:
            pass

    # Safety net: guarantee the shipped script contains no banned phrase, even if the
    # model never produced a clean one. Mapped words → synonyms; others stripped.
    if _find_banned(best["script"]):
        before = _find_banned(best["script"])
        best["script"] = _repair_banned(best["script"])
        after = _find_banned(best["script"])
        print(f"[gpt] repaired banned phrase '{before}'"
              + (f" (still found '{after}')" if after else " — clean"))

    # Fact gate: verify grounding + no misinformation before the script can
    # reach the upload path. A FAIL doesn't kill the render — it caps the score
    # below the auto-upload threshold, so the video is saved for human review.
    fact_ok, fact_reason, fact_cost = _fact_gate(client, seed, best["script"])
    total_cost += fact_cost
    final_fact_ok  = fact_ok      # may flip True below if the rewrite passes
    final_fact_why = fact_reason
    if not fact_ok:
        capped = min(best["score"], score_min - 3)
        print(f"[gpt] ⚠ FACT GATE FAILED: {fact_reason}")
        # Self-recovery: give the gate that detected the problem ONE grounded
        # rewrite of its own (see _grounded_rewrite), instead of leaving a dead
        # capped 5/10 whenever main.py's separate supervisor gate happens to
        # disagree and not trigger its rewrite.
        rewrite = _grounded_rewrite(
            client, system=system, base_usr=base_usr, body_model=body_model,
            fact_reason=fact_reason, winning_hook=winning_hook, seed=seed,
            run_id=run_id, active=active)
        total_cost += (rewrite or {}).get("cost", 0.0)
        if rewrite and rewrite["score"] >= capped:
            print(f"[gpt]   grounded rewrite passed fact-check → "
                  f"{rewrite['score']}/10 (replacing the capped {capped}/10 draft)")
            best.update(script=rewrite["script"], score=rewrite["score"],
                        crits=rewrite["crits"], reasoning=rewrite["reasoning"])
            final_fact_ok, final_fact_why = True, ""   # the rewrite is grounded
        else:
            print(f"[gpt]   score capped {best['score']} → {capped} "
                  f"(upload will be held for review)")
            best["reasoning"] = (f"FACT GATE: {fact_reason} | "
                                 + _restate_total(best.get("reasoning") or "",
                                                  best["score"], capped))
            best["score"] = capped

    if best["score"] < score_min:
        print(f"[gpt] ⚠ best score was {best['score']}/10 (target ≥{score_min}) — using best attempt")

    # Final log row
    log_attempt({
        "run_id": run_id, "niche": active,
        "seed_type": seed.get("type") if seed else None,
        "phase": "final", "attempt_n": best["attempt_n"],
        "hook": winning_hook,
        "winning_hook_score": winning_hook_score,
        "body": best["script"], "total_score": best["score"],
        "criterion_scores": best["crits"], "reasoning": best["reasoning"],
        "temperature": best["temperature"], "cost_usd": total_cost,
        "accepted": True,
    })
    print(f"[gpt] final: {best['score']}/10 @ temp={best['temperature']} "
          f"(attempts={best['attempt_n']}, cost=${total_cost:.4f})")

    return {
        "script": best["script"],
        "run_id": run_id,
        "score": best["score"],
        # Did the shipped script clear the fact gate (either first pass or via
        # the grounded rewrite)? write_script_until_good loops on this — a
        # capped score alone can't distinguish "weak writing" from "wrong
        # facts", and those need different escalations.
        "fact_ok": final_fact_ok,
        "fact_reason": final_fact_why,
        # Set when the story plan never produced a filmable moment — a
        # concept-shaped script with no event in it. Held, not failed: the
        # video renders and a human decides.
        "scene_weak": LAST_SCENE_WEAKNESS,
        "hook": winning_hook,
        "criterion_scores": best["crits"],
        "attempts_used": best["attempt_n"],
        "final_temperature": best["temperature"],
        "reasoning": best["reasoning"],
        "cost_usd": total_cost,
    }


def _restate_total(reasoning: str, raw: int, capped: int) -> str:
    """Rewrite the critic's own "TOTAL: n/10" line to show the fact-gate cap.

    The cap itself is deliberate and correct — a script the fact gate rejected
    must not be able to present as publishable. What was NOT correct is that the
    critic's verbatim reasoning was kept alongside it, still ending "TOTAL:
    8/10" while the header showed 4/10. A reviewer reading the dashboard sees
    two different scores for the same video and has no way to tell which one
    the pipeline acted on; every external review of these runs flagged it as a
    scoring bug. Both numbers are shown, with the reason, so the cap reads as
    the decision it is."""
    if raw == capped:
        return reasoning
    return re.sub(
        r"(?i)\bTOTAL:\s*\d+\s*/\s*10\b",
        f"TOTAL: {capped}/10 (critic scored {raw}/10, capped by the fact gate)",
        reasoning, count=1)


def _grounded_rewrite(client: OpenAI, *, system: str, base_usr: str,
                      body_model: str, fact_reason: str, winning_hook: str,
                      seed: dict | None, run_id: str, active: str) -> dict | None:
    """One grounded rewrite after the in-writer fact gate fails.

    Why this exists: the fact gate CAPS the score (→5/10, held) but the only
    rewrite path used to live in main.py's SEPARATE supervisor gate
    (judge_script_facts). The two graders can disagree — the in-writer gate
    flags a script the supervisor passes — and when they did, the script was
    capped to 5/10 with NO rewrite attempt at all (seen live on the money_history
    'Hanseatic League' run: 8/10 → capped 5, no recovery). This gives the gate
    that detected the problem its own recovery, feeding the exact objection back
    as a hard grounding constraint.

    Returns a result dict (script/score/crits/reasoning/cost) only if the rewrite
    PASSES the fact gate; else None so the caller keeps the capped original."""
    correction = (
        f"\n\nFACTUAL CORRECTION REQUIRED. The previous draft was rejected by the "
        f"fact-check for: {fact_reason}\n"
        f"Rewrite the COMPLETE SCRIPT making ONLY claims directly supported by the "
        f"source and pre-analysis above. Do NOT invent figures, secret motives, "
        f"hidden deals, or sweeping superlatives ('reshaped forever', 'changed "
        f"everything') the source does not state — every sentence must trace to a "
        f"real detail. Keep the hook on line 1 and the CTA on the last line."
    )
    try:
        script, cost, _ms, _pt, _ct = _generate(
            client, system, base_usr + correction, model=body_model, temperature=0.5)
    except Exception:
        return None
    lines = [l.strip() for l in script.split("\n") if l.strip()]
    if lines and not _hook_already_present(lines[0], winning_hook):
        lines.insert(0, winning_hook)
        script = "\n".join(lines)
    if _find_banned(script):
        script = _repair_banned(script)
    if _body_pre_check(script):
        return None   # rewrite broke a structural rule — keep the capped original
    fact_ok, _reason2, fc_cost = _fact_gate(client, seed, script)
    cost += fc_cost
    if not fact_ok:
        return None   # still ungrounded — keep the capped original
    total, crits, reasoning, sc_cost, _ = _score(
        client, script, seed, winning_hook, run_id, active)
    cost += sc_cost
    return {"script": script, "score": total, "crits": crits,
            "reasoning": reasoning, "cost": cost}


# ── Fact gate ───────────────────────────────────────────────────────────────────

def _fact_gate(client: OpenAI, seed: dict | None, script: str) -> tuple[bool, str, float]:
    """Grounding + misinformation check on the FINAL script.

    An educational channel lives or dies on accuracy. GPT is told to use only
    real details from the seed, but it can still invent specifics — and a seed
    itself can carry conspiracy framing (this happened live: a history.SE
    question about Benjamin Freedman's 1961 speech, a known antisemitic
    conspiracy source, became an 8/10 script about a "$50B deception").

    Rule 3 is deliberately NARROW (secret/covert motive, not ordinary
    editorializing) after a live pattern of good scripts (8/10, 10/10) getting
    capped to 5/10 for completely benign explanatory language: "simpler for
    trade," "reflected a shift in strategy," "redefined modern finance" all got
    flagged as "attributes motives... not supported" by the checker model, even
    though none of them assert a hidden agenda — they're just normal "why this
    happened" narration, which every history-education script needs to not read
    as a dry fact list. Rule 3 now targets the actual conspiracy-adjacent
    pattern it was built for (secret deals, hidden agendas, covert plans) and
    explicitly carves out ordinary cause-and-effect commentary, so the fact
    gate stops punishing normal narration while still catching invented
    motives and conspiracy framing.

    Returns (passed, reason, cost_usd). Fail-open on API errors: the gate must
    never break a render — a failed CHECK is not a failed SCRIPT.
    """
    seed_blk = _seed_block(seed) if seed else "(no source material)"
    prompt = (
        "You are a strict fact-checker for a history-education YouTube channel.\n\n"
        f"{seed_blk}\n"
        f"SCRIPT TO VERIFY:\n{script}\n\n"
        "THE SOURCE IS ONE ENCYCLOPEDIA EXCERPT, NOT THE SUM OF HISTORY.\n"
        "\"Not in the excerpt\" and \"false\" are different findings, and treating "
        "them as one is the single most common way this check goes wrong. Every "
        "recent wrong rejection said some version of \"unsupported by the source "
        "material\" about a claim that was perfectly true: the Gold Standard Act "
        "of 1900 is real, the Latin Monetary Union really was undone by swings in "
        "metal value, panic really did hit Paris in 1720. Rejecting those teaches "
        "the writer to produce a dry list of excerpt quotations, which is not the "
        "job.\n\n"
        "For each questionable claim, decide WHICH of these it is:\n"
        "  CONTRADICTED — the source says otherwise. (Source: \"70,000 tons of "
        "ORE\"; script: \"70,000 tons of SILVER\".)\n"
        "                 Before you use this, put the source's words and the "
        "script's words side by side and check they actually DISAGREE. A live "
        "rejection read \"the script says bad money circulates, but the source "
        "says good money is retained while bad money circulates\" — those are "
        "the same statement. Agreement restated in different words is a PASS.\n"
        "  INVENTED     — a specific number, date, name or quote that is in "
        "neither the source nor mainstream history. (\"hawala dates back to "
        "1327\" — that year exists nowhere.)\n"
        "  MIND-READ    — attributes an INTERNAL state or hidden motive to a "
        "NAMED actor as the EXPLANATION for why they acted. (\"policymakers "
        "were scared to act\", \"Comstock merely took credit\", \"silenced by "
        "those who feared inflation\".)\n"
        "                 NOT mind-reading: describing what people were "
        "OBSERVED to do, even when the behaviour has an emotional name. "
        "\"Drivers queued for hours\", \"panic buying emptied the pumps\", "
        "\"crowds ran on the banks\", \"desperate customers demanded metal\" "
        "are documented collective behaviours, not claims about anyone's inner "
        "life. THE TEST: could a camera have filmed it? Then it is an EVENT, "
        "and events are what this channel is made of. Only an unfilmable "
        "claim — what a person privately felt, feared or intended — fails.\n"
        "                 TWO MORE THINGS THAT ARE NOT MIND-READING, both of "
        "which have wrongly failed real scripts:\n"
        "                 (a) An UNNAMED aggregate. \"People\", \"traders\", "
        "\"the public\", \"savers\" are not named actors — nobody's private "
        "mind is being claimed, because nobody in particular is being "
        "described. This category needs a specific person or body.\n"
        "                 (b) RESTATING THE SOURCE'S OWN MECHANISM. If the "
        "excerpt asserts a behavioural principle, saying what that principle "
        "means in plain words is quoting the source, not reading a mind. "
        "Gresham's law IS \"people keep the good coin and spend the bad\" — so "
        "\"people noticed and stashed the good ones away\" restates the "
        "source; it does not go beyond it. Failing that is failing the excerpt "
        "for saying what the excerpt says.\n"
        "  CONSPIRACY   — hidden cabals, 'what they don't want you to know', or "
        "framing drawn from a known misinformation source.\n"
        "  ABSENT       — true, or ordinary mainstream history, simply not in "
        "this excerpt.\n\n"
        "FAIL only for CONTRADICTED, INVENTED, MIND-READ or CONSPIRACY.\n"
        "ABSENT is a PASS. So is ordinary cause-and-effect narration — \"it was "
        "simpler for trade\", \"this reflected a shift in strategy\", \"the union "
        "could not survive the swings\" — that is how history is explained, not a "
        "factual violation.\n"
        "Emotional writing about what HAPPENED is also a PASS: \"people carted "
        "wheelbarrows of worthless notes to the shops and still went home "
        "hungry\" is vivid AND factual. Only certainty about what someone was "
        "THINKING is a violation.\n\n"
        "Reply with EXACTLY one line:\n"
        "PASS\n"
        "or\n"
        "FAIL: <CATEGORY> — <one short sentence naming the worst violation>"
    )
    model = _standards()["models"].get("fact_check", "gpt-4o-mini")
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0, max_tokens=80, timeout=60,
        )
        usage = resp.usage
        cost  = estimate_cost(model, usage.prompt_tokens, usage.completion_tokens)
        text  = (resp.choices[0].message.content or "").strip()
        if text.upper().startswith("PASS"):
            return True, "", cost
        reason = text.split(":", 1)[1].strip() if ":" in text else text
        # ABSENT IS A PASS, AND THAT IS DECIDED HERE, NOT BY THE MODEL. The
        # prompt says so in as many words, and the checker still returned
        # "FAIL: ABSENT — the script does not mention the denarius's last
        # issuance in bronze under Aurelian" — it classified correctly and then
        # failed on its own PASS category. Asking a model to apply a rule it
        # just stated is not the same as enforcing the rule; the classification
        # is what it is good at, so take that and decide in code.
        if reason.strip().upper().startswith("ABSENT"):
            print(f"[gpt] fact gate said ABSENT (a pass): {reason}")
            return True, "", cost
        return False, reason, cost
    except Exception as e:
        print(f"[gpt] fact gate skipped ({e})")
        return True, "", 0.0


# ── Blacklist ───────────────────────────────────────────────────────────────────

def _blacklist_key(script: str) -> str:
    return " ".join(script.lower().split()[:20])


# ── Semantic near-duplicate gate (embeddings) ────────────────────────────────────
# The key-based blacklist above only catches an EXACT first-20-words repeat.
# At daily scale, two different seeds about similar facts produce differently-
# worded scripts that are still the same video to a viewer — and to YouTube's
# inauthentic-content reviewers. text-embedding-3-small costs ~$0.000002 per
# script (free at our scale) and catches paraphrase-level duplication.

EMBEDDINGS_FILE   = CONFIG_DIR / "script_embeddings.json"
EMBED_MODEL       = "text-embedding-3-small"
EMBED_HISTORY     = 150     # most-recent embeddings kept per channel
SIM_THRESHOLD     = 0.90    # cosine similarity above this = same video, reject


def _embed_script(script: str) -> list[float] | None:
    """Embedding for a script, or None on any failure (fail-open — a broken
    embedding call must never block a render, same policy as the fact gate)."""
    key = _load_key()
    if not key:
        return None
    try:
        from openai import OpenAI
        resp = OpenAI(api_key=key).embeddings.create(
            model=EMBED_MODEL, input=script[:4000], timeout=20)
        return resp.data[0].embedding
    except Exception as e:
        print(f"[gpt] embedding skipped ({e})")
        return None


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na  = sum(x * x for x in a) ** 0.5
    nb  = sum(y * y for y in b) ** 0.5
    return dot / (na * nb) if na and nb else 0.0


def _load_embeddings() -> list[dict]:
    if not EMBEDDINGS_FILE.exists():
        return []
    try:
        return json.loads(EMBEDDINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        return []


def check_similarity(script: str, channel: str = "main_en") -> tuple[bool, float, list | None]:
    """(is_near_duplicate, max_similarity, embedding). Fail-open on no
    embedding. The embedding is returned so the caller can store it via
    add_embedding without paying for a second API call."""
    vec = _embed_script(script)
    if vec is None:
        return False, 0.0, None
    sims = [_cosine(vec, e["vec"]) for e in _load_embeddings()
            if e.get("channel") == channel and e.get("vec")]
    mx = max(sims, default=0.0)
    return mx >= SIM_THRESHOLD, mx, vec


def add_embedding(vec: list | None, channel: str = "main_en") -> None:
    """Record an accepted script's embedding (rounded — 5dp is plenty for
    cosine at our threshold and keeps the JSON ~3x smaller). Keeps the most
    recent EMBED_HISTORY entries PER CHANNEL so one busy channel can't evict
    another's history."""
    if not vec:
        return
    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    entries = _load_embeddings()
    entries.append({"channel": channel, "vec": [round(x, 5) for x in vec]})
    kept: list[dict] = []
    by_channel: dict[str, int] = {}
    for e in reversed(entries):                      # newest first
        ch = e.get("channel", "main_en")
        if by_channel.get(ch, 0) < EMBED_HISTORY:
            kept.append(e)
            by_channel[ch] = by_channel.get(ch, 0) + 1
    kept.reverse()
    EMBEDDINGS_FILE.write_text(json.dumps(kept), encoding="utf-8")


# ── Topic clustering (beyond wording-level dedup) ─────────────────────────────
# check_similarity/add_embedding above catch a script that's WORDED
# differently but says the same thing. They can still miss a script that's
# semantically distinct — different examples, different framing — but keeps
# landing on the same underlying topic (three videos on "compound interest"
# in two weeks, each written differently). This clusters by TOPIC instead of
# by full-script wording, and is time-windowed rather than count-windowed —
# covering the same topic is fine again once it's not recent anymore.

TOPIC_EMBEDDINGS_FILE = CONFIG_DIR / "topic_embeddings.json"
TOPIC_SIM_THRESHOLD   = 0.88   # short-phrase embeddings run hotter than full-script ones
TOPIC_WINDOW_DAYS     = 14
TOPIC_HISTORY_CAP     = 500    # defensive cap per channel regardless of window (high-frequency schedules)

_CORE_LINE_RE = re.compile(r"(?im)^\s*\d*\.?\s*CORE:\s*(.+)$")


def extract_core_topic(analysis: str) -> str:
    """Pull the CORE line out of pre-analysis's structured output ('3. CORE:
    ...'). Falls back to the first non-empty line if the format ever drifts
    (fail-open — a missing topic tag just means this check silently no-ops,
    never blocks a render)."""
    if not analysis:
        return ""
    m = _CORE_LINE_RE.search(analysis)
    if m:
        return m.group(1).strip()
    for line in analysis.splitlines():
        if line.strip():
            return line.strip()
    return ""


def _load_topic_embeddings() -> list[dict]:
    if not TOPIC_EMBEDDINGS_FILE.exists():
        return []
    try:
        return json.loads(TOPIC_EMBEDDINGS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        return []


def check_topic_similarity(core_topic: str, channel: str = "main_en",
                           now: float = None) -> tuple[bool, float, list | None]:
    """(is_recent_duplicate_topic, max_similarity, embedding). Only compares
    against entries within TOPIC_WINDOW_DAYS — the same topic is fair game
    again once it's not recent. Fail-open (no embedding, no history) like
    every other gate in this file. `now` is injectable for tests (this
    module avoids datetime.now() churn in the same spirit as the rest of the
    codebase's fail-open timing checks)."""
    if not core_topic:
        return False, 0.0, None
    vec = _embed_script(core_topic)
    if vec is None:
        return False, 0.0, None
    now = now if now is not None else time.time()
    cutoff = now - TOPIC_WINDOW_DAYS * 86400
    sims = [
        _cosine(vec, e["vec"]) for e in _load_topic_embeddings()
        if e.get("channel") == channel and e.get("vec")
        and e.get("ts", 0) >= cutoff
    ]
    mx = max(sims, default=0.0)
    return mx >= TOPIC_SIM_THRESHOLD, mx, vec


def add_topic_embedding(vec: list | None, channel: str = "main_en",
                        now: float = None) -> None:
    """Record an accepted script's core-topic embedding with a timestamp.
    Prunes entries older than the window PLUS a defensive per-channel count
    cap (a multi-run-per-day schedule could otherwise grow this file
    unbounded even with time-pruning if the window were ever misconfigured)."""
    if not vec:
        return
    now = now if now is not None else time.time()
    TOPIC_EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    entries = _load_topic_embeddings()
    entries.append({"channel": channel, "vec": [round(x, 5) for x in vec], "ts": now})
    cutoff = now - TOPIC_WINDOW_DAYS * 86400
    kept: list[dict] = []
    by_channel: dict[str, int] = {}
    for e in reversed(entries):   # newest first
        ch = e.get("channel", "main_en")
        if e.get("ts", 0) < cutoff:
            continue
        if by_channel.get(ch, 0) < TOPIC_HISTORY_CAP:
            kept.append(e)
            by_channel[ch] = by_channel.get(ch, 0) + 1
    kept.reverse()
    TOPIC_EMBEDDINGS_FILE.write_text(json.dumps(kept), encoding="utf-8")


def check_blacklist(script: str) -> bool:
    if not BLACKLIST_FILE.exists():
        return False
    try:
        items = json.loads(BLACKLIST_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        return False
    return _blacklist_key(script) in items


def add_to_blacklist(script: str) -> None:
    BLACKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        items = json.loads(BLACKLIST_FILE.read_text(encoding="utf-8")) if BLACKLIST_FILE.exists() else []
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        items = []
    key   = _blacklist_key(script)
    if key not in items:
        items.append(key)
    BLACKLIST_FILE.write_text(json.dumps(items[-500:], indent=2), encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_writer.py '<scene description>'")
        sys.exit(1)
    desc = " ".join(sys.argv[1:])

    from research import get_seed
    print("[cli] fetching real seed...")
    seed = get_seed()

    result = write_script(desc, seed=seed)
    print(f"\n{'='*60}\nSCRIPT (score {result['score']}/10):\n{result['script']}\n{'='*60}")
    print(f"run_id={result['run_id']}  cost=${result['cost_usd']:.4f}  attempts={result['attempts_used']}")
