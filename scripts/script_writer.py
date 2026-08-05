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
        _standards_cache = json.loads(STANDARDS_FILE.read_text())
    return _standards_cache


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
    """Per-channel learnings (channel_config resolves the path; legacy installs
    keep reading config/learnings.json via the shim)."""
    try:
        from channel_config import load_channel
        path = load_channel().learnings_path
    except Exception:
        path = LEARNINGS_FILE
    if path.exists():
        try:
            return json.loads(path.read_text())
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
    data = json.loads(GOLD_EXAMPLES_FILE.read_text())
    return data.get(niche_name, [])


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
    for i, ex in enumerate(examples[:2], 1):
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
    src_num_tokens = {n.replace(",", "") for n in _HOOK_NUMBER_RE.findall(source_text)}
    for num in _HOOK_NUMBER_RE.findall(hook):
        if num.replace(",", "") not in src_num_tokens:
            return f"number '{num}' not in source/analysis (invented figure)"
    return None


# Comma-grouped digits are ONE number, not several. The old pattern was
# r"\b\d{3,}\b", which splits "10,000,000" on its commas into three separate
# "000" tokens and then reports the script as repeating "000" 3x — a script
# that in fact contains that figure exactly once. Observed live: three
# consecutive false rejections on the Venezuela hyperinflation script
# (2026-07-31 10:43), each one an entire wasted generation spent rewriting a
# script that was never actually wrong. Matching the whole comma-grouped run
# and stripping the separators is what makes the count mean what it claims.
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
    repeats = {n: c for n, c in counts.items() if c >= 2}
    if not repeats:
        return None
    worst = max(repeats, key=repeats.get)
    return (f"number '{worst}' repeated {repeats[worst]}x — restate with a "
            f"NEW specific each time, not the same figure")


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
            "Wrong: 'A Reddit user saved $2.4M.' Right: '$2.4M by 38. Still scared to retire.'\n"
            "3. CORE: One sentence — the insight this source proves.\n"
            "4. EMOTIONAL STAKES: One sentence — what does the average person lose or fear if they ignore this insight?\n"
            "5. CONCRETE DETAIL: The single most specific, vivid detail from the source (a number, name, date, or documented outcome).\n"
            "6. LOOP ANGLE: One question for the second-to-last line.\n"
            "7. VIDEO QUERIES: 3 comma-separated stock footage search terms that visually match the hook angle "
            "(e.g. for a frugal savings story: 'hardware store tools, leaky faucet repair, money saving jar').\n"
            "8. SENSORY ANCHOR: One physical sensation, concrete image, or specific moment from this source that "
            "a viewer can feel in their body. Not abstract emotions — a specific scene. "
            "Example: 'the envelope from the IRS, unopened on the kitchen counter for three weeks'.\n\n"
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

    forbidden_str = ", ".join(f"'{x}'" for x in hs["forbidden_openers"][:8])
    novelty_blk   = _novelty_block(niche_name)
    numbers_blk   = _allowed_numbers_block(seed_blk, analysis)

    prompt = (
        f"{seed_blk}\n"
        f"NICHE: {niche_name}\n"
        f"PRE-ANALYSIS:\n{analysis}\n\n"
        f"{novelty_blk}"
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
        f"- Must surface a contradiction, paradox, or pattern interrupt — not a report\n"
        f"- Must NOT start with any of: {forbidden_str}\n"
        f"- Must NOT use vague generalities — every word earns its place\n\n"
        f"Each of the {n_hooks} hooks should attack the source from a DIFFERENT angle "
        f"(every example below assumes its number/name IS in the source — always use "
        f"the source's own figures, never these):\n"
        "1. Number-first  — lead with the source's most devastating number. "
        "(the tension is in the contradiction, not the number itself)\n"
        "2. Name-first    — the source's real person/place makes it instantly credible. "
        "(e.g. 'Buffett's worst trade made him $25B.' — the reversal IS the hook)\n"
        "3. Time-contrast — the source's date reveals how long the pattern has existed. "
        "(e.g. '2,000 years ago, Seneca described your anxiety exactly.' — the gap creates the itch)\n"
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
    seed_text = (seed.get("content") or "")[:300] if seed else ""
    # Full grounding corpus for the invented-number check — the whole seed
    # (not the 300-char scoring excerpt) plus the pre-analysis, since a
    # legitimate hook may cite a figure the analysis surfaced from the source.
    grounding = " ".join(filter(None, [
        (seed.get("content") or "") if seed else "",
        (seed.get("title") or "") if seed else "",
        _strip_list_markers(analysis or ""),
    ]))

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
        "  • States or implies the OPPOSITE of common belief (contradiction/paradox)?\n"
        "  • ≤10 words?\n"
        "If all three pass, proceed to Step 2. If any fail → score 1-3 and stop.\n\n"
        "STEP 2 — SURPRISE INTENSITY (only when all gates pass — score 4-10):\n"
        "  • LOW surprise — viewer half-expected this, mild paradox: 4-6\n"
        "  • MEDIUM surprise — viewer wouldn't have predicted this: 7-8\n"
        "  • HIGH surprise — viewer actively thinks 'wait, is that actually true?': 9-10\n\n"
        "A 9/10 hook makes a viewer question something they believed with certainty.\n"
        "A 7/10 is solid. A 5/10 fails. A 3/10 gets cut.\n\n"
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
# One cheap pass BEFORE any drafting: pins down the single most compelling,
# source-grounded angle, the exact moment the reversal should hinge on, and
# why THIS telling matters right now — instead of Phase C writing blind from
# raw pre-analysis and hoping a good shape falls out. Feeds every attempt (not
# just the first), so retries have a real plan to hew to, not just corrections.
# RUFUS_SCRIPT_ARCHITECT=0 disables (fail-open — a plan-less run just writes
# exactly as before).

def _architect_enabled() -> bool:
    return os.environ.get("RUFUS_SCRIPT_ARCHITECT", "1").strip().lower() \
        not in ("0", "false", "no", "off")


def _story_architect(client: OpenAI, seed: dict, analysis: str, hook: str,
                     run_id: str, niche_name: str) -> tuple[str, float]:
    """Returns (plan_text, cost_usd) — a plan already checked against the
    source, not just written with a grounding instruction and hoped for.

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
        "In under 130 words, reply in exactly 4 short labeled lines:\n"
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
        "hold the whole video if you do."
    )
    # 2 attempts, not 3+: this is a cheap pre-check, not the main quality gate
    # (the body still goes through the full fact gate regardless). Two shots
    # at a ~130-word plan is enough to shake loose an invented motive without
    # turning a cost-saving measure into its own expensive loop.
    MAX_PLAN_ATTEMPTS = 2
    total_cost = 0.0
    last_plan, last_reason = "", ""

    for attempt in range(1, MAX_PLAN_ATTEMPTS + 1):
        this_prompt = prompt
        if attempt > 1 and last_reason:
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
            return plan, total_cost

        print(f"[gpt] story architect attempt {attempt}/{MAX_PLAN_ATTEMPTS} "
              f"ungrounded ({reason}) — {'retrying' if attempt < MAX_PLAN_ATTEMPTS else 'using anyway'}")
        last_plan, last_reason = plan, reason

    # Exhausted retries: use the last plan anyway rather than blocking the
    # render — the body's OWN fact gate still runs at the end regardless, so
    # this pre-check can only save cost, never be the sole line of defense.
    return last_plan, total_cost


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

    return f"""You are the most exacting short-form script writer working today.
Your standard: if a line does not earn its place, cut it. If a word is vague, replace it with something specific.

NICHE:
{niche_context}

YOUR JOB:
You are given REAL source material and a HOOK that has already been chosen. Write the body of a 35-50 second YouTube Short that delivers on the hook.

VOICE:
- Sound like someone who has been in this field for 20 years and is slightly impatient with people who haven't figured this out.
- Make the viewer FEEL something specific — real fear or real anger at what happened, not just learn a fact. A viewer who finishes the video informed but unmoved is a script that failed, even if every number in it checks out.
- Every beat should carry a little dread or indignation: what SHOULD have scared the people in this story and didn't, what SHOULD outrage the viewer about how it played out. That's the emotional register — not sadness, not inspiration.
- Specific always beats vague. A name beats "someone". A number beats "many". A year beats "recently".
- Short sentences ({body['min_avg_sentence_words']}-{body['max_avg_sentence_words']} words avg). Vary rhythm deliberately.
- Never moralize. Never summarize. Trust the audience.

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

BODY ({body['min_words']}-{body['max_words']} words total including hook and CTA):
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
    if crits.get("human", 1) < 1:
        fixes.append(f"CRITICAL: sound like a person with a real opinion — use one "
                     f"of: {opinion_all}. No neutral, encyclopedia-style description.")
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

_CRIT_RE = re.compile(
    r"^(SPECIFICITY|HOOK|COMPRESSION|LOOP|HUMAN|TOTAL):\s*(\d+)",
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
    specificity_criterion = (
        "SPECIFICITY 0-3: Does the script ground claims in real, verifiable history? "
        "Well-documented facts ARE the specifics. 0=vague, 1=one fact, 2=several, 3=every claim grounded.\n"
        if is_wisdom else
        "SPECIFICITY 0-3: Does the script use real details from source? "
        "0=invented/vague, 1=one specific, 2=several, 3=every claim grounded.\n"
    )

    prompt = (
        f"SCRIPT:\n\"{script}\"\n\n"
        f"FIXED HOOK (line 1): \"{hook}\"\n"
        f"SOURCE ({seed_type} seed): \"{seed_text}\"\n\n"
        "You are a ruthless short-form editor. Score the SCRIPT BODY (the hook is pre-vetted).\n\n"
        "STEP 1 — DISQUALIFIERS (any one → final ≤4):\n"
        + invented_disqualifier +
        "□ Script uses placeholder names (John/Sarah/Mike/Alex) as if real\n"
        "□ Script adopts first-person voice of someone in the source\n"
        "□ Script has zero specifics (no number, name, date, or verbatim detail)\n"
        "□ Loop line (second-to-last) shares zero content words with the hook\n"
        "□ BORING: Body has no tension, contradiction, or turning point — reads like a neutral Wikipedia summary\n"
        "□ NO EARLY SENSORY DETAIL: zero concrete physical detail a viewer could see, hear, "
        "feel, smell, or taste appears in the FIRST THIRD of the body (the setup, right after "
        "the hook) — a sensory detail buried near the end doesn't stop the swipe; it has to "
        "land while the viewer is still deciding whether to keep watching\n\n"
        "STEP 2 — SCORE EACH (only if no disqualifiers):\n"
        + specificity_criterion +
        "HOOK 0-2: Does the body deliver on the cognitive itch the hook opened? 0=unanswered, 1=partial, 2=paid off in loop.\n"
        "COMPRESSION 0-2: Every sentence earns its place. Penalize avg sentence >12 words and hedging (maybe/perhaps/could/might). 0=padded, 1=mostly tight, 2=every word counts.\n"
        "LOOP 0-2: Does the second-to-last line mirror the hook's structure or pose the question the hook answered? Token-echo required. 0=no echo, 1=thematic only, 2=structural mirror.\n"
        "HUMAN 0-1: Sounds like a real expert with opinions. Reward opinion words (worst/wrong/smartest/scared). Penalize neutral description. 0=AI/generic, 1=genuine voice.\n\n"
        "STEP 3 — REPLY EXACTLY:\n"
        "DISQUALIFIERS: [list, or 'none']\n"
        "SPECIFICITY: [0-3]/3 — [explain]\n"
        "HOOK: [0-2]/2 — [explain]\n"
        "COMPRESSION: [0-2]/2 — [explain]\n"
        "LOOP: [0-2]/2 — [quote the loop line, explain echo]\n"
        "HUMAN: [0-1]/1 — [explain]\n"
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
    client    = OpenAI(api_key=_load_key())
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
    client         = OpenAI(api_key=_load_key())
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

    # ── Phase C: body generation ──────────────────────────────────────────────
    system = _build_system(niche, active, cta, winning_hook)
    seed_blk       = _seed_block(seed) if seed else ""
    hook_tokens    = _content_tokens(winning_hook)
    hook_token_str = ", ".join(sorted(hook_tokens)) or "(none)"
    opinion_all    = ", ".join(std["opinion_pool"])
    plan_blk       = f"STORY PLAN (write to this shape):\n{story_plan}\n\n" if story_plan else ""
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
            best["score"] = capped
            best["reasoning"] = f"FACT GATE: {fact_reason} | " + (best.get("reasoning") or "")

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
        "hook": winning_hook,
        "criterion_scores": best["crits"],
        "attempts_used": best["attempt_n"],
        "final_temperature": best["temperature"],
        "reasoning": best["reasoning"],
        "cost_usd": total_cost,
    }


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
        "FAIL this script if ANY of these hold:\n"
        "1. A specific claim (number, date, name, event, quote) is neither supported "
        "by the source material above nor well-established mainstream history.\n"
        "2. It presents conspiracy-theory claims or framing as fact (hidden cabals, "
        "'what they don't want you to know', claims from known misinformation sources).\n"
        "3. It asserts a SPECIFIC secret/covert motive or hidden deal that mainstream "
        "historiography does not support (e.g. 'they secretly conspired to...', "
        "'the real reason, hidden from the public, was...').\n\n"
        "Do NOT fail for ordinary editorial narration explaining why something happened "
        "or why it matters (e.g. 'it was simpler for trade', 'this reflected a shift in "
        "strategy', 'it reshaped the industry') — that is normal explanatory writing, not "
        "a factual violation, UNLESS it also invents a specific unsupported fact already "
        "covered by rule 1. Only fail rule 3 for an actual SECRET/COVERT motive claim.\n\n"
        "Reply with EXACTLY one line:\n"
        "PASS\n"
        "or\n"
        "FAIL: <one short sentence naming the worst violation>"
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
        return json.loads(EMBEDDINGS_FILE.read_text())
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
    EMBEDDINGS_FILE.write_text(json.dumps(kept))


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
        return json.loads(TOPIC_EMBEDDINGS_FILE.read_text())
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
    TOPIC_EMBEDDINGS_FILE.write_text(json.dumps(kept))


def check_blacklist(script: str) -> bool:
    if not BLACKLIST_FILE.exists():
        return False
    try:
        items = json.loads(BLACKLIST_FILE.read_text())
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        return False
    return _blacklist_key(script) in items


def add_to_blacklist(script: str) -> None:
    BLACKLIST_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        items = json.loads(BLACKLIST_FILE.read_text()) if BLACKLIST_FILE.exists() else []
    except (json.JSONDecodeError, OSError, UnicodeDecodeError, ValueError):
        items = []
    key   = _blacklist_key(script)
    if key not in items:
        items.append(key)
    BLACKLIST_FILE.write_text(json.dumps(items[-500:], indent=2))


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
