#!/usr/bin/env python3
"""
supervisor.py — a cheap AI judge that can force ONE retry of a pipeline stage
before Rufus moves on, instead of only gating at the very end.

Why: the static score>=8 gate at upload time catches a bad SCRIPT, but a stale
research seed or off-target image prompts burn a full render's worth of time
(and GPU minutes, for comfy/sd) before anything catches it. This adds two
early, cheap judge calls (gpt-4o-mini, a fraction of a cent each) that can
reject and retry ONCE:

  judge_seed(seed, niche_name)              — after research, before scripting
  judge_footage_prompts(prompts, ...)       — after prompt-writing, before FLUX/SD render
  judge_script_facts(script, seed)          — after scripting: factual integrity vs source

Disable entirely with RUFUS_SUPERVISOR=0 (env) if the extra API calls aren't
worth it for your run. Fails OPEN on any error (no key, API down, bad JSON) —
a broken judge must never block a render.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import db_manager

CONFIG_DIR = Path(__file__).parent.parent / "config"
KEYS_FILE  = CONFIG_DIR / "keys.json"
MODEL      = "gpt-4o-mini"


def enabled() -> bool:
    return os.environ.get("RUFUS_SUPERVISOR", "1").strip().lower() not in ("0", "false", "no", "off")


def _load_key() -> str:
    try:
        key = json.loads(KEYS_FILE.read_text(encoding="utf-8")).get("openai", "")
        if key and not key.startswith("YOUR_") and not key.startswith("FILL_"):
            return key
    except Exception:
        pass
    return ""


def _log_verdict(phase: str, ok: bool, reason: str, niche: str = None,
                 seed_type: str = None, run_id: str = None) -> None:
    """Persist a supervisor gate's verdict to script_attempts, same table the
    hook/body phases already log to — so the dashboard's bottleneck breakdown
    can actually answer "is Hook Scorer or Fact-check the real bottleneck",
    instead of only ever seeing script_writer's own two phases. Best-effort:
    a logging failure must never affect the actual gate decision."""
    try:
        db_manager.save_attempt(
            run_id=run_id or "", niche=niche or "", seed_type=seed_type or "",
            phase=phase, attempt_n=1,
            rejected_reason=(None if ok else reason),
            accepted=ok, cost_usd=0.0, ms=0,
        )
    except Exception:
        pass


def _judge(prompt: str, *, phase: str = None, niche: str = None,
          seed_type: str = None, run_id: str = None) -> tuple[bool, str]:
    """Ask gpt-4o-mini an APPROVE|REJECT question. Returns (approved, reason).
    Fails open (approved=True) on missing key, API error, or any malformed
    reply — only an explicit, well-formed REJECT holds up the pipeline.

    Pass `phase` to also log the verdict (see _log_verdict) — omitted by
    default so this stays a pure function for anyone calling it directly."""
    key = _load_key()
    if not key:
        if phase:
            _log_verdict(phase, True, "no OpenAI key", niche, seed_type, run_id)
        return True, "no OpenAI key — supervisor skipped"
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=80,
            temperature=0,
            timeout=30,
        )
        raw = (resp.choices[0].message.content or "").strip()
        verdict, _, reason = raw.partition("|")
        verdict = verdict.strip().upper()
        reason  = reason.strip() or raw[:150]
        ok = not verdict.startswith("REJECT")
        if phase:
            _log_verdict(phase, ok, reason, niche, seed_type, run_id)
        return ok, reason
    except Exception as e:
        reason = f"supervisor error ({e}) — fail-open, approved"
        if phase:
            _log_verdict(phase, True, reason, niche, seed_type, run_id)
        return True, reason


import re

# ── groundability: can a dated, human story be written from this at all ──────
#
# THE FAILURE THIS ANSWERS, read off a rejection log rather than guessed at.
# Almost every fact-gate rejection is one of two sentences:
#
#   "not supported by the source, which does not provide this specific figure"
#   "MIND-READ — the script claims the government 'rigged the game'"
#
# Both come from the same place. The script writer is required to open on a
# dated moment with a named person and a hard number. When the seed is a
# StackExchange discussion — "In what ways was the Gold Confiscation Act
# beneficial" — it contains an argument, not an event: no date, no person, no
# figure. The writer cannot satisfy its rules from that material, so it supplies
# the missing specifics from its own knowledge, and the fact gate correctly
# kills the result. The gates are right, the writer is competent, and the INPUT
# is wrong.
#
# So this asks the only question that predicts the outcome: does the source
# text physically contain the raw materials — a year, a second number, and
# proper nouns? It is deterministic, free, and runs before the judge call, so
# a source that could never have worked costs nothing instead of a full script
# cycle. RUFUS_SEED_TRIES then fetches the next source, which is how the chain
# reaches Wikipedia (dense with dates and figures) instead of stopping at the
# first discussion thread it finds.
_YEAR_RE = re.compile(r"\b(1[0-9]{3}|20[0-9]{2})\b")
_NUMBER_RE = re.compile(r"\b\d[\d,.]*\b")
# Words that are capitalised for grammar or formatting rather than because they
# name somebody. Without this, "The" and "In" at the head of every sentence read
# as a cast of characters.
_NOT_A_NAME = {
    "the", "a", "an", "in", "on", "at", "by", "for", "from", "to", "of",
    "and", "but", "or", "if", "when", "while", "after", "before", "this",
    "that", "these", "those", "it", "he", "she", "they", "we", "you", "i",
    "there", "here", "what", "why", "how", "who", "which", "was", "were",
    "is", "are", "his", "her", "their", "its", "some", "many", "most",
    "however", "although", "because", "since", "during", "under", "over",
    "question", "answer", "edit", "update", "note", "wikipedia", "stack",
    "exchange", "reddit", "source", "title", "content",
}

MIN_PROPER_NOUNS = 2


def groundability(seed: dict) -> tuple[bool, str]:
    """Whether this source physically contains what a dated story needs.

    Returns (ok, reason). Never raises; an unreadable seed passes, because a
    check that cannot read something must not be the thing that rejects it.
    """
    text = " ".join(str(seed.get(k) or "") for k in ("title", "content"))
    if len(text.strip()) < 200:
        return False, ("the source is too short to build a story on "
                       f"({len(text.strip())} characters)")

    years = set(_YEAR_RE.findall(text))
    if not years:
        return False, ("the source names no year, so any date in the script "
                       "would be invented — this is where MIND-READ and "
                       "INVENTED rejections come from")

    numbers = {n for n in _NUMBER_RE.findall(text)
               if n.replace(",", "").replace(".", "") not in years}
    if not numbers:
        return False, ("the source has a date but no other figure, so the "
                       "hook's number would have to be invented")

    names = {w for w in re.findall(r"\b[A-Z][a-z]{2,}\b", text)
             if w.lower() not in _NOT_A_NAME}
    if len(names) < MIN_PROPER_NOUNS:
        return False, (f"the source names {len(names)} proper noun(s); a scene "
                       f"needs people and a place, and a writer with neither "
                       f"invents both")
    return True, ""


def judge_seed(seed: dict, niche_name: str, run_id: str = None) -> tuple[bool, str]:
    """Reject a research seed that's too thin/generic/off-topic to build a
    real story on, OR that has no genuine "knowledge gap" — cheaper to catch
    here than after a full script + render.

    The knowledge-gap check (added after a fresh-eyes review): accuracy
    alone doesn't make content interesting. A seed can be perfectly
    concrete/on-topic and still be a flat restatement of something the
    viewer already assumes — no surprise, no reason to keep watching. This
    asks the SAME judge call to also confirm the seed contains something
    that contradicts a viewer's likely mental model, not just a fact."""
    if not enabled():
        return True, "supervisor disabled"

    # DETERMINISTIC FIRST, and free. See groundability: a source with no year,
    # no figure and no names cannot produce a dated story, and asking a model
    # whether it is "interesting" spends a call on a question that is already
    # answered. Recorded in the same rejection table as the judge's own verdicts
    # so the Failures page shows one story, not two.
    ok, why = groundability(seed)
    if not ok:
        reason = f"ungroundable source: {why}"
        try:
            db_manager.save_attempt(
                run_id=run_id, niche=niche_name,
                seed_type=seed.get("type"), phase="seed_gate", attempt_n=1,
                rejected_reason=reason, accepted=False)
        except Exception:
            pass
        return False, reason

    stype   = seed.get("type", "")
    title   = seed.get("title", "") or ""
    content = (seed.get("content", "") or "")[:400]
    source  = seed.get("source", "")

    prompt = (
        f"You are a strict story editor for a {niche_name} YouTube Shorts channel.\n"
        f"SEED (type={stype}, source={source}):\nTITLE: {title}\nCONTENT: {content}\n\n"
        "Reject this seed if EITHER is true:\n"
        "1. It's genuinely unusable: no concrete facts/numbers/names to build a story on, "
        "generic filler with no hook potential, or clearly off-topic for the niche. Do NOT "
        "reject just because it's a modest story — modest but concrete beats vague.\n"
        "2. KNOWLEDGE GAP TEST: it contains no counter-intuitive fact that would break a "
        "typical viewer's mental model — i.e. nothing here would make someone stop and think "
        "\"wait, really?\". A seed can be accurate and on-topic and still fail this if it's "
        "just a flat, expected restatement of common knowledge with no surprise in it.\n\n"
        "Reply with EXACTLY: APPROVE|<one-sentence reason>  or  REJECT|<one-sentence reason "
        "naming which of the two failed>"
    )
    return _judge(prompt, phase="seed_gate", niche=niche_name,
                 seed_type=stype, run_id=run_id)


def judge_script_facts(script: str, seed: dict, niche_name: str = None,
                       run_id: str = None) -> tuple[bool, str]:
    """Factual-integrity gate: does the finished script contradict or fabricate
    beyond its source material? The script-writer PROMPT forbids inventing
    names/numbers/dates — this verifies it actually complied. The one judge
    whose rejection should ultimately HOLD an upload: publishing wrong facts
    costs viewer trust that a channel never gets back.

    NOTE — two-layer design, on purpose: script_writer._fact_gate runs INSIDE
    every write_script call (including the corrective rewrite this judge
    triggers) and caps the score below the auto-upload threshold; this judge
    runs at the main.py stage boundary and drives ONE rewrite with the
    objection fed back, then holds the upload if still flagged. Both are
    gpt-4o-mini (~$0.001 each) — cheap defense in depth, not duplication."""
    if not enabled():
        return True, "supervisor disabled"

    stype   = seed.get("type", "")
    source  = (f"TITLE: {seed.get('title', '') or ''}\n"
               f"CONTENT: {(seed.get('content', '') or '')[:1200]}\n"
               f"FROM: {seed.get('source', '')}")

    wisdom_note = (
        "\nThis is a WISDOM-QUOTE seed: well-documented historical facts "
        "(famous figures' biographies, S&P long-run returns, dated public events) "
        "are ALLOWED as illustration even though they aren't in the source — "
        "reject only claims that are actually false or invented specifics "
        "(a made-up person, amount, or event) presented as fact."
        if stype == "wisdom" else ""
    )

    prompt = (
        "You are a fact-checker for a short informative video. Compare the SCRIPT "
        "against its SOURCE.\n\n"
        f"SOURCE (type={stype}):\n{source}\n\nSCRIPT:\n{script}\n\n"
        "REJECT only if the script states a specific checkable claim — a name, "
        "number, date, dollar amount, place, or event — that CONTRADICTS the source "
        "or is fabricated (not in the source and not verifiable common knowledge). "
        "Also REJECT if it presents conspiracy-theory claims or framing as fact "
        "(hidden cabals, secret deals mainstream historiography does not support, "
        "claims sourced from known misinformation)."
        f"{wisdom_note}\n"
        "Do NOT reject for: opinions, vague statements, dramatic framing, rounding, "
        "or reasonable paraphrase. This is an integrity check, not a style review.\n\n"
        "Reply with EXACTLY: APPROVE|<one-sentence reason>  or  "
        "REJECT|<the specific false/fabricated claim>"
    )
    return _judge(prompt, phase="fact_check", niche=niche_name,
                 seed_type=stype, run_id=run_id)


def judge_footage_prompts(prompts: list[str], niche_name: str, hook: str,
                          run_id: str = None) -> tuple[bool, str]:
    """Reject a batch of beat-image prompts that clearly won't track the
    story. Checked against the PROMPTS (text), not rendered pixels — cheap,
    and catches prompt-builder drift before burning FLUX/SD generation time."""
    if not enabled():
        return True, "supervisor disabled"
    if not prompts:
        return True, "no prompts to judge"

    listed = "\n".join(f"{i+1}. {p}" for i, p in enumerate(prompts))
    prompt = (
        f"You are a strict art director for a {niche_name} YouTube Short.\n"
        f"HOOK: \"{hook}\"\n\nIMAGE PROMPTS to be generated, one per beat, in order:\n{listed}\n\n"
        "Reject ONLY if these prompts clearly fail the story: near-duplicate prompts with no "
        "visual variety, generic scenes with no connection to the hook, or imagery that "
        "contradicts the hook's subject. Approve anything reasonably on-topic — this is a "
        "coarse safety net, not a taste judge.\n\n"
        "Reply with EXACTLY: APPROVE|<one-sentence reason>  or  REJECT|<one-sentence reason>"
    )
    return _judge(prompt, phase="footage_gate", niche=niche_name, run_id=run_id)
