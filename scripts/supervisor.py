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
from pathlib import Path

CONFIG_DIR = Path(__file__).parent.parent / "config"
KEYS_FILE  = CONFIG_DIR / "keys.json"
MODEL      = "gpt-4o-mini"


def enabled() -> bool:
    return os.environ.get("RUFUS_SUPERVISOR", "1").strip().lower() not in ("0", "false", "no", "off")


def _load_key() -> str:
    try:
        key = json.loads(KEYS_FILE.read_text()).get("openai", "")
        if key and not key.startswith("YOUR_") and not key.startswith("FILL_"):
            return key
    except Exception:
        pass
    return ""


def _judge(prompt: str) -> tuple[bool, str]:
    """Ask gpt-4o-mini an APPROVE|REJECT question. Returns (approved, reason).
    Fails open (approved=True) on missing key, API error, or any malformed
    reply — only an explicit, well-formed REJECT holds up the pipeline."""
    key = _load_key()
    if not key:
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
        if verdict.startswith("REJECT"):
            return False, reason
        return True, reason
    except Exception as e:
        return True, f"supervisor error ({e}) — fail-open, approved"


def judge_seed(seed: dict, niche_name: str) -> tuple[bool, str]:
    """Reject a research seed that's too thin/generic/off-topic to build a
    real story on — cheaper to catch here than after a full script + render."""
    if not enabled():
        return True, "supervisor disabled"

    stype   = seed.get("type", "")
    title   = seed.get("title", "") or ""
    content = (seed.get("content", "") or "")[:400]
    source  = seed.get("source", "")

    prompt = (
        f"You are a strict story editor for a {niche_name} YouTube Shorts channel.\n"
        f"SEED (type={stype}, source={source}):\nTITLE: {title}\nCONTENT: {content}\n\n"
        "Reject this seed ONLY if it is genuinely unusable: no concrete facts/numbers/names "
        "to build a story on, generic filler with no hook potential, or clearly off-topic "
        "for the niche. Do NOT reject just because it's a modest story — modest but concrete "
        "beats vague.\n\n"
        "Reply with EXACTLY: APPROVE|<one-sentence reason>  or  REJECT|<one-sentence reason>"
    )
    return _judge(prompt)


def judge_script_facts(script: str, seed: dict) -> tuple[bool, str]:
    """Factual-integrity gate: does the finished script contradict or fabricate
    beyond its source material? The script-writer PROMPT forbids inventing
    names/numbers/dates — this verifies it actually complied. The one judge
    whose rejection should ultimately HOLD an upload: publishing wrong facts
    costs viewer trust that a channel never gets back."""
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
        "or is fabricated (not in the source and not verifiable common knowledge)."
        f"{wisdom_note}\n"
        "Do NOT reject for: opinions, vague statements, dramatic framing, rounding, "
        "or reasonable paraphrase. This is an integrity check, not a style review.\n\n"
        "Reply with EXACTLY: APPROVE|<one-sentence reason>  or  "
        "REJECT|<the specific false/fabricated claim>"
    )
    return _judge(prompt)


def judge_footage_prompts(prompts: list[str], niche_name: str, hook: str) -> tuple[bool, str]:
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
    return _judge(prompt)
