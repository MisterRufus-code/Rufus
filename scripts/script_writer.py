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


def _find_banned(script: str) -> str | None:
    text = script.lower()
    for phrase in _standards()["banned_phrases"]:
        if re.search(r"\b" + re.escape(phrase.lower()) + r"\b", text):
            return phrase
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


def _has_opinion_word(text: str) -> bool:
    text_lower = text.lower()
    return any(re.search(r"\b" + re.escape(w) + r"\b", text_lower)
               for w in _standards()["opinion_pool"])


def _body_pre_check(script: str) -> str | None:
    """Return rejection reason or None for the full body. Runs AFTER banned check."""
    body = _standards()["body"]
    words = len(script.split())
    if words < body["min_words"]:
        return f"too short ({words} words, need ≥{body['min_words']})"
    if words > body["max_words"]:
        return f"too long ({words} words, cap {body['max_words']})"

    density = _specificity_density(script)
    if density < body["specificity_per_25_words"]:
        return f"low specificity ({density:.2f}/25w, need ≥{body['specificity_per_25_words']})"

    avg, n = _sentence_stats(script)
    if n < 3:
        return f"too few sentences ({n})"
    if avg > body["max_avg_sentence_words"]:
        return f"sentences too long (avg {avg:.1f} words, cap {body['max_avg_sentence_words']})"
    if avg < body["min_avg_sentence_words"]:
        return f"sentences too short (avg {avg:.1f} words, floor {body['min_avg_sentence_words']})"

    echoes, _ = _loop_echoes_hook(script)
    if not echoes:
        return "loop no echo (second-to-last line shares no content tokens with hook)"

    if not _has_opinion_word(script):
        return "no opinion word (need ≥1 from opinion_pool)"

    if (h := _find_hedging(script)):
        return f"hedging word: '{h}'"

    return None


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

    prompt = (
        f"{seed_blk}\n"
        f"NICHE: {niche_name}\n"
        f"PRE-ANALYSIS:\n{analysis}\n\n"
        f"{novelty_blk}"
        f"Generate exactly {n_hooks} numbered HOOK LINES for a YouTube Short.\n\n"
        f"HOOK RULES — every line must obey ALL of these:\n"
        f"- Length: {hs['min_words']}–{hs['max_words']} words (HARD CAP {hs['hard_max_words']})\n"
        f"- Must contain at least one of: a number, a dollar amount, a proper noun (real person/place), or a year\n"
        f"- Must surface a contradiction, paradox, or pattern interrupt — not a report\n"
        f"- Must NOT start with any of: {forbidden_str}\n"
        f"- Must NOT use vague generalities — every word earns its place\n\n"
        f"Each of the {n_hooks} hooks should attack the source from a DIFFERENT angle:\n"
        "1. Number-first  — lead with the specific, devastating number. "
        "(e.g. '$2.4M by 38. Still scared to retire.' — the tension is in the contradiction, not the number)\n"
        "2. Name-first    — real person's name makes it immediately credible. "
        "(e.g. 'Buffett's worst trade made him $25B.' — the reversal IS the hook)\n"
        "3. Time-contrast — a date reveals how long the pattern has existed. "
        "(e.g. '2,000 years ago, Seneca described your 2024 anxiety exactly.' — the gap creates the itch)\n"
        "4. Identity hit  — names what the viewer is actually doing wrong. "
        "(e.g. 'You're not broke. You're making one $340/month mistake.' — blame+specificity+fix)\n"
        "5. Counter-claim — the thing everyone believes that is factually backwards. "
        "(e.g. 'The less you work, the more you earn. Here's the math.' — must be provable)\n"
        "6. Scene-drop    — drop the viewer into a specific moment, no context given. "
        "(e.g. '3am. $840K in debt. He opened his laptop and typed one email.' — mystery drives completion)\n"
        "7. Loaded question — a question the viewer cannot NOT answer. "
        "(e.g. 'Why do doctors have the lowest savings rate of any profession?' — specific, counterintuitive)\n"
        "8. Confession    — first-person, specific cost, earned credibility. "
        "(e.g. 'I earned $1.2M in 18 months and had $14 in checking.' — the gap is the hook)\n\n"
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
                 run_id: str) -> tuple[int, float, str, float]:
    """Score hooks (regex pre-filter then LLM). Returns (winner_idx, score, reason, cost)."""
    std       = _standards()
    model     = std["models"]["hook_score"]
    seed_text = (seed.get("content") or "")[:300] if seed else ""

    # 1. Regex pre-filter
    survivors: list[tuple[int, str]] = []
    for i, h in enumerate(hooks):
        reason = _hook_pre_check(h)
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


# ── Phase C: Body generator ─────────────────────────────────────────────────────

def _build_system(niche_cfg: dict, niche_name: str, cta: str, hook: str) -> str:
    gold_examples = _load_gold_examples(niche_name)
    gold_block    = _build_gold_block(gold_examples)
    niche_context = niche_cfg.get("gpt_system", "")
    std           = _standards()
    body          = std["body"]
    banned_all    = ", ".join(f"'{p}'" for p in std["banned_phrases"])
    opinion_all   = ", ".join(std["opinion_pool"])
    hedging_all   = ", ".join(std["hedging_words"])

    return f"""You are the most exacting short-form script writer working today.
Your standard: if a line does not earn its place, cut it. If a word is vague, replace it with something specific.

NICHE:
{niche_context}

YOUR JOB:
You are given REAL source material and a HOOK that has already been chosen. Write the body of a 35-50 second YouTube Short that delivers on the hook.

VOICE:
- Sound like someone who has been in this field for 20 years and is slightly impatient with people who haven't figured this out.
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
        "□ BORING: Body has no tension, contradiction, or turning point — reads like a neutral Wikipedia summary\n\n"
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


def write_script(scene_description: str, seed: dict | None = None,
                 precomputed_analysis: str = None, run_id: str = None) -> dict:
    """Three-phase hook-first script writer. Returns dict with full metadata.

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

        idx, score, reason, c = _hook_scorer(client, hooks, seed, active, run_id)
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

    # ── Phase C: body generation ──────────────────────────────────────────────
    system = _build_system(niche, active, cta, winning_hook)
    seed_blk       = _seed_block(seed) if seed else ""
    hook_tokens    = _content_tokens(winning_hook)
    hook_token_str = ", ".join(sorted(hook_tokens)) or "(none)"
    opinion_all    = ", ".join(std["opinion_pool"])
    base_usr = (
        f"{seed_blk}\n"
        f"Background scene: {scene_description}\n\n"
        f"PRE-ANALYSIS:\n{analysis}\n\n"
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
        """Map a rejection reason → a one-line CRITICAL correction for the next prompt."""
        if rejection.startswith("banned"):
            bad = rejection.split("'")[1] if "'" in rejection else ""
            return (f"CRITICAL: '{bad}' is a BANNED phrase — never use it or any variation.")
        if "loop no echo" in rejection:
            return (f"CRITICAL: the second-to-last line must echo a word from the hook "
                    f"({hook_token_str}).")
        if "opinion word" in rejection:
            return f"CRITICAL: include at least one opinion word: {opinion_all}."
        if "hedging" in rejection:
            bad = rejection.split("'")[1] if "'" in rejection else ""
            return f"CRITICAL: remove '{bad}' — no hedging language."
        if "too short" in rejection:
            return f"CRITICAL: write at least {std['body']['min_words']} words total (aim for ~100)."
        if "too long" in rejection:
            return f"CRITICAL: keep it under {std['body']['max_words']} words."
        if "specificity" in rejection:
            return "CRITICAL: add concrete specifics — a name, year, or dollar amount per sentence."
        if "sentences too long" in rejection:
            return f"CRITICAL: shorten sentences (avg ≤{std['body']['max_avg_sentence_words']} words)."
        return ""

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

        # Ensure hook is on line 1 — INSERT if missing (don't replace a body line)
        lines = [l.strip() for l in script.split("\n") if l.strip()]
        if lines:
            hook_clean = winning_hook.strip().strip('"').strip("'")
            if lines[0].strip().strip('"').strip("'") != hook_clean:
                # Hook not on line 1: insert at top (body-only output or rephrase)
                lines.insert(0, winning_hook)
                script = "\n".join(lines)

        # Pre-score regex rejections (cheap)
        _banned = _find_banned(script)
        rejection = f"banned phrase: '{_banned}'" if _banned else None
        if not rejection:
            rejection = _body_pre_check(script)

        if rejection:
            last_rejection = rejection  # carry forward to next attempt's push
            fix = _fix_for(rejection)
            if fix and fix not in accumulated_fixes:
                accumulated_fixes.append(fix)   # remember it for ALL future attempts
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
    if not fact_ok:
        capped = min(best["score"], score_min - 3)
        print(f"[gpt] ⚠ FACT GATE FAILED: {fact_reason}")
        print(f"[gpt]   score capped {best['score']} → {capped} (upload will be held for review)")
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
        "criterion_scores": best["crits"],
        "attempts_used": best["attempt_n"],
        "final_temperature": best["temperature"],
        "reasoning": best["reasoning"],
        "cost_usd": total_cost,
    }


# ── Fact gate ───────────────────────────────────────────────────────────────────

def _fact_gate(client: OpenAI, seed: dict | None, script: str) -> tuple[bool, str, float]:
    """Grounding + misinformation check on the FINAL script.

    An educational channel lives or dies on accuracy. GPT is told to use only
    real details from the seed, but it can still invent specifics — and a seed
    itself can carry conspiracy framing (this happened live: a history.SE
    question about Benjamin Freedman's 1961 speech, a known antisemitic
    conspiracy source, became an 8/10 script about a "$50B deception").

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
        "3. It attributes motives or secret deals that mainstream historiography does "
        "not support.\n\n"
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
