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
    if LEARNINGS_FILE.exists():
        return json.loads(LEARNINGS_FILE.read_text())
    return {}


def _load_gold_examples(niche_name: str) -> list[dict]:
    if not GOLD_EXAMPLES_FILE.exists():
        return []
    data = json.loads(GOLD_EXAMPLES_FILE.read_text())
    return data.get(niche_name, [])


def _pick_cta(niche_cfg: dict) -> str:
    pool = niche_cfg.get("cta_pool") or [niche_cfg.get("cta", "Follow for more.")]
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
                 niche: str) -> tuple[str, float]:
    """Cheap pre-pass: extract hook angle + structural cues. Returns (text, cost)."""
    is_wisdom = seed and seed.get("type") == "wisdom"
    model     = _standards()["models"]["pre_analyze"]

    if is_wisdom:
        quote  = seed.get("content", "")
        author = seed.get("source", "Unknown")
        prompt = (
            f"QUOTE: \"{quote}\"\nAUTHOR: {author}\n\n"
            "You are a historical researcher finding hook material for a 40-second YouTube Short.\n\n"
            "1. BIOGRAPHICAL FACT: One real, verifiable fact about this person's life that PROVES the quote through their actions. "
            "Must be concrete — include a number, year, event, or documented outcome.\n"
            "2. BEHAVIOR CONDEMNED: What specific thing do most people do that this quote calls wrong? One sentence.\n"
            "3. PARADOX: 'Most people [X]. This quote reveals [Y instead].' Must be counterintuitive.\n"
            "4. EMOTIONAL STAKES: One sentence — what does someone lose, fear, or regret if they never understand this quote?\n"
            "5. HOOK ANGLE: One ≤8-word seed phrase that leads with the BIOGRAPHICAL FACT — not the quote text.\n"
            "6. LOOP ANGLE: One question for the second-to-last line that makes viewers want to replay from line 1.\n"
            "7. VIDEO QUERIES: 3 comma-separated stock footage search terms that visually match the hook angle "
            "(e.g. for a 2008 crisis hook: 'stock market crash, trading floor panic, financial chart red').\n\n"
            "Reply ONLY with these 7 numbered items. No full script."
        )
        max_toks = 300
    else:
        seed_blk = _seed_block(seed) if seed else f"Scene description: {scene}"
        prompt = (
            f"{seed_blk}\nBackground scene: {scene}\n\n"
            "Before writing the script, find these things in the source. Use REAL details — no invention.\n\n"
            "1. CONTRADICTION: One sentence — the surprising paradox in this source.\n"
            "2. HOOK ANGLE: One ≤8-word seed phrase. Lead with the number/name/contradiction. "
            "Wrong: 'A Reddit user saved $2.4M.' Right: '$2.4M by 38. Still scared to retire.'\n"
            "3. CORE: One sentence — the insight this source proves.\n"
            "4. EMOTIONAL STAKES: One sentence — what does the average person lose or fear if they ignore this insight?\n"
            "5. CONCRETE DETAIL: The single most specific, vivid detail from the source (a number, name, date, or documented outcome).\n"
            "6. LOOP ANGLE: One question for the second-to-last line.\n"
            "7. VIDEO QUERIES: 3 comma-separated stock footage search terms that visually match the hook angle "
            "(e.g. for a frugal savings story: 'hardware store tools, leaky faucet repair, money saving jar').\n\n"
            "Reply with ONLY these 7 numbered items. No full script."
        )
        max_toks = 250

    try:
        t0 = time.time()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=max_toks,
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

    prompt = (
        f"{seed_blk}\n"
        f"NICHE: {niche_name}\n"
        f"PRE-ANALYSIS:\n{analysis}\n\n"
        f"Generate exactly {n_hooks} numbered HOOK LINES for a YouTube Short.\n\n"
        f"HOOK RULES — every line must obey ALL of these:\n"
        f"- Length: {hs['min_words']}–{hs['max_words']} words (HARD CAP {hs['hard_max_words']})\n"
        f"- Must contain at least one of: a number, a dollar amount, a proper noun (real person/place), or a year\n"
        f"- Must surface a contradiction, paradox, or pattern interrupt — not a report\n"
        f"- Must NOT start with any of: {forbidden_str}\n"
        f"- Must NOT use vague generalities — every word earns its place\n\n"
        f"Each of the {n_hooks} hooks should attack the source from a DIFFERENT angle:\n"
        "1. Number-first  (e.g. '$2.4M by 38. Still scared to retire.')\n"
        "2. Name-first    (e.g. 'Buffett's worst trade made him $25B.')\n"
        "3. Time-first    (e.g. '2,000 years ago, Seneca solved your anxiety.')\n"
        "4. Identity hit  (e.g. 'You're not disciplined. You're scared.')\n"
        "5. Counter-claim (e.g. 'The richest investors never beat the market.')\n"
        "6. Pattern break (e.g. 'Stop scrolling. This is the trade that broke Buffett.')\n"
        "7. Question      (e.g. 'Why do most lottery winners go broke?')\n"
        "8. Confession    (e.g. 'I made $2.4M and still cried at night.')\n\n"
        f"Output FORMAT — exactly one hook per line, numbered 1-{n_hooks}, no commentary:\n"
        f"1. <hook>\n2. <hook>\n...\n{n_hooks}. <hook>"
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=400,
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
        "SCORING CRITERIA (0-10 total):\n"
        "- Contradiction strength (cognitive gap the brain CANNOT ignore — feels wrong, demands resolution): 0-4\n"
        "- Pattern interrupt (stops the scroll — opens a question the viewer must answer): 0-3\n"
        "- Specificity (real number/name/year that earns instant credibility): 0-2\n"
        "- Brevity & punch (every word earns its place, no filler): 0-1\n\n"
        "DISQUALIFY (score 0-3) if: hook is purely descriptive with no tension, no contradiction, "
        "no cognitive gap — just a statement of fact or neutral observation.\n\n"
        "Reply ONLY with this JSON array, one object per hook, in order:\n"
        '[{"i": 1, "score": 0-10, "reason": "one-sentence why"}, ...]'
    )

    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=600,
        response_format={"type": "json_object"} if False else None,  # JSON array — not object
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
            scores = scores.get("results") or scores.get("hooks") or list(scores.values())[0]
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
        best_idx   = survivors[0][0]
        best_score = 0
        best_reason = "all scoring entries malformed — fell back to first survivor"

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

BEAT 1 — SETUP (lines 2-3): Ground the viewer in a specific fact. Use a number, name, or date. No vague context.
BEAT 2 — TURN (lines 4-5): The unexpected reversal or contradiction. Start with "But" or "Until" or "Then". This is the tension that creates emotion.
BEAT 3 — PAYOFF (lines 6-7): Name the mechanism. Reveal WHY the turn happened. No advice. Show the truth.

BODY ({body['min_words']}-{body['max_words']} words total including hook and CTA):
- Every sentence either adds evidence or builds tension. No filler.
- Use specific names, numbers, dates, dollar amounts. At least one specific per 25 words.
- OPINION WORD (required): body must contain at least one of these exact words: {opinion_all}

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
        max_tokens=350,
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
    analysis, cost = _pre_analyze(client, seed, scene, run_id, active)
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

    # Pre-analysis — skip if already done before video selection
    if precomputed_analysis:
        analysis = precomputed_analysis
    else:
        analysis, cost = _pre_analyze(client, seed, scene_description, run_id, active)
        total_cost += cost
        if analysis:
            print(f"[gpt] analysis:\n{analysis}")

    # CTA
    cta = _pick_cta(niche)
    print(f"[gpt] cta: {cta}")

    # ── Phase A + B: get a winning hook (1 retry allowed) ─────────────────────
    winning_hook = None
    winning_hook_score = 0
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

        winning_hook = hooks[idx]
        winning_hook_score = score
        print(f"[gpt] Phase B winner ({score}/10): {winning_hook}")
        print(f"[gpt]   reason: {reason}")
        break

    if not winning_hook:
        # Last-resort: take whatever hook we have, even if weak
        if hooks:
            winning_hook = hooks[0]
            winning_hook_score = 0
            print(f"[gpt] ⚠ fallback hook (no candidate passed filter): {winning_hook}")
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
    last_rejection = ""  # tracks what failed in the previous attempt

    best = {"script": "", "score": 0, "crits": {}, "reasoning": "",
            "temperature": temps[0], "attempt_n": 0}

    for attempt in range(1, max_attempts + 1):
        temp = temps[min(attempt - 1, len(temps) - 1)]

        # Retry pressure — escalate based on what failed
        if attempt == 1:
            push = ""
        else:
            crit_note = (
                f" (prev score {best['score']}/10 — "
                f"spec={best['crits'].get('specificity','?')}, "
                f"hook={best['crits'].get('hook','?')}, "
                f"loop={best['crits'].get('loop','?')})"
                if best["score"] > 0 else ""
            )
            # Specific correction based on exactly what failed last time
            specific_fix = ""
            if last_rejection.startswith("banned"):
                bad_phrase = last_rejection.split("'")[1] if "'" in last_rejection else ""
                specific_fix = (
                    f" CRITICAL: Your previous script used '{bad_phrase}' — that phrase is BANNED."
                    f" Do not use it or any variation of it."
                )
            elif "loop no echo" in last_rejection:
                specific_fix = (
                    f" CRITICAL: Your second-to-last line must contain at least one of: {hook_token_str}."
                    f" Quote or echo a word from the hook: '{winning_hook}'"
                )
            elif "opinion word" in last_rejection:
                specific_fix = (
                    f" CRITICAL: Your body had no opinion word. Use at least one of: {opinion_all}."
                )
            elif "hedging" in last_rejection:
                bad_hedge = last_rejection.split("'")[1] if "'" in last_rejection else ""
                specific_fix = f" CRITICAL: Remove '{bad_hedge}' — no hedging language allowed."
            elif "too short" in last_rejection:
                specific_fix = f" CRITICAL: Write at least {std['body']['min_words']} words total."
            push = (
                f"\n\nAttempt {attempt}{crit_note}.{specific_fix} "
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
        rejection = _find_banned(script) and f"banned phrase: '{_find_banned(script)}'"
        if not rejection:
            rejection = _body_pre_check(script)

        if rejection:
            last_rejection = rejection  # carry forward to next attempt's push
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
        print(f"[gpt] ⚠ no body attempt produced output — using last raw script")
        best["script"] = script
        best["attempt_n"] = max_attempts

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


# ── Blacklist ───────────────────────────────────────────────────────────────────

def _blacklist_key(script: str) -> str:
    return " ".join(script.lower().split()[:10])


def check_blacklist(script: str) -> bool:
    if not BLACKLIST_FILE.exists():
        return False
    try:
        items = json.loads(BLACKLIST_FILE.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    return _blacklist_key(script) in items


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
    desc = " ".join(sys.argv[1:])

    from research import get_seed
    print("[cli] fetching real seed...")
    seed = get_seed()

    result = write_script(desc, seed=seed)
    print(f"\n{'='*60}\nSCRIPT (score {result['score']}/10):\n{result['script']}\n{'='*60}")
    print(f"run_id={result['run_id']}  cost=${result['cost_usd']:.4f}  attempts={result['attempts_used']}")
