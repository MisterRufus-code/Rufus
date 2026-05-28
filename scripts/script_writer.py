#!/usr/bin/env python3
"""
script_writer.py
Writes a value-focused Shorts script from real seed material + scene description.

Architecture:
  1. Pre-analysis pass (gpt-4o-mini, 120 tokens) — identifies hook angle, core claim, loop line
  2. Write pass (gpt-4o, 300 tokens) — full script using analysis + gold examples + niche context
  3. Scorer (gpt-4o, chain-of-thought) — 5-criterion rubric with hard disqualifiers
  4. Up to 4 attempts, keep best ≥ 8/10
"""

import json
import os
import random
import re
import sys
from pathlib import Path

from openai import OpenAI

CONFIG_DIR          = Path(__file__).parent.parent / "config"
NICHES_FILE         = CONFIG_DIR / "niches.json"
KEYS_FILE           = CONFIG_DIR / "keys.json"
BLACKLIST_FILE      = CONFIG_DIR / "blacklist.json"
LEARNINGS_FILE      = CONFIG_DIR / "learnings.json"
GOLD_EXAMPLES_FILE  = CONFIG_DIR / "gold_examples.json"

BANNED_PHRASES = [
    # Creator-slop openers
    "buckle up", "let's dive in", "let me tell you", "imagine this", "picture this",
    "here's the thing", "here's why", "the truth is", "let's talk about",
    "in this video", "in today's video", "in today's world",
    "did you know", "have you ever", "what if i told you",
    "most people don't know", "nobody talks about", "the secret is",
    "one simple trick", "this changes everything",
    # Motivational filler
    "game-changer", "game changer", "unlock", "skyrocket", "leverage",
    "dive deep", "delve", "revolutionize", "groundbreaking", "cutting-edge",
    "seamlessly", "robust", "empower", "synergy", "hustle harder",
    # Corporate buzzwords
    "it's important to note", "at the end of the day", "paradigm", "disrupt",
    "journey", "navigating", "landscape", "crucial", "vital", "actionable",
    # Hedging
    "in many ways", "in some sense", "you could argue",
    # AI essay structure
    "first and foremost", "moreover", "furthermore", "in conclusion",
    "to sum up", "all in all", "having said that",
    # Moralizing openers
    "remember", "always remember", "never forget",
]

SCORE_MIN    = 8     # ruthless — only genuinely strong scripts pass
MAX_ATTEMPTS = 4
MIN_WORDS    = 80    # 35-50s at ~150wpm
MAX_WORDS    = 130
HOOK_MAX_WORDS = 10  # hooks over 10 words are rejected before scoring


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
    """Pick a random CTA from the niche's cta_pool so videos don't all end identically."""
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


def _build_system(niche_cfg: dict, niche_name: str, cta: str) -> str:
    gold_examples = _load_gold_examples(niche_name)
    gold_block    = _build_gold_block(gold_examples)
    niche_context = niche_cfg.get("gpt_system", "")

    return f"""You are the most exacting short-form script writer working today.
Your standard: if a line does not earn its place, cut it. If a word is vague, replace it with something specific. If the hook doesn't create a cognitive itch the brain cannot ignore, rewrite it.

NICHE:
{niche_context}

YOUR JOB:
You are given REAL source material. Your job is to compress and sharpen it into a 35-50 second YouTube Short. You do not invent. You do not generalize. You use what is in the source.

VOICE:
- Sound like someone who has been in this field for 20 years and is slightly impatient with people who haven't figured this out yet.
- Specific always beats vague. A name beats "someone". A number beats "many". A year beats "recently".
- Short sentences — but not robotic fragments. Vary rhythm deliberately to create punch.
- Never moralize. Never summarize. Never explain the point. Trust the audience.
- Attribution is natural: not "Buffett said 'price is what you pay'" but "Buffett has owned Coca-Cola since 1988."

STRUCTURE — NON-NEGOTIABLE:
LINE 1 (HOOK): ≤{HOOK_MAX_WORDS} words. Creates a cognitive open loop — a question the brain cannot rest until it answers. Anchors on a specific person, number, or verifiable fact. No setup. No "did you know". No "I want to talk about". No "have you ever".
BODY: Every sentence either adds evidence or builds tension. Nothing can be cut without losing meaning. If the source has a name, number, date, or direct fact — it belongs in the body.
SECOND-TO-LAST LINE (LOOP): A question or reframing that makes the viewer want to go back to line 1. This is what drives replays.
LAST LINE (CTA): Always exactly this, on its own line: "{cta}"

ANTI-HALLUCINATION (HARD RULE):
Never invent: a person's first name, a dollar amount, a percentage, a date, a company-specific event, or a quote that is not in the source material. If the source is only a scene description with no concrete facts, restrict yourself to well-documented historical truths (e.g. "the S&P 500 has never been negative over any 20-year rolling period") or to widely-attributed quotes from named historical figures you are 100% certain said them. When uncertain, remove the specific. Vague-but-true is always better than specific-and-invented.

NARRATION VOICE (HARD RULE):
The script will be voiced by a creator who is NOT the person in the source material.
- HOOK (line 1 ONLY): can be a punchy fact-led fragment. NO explicit attribution needed. Examples: "$2.4 million by 38. Still scared to retire." or "He ran 100 miles on two broken feet." The "who" is revealed in line 2 or 3.
- BODY (line 2+): third person, with clear attribution by line 3 at latest. "A Reddit user on r/FIRE broke down their portfolio…" Never first-person impersonation.
- The viewer must understand by line 3 that this is a real person's story being reported, not the creator's confession.

HOOK STRENGTH — what separates a 2/2 hook from a 1/2 hook:
The weak version reports the FACT. The strong version reveals the CONTRADICTION inside the fact. The brain ignores facts. It cannot ignore a contradiction.

Weak (1/2): "A Reddit user saved $2.4 million by 38."
Strong (2/2): "$2.4 million by 38. Still scared to retire."

Weak: "Buffett has owned Coca-Cola since 1988."
Strong: "Buffett's worst trade made him $25 billion."

Weak: "Seneca wrote about anxiety 2000 years ago."
Strong: "Seneca solved your anxiety 2000 years ago."

Weak: "Goggins ran a hundred miles."
Strong: "He ran 100 miles on two broken feet."

Weak: "Jung studied projection."
Strong: "Your loudest opinions are confessions in disguise."

Build the hook around the SHARPEST contradiction in the source. If the source has no contradiction, find the unexpected number, the unexpected pairing, or the gap between expectation and reality.

HARD RULES:
- {MIN_WORDS}-{MAX_WORDS} words total
- Real attribution when source is a quote: weave the author into the body naturally
- HOOK must NOT open with "A Reddit user", "Someone", "A person" — lead with the number, name, or contradiction itself. Wrong: "A Reddit user saved $2.4M." Right: "$2.4M by 38. Still scared to retire."
- Never use: "picture this", "here's why", "the truth is", "let me tell you", "imagine this", "in this video", "did you know", "what if I told you", "most people don't know", "nobody talks about", "buckle up", "let's dive in", "game-changer", "unlock", "skyrocket", "leverage", "delve", "dive deep", "paradigm", "journey", "landscape", "crucial", "vital", "actionable"
- Never use placeholder names (John, Sarah, Mike, Alex) as if they were real people
- Output ONLY the script text. No labels. No "Here is the script:". No quotes around it.
{gold_block}"""


def _pre_analyze(client: OpenAI, seed: dict, scene: str) -> str:
    """Fast pre-pass: identify hook angle, core claim, and loop line before writing.

    Cheap call (gpt-4o-mini, 120 tokens). Injects the result into the writer prompt
    so the model doesn't have to do structural thinking AND prose writing simultaneously.
    """
    seed_blk = _seed_block(seed) if seed else f"Scene description: {scene}"
    prompt = (
        f"{seed_blk}\n"
        f"Background scene: {scene}\n\n"
        "Before writing the script, find these four things in the source. Use REAL details from the source — no invention.\n\n"
        "1. CONTRADICTION: What is the surprising contradiction, paradox, or unexpected pairing in this source? "
        "(e.g. 'rich but scared', 'worst trade made him richest', 'survived camps by finding meaning, not strength'). "
        "One sentence. This is what makes the story worth watching.\n"
        "2. HOOK: Write line 1 as a punchy fact-led fragment ≤10 words that surfaces the contradiction. "
        "DO NOT start with 'A Reddit user' or 'Someone'. Lead with the number/name/contradiction. "
        "Example weak: 'A Reddit user saved $2.4 million by 38.' "
        "Example strong: '$2.4 million by 38. Still scared to retire.'\n"
        "3. CORE: The one insight this source proves. One sentence. Specific and bold.\n"
        "4. LOOP: A question or restatement of the hook for the second-to-last line. One sentence.\n\n"
        "Reply with ONLY these 4 numbered items. Do not write the full script."
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=120,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return ""


def _generate(client: OpenAI, system: str, user: str, temperature: float = 0.9) -> str:
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        temperature=temperature,
        max_tokens=300,
    )
    return resp.choices[0].message.content.strip()


def _find_banned(script: str) -> str | None:
    text = script.lower()
    for phrase in BANNED_PHRASES:
        pattern = r"\b" + re.escape(phrase.lower()) + r"\b"
        if re.search(pattern, text):
            return phrase
    return None


def _hook_too_long(script: str) -> bool:
    """Return True if the first line exceeds HOOK_MAX_WORDS."""
    first_line = script.strip().split("\n")[0]
    return len(first_line.split()) > HOOK_MAX_WORDS


def _score(client: OpenAI, script: str, seed: dict) -> tuple[int, str]:
    """Chain-of-thought scoring with hard disqualifiers. Returns (score, reasoning).

    Uses gpt-4o — different model class from writer (gpt-4o is same family but this
    forces structured evaluation; scorer temperature is 0.0 to eliminate guessing).
    """
    seed_text = (seed.get("content", "") or "")[:500] if seed else ""
    seed_type = (seed.get("type") or "unknown") if seed else "unknown"
    is_wisdom = seed_type in ("wisdom",)

    # Wisdom seeds are abstract quotes — the writer MUST add historical context.
    # Penalizing that context as "invented" destroys the score unfairly.
    invented_disqualifier = (
        "□ Script invents fictional characters or made-up people not verifiable as real historical figures\n"
        "   NOTE: For this quote-based seed, well-documented historical facts (market returns, verified dates,\n"
        "   real investor track records) are EXPECTED and are NOT invented — only flag made-up people or fake events.\n"
        if is_wisdom else
        "□ Script invents a person, dollar amount, percentage, or date not present in the source material\n"
    )

    specificity_criterion = (
        "SPECIFICITY 0-3: Does the script ground claims in real, verifiable history? "
        "For a quote-based seed, well-documented facts (e.g. S&P 500 returns, historical crashes, "
        "verified investor records, named authors' biographies) ARE the specifics — they illustrate the quote. "
        "0=vague generalities only, 1=one real historical fact, 2=several verifiable facts, 3=every claim historically grounded\n"
        if is_wisdom else
        "SPECIFICITY 0-3: Does the script use real details from source (names, numbers, dates, direct facts)? "
        "0=invented/vague, 1=one weak specific, 2=several, 3=every claim grounded in source\n"
    )

    prompt = (
        f"SCRIPT TO EVALUATE:\n\"{script}\"\n\n"
        f"SOURCE IT WAS BASED ON ({seed_type} seed):\n\"{seed_text}\"\n\n"
        "You are a ruthless short-form content editor. Protect the audience from mediocre content.\n\n"
        "STEP 1 — AUTOMATIC DISQUALIFIERS (any one = final score ≤ 4, stop evaluating further):\n"
        f"□ Hook (first line) is longer than {HOOK_MAX_WORDS} words\n"
        "□ Hook starts with: Did you know / Have you ever / I want to / Let me / Imagine / Picture this / What if / A Reddit user / Someone\n"
        "□ Script contains banned phrases: 'picture this', 'here's why', 'the truth is', 'let me tell you', 'most people', 'nobody talks about', 'what if I told you', 'buckle up', 'game-changer', 'paradigm', 'journey'\n"
        + invented_disqualifier +
        "□ Script uses a placeholder name (John/Sarah/Mike/Alex) as if it were a real person\n"
        "□ Script adopts first-person voice of someone in the source (e.g. 'I saved', 'my portfolio') when the source is a Reddit story or someone else's quote — the creator is reporting, not confessing\n"
        "□ Script contains zero specifics (no name, number, date, or verbatim detail from source or history)\n\n"
        "STEP 2 — SCORE EACH CRITERION (only if no disqualifiers):\n"
        + specificity_criterion +
        "HOOK 0-2: First line ≤10 words? Creates a question the brain cannot answer without watching? Uses a specific? 0=generic/setup, 1=decent, 2=irresistible\n"
        "COMPRESSION 0-2: Every sentence earns its place. No filler, no hedging, no setup. 0=padded, 1=mostly tight, 2=every word counts\n"
        "LOOP 0-2: Second-to-last line connects back to hook, makes viewer want to replay from start. 0=absent/weak, 1=decent, 2=precise and powerful\n"
        "HUMAN 0-1: Sounds like a real expert with opinions, not AI narration or generic motivation. 0=AI/generic, 1=genuine voice\n\n"
        "STEP 3 — REPLY IN THIS EXACT FORMAT:\n"
        "DISQUALIFIERS: [list any triggered, or 'none']\n"
        "SPECIFICITY: [0-3]/3 — [one sentence explanation]\n"
        "HOOK: [0-2]/2 — [quote the hook, explain why it works or fails]\n"
        "COMPRESSION: [0-2]/2 — [identify any padded line, or confirm tight]\n"
        "LOOP: [0-2]/2 — [quote the loop line, explain]\n"
        "HUMAN: [0-1]/1 — [explain]\n"
        "TOTAL: [sum]/10"
    )
    try:
        result = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=400,
        )
        reasoning = result.choices[0].message.content.strip()
        for line in reasoning.splitlines():
            stripped = line.strip()
            if stripped.upper().startswith("TOTAL:"):
                score_str = stripped.split(":")[-1].strip().split("/")[0].strip()
                return int(score_str), reasoning
        return 5, reasoning
    except Exception as e:
        return 5, f"scorer error: {e}"


def write_script(scene_description: str, seed: dict | None = None) -> str:
    niche, active = _load_niche()
    client        = OpenAI(api_key=_load_key())
    learnings     = _load_learnings()

    if seed:
        # seed['source'] already includes the "r/" prefix for reddit type
        print(f"[gpt] seed: {seed.get('type', '?')} from {seed.get('source', 'Unknown')}")

    # Pre-analysis: extract hook angle, core claim, loop line before writing
    analysis = _pre_analyze(client, seed, scene_description)
    if analysis:
        print(f"[gpt] analysis:\n{analysis}")

    # Random CTA from the niche's pool — picked once, used by all attempts
    cta = _pick_cta(niche)
    print(f"[gpt] cta: {cta}")

    system = _build_system(niche, active, cta)

    winning_hooks = learnings.get("winning_hooks", [])[:3]
    avoid_hooks   = learnings.get("losing_hooks",  [])[:3]
    hook_hint     = ""
    if winning_hooks:
        hook_hint += f"\n\nHook patterns that previously worked well: {winning_hooks}"
    if avoid_hooks:
        hook_hint += f"\nHook patterns to avoid: {avoid_hooks}"

    seed_blk = _seed_block(seed) if seed else ""
    base_usr = (
        f"{seed_blk}\n"
        f"Background scene: {scene_description}\n\n"
        f"PRE-ANALYSIS (use this as your structural guide):\n{analysis}\n\n"
        f"Now write the script. {MIN_WORDS}-{MAX_WORDS} words. "
        f"End with this exact line: \"{cta}\"\n"
        f"The line before the CTA must be the loop line from your analysis."
        f"{hook_hint}"
    )

    best_script   = ""
    best_score    = 0
    best_reasoning = ""
    last_script   = ""

    # Climb temperature across attempts: faithful first, more creative on retries
    temps = [0.7, 0.9, 1.05, 1.15]

    for attempt in range(1, MAX_ATTEMPTS + 1):
        push = "" if attempt == 1 else (
            f"\n\nAttempt {attempt}: Previous score {best_score}/10. "
            "The hook is too safe. Lead with the CONTRADICTION, not the fact. "
            "Cut any sentence that just reports — keep only the ones that reveal. "
            "Make the hook ≤8 words and surface the paradox in the source."
        )
        temp        = temps[min(attempt - 1, len(temps) - 1)]
        script      = _generate(client, system, base_usr + push, temperature=temp)
        last_script = script

        # Hard pre-score rejections (no API call needed)
        banned = _find_banned(script)
        if banned:
            print(f"[gpt] attempt {attempt}/{MAX_ATTEMPTS} – rejected (banned: '{banned}')")
            continue

        word_count = len(script.split())
        if word_count < MIN_WORDS:
            print(f"[gpt] attempt {attempt}/{MAX_ATTEMPTS} – too short ({word_count} words, need ≥{MIN_WORDS})")
            continue

        if _hook_too_long(script):
            hook_text = script.strip().split("\n")[0]
            print(f"[gpt] attempt {attempt}/{MAX_ATTEMPTS} – hook too long: \"{hook_text}\"")
            continue

        # Chain-of-thought scoring
        score, reasoning = _score(client, script, seed)
        print(f"[gpt] attempt {attempt}/{MAX_ATTEMPTS} – score: {score}/10 – {word_count} words")
        # Print the key scoring lines for debugging
        for line in reasoning.splitlines():
            if any(line.strip().startswith(k) for k in ("DISQUALIFIERS:", "HOOK:", "TOTAL:")):
                print(f"      {line.strip()}")

        if score > best_score:
            best_score    = score
            best_script   = script
            best_reasoning = reasoning

        if score >= SCORE_MIN:
            break

    if not best_script:
        print(f"[gpt] ⚠ no attempt passed all filters – using last output")
        best_script = last_script

    if best_score < SCORE_MIN:
        print(f"[gpt] ⚠ best score was {best_score}/10 (target ≥{SCORE_MIN}) – using best attempt")

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
    desc = " ".join(sys.argv[1:])

    # CLI mode auto-fetches a real seed so the writer never hallucinates from
    # an empty source (which is how invented people like "John lost $50k" appear).
    sys.path.insert(0, str(Path(__file__).parent))
    from research import get_seed
    print("[cli] fetching real seed (Reddit first, wisdom fallback)...")
    seed = get_seed()

    script = write_script(desc, seed=seed)
    print(f"\n{'='*60}\nSCRIPT:\n{script}\n{'='*60}")
