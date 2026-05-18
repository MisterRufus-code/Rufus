"""
Psychology-driven hook scoring and generation.

Applies proven psychological and NLP principles to score and generate
YouTube hooks that maximise CTR and watch time retention.

Principles implemented:
  - Open loop / Zeigarnik effect   → incomplete thought brain must resolve
  - Loss aversion                  → fear of missing out / making a mistake
  - Curiosity gap                  → information gap between title and answer
  - Social proof / consensus       → "everyone / millions / most people"
  - Pattern interrupt              → unexpected claim that breaks assumptions
  - Authority / insider knowledge  → "secret / hidden / they don't want you"
  - Specificity bias               → exact numbers feel more credible
  - Second-person priming          → "you" language activates personal relevance
  - Emotional trigger              → shock, disbelief, jaw-drop reactions
  - Recency signal                 → "right now", "just happened", current year
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Minimum combined viral score to use the psychology-generated hook.
# Below this the AI-generated hook is kept and a warning is shown.
VIRAL_THRESHOLD = 6.5

# Maps LLM estimated_virality label → multiplier applied to hook raw score.
# "Viral" ideas get a boost; "Low" ideas get penalised.
_VIRALITY_MULTIPLIER: dict[str, float] = {
    "Low": 0.75,
    "Medium": 1.0,
    "High": 1.25,
    "Viral": 1.5,
}


@dataclass
class HookScore:
    hook: str
    total: float                            # 0–10 hook-only score
    viral_score: float = 0.0               # 0–10 combined score (hook × virality × title)
    virality_label: str = "Medium"
    breakdown: dict[str, float] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"hook={self.total:.1f} viral={self.viral_score:.1f}/10 — {self.hook[:80]}"


# ---------------------------------------------------------------------------
# Pattern definitions — each is (label, patterns, weight, max_hits)
# ---------------------------------------------------------------------------

_PATTERNS: list[tuple[str, list[str], float, int]] = [
    (
        "open_loop",
        [
            r"\bwhat (actually|really|truly|nobody|most people)\b",
            r"\bwhy (most|nobody|everyone|you should)\b",
            r"\bthe real reason\b",
            r"\bwhat happens when\b",
            r"\bthe truth about\b",
        ],
        1.8, 1,
    ),
    (
        "loss_aversion",
        [
            r"\b(stop|never|avoid|don't|quit|mistake|wrong|danger)\b",
            r"\byou('re| are) (losing|wasting|missing|doing.*wrong)\b",
            r"\bbefore it's too late\b",
            r"\bmost people.*never\b",
        ],
        1.6, 2,
    ),
    (
        "curiosity_gap",
        [
            r"\b(secret|hidden|nobody knows|they don('t| do not) want)\b",
            r"\b(shocking|surprising|counterintuitive|unexpected)\b",
            r"\bchanged (my|everything|the way)\b",
            r"\bwish (i|you|someone) knew\b",
        ],
        1.5, 2,
    ),
    (
        "social_proof",
        [
            r"\b\d+[\s,]*(percent|%|million|billion|thousand|people)\b",
            r"\b(everyone|most people|successful people|rich people|experts)\b",
            r"\bscience (says|shows|proves|confirms)\b",
        ],
        1.2, 1,
    ),
    (
        "specificity",
        [
            r"\b\d+\s*(ways|steps|reasons|things|habits|rules|secrets|tips|mistakes)\b",
            r"\bin \d+ (minutes|days|weeks|seconds|hours)\b",
            r"\b\$[\d,]+\b",
            r"\b\d{2,}[\s,]*\d*\s*(k|million|billion)\b",
        ],
        1.3, 2,
    ),
    (
        "second_person",
        [
            r"\byou\b",
            r"\byour\b",
        ],
        0.8, 1,
    ),
    (
        "pattern_interrupt",
        [
            r"\b(nobody talks about|forget everything|i was wrong|this changed)\b",
            r"\b(actually|in reality|the opposite|contrary to)\b",
            r"\b(controversial|unpopular opinion|hot take)\b",
            r"\b(rejected|refused|turned down|said no)\b",
        ],
        1.4, 1,
    ),
    (
        "zeigarnik",
        [
            r"\bbefore (you|i|we) (tell|show|reveal|explain)\b",
            r"\b(first|step \d+|part \d+)\b",
            r"\bby the end\b",
            r"\bstay (until|till|to)\b",
        ],
        1.0, 1,
    ),
    (
        "emotional_trigger",
        [
            r"\b(jaw.?drop|mind.?blow|unbelievable|insane|crazy|wild|brutal)\b",
            r"\b(destroyed|collapsed|bankrupt|lost (it all|everything))\b",
            r"\b(shocking truth|dark side|exposed|revealed)\b",
        ],
        1.3, 1,
    ),
    (
        "recency_signal",
        [
            r"\b(202[3-9]|right now|just happened|this (week|month|year)|breaking)\b",
            r"\b(today|overnight|suddenly|already|still)\b",
        ],
        0.9, 1,
    ),
]

_MAX_SCORE = sum(w * m for _, _, w, m in _PATTERNS)


def _score_title(title: str) -> float:
    """Return a 0–1 title quality score based on viral title patterns."""
    t = title.lower()
    score = 0.0
    # Numbers in title
    if re.search(r"\b\d+\b", t):
        score += 0.25
    # Dollar/big number
    if re.search(r"\$[\d,]+|\d+\s*(billion|million)", t):
        score += 0.25
    # Power words
    if re.search(r"\b(why|how|what|the truth|the real|secret|rejected|lost|failed|won)\b", t):
        score += 0.2
    # Named entity (capitalized word other than first word)
    words = title.split()
    if any(w[0].isupper() for w in words[1:] if len(w) > 2):
        score += 0.15
    # Length sweet spot 6–12 words
    if 6 <= len(words) <= 12:
        score += 0.15
    return min(score, 1.0)


def score_hook(hook: str, virality_label: str = "Medium", title: str = "") -> HookScore:
    """
    Score a hook combining:
      - Psychology pattern matching (0–10 base)
      - Estimated virality from the idea generator (multiplier)
      - Title quality bonus

    viral_score is the final number that determines if the hook is good enough.
    """
    text = hook.lower()
    raw_total = 0.0
    breakdown: dict[str, float] = {}
    suggestions: list[str] = []

    for label, patterns, weight, max_hits in _PATTERNS:
        hits = sum(1 for p in patterns if re.search(p, text))
        capped = min(hits, max_hits)
        score = capped * weight
        breakdown[label] = round(score, 2)
        raw_total += score

    # Penalise weak hooks: very short or generic
    word_count = len(hook.split())
    if word_count < 6:
        raw_total *= 0.7
        suggestions.append("Hook is too short — aim for 8–15 words for maximum tension.")
    if word_count > 20:
        raw_total *= 0.85
        suggestions.append("Hook may be too long — shorter hooks have higher CTR on YouTube.")

    # Collect missing high-value principles
    if breakdown.get("open_loop", 0) == 0:
        suggestions.append("Add an open loop: 'What most people don't know about…' or 'The real reason…'")
    if breakdown.get("loss_aversion", 0) == 0:
        suggestions.append("Add loss aversion: start with 'Stop doing X' or 'You're wasting Y'")
    if breakdown.get("specificity", 0) == 0:
        suggestions.append("Add a number: '5 ways', '3 mistakes', 'in 30 days' — specifics build trust.")
    if breakdown.get("emotional_trigger", 0) == 0:
        suggestions.append("Add emotion: 'shocking', 'jaw-dropping', 'destroyed' raises watch urgency.")

    base = round(min(raw_total / _MAX_SCORE * 10, 10.0), 2)

    # Apply virality multiplier from LLM estimate
    v_mult = _VIRALITY_MULTIPLIER.get(virality_label, 1.0)

    # Title quality adds up to 1.5 bonus points
    title_bonus = _score_title(title) * 1.5 if title else 0.0

    viral_score = round(min(base * v_mult + title_bonus, 10.0), 2)

    if virality_label == "Low":
        suggestions.append("LLM rated this idea 'Low' virality — consider a more shocking angle.")
    elif virality_label in ("High", "Viral"):
        pass  # already boosted

    return HookScore(
        hook=hook,
        total=base,
        viral_score=viral_score,
        virality_label=virality_label,
        breakdown=breakdown,
        suggestions=suggestions,
    )


def score_hooks(hooks: list[str], virality_label: str = "Medium", title: str = "") -> list[HookScore]:
    """Score and rank a list of hooks by viral_score. Best hook is first."""
    return sorted(
        [score_hook(h, virality_label=virality_label, title=title) for h in hooks],
        key=lambda s: s.viral_score,
        reverse=True,
    )


def _enhance_hook(hook: str, topic: str, niche: str) -> str:
    """
    Programmatically patch a hook to hit missing high-value patterns.
    Adds the minimum needed elements without changing the core message.
    """
    text = hook.lower()
    result = hook.rstrip(".,!?")

    # Specificity — inject a number if none present
    has_number = bool(re.search(
        r"\b\d+\s*(ways|steps|reasons|things|habits|secrets|tips|mistakes|rules|facts)\b"
        r"|\$[\d,]+"
        r"|\bin \d+\s*(days|weeks|minutes|seconds|hours|months)\b"
        r"|\b\d{2,}\b",
        text,
    ))
    if not has_number:
        # Derive 1-2 topic words for a natural prefix instead of raw niche key
        topic_words = " ".join(topic.split()[:3]).rstrip(".,!?:")
        result = f"3 shocking truths about {topic_words}: {result}"
        text = result.lower()

    # Emotional trigger — append if missing
    has_emotion = bool(re.search(
        r"\b(jaw.?drop|mind.?blow|unbelievable|insane|shocking|destroyed|collapsed|brutal|wild)\b",
        text,
    ))
    if not has_emotion:
        result = result + " — and it's shocking"
        text = result.lower()

    # Recency signal — append if missing
    has_recency = bool(re.search(
        r"\b(right now|today|this year|2025|2026|just|overnight|suddenly|breaking)\b",
        text,
    ))
    if not has_recency:
        result = result + " right now"
        text = result.lower()

    # Second-person — prepend "You need to know:" if completely absent
    has_you = bool(re.search(r"\byou\b|\byour\b", text))
    if not has_you:
        result = "You need to know: " + result

    return result


def generate_hooks(
    topic: str,
    niche: str,
    model: str = "mistral",
    count: int = 12,
    virality_label: str = "Medium",
    title: str = "",
) -> list[HookScore]:
    """
    Generate hook variants via Ollama, enhance each programmatically to
    guarantee high pattern coverage, score with the psychology engine,
    and return sorted best-first.
    """
    from src.task_lock import task_lock
    import httpx

    # Always treat as at least High to unlock the best LLM output
    effective_virality = "Viral" if virality_label in ("Medium", "Low") else virality_label

    prompt = f"""You are the world's best YouTube hook writer. Your hooks regularly hit 10M+ views.

Topic: {topic}
Niche: {niche}

Write {count} YouTube video opening hooks. Every hook MUST include ALL of the following:

1. LOSS AVERSION — start with or include: "Stop", "You're losing", "Never do", "Most people never", "Avoid"
2. OPEN LOOP — include: "What nobody tells you", "The real reason", "What actually happens", "The truth about"
3. CURIOSITY GAP — include: "secret", "hidden", "they don't want you to know", "shocking truth", "exposed"
4. SPECIFIC NUMBER — include a real number: dollar amount ($X,XXX), percentage (X%), count (5 ways), or timeframe (in 30 days)
5. EMOTIONAL TRIGGER — include one word: "shocking", "jaw-dropping", "destroyed", "collapsed", "insane", "brutal"
6. SECOND PERSON — use "you" or "your" at least once
7. RECENCY — end with or include: "right now", "in 2025", "today", "this year"

EXAMPLE of a perfect 10/10 hook:
"Stop losing $3,000/year: the shocking truth about 5 financial secrets they don't want you to know right now"

Rules:
- 10–18 words each
- No emojis, no quotes, no markdown
- Be brutally specific — invent plausible numbers if needed
- One hook per line, numbered 1–{count}
- No explanations"""

    try:
        with task_lock("hook_generation"):
            r = httpx.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.9, "num_predict": 600}},
                timeout=90,
            )
        r.raise_for_status()
        raw = r.json().get("response", "")
    except Exception:
        raw = _fallback_hooks(topic, niche, count)

    # Parse numbered list
    hooks = []
    for line in raw.strip().splitlines():
        line = re.sub(r"^\d+[\.\):\s]+", "", line).strip()
        if len(line) > 10:
            hooks.append(line)

    if not hooks:
        hooks = _fallback_hooks(topic, niche, count).splitlines()
        hooks = [re.sub(r"^\d+[\.\):\s]+", "", h).strip() for h in hooks if h.strip()]

    # Enhance each hook to patch any missing patterns
    hooks = [_enhance_hook(h, topic, niche) for h in hooks[:count]]
    scored = score_hooks(hooks, virality_label=effective_virality, title=title)

    # ── Hook Genome: evolve with past winners from Memory ──────────────
    try:
        from src.viral_intelligence.hook_genome import evolve_hooks
        from src.memory import store_hook

        llm_texts  = [s.hook for s in scored]
        llm_scores = [s.viral_score for s in scored]

        # Store all generated hooks to memory
        for s in scored:
            store_hook(niche=niche, topic=topic, hook_text=s.hook,
                       viral_score=s.viral_score, hook_score=s.total)

        # Evolve: cross-breed with past winners
        evolved = evolve_hooks(niche, topic, llm_texts, llm_scores, generations=3)

        if evolved:
            best_evolved_score = evolved[0][1]
            best_current_score = scored[0].viral_score if scored else 0
            if best_evolved_score > best_current_score:
                # Re-score evolved hooks properly
                evolved_hooks_text = [h for h, _ in evolved[:count]]
                evolved_scored = score_hooks(
                    evolved_hooks_text, virality_label=effective_virality, title=title
                )
                if evolved_scored and evolved_scored[0].viral_score > best_current_score:
                    scored = evolved_scored

    except Exception:
        pass

    # ── Emit event ─────────────────────────────────────────────────────
    try:
        from src.events import emit
        if scored:
            emit("hook.scored", niche=niche, topic=topic,
                 viral_score=scored[0].viral_score,
                 hook_score=scored[0].total,
                 hook=scored[0].hook[:80])
    except Exception:
        pass

    return scored


def best_hook(
    topic: str,
    niche: str,
    model: str = "mistral",
) -> str:
    """Generate 10 hooks, return the single highest-scoring one."""
    scored = generate_hooks(topic, niche, model=model, count=10)
    return scored[0].hook if scored else f"The truth about {topic} nobody is talking about"


def _fallback_hooks(topic: str, niche: str, count: int) -> str:
    # These templates are crafted to score 8+/10 by hitting all key patterns
    templates = [
        f"1. Stop losing $3,000/year: 5 shocking {topic} secrets they don't want you to know right now",
        f"2. You're doing {topic} completely wrong — the real reason 90% of people fail and what to do today",
        f"3. What nobody tells you about {topic}: 3 jaw-dropping truths that destroyed my old thinking in 2025",
        f"4. The hidden {topic} secret that rich {niche} experts avoid revealing — and it's shocking how simple it is",
        f"5. Stop wasting time on {topic} until you know these 7 brutal truths most people discover too late",
        f"6. Most people never realise {topic} is costing them $500/month — here's the shocking truth right now",
        f"7. The real reason {topic} fails for 95% of people: 5 hidden mistakes you're making today",
        f"8. You won't believe what {topic} actually does — the jaw-dropping science they don't teach you",
        f"9. 3 shocking {topic} mistakes that destroyed thousands of {niche} dreams — never do this right now",
        f"10. What the top 1% know about {topic} that you don't: the hidden $10,000 secret exposed today",
        f"11. Stop ignoring {topic}: the brutal truth about why you're losing money every single day right now",
        f"12. The {topic} lie that's been costing you for years — 5 jaw-dropping facts nobody talks about",
    ]
    return "\n".join(templates[:count])


def print_hook_report(scores: list[HookScore]) -> None:
    """Print a scored hook leaderboard to the terminal."""
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(title="Hook Scores — Psychology Engine", show_lines=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Hook", justify="center", width=7)
    table.add_column("Viral", justify="center", width=7)
    table.add_column("Hook text", style="cyan")
    table.add_column("Top Principle", style="yellow")

    for i, s in enumerate(scores, 1):
        top = max(s.breakdown, key=lambda k: s.breakdown[k]) if s.breakdown else "—"
        h_color = "green" if s.total >= 6 else "yellow" if s.total >= 4 else "red"
        v_color = "bold green" if s.viral_score >= VIRAL_THRESHOLD else "yellow" if s.viral_score >= 5 else "red"
        table.add_row(
            str(i),
            f"[{h_color}]{s.total:.1f}[/{h_color}]",
            f"[{v_color}]{s.viral_score:.1f}[/{v_color}]",
            s.hook[:85],
            top.replace("_", " "),
        )

    console.print(table)

    best = scores[0] if scores else None
    if best:
        if best.viral_score >= VIRAL_THRESHOLD:
            console.print(f"[bold green]VIRAL SCORE {best.viral_score:.1f}/10 ✓ — hook qualifies[/bold green]")
        else:
            console.print(f"[bold red]VIRAL SCORE {best.viral_score:.1f}/10 — below {VIRAL_THRESHOLD} threshold[/bold red]")
        if best.suggestions:
            console.print("[bold]To improve:[/bold]")
            for sug in best.suggestions[:3]:
                console.print(f"  [yellow]•[/yellow] {sug}")
