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


def generate_hooks(
    topic: str,
    niche: str,
    model: str = "mistral",
    count: int = 10,
    virality_label: str = "Medium",
    title: str = "",
) -> list[HookScore]:
    """
    Generate `count` hook variants via Ollama, score each with the
    psychology engine, and return them sorted best-first.
    """
    from src.task_lock import task_lock
    import httpx

    virality_instruction = {
        "Low":    "Make the hooks more provocative — this topic needs extra shock value.",
        "Medium": "Balance curiosity and specificity.",
        "High":   "Push for maximum emotional impact and curiosity gap.",
        "Viral":  "Go all-in: the most shocking, jaw-dropping, specific hooks possible. Think MrBeast-level attention grabbing.",
    }.get(virality_label, "")

    prompt = f"""You are a viral YouTube hook writer trained in psychology, NLP, and copywriting.

Topic: {topic}
Niche: {niche}
Virality target: {virality_label}
Instruction: {virality_instruction}

Write {count} different YouTube video opening hooks.
REQUIRED — each hook must combine AT LEAST 2 of these:
- Open loop: "What nobody tells you about…", "The real reason…"
- Loss aversion: "Stop", "You're losing", "Most people never"
- Curiosity gap: "The secret", "Hidden truth", "They don't want you to know"
- Specific numbers: exact dollar amounts, percentages, durations
- Pattern interrupt: counterintuitive or shocking claim
- Emotional trigger: "jaw-dropping", "destroyed", "collapsed", "shocking"

Rules:
- 8–15 words each
- No emojis
- No quotes around the hook
- Be SPECIFIC — use real numbers, real consequences, real stakes
- Return ONLY the list, one hook per line, numbered 1–{count}
- No explanations, no intros"""

    try:
        with task_lock("hook_generation"):
            r = httpx.post(
                "http://localhost:11434/api/generate",
                json={"model": model, "prompt": prompt, "stream": False,
                      "options": {"temperature": 0.95, "num_predict": 512}},
                timeout=90,
            )
        r.raise_for_status()
        raw = r.json().get("response", "")
    except Exception:
        # Fallback: return template hooks if Ollama fails
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

    return score_hooks(hooks[:count], virality_label=virality_label, title=title)


def best_hook(
    topic: str,
    niche: str,
    model: str = "mistral",
) -> str:
    """Generate 10 hooks, return the single highest-scoring one."""
    scored = generate_hooks(topic, niche, model=model, count=10)
    return scored[0].hook if scored else f"The truth about {topic} nobody is talking about"


def _fallback_hooks(topic: str, niche: str, count: int) -> str:
    templates = [
        f"1. Stop making this {topic} mistake — it's costing you everything",
        f"2. What successful {niche} people know about {topic} that you don't",
        f"3. The real reason most people fail at {topic}",
        f"4. I was completely wrong about {topic} until I discovered this",
        f"5. {topic} — what nobody tells you before it's too late",
        f"6. 3 {topic} habits that changed everything in 30 days",
        f"7. The hidden truth about {topic} that experts don't want you to know",
        f"8. You're wasting your time on {topic} without knowing this",
        f"9. Most people get {topic} completely backwards — here's why",
        f"10. The {topic} secret that took me years to figure out",
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
