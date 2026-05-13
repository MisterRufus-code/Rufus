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
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class HookScore:
    hook: str
    total: float                            # 0–10 composite
    breakdown: dict[str, float] = field(default_factory=dict)
    suggestions: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.total:.1f}/10 — {self.hook[:80]}"


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
            r"\b\d+[\s,]*(percent|%|million|thousand|people)\b",
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
]

_MAX_SCORE = sum(w * m for _, _, w, m in _PATTERNS)


def score_hook(hook: str) -> HookScore:
    """
    Score a hook string on psychological engagement principles.
    Returns a HookScore with total 0–10 and per-principle breakdown.
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

    total = round(min(raw_total / _MAX_SCORE * 10, 10.0), 2)
    return HookScore(hook=hook, total=total, breakdown=breakdown, suggestions=suggestions)


def score_hooks(hooks: list[str]) -> list[HookScore]:
    """Score and rank a list of hooks. Best hook is first."""
    return sorted([score_hook(h) for h in hooks], key=lambda s: s.total, reverse=True)


def generate_hooks(
    topic: str,
    niche: str,
    model: str = "mistral",
    count: int = 10,
) -> list[HookScore]:
    """
    Generate `count` hook variants via Ollama, score each with the
    psychology engine, and return them sorted best-first.
    """
    from src.task_lock import task_lock
    import httpx

    prompt = f"""You are a YouTube hook writer trained in psychology and copywriting.

Topic: {topic}
Niche: {niche}

Write {count} different YouTube video hooks (opening sentences / titles).
Each hook must use at least one of these psychological principles:
- Open loop (incomplete thought the brain must resolve)
- Loss aversion (stop, mistake, avoid, wasting)
- Curiosity gap (secret, hidden, nobody knows)
- Social proof (specific numbers, "most people")
- Pattern interrupt (counterintuitive, surprising claim)
- Second person ("you", "your")

Rules:
- 8–15 words each
- No emojis
- No quotes around the hook
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
        line = re.sub(r"^\d+[\.\)]\s*", "", line).strip()
        if len(line) > 10:
            hooks.append(line)

    if not hooks:
        hooks = _fallback_hooks(topic, niche, count).splitlines()
        hooks = [re.sub(r"^\d+[\.\)]\s*", "", h).strip() for h in hooks if h.strip()]

    return score_hooks(hooks[:count])


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
    table.add_column("Score", justify="center", width=7)
    table.add_column("Hook", style="cyan")
    table.add_column("Top Principle", style="yellow")

    for i, s in enumerate(scores, 1):
        top = max(s.breakdown, key=lambda k: s.breakdown[k]) if s.breakdown else "—"
        color = "green" if s.total >= 6 else "yellow" if s.total >= 4 else "red"
        table.add_row(
            str(i),
            f"[{color}]{s.total:.1f}[/{color}]",
            s.hook[:90],
            top.replace("_", " "),
        )

    console.print(table)

    best = scores[0] if scores else None
    if best and best.suggestions:
        console.print("\n[bold]Improvement suggestions for top hook:[/bold]")
        for sug in best.suggestions:
            console.print(f"  [yellow]•[/yellow] {sug}")
