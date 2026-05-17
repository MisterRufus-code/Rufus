"""
Hook Genome — evolutionary algorithm for viral hook generation.

Instead of generating hooks from scratch every run, the Genome:
  1. Seeds itself with every hook ever generated (from Memory)
  2. Selects the fittest (highest viral_score + retention + views)
  3. Crosses over pairs of winners to breed new candidates
  4. Mutates offspring with targeted word swaps and structure changes
  5. Scores the new generation with the psychology engine
  6. Returns the single best hook — better than any LLM can produce alone

Over time the gene pool fills with proven winners. The system
converges on niche-specific language that actually retains viewers.

Biology analogy:
  Gene     = a hook string
  Fitness  = viral_score × retention × view_weight
  Mutation = word substitution, structure swap, number injection
  Crossover= take first clause of parent A + second clause of parent B
  Selection= tournament (top-k compete, winner advances)
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional

from rich.console import Console

console = Console()


# ---------------------------------------------------------------------------
# Mutation operators
# ---------------------------------------------------------------------------

_POWER_STARTS = [
    "Stop", "Never", "You're losing", "Most people never",
    "The shocking truth about", "What nobody tells you about",
    "The real reason", "You won't believe what",
    "They don't want you to know about", "I was completely wrong about",
]

_EMOTIONAL_SWAPS = {
    "important":   "shocking",
    "interesting": "jaw-dropping",
    "bad":         "devastating",
    "good":        "game-changing",
    "big":         "massive",
    "small":       "tiny",
    "wrong":       "destroying your",
    "problem":     "costly mistake",
    "tip":         "secret",
    "way":         "method",
    "thing":       "truth",
}

_NUMBER_INJECTIONS = [
    ("N ways",         lambda: f"{random.choice([3,5,7,9,10])} ways"),
    ("N secrets",      lambda: f"{random.choice([3,5,7])} secrets"),
    ("N mistakes",     lambda: f"{random.choice([3,5,7])} mistakes"),
    ("in N days",      lambda: f"in {random.choice([7,14,21,30])} days"),
    ("$N",             lambda: f"{random.choice([100, 500, 1000, 5000, 10000])} dollars"),
    ("N%",             lambda: f"{random.choice([80,90,95,99])}%"),
]

_RECENCY_SUFFIXES = [
    "right now", "in 2025", "today", "this year",
    "before it's too late", "and it's shocking",
]


def _mutate(hook: str, strength: float = 0.3) -> str:
    """Apply random mutations to a hook string."""
    result = hook

    # 1. Power start replacement
    if random.random() < strength:
        first_word = result.split()[0].lower() if result.split() else ""
        if first_word not in {"stop", "never", "you're", "most", "the", "what"}:
            new_start = random.choice(_POWER_STARTS)
            # Keep the rest of the hook after the first clause
            parts = re.split(r"[,:—–]", result, maxsplit=1)
            rest  = parts[1].strip() if len(parts) > 1 else result
            result = f"{new_start} {rest}"

    # 2. Emotional word swap
    if random.random() < strength:
        for bland, vivid in _EMOTIONAL_SWAPS.items():
            pattern = rf"\b{re.escape(bland)}\b"
            if re.search(pattern, result, re.IGNORECASE):
                result = re.sub(pattern, vivid, result, count=1, flags=re.IGNORECASE)
                break

    # 3. Number injection (if no number present) — prefix format, not mid-sentence insert
    if random.random() < strength:
        if not re.search(r"\b\d+\b|\$", result):
            label, make_num = random.choice(_NUMBER_INJECTIONS)
            num = make_num()
            if label.startswith("N ways"):
                result = f"{num} to {result[0].lower()}{result[1:]}"
            elif label.startswith("N secrets"):
                result = f"{num} {result[0].lower()}{result[1:]}"
            elif label.startswith("in N"):
                result = result.rstrip(".,!?") + f" — achieve this {num}"
            elif label.startswith("$"):
                result = result.rstrip(".,!?") + f" (costs you {num}/year)"
            elif label.startswith("N%"):
                result = f"{num} of people don't know: {result[0].lower()}{result[1:]}"
            else:
                result = result.rstrip(".,!?") + f" — {num}"

    # 4. Recency suffix
    if random.random() < strength * 0.5:
        has_recency = re.search(
            r"\b(right now|today|this year|2025|before it)\b", result, re.I
        )
        if not has_recency:
            result = result.rstrip(".,!?") + " " + random.choice(_RECENCY_SUFFIXES)

    return result.strip()


def _crossover(parent_a: str, parent_b: str) -> str:
    """
    Breed two hooks: take the opening clause of A + the closing clause of B.
    Falls back to word-level midpoint if no clause boundary found.
    """
    # Try clause split on punctuation
    for delim in [",", ":", "—", "–", " — "]:
        if delim in parent_a and delim in parent_b:
            a_start = parent_a.split(delim)[0].strip()
            b_end   = parent_b.split(delim, 1)[-1].strip()
            child   = f"{a_start}{delim} {b_end}"
            if 6 <= len(child.split()) <= 22:
                return child

    # Word-level midpoint
    words_a = parent_a.split()
    words_b = parent_b.split()
    mid_a   = len(words_a) // 2
    mid_b   = len(words_b) // 2
    child   = " ".join(words_a[:mid_a] + words_b[mid_b:])
    return child if len(child.split()) >= 5 else parent_a


# ---------------------------------------------------------------------------
# Genome class
# ---------------------------------------------------------------------------

@dataclass
class Gene:
    hook:     str
    score:    float  = 0.0
    niche:    str    = ""
    hook_id:  int    = 0

    def fitness(self) -> float:
        return self.score


class HookGenome:
    """
    Evolves a population of hooks using selection → crossover → mutation.
    Seeds from Memory (past winners) + fresh LLM-generated candidates.
    """

    def __init__(self, niche: str, population_size: int = 20):
        self.niche           = niche
        self.population_size = population_size
        self._pool: list[Gene] = []

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_from_memory(self) -> int:
        """Load top past hooks from the Memory Layer as initial population."""
        try:
            from src.memory import top_hooks
            memories = top_hooks(self.niche, min_score=5.0, limit=self.population_size)
            for m in memories:
                self._pool.append(Gene(
                    hook    = m.hook_text,
                    score   = m.fitness(),
                    niche   = self.niche,
                    hook_id = m.id,
                ))
            return len(memories)
        except Exception:
            return 0

    def seed_from_list(self, hooks: list[str], scores: list[float]) -> None:
        """Add fresh LLM-generated hooks to the pool."""
        for hook, score in zip(hooks, scores):
            self._pool.append(Gene(hook=hook, score=score, niche=self.niche))

    # ------------------------------------------------------------------
    # Evolution
    # ------------------------------------------------------------------

    def _tournament_select(self, k: int = 3) -> Gene:
        """Select a winner via tournament — k random candidates compete."""
        contestants = random.sample(self._pool, min(k, len(self._pool)))
        return max(contestants, key=lambda g: g.fitness())

    def _score_hook(self, hook: str, topic: str = "") -> float:
        """Score a hook using the psychology engine."""
        try:
            from src.psychology.hooks import score_hook
            result = score_hook(hook, virality_label="Viral", title=topic)
            return result.viral_score
        except Exception:
            return 5.0

    def evolve(self, generations: int = 4, topic: str = "") -> list[Gene]:
        """
        Run the evolutionary loop and return the final sorted population.

        generations: number of evolution cycles
        topic:       used for scoring context
        """
        if len(self._pool) < 2:
            return self._pool

        for gen in range(generations):
            offspring: list[Gene] = []

            # Elitism — keep top 30% unchanged
            self._pool.sort(key=lambda g: g.fitness(), reverse=True)
            elite_count = max(2, len(self._pool) // 3)
            offspring.extend(self._pool[:elite_count])

            # Fill rest with crossover + mutation
            while len(offspring) < self.population_size:
                parent_a = self._tournament_select()
                # Force different parent for crossover diversity
                for _ in range(5):
                    parent_b = self._tournament_select()
                    if parent_b.hook != parent_a.hook:
                        break

                if parent_a.hook == parent_b.hook:
                    # Pure mutation when pool has low diversity
                    child_hook = _mutate(parent_a.hook, strength=0.9)
                else:
                    child_hook = _crossover(parent_a.hook, parent_b.hook)
                    child_hook = _mutate(child_hook, strength=0.4)

                child_score = self._score_hook(child_hook, topic=topic)
                offspring.append(Gene(
                    hook  = child_hook,
                    score = child_score,
                    niche = self.niche,
                ))

            self._pool = offspring

        # Deduplicate — keep only unique hooks (by normalized text)
        seen: set[str] = set()
        unique: list[Gene] = []
        for g in sorted(self._pool, key=lambda x: x.fitness(), reverse=True):
            key = g.hook.lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(g)
        self._pool = unique

        best = self._pool[0] if self._pool else None
        if best:
            console.print(
                f"[cyan]HookGenome[/cyan] gen={generations} "
                f"best=[bold]{best.score:.2f}/10[/bold] "
                f"unique={len(self._pool)}"
            )

        return self._pool

    def best(self) -> Optional[str]:
        """Return the single best hook from the current population."""
        if self._pool:
            return self._pool[0].hook
        return None

    def top_n(self, n: int = 5) -> list[tuple[str, float]]:
        """Return top N unique (hook, score) pairs."""
        return [(g.hook, g.score) for g in self._pool[:n]]


# ---------------------------------------------------------------------------
# High-level convenience
# ---------------------------------------------------------------------------

def evolve_hooks(
    niche:    str,
    topic:    str,
    llm_hooks: list[str],
    llm_scores: list[float],
    generations: int = 4,
) -> list[tuple[str, float]]:
    """
    Full pipeline:
      1. Seed genome with past winners from Memory
      2. Add fresh LLM-generated hooks
      3. Evolve for N generations
      4. Return top hooks sorted by fitness

    Returns: list of (hook_text, score) tuples
    """
    genome = HookGenome(niche=niche)
    seeded = genome.seed_from_memory()
    genome.seed_from_list(llm_hooks, llm_scores)

    if seeded > 0:
        console.print(f"[dim]HookGenome: {seeded} winners loaded from memory[/dim]")

    genome.evolve(generations=generations, topic=topic)
    return genome.top_n(n=10)
