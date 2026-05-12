"""
ML Optimization Layer.

Analyses the feedback database to:
  1. Identify high-performing niches, topics, and prompt styles
  2. Recommend better Ollama prompts based on past success
  3. Adjust entropy thresholds automatically
  4. Score and rank niche opportunities

Uses scikit-learn (local, free) — no external ML service needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.ml.feedback import load_all_feedback, VideoFeedback

console = Console()


@dataclass
class OptimizationInsights:
    best_niches: list[tuple[str, float]]            # (niche, avg_score)
    best_models: list[tuple[str, float]]            # (model_name, avg_score)
    recommended_style: str
    recommended_entropy_threshold: float
    top_topics: list[str]
    suggestions: list[str] = field(default_factory=list)


def analyse_feedback() -> OptimizationInsights:
    """
    Read all feedback records and derive optimization recommendations.
    """
    records = load_all_feedback()

    if not records:
        return OptimizationInsights(
            best_niches=[],
            best_models=[],
            recommended_style="engaging and conversational",
            recommended_entropy_threshold=0.55,
            top_topics=[],
            suggestions=["No feedback data yet. Run the pipeline and record results first."],
        )

    # --- Niche performance ---
    niche_scores: dict[str, list[float]] = {}
    for r in records:
        niche_scores.setdefault(r.niche, []).append(r.performance_score)
    best_niches = sorted(
        [(n, round(float(np.mean(s)), 3)) for n, s in niche_scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    # --- Model performance ---
    model_scores: dict[str, list[float]] = {}
    for r in records:
        model_scores.setdefault(r.prompt_model, []).append(r.performance_score)
    best_models = sorted(
        [(m, round(float(np.mean(s)), 3)) for m, s in model_scores.items()],
        key=lambda x: x[1],
        reverse=True,
    )

    # --- Top topics from high performers ---
    high_performers = [r for r in records if r.performance_score >= 0.6]
    top_topics = list({r.topic for r in high_performers})[:10]

    # --- Retention-driven entropy threshold ---
    # If average retention is high → keep threshold as-is
    # If retention is low → raise threshold (force more variety)
    avg_retention = float(np.mean([r.retention_rate for r in records]))
    if avg_retention < 0.35:
        rec_threshold = 0.65
    elif avg_retention < 0.50:
        rec_threshold = 0.55
    else:
        rec_threshold = 0.50

    # --- Style recommendation (simple heuristic) ---
    # Pick the style that correlates with best CTR
    ctr_values = [r.ctr for r in records]
    avg_ctr = float(np.mean(ctr_values)) if ctr_values else 0.0
    if avg_ctr < 0.04:
        style = "shocking and controversial — start with a bold statement"
    elif avg_ctr < 0.07:
        style = "curious and engaging — lead with a surprising question"
    else:
        style = "energetic and direct — get to the value immediately"

    # --- Suggestions ---
    suggestions = []
    if best_niches and best_niches[0][1] < 0.4:
        suggestions.append("All niches performing poorly. Try different topics or improve thumbnail.")
    if avg_retention < 0.4:
        suggestions.append("Low retention. Shorten scripts, add more scene cuts, improve hook.")
    if avg_ctr < 0.03:
        suggestions.append("Low CTR. Test bolder titles and more eye-catching thumbnails.")
    if len(records) < 5:
        suggestions.append("Need more data — run at least 5 videos before trusting these insights.")

    return OptimizationInsights(
        best_niches=best_niches,
        best_models=best_models,
        recommended_style=style,
        recommended_entropy_threshold=rec_threshold,
        top_topics=top_topics,
        suggestions=suggestions,
    )


def get_keyword_context(niche: str, topic: str) -> str:
    """
    Return a short ML context string for keyword generation.
    Tells Ollama which types of footage performed well in the past.
    """
    try:
        records = load_all_feedback()
        relevant = [r for r in records if r.niche == niche or r.topic == topic]
        if not relevant:
            return ""
        high = [r for r in relevant if r.performance_score >= 0.6]
        if not high:
            return ""
        # Pull keywords_used from raw records
        from src.ml.feedback import _load_db
        raw_records = _load_db()
        good_keywords: list[str] = []
        for raw in raw_records:
            if raw.get("niche") == niche and raw.get("entropy_score", 0) >= 0.55:
                good_keywords.extend(raw.get("keywords_used", []))
        if not good_keywords:
            return ""
        from collections import Counter
        top = [kw for kw, _ in Counter(good_keywords).most_common(6)]
        return f"Footage types that worked well before: {', '.join(top)}"
    except Exception:
        return ""


def get_optimized_prompt_prefix(niche: str) -> str:
    """
    Return a learned prompt prefix for the given niche based on past feedback.
    Falls back to a sensible default.
    """
    records = [r for r in load_all_feedback() if r.niche == niche]
    if not records:
        return "Create content that is engaging, educational, and shareable."

    best = max(records, key=lambda r: r.performance_score)
    return (
        f"Based on past successful videos in the '{niche}' niche "
        f"(best CTR: {best.ctr:.1%}, retention: {best.retention_rate:.1%}), "
        f"create content with similar energy and style."
    )


def print_insights(insights: OptimizationInsights) -> None:
    # Niche table
    if insights.best_niches:
        t = Table(title="Niche Performance", show_lines=True)
        t.add_column("Niche", style="cyan")
        t.add_column("Avg Score", justify="right", style="green")
        for niche, score in insights.best_niches:
            color = "green" if score >= 0.6 else ("yellow" if score >= 0.4 else "red")
            t.add_row(niche, f"[{color}]{score}[/{color}]")
        console.print(t)

    console.print(
        Panel(
            f"[cyan]Recommended style:[/cyan] {insights.recommended_style}\n"
            f"[cyan]Entropy threshold:[/cyan] {insights.recommended_entropy_threshold}\n"
            f"[cyan]Top topics:[/cyan] {', '.join(insights.top_topics) or 'N/A'}\n\n"
            + "\n".join(f"[yellow]• {s}[/yellow]" for s in insights.suggestions),
            title="ML Optimization Insights",
            border_style="blue",
        )
    )
