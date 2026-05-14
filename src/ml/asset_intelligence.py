"""Asset intelligence — learns which footage properties drive retention and CTR.

Analyses historical feedback to identify what visual characteristics (motion,
duration, detected objects, dominant colours) correlate with high-performing
videos, then uses those patterns to score new assets before they are used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from src.logging_config import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class AssetInsight:
    """Learned intelligence about which asset properties drive performance."""

    optimal_motion_score_range: tuple[float, float]   # (min, max) e.g. (0.4, 0.7)
    optimal_duration_range: tuple[float, float]        # seconds, e.g. (5.0, 15.0)
    top_performing_objects: list[str]                  # detected objects in top-25% videos
    top_performing_colors: list[str]                   # dominant colours in top-25% videos
    performance_by_asset_type: dict[str, float]        # {"video": 0.65, "image": 0.42}
    confidence: float                                  # 0–1, min(1.0, sample_count / 20)


# Neutral / empty defaults returned when there is insufficient data
_NEUTRAL_INSIGHT = AssetInsight(
    optimal_motion_score_range=(0.0, 1.0),
    optimal_duration_range=(0.0, 999.0),
    top_performing_objects=[],
    top_performing_colors=[],
    performance_by_asset_type={},
    confidence=0.0,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _quartile_boundary(values: list[float], q: float) -> float:
    """Return the q-th quantile (0–1) of *values* without numpy dependency."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = q * (len(sorted_vals) - 1)
    lo = int(idx)
    hi = lo + 1
    if hi >= len(sorted_vals):
        return sorted_vals[-1]
    frac = idx - lo
    return sorted_vals[lo] + frac * (sorted_vals[hi] - sorted_vals[lo])


def _top_n_counts(items: list[str], n: int = 5) -> list[str]:
    """Return up to *n* most-frequent strings from *items*."""
    if not items:
        return []
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return [k for k, _ in sorted(counts.items(), key=lambda x: x[1], reverse=True)[:n]]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class AssetIntelligence:
    """
    Learns which footage characteristics correlate with video performance.

    Call ``analyse()`` to get an ``AssetInsight``, or ``score_asset()`` to
    rate a single ``MediaAsset`` against the learned model.
    """

    def __init__(self) -> None:
        self._cached_insight: Optional[AssetInsight] = None
        self._cached_niche: Optional[str] = None

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyse(self, niche: str = "general") -> AssetInsight:
        """
        Build an AssetInsight from feedback + vector store metadata.

        Steps:
        1. Load all feedback records.
        2. Find asset_ids used in top-25% and bottom-25% performing videos.
        3. Pull asset metadata from the vector store.
        4. Bin motion_score into quartiles; find which quartile top videos
           predominantly sit in.
        5. Bin duration into short (<5s) / medium (5–15s) / long (>15s).
        6. Collect detected objects and dominant colours from top videos.
        7. Calculate per-asset-type average performance.

        Returns neutral defaults with confidence=0 if there is no data.
        """
        # ---- Step 1: load feedback ----------------------------------------
        try:
            from src.ml.feedback import load_all_feedback
            all_feedback = load_all_feedback()
        except Exception as exc:
            log.warning("asset_intelligence: could not load feedback", error=str(exc))
            return _NEUTRAL_INSIGHT

        # Filter by niche (or use all records if "general" or no niche match)
        niche_records = [r for r in all_feedback if r.niche == niche]
        records = niche_records if niche_records else all_feedback

        if not records:
            log.debug("asset_intelligence: no feedback data available — returning neutral defaults")
            return _NEUTRAL_INSIGHT

        sample_count = len(records)
        confidence = min(1.0, sample_count / 20)

        # ---- Step 2: split top-25% / bottom-25% ----------------------------
        sorted_by_score = sorted(records, key=lambda r: r.performance_score)
        cutoff_n = max(1, len(sorted_by_score) // 4)
        bottom_records = sorted_by_score[:cutoff_n]
        top_records = sorted_by_score[-cutoff_n:]

        top_asset_ids: set[str] = {aid for r in top_records for aid in r.asset_ids}
        bottom_asset_ids: set[str] = {aid for r in bottom_records for aid in r.asset_ids}

        # ---- Step 3: load asset metadata from vector store -----------------
        top_assets = []
        asset_map: dict = {}
        try:
            from src.database.vector_store import get_vector_store
            store = get_vector_store()
            all_store_assets = store.get_all_assets()
            asset_map = {a.asset_id: a for a in all_store_assets}
            top_assets = [asset_map[aid] for aid in top_asset_ids if aid in asset_map]
        except Exception as exc:
            log.warning(
                "asset_intelligence: could not reach vector store — "
                "motion/duration/object analysis skipped",
                error=str(exc),
            )

        # ---- Step 4: optimal motion score range ----------------------------
        motion_scores = [a.motion_score for a in top_assets if 0.0 <= a.motion_score <= 1.0]
        if motion_scores:
            opt_motion = (
                round(_quartile_boundary(motion_scores, 0.25), 3),
                round(_quartile_boundary(motion_scores, 0.75), 3),
            )
        else:
            opt_motion = (0.0, 1.0)

        # ---- Step 5: optimal duration range --------------------------------
        durations = [a.duration_seconds for a in top_assets if a.duration_seconds > 0]
        short_thresh, long_thresh = 5.0, 15.0

        if durations:
            bins: dict[str, list[float]] = {"short": [], "medium": [], "long": []}
            for d in durations:
                if d < short_thresh:
                    bins["short"].append(d)
                elif d <= long_thresh:
                    bins["medium"].append(d)
                else:
                    bins["long"].append(d)

            # Pick the bucket with the most top-performing assets
            best_bucket = max(bins, key=lambda k: len(bins[k]))
            if best_bucket == "short":
                opt_duration = (0.0, short_thresh)
            elif best_bucket == "medium":
                opt_duration = (short_thresh, long_thresh)
            else:
                opt_duration = (long_thresh, 9999.0)
        else:
            opt_duration = (0.0, 9999.0)

        # ---- Step 6: top objects and colours from top videos ---------------
        all_top_objects: list[str] = []
        all_top_colors: list[str] = []
        for a in top_assets:
            all_top_objects.extend(a.detected_objects)
            all_top_colors.extend(a.dominant_colors)

        top_objects = _top_n_counts(all_top_objects, n=10)
        top_colors = _top_n_counts(all_top_colors, n=5)

        # Remove objects that also appear heavily in bottom-performing assets
        # (they are not discriminative)
        if bottom_asset_ids:
            bottom_assets = [asset_map[aid] for aid in bottom_asset_ids if aid in asset_map]
            bottom_objects = set(_top_n_counts(
                [o for a in bottom_assets for o in a.detected_objects], n=10
            ))
            top_objects = [o for o in top_objects if o not in bottom_objects]

        # ---- Step 7: performance by asset type -----------------------------
        type_scores: dict[str, list[float]] = {}
        for r in records:
            for aid in r.asset_ids:
                if aid in asset_map:
                    atype = asset_map[aid].asset_type
                    type_scores.setdefault(atype, []).append(r.performance_score)

        perf_by_type: dict[str, float] = {
            atype: round(sum(scores) / len(scores), 3)
            for atype, scores in type_scores.items()
            if scores
        }

        insight = AssetInsight(
            optimal_motion_score_range=opt_motion,
            optimal_duration_range=opt_duration,
            top_performing_objects=top_objects,
            top_performing_colors=top_colors,
            performance_by_asset_type=perf_by_type,
            confidence=round(confidence, 3),
        )

        # Cache for score_asset() calls
        self._cached_insight = insight
        self._cached_niche = niche

        log.info(
            "asset_intelligence: analysis complete",
            niche=niche,
            sample_count=sample_count,
            confidence=insight.confidence,
            top_objects=top_objects[:5],
        )
        return insight

    # ------------------------------------------------------------------
    # Asset scoring
    # ------------------------------------------------------------------

    def score_asset(self, asset, niche: str = "general") -> float:
        """
        Score a MediaAsset 0–1 based on learned intelligence.

        Components (equal weight when data is available):
        - motion score within optimal range    → +0.33
        - duration within optimal range        → +0.33
        - detected objects overlap with top    → +0.33

        Returns 0.5 (neutral) if no data or confidence is zero.
        """
        # Use cached insight if available for the same niche
        if self._cached_insight is None or self._cached_niche != niche:
            insight = self.analyse(niche)
        else:
            insight = self._cached_insight

        if insight.confidence == 0.0:
            return 0.5

        scores: list[float] = []

        # Motion score component
        m_lo, m_hi = insight.optimal_motion_score_range
        if m_hi > m_lo:  # range is meaningful
            in_range = m_lo <= asset.motion_score <= m_hi
            scores.append(1.0 if in_range else 0.2)

        # Duration component
        d_lo, d_hi = insight.optimal_duration_range
        if d_hi < 9999.0 or d_lo > 0.0:
            in_range = d_lo <= asset.duration_seconds <= d_hi
            scores.append(1.0 if in_range else 0.2)

        # Object overlap component
        if insight.top_performing_objects and asset.detected_objects:
            top_set = set(insight.top_performing_objects)
            asset_objs = set(asset.detected_objects)
            overlap = len(top_set & asset_objs) / max(len(top_set), 1)
            scores.append(min(1.0, overlap * 2))  # scale: 50% overlap → 1.0
        elif insight.top_performing_objects:
            scores.append(0.3)  # penalise slightly for no detected objects

        if not scores:
            return 0.5

        raw = sum(scores) / len(scores)
        # Blend with neutral based on confidence
        return round(0.5 * (1 - insight.confidence) + raw * insight.confidence, 3)

    # ------------------------------------------------------------------
    # Rich report
    # ------------------------------------------------------------------

    def print_report(self, niche: str = "general") -> None:
        """Print a Rich panel summarising the asset intelligence for *niche*."""
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table
            console = Console()
        except ImportError:
            print("rich not installed — cannot print report")
            return

        insight = self.analyse(niche)

        if insight.confidence == 0.0:
            console.print(
                Panel(
                    "[yellow]No feedback data available yet.[/yellow]\n"
                    "Run the pipeline and upload videos to collect performance data.",
                    title=f"Asset Intelligence — {niche}",
                    border_style="yellow",
                )
            )
            return

        m_lo, m_hi = insight.optimal_motion_score_range
        d_lo, d_hi = insight.optimal_duration_range
        d_hi_str = f"{d_hi:.0f}s" if d_hi < 9999.0 else "∞"

        lines = [
            f"[cyan]Confidence:[/cyan]            {insight.confidence:.0%} "
            f"([dim]based on ≥{max(1, round(insight.confidence * 20))} videos[/dim])",
            "",
            f"[cyan]Optimal motion score:[/cyan]  {m_lo:.2f} – {m_hi:.2f}",
            f"[cyan]Optimal clip duration:[/cyan] {d_lo:.0f}s – {d_hi_str}",
        ]

        if insight.top_performing_objects:
            lines.append(
                f"[cyan]Top objects:[/cyan]           "
                + ", ".join(insight.top_performing_objects[:8])
            )
        if insight.top_performing_colors:
            lines.append(
                f"[cyan]Top colours:[/cyan]           "
                + ", ".join(insight.top_performing_colors)
            )

        if insight.performance_by_asset_type:
            type_strs = [
                f"{t}: {s:.2f}" for t, s in insight.performance_by_asset_type.items()
            ]
            lines.append(f"[cyan]Performance by type:[/cyan]   " + "  |  ".join(type_strs))

        # Confidence bar
        bar_filled = int(insight.confidence * 20)
        bar = "[green]" + "█" * bar_filled + "[/green]" + "░" * (20 - bar_filled)
        lines.extend(["", f"[dim]Confidence:[/dim] {bar}"])

        table: Table | None = None
        if insight.top_performing_objects:
            table = Table(title="Top-performing detected objects", show_lines=False, box=None)
            table.add_column("Rank", style="dim", width=5)
            table.add_column("Object", style="cyan")
            for i, obj in enumerate(insight.top_performing_objects[:10], 1):
                table.add_row(str(i), obj)

        console.print(
            Panel(
                "\n".join(lines),
                title=f"[bold]Asset Intelligence — {niche}[/bold]",
                border_style="blue",
            )
        )
        if table is not None:
            console.print(table)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_intelligence: Optional[AssetIntelligence] = None


def get_asset_intelligence() -> AssetIntelligence:
    """Return the process-wide AssetIntelligence singleton."""
    global _intelligence
    if _intelligence is None:
        _intelligence = AssetIntelligence()
    return _intelligence
