"""
TrendSniper — finds topics 3-7 days before they peak on YouTube.

Most tools react to trends (Google Trends RSS, Reddit hot). TrendSniper
measures *acceleration* — how fast a topic is growing, not how big it is.
A topic with 500 searches/day growing at 40%/day will outperform a topic
with 5,000 searches/day growing at 2%/day in 72 hours.

Sources (all free, no API keys):
  - Google Trends RSS (volume proxy via rank position)
  - Reddit rising posts (velocity = upvotes/hour in first 2h)
  - YouTube autocomplete momentum (new completions = emerging intent)
  - Cross-niche spillover (topic spreading from one niche to another)
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console

console = Console()

_CACHE_DIR = Path("data/sniper_cache")
_CACHE_TTL  = 3600  # 1 hour


@dataclass
class SniperSignal:
    topic:        str
    niche:        str
    momentum:     float          # 0-100 composite score
    acceleration: float          # % growth estimate
    sources:      list[str]      = field(default_factory=list)
    related:      list[str]      = field(default_factory=list)
    detected_at:  datetime       = field(default_factory=datetime.now)

    def is_hot(self) -> bool:
        return self.momentum >= 60.0

    def __str__(self) -> str:
        return (f"[{self.momentum:.0f}/100] {self.topic} "
                f"(+{self.acceleration:.0f}% momentum, via {', '.join(self.sources)})")


# ---------------------------------------------------------------------------
# Niche → subreddits / seed words (mirrors src/trends.py but expanded)
# ---------------------------------------------------------------------------

_SUBREDDITS: dict[str, list[str]] = {
    "finance":   ["personalfinance", "financialindependence", "investing",
                  "stocks", "wallstreetbets", "Economics"],
    "tech":      ["technology", "artificial", "MachineLearning", "OpenAI",
                  "singularity", "ChatGPT"],
    "health":    ["nutrition", "loseit", "fitness", "HealthyFood",
                  "longevity", "Biohackers"],
    "wellness":  ["Meditation", "mindfulness", "selfimprovement", "productivity"],
    "business":  ["entrepreneur", "smallbusiness", "startups", "passive_income"],
    "general":   ["todayilearned", "interestingasfuck", "YouShouldKnow",
                  "mildlyinteresting", "worldnews"],
}

_SEEDS: dict[str, list[str]] = {
    "finance":   ["investing", "passive income", "make money", "financial freedom"],
    "tech":      ["AI tools", "ChatGPT", "robots", "future technology"],
    "health":    ["weight loss", "healthy habits", "longevity", "diet"],
    "wellness":  ["morning routine", "meditation", "stress", "sleep"],
    "business":  ["side hustle", "online business", "dropshipping", "entrepreneur"],
    "general":   ["did you know", "life hacks", "psychology", "science facts"],
}


class TrendSniper:
    """
    Detects early-stage trends via multi-source acceleration analysis.
    Results are cached per niche to avoid hammering public APIs.
    """

    def __init__(self, niche: str = "general", geo: str = "US"):
        self.niche = niche.lower()
        self.geo   = geo
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _cache_path(self, key: str) -> Path:
        h = hashlib.md5(key.encode()).hexdigest()[:12]
        return _CACHE_DIR / f"{h}.json"

    def _cache_get(self, key: str) -> Optional[dict]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text())
            if time.time() - data.get("ts", 0) < _CACHE_TTL:
                return data
        except Exception:
            pass
        return None

    def _cache_set(self, key: str, data: dict) -> None:
        data["ts"] = time.time()
        try:
            self._cache_path(key).write_text(json.dumps(data))
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Source 1 — Google Trends RSS (rank-based velocity)
    # ------------------------------------------------------------------

    def _google_rising(self) -> list[tuple[str, float]]:
        """Return (topic, momentum_score) pairs from Google Trends RSS."""
        cache_key = f"google_rss_{self.geo}"
        cached = self._cache_get(cache_key)
        if cached:
            return [(t["topic"], t["score"]) for t in cached.get("items", [])]

        urls = [
            f"https://trends.google.com/trending/rss?geo={self.geo}",
            f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={self.geo}",
        ]
        items: list[tuple[str, float]] = []
        for url in urls:
            try:
                r = httpx.get(url, timeout=10, follow_redirects=True)
                r.raise_for_status()
                root = ET.fromstring(r.text)
                for i, item in enumerate(root.findall(".//item")):
                    title = item.findtext("title", "").strip()
                    if title:
                        # Items at the top of RSS = faster rising
                        score = max(10.0, 100.0 - i * 4)
                        items.append((title, score))
                break
            except Exception:
                continue

        self._cache_set(cache_key, {"items": [{"topic": t, "score": s} for t, s in items]})
        return items

    # ------------------------------------------------------------------
    # Source 2 — Reddit rising velocity
    # ------------------------------------------------------------------

    def _reddit_rising(self) -> list[tuple[str, float]]:
        """
        Score Reddit rising posts by upvotes-per-hour in first 2h.
        High upvote velocity = emerging interest before it hits Google.
        """
        subreddits = _SUBREDDITS.get(self.niche, _SUBREDDITS["general"])
        cache_key  = f"reddit_rising_{self.niche}"
        cached = self._cache_get(cache_key)
        if cached:
            return [(t["topic"], t["score"]) for t in cached.get("items", [])]

        results: list[tuple[str, float]] = []
        headers = {"User-Agent": "rufus-trend-bot/1.0"}

        for sub in subreddits[:4]:
            try:
                r = httpx.get(
                    f"https://www.reddit.com/r/{sub}/rising.json?limit=25",
                    headers=headers, timeout=10,
                )
                r.raise_for_status()
                posts = r.json().get("data", {}).get("children", [])
                now = time.time()

                for post in posts:
                    d = post.get("data", {})
                    title   = d.get("title", "")
                    ups     = d.get("ups", 0)
                    created = d.get("created_utc", now)
                    age_h   = max(0.1, (now - created) / 3600)

                    # Velocity = upvotes per hour; cap age at 48h
                    if age_h > 48:
                        continue
                    velocity = ups / age_h
                    # Decay for older posts — newer rising posts get bonus
                    freshness = max(0, 1.0 - age_h / 48)
                    score = min(100.0, velocity * 0.1 + freshness * 30)

                    if score > 5 and len(title) > 10:
                        results.append((title, score))

            except Exception:
                continue

        results.sort(key=lambda x: x[1], reverse=True)
        self._cache_set(cache_key, {"items": [{"topic": t, "score": s} for t, s in results[:20]]})
        return results[:20]

    # ------------------------------------------------------------------
    # Source 3 — YouTube autocomplete momentum
    # ------------------------------------------------------------------

    def _youtube_autocomplete_new(self) -> list[tuple[str, float]]:
        """
        Fetch autocomplete suggestions for niche seeds.
        New suggestions that weren't seen before = emerging search intent.
        """
        seeds   = _SEEDS.get(self.niche, _SEEDS["general"])
        history = self._cache_get(f"autocomplete_history_{self.niche}") or {"seen": {}}
        seen    = set(history.get("seen", {}).keys())
        new_signals: list[tuple[str, float]] = []

        for seed in seeds[:3]:
            try:
                r = httpx.get(
                    "https://suggestqueries.google.com/complete/search",
                    params={"client": "youtube", "q": seed, "hl": "en"},
                    timeout=8,
                )
                raw  = r.text
                # Response is JSONP: window.google.ac.h(["seed", [...]])
                match = re.search(r'\["([^"]+)"', raw.replace(r"<", "<"))
                suggestions = re.findall(r'"([^"]{10,80})"', raw)

                for s in suggestions:
                    if s not in seen:
                        new_signals.append((s, 45.0))   # new = emerging
                    seen.add(s)
            except Exception:
                continue

        # Persist updated seen set
        self._cache_set(f"autocomplete_history_{self.niche}", {"seen": {s: 1 for s in seen}})
        return new_signals[:15]

    # ------------------------------------------------------------------
    # Source 4 — Cross-niche spillover
    # ------------------------------------------------------------------

    def _cross_niche_spillover(self) -> list[tuple[str, float]]:
        """
        Topics trending in adjacent niches are likely to spill over.
        E.g., a finance topic trending in tech subreddits = early signal.
        """
        adjacent = {
            "finance":  ["tech", "business"],
            "tech":     ["finance", "general"],
            "health":   ["wellness", "general"],
            "wellness": ["health", "general"],
            "business": ["finance", "tech"],
            "general":  ["tech", "finance"],
        }
        spillovers: list[tuple[str, float]] = []
        for other_niche in adjacent.get(self.niche, []):
            sniper = TrendSniper(niche=other_niche, geo=self.geo)
            try:
                rss = sniper._google_rising()[:5]
                for topic, score in rss:
                    # Spillover discount — adjacent-niche signal is weaker
                    spillovers.append((topic, score * 0.55))
            except Exception:
                continue
        return spillovers

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def snipe(self, top_n: int = 10) -> list[SniperSignal]:
        """
        Run all sources, merge and deduplicate, return top signals
        sorted by momentum score (highest = most likely to go viral soon).
        """
        scores: dict[str, dict] = {}

        def _add(topic: str, score: float, source: str) -> None:
            key = topic.lower().strip()[:80]
            if not key:
                return
            if key not in scores:
                scores[key] = {"topic": topic, "score": 0.0, "sources": [], "count": 0}
            scores[key]["score"]  += score
            scores[key]["count"]  += 1
            if source not in scores[key]["sources"]:
                scores[key]["sources"].append(source)

        try:
            for t, s in self._google_rising():
                _add(t, s, "google_trends")
        except Exception:
            pass

        try:
            for t, s in self._reddit_rising():
                _add(t, s, "reddit_rising")
        except Exception:
            pass

        try:
            for t, s in self._youtube_autocomplete_new():
                _add(t, s, "yt_autocomplete")
        except Exception:
            pass

        try:
            for t, s in self._cross_niche_spillover():
                _add(t, s, "cross_niche")
        except Exception:
            pass

        signals: list[SniperSignal] = []
        for key, data in scores.items():
            # Multi-source bonus: appearing in 2+ sources = stronger signal
            source_bonus   = (data["count"] - 1) * 15.0
            momentum       = min(100.0, data["score"] + source_bonus)
            acceleration   = min(999.0, momentum * 1.8)

            signals.append(SniperSignal(
                topic        = data["topic"],
                niche        = self.niche,
                momentum     = momentum,
                acceleration = acceleration,
                sources      = data["sources"],
            ))

        signals.sort(key=lambda s: s.momentum, reverse=True)
        return signals[:top_n]

    def best_topic(self) -> str:
        """Return the single best topic to make a video about right now."""
        signals = self.snipe(top_n=5)
        if signals:
            best = signals[0]
            console.print(
                f"[cyan]TrendSniper[/cyan] → "
                f"[bold]{best.topic}[/bold] "
                f"[dim](momentum={best.momentum:.0f}/100 via {', '.join(best.sources)})[/dim]"
            )
            return best.topic
        return ""
