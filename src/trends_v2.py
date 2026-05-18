"""
trends_v2.py — multi-source trend aggregation for true viral signal.

The existing `src.trends` module (Google Trends + YouTube trending) lags the
real viral wave by 12-48 hours. By the time something is on Google Trends,
the early movers have already captured the audience.

This module pulls from 6 sources, scores momentum, dedupes against memory,
and returns ranked topic candidates. All sources are free, no API keys
required for the basic configuration.

Sources (ordered by leverage):
  1. Reddit /r/all + niche subs              ←  2-6h lead, momentum signal
  2. Nitter (X/Twitter mirror)               ←  <1h, real-time
  3. Google News RSS by niche query          ←  30min, hard news
  4. YouTube Data API mostPopular            ←  12-24h, confirmed viral
  5. HackerNews top + Show HN (tech/AI)      ←  6h, niche signal
  6. Wikipedia top trending pages            ←  daily, search volume proxy

Usage:
    from src.trends_v2 import discover_trending_topics
    topics = discover_trending_topics(niche="dark_triad_ai", limit=20)
    # → [TrendingTopic(...), ...] sorted by momentum, deduped against memory

Drop this in as `src/trends_v2.py`. The orchestrator can be wired up by
replacing the import at orchestrator.py:2771 with:
    from src.trends_v2 import discover_trending_topics
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

# httpx is already a project dependency
import httpx


# ---------------------------------------------------------------------------
# Niche → seed query map. Add new niches as you create their YAMLs.
# ---------------------------------------------------------------------------
NICHE_QUERIES: dict[str, dict[str, list[str]]] = {
    "dark_triad_ai": {
        "subreddits": ["singularity", "ArtificialInteligence", "Futurology",
                       "economy", "collapse", "BasicIncome", "geopolitics"],
        "google_news": ["AI job displacement", "CBDC central bank digital",
                        "AI automation jobs", "tech billionaire power",
                        "OpenAI Anthropic regulation"],
        "twitter": ["AI jobs", "CBDC", "automation", "wealth gap"],
        "yt_categories": ["25"],  # News & Politics
    },
    "finance": {
        "subreddits": ["personalfinance", "investing", "stocks", "wallstreetbets",
                       "Economics", "financialindependence"],
        "google_news": ["stock market", "Federal Reserve", "inflation",
                        "interest rates", "recession"],
        "twitter": ["markets", "stocks", "earnings"],
        "yt_categories": ["25"],
    },
    "tech": {
        "subreddits": ["technology", "MachineLearning", "programming",
                       "LocalLLaMA", "OpenAI", "ChatGPT"],
        "google_news": ["AI breakthrough", "tech layoffs", "open source AI",
                        "GPU shortage"],
        "twitter": ["AI", "ChatGPT", "Claude"],
        "yt_categories": ["28"],  # Science & Tech
    },
    "health": {
        "subreddits": ["nutrition", "loseit", "fitness", "longevity",
                       "supplements", "ScienceBasedParenting"],
        "google_news": ["weight loss", "longevity research", "supplement",
                        "Ozempic", "biohacking"],
        "twitter": ["health", "longevity", "Ozempic"],
        "yt_categories": ["26"],  # Howto & Style
    },
    "general": {
        "subreddits": ["all"],
        "google_news": ["trending", "viral"],
        "twitter": [],
        "yt_categories": ["22"],
    },
}


# Nitter instances — these go up and down. Health-check before use.
NITTER_INSTANCES = [
    "https://nitter.net",
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
]


@dataclass
class TrendingTopic:
    keyword: str
    source: str           # "reddit", "nitter", "gnews", "youtube", "hn", "wikipedia"
    momentum: float       # 0-100, higher = more momentum NOW
    raw_score: float      # source-specific (upvotes, view count, etc.)
    url: str = ""
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    def fingerprint(self) -> str:
        """Lowercase normalized form for dedup."""
        return re.sub(r"[^\w\s]", "", self.keyword.lower()).strip()


# ---------------------------------------------------------------------------
# Source 1: Reddit
# ---------------------------------------------------------------------------

def _reddit_rising(subreddit: str, limit: int = 25, timeout: float = 10.0) -> list[TrendingTopic]:
    """Pull rising posts from a subreddit. No auth needed for public JSON."""
    url = f"https://www.reddit.com/r/{subreddit}/rising.json?limit={limit}"
    headers = {"User-Agent": "Rufus-Trends/2.0"}
    out: list[TrendingTopic] = []
    try:
        r = httpx.get(url, headers=headers, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        now = time.time()
        for child in data.get("data", {}).get("children", []):
            p = child.get("data", {})
            title = (p.get("title") or "").strip()
            if not title or len(title) < 15:
                continue
            ups = p.get("ups", 0)
            age_hours = max(1, (now - p.get("created_utc", now)) / 3600)
            # Momentum = upvotes per hour, capped
            ups_per_hour = ups / age_hours
            momentum = min(100, ups_per_hour / 5)  # 500 ups/hr → 100
            out.append(TrendingTopic(
                keyword=title,
                source=f"reddit/r/{subreddit}",
                momentum=momentum,
                raw_score=float(ups),
                url=f"https://reddit.com{p.get('permalink', '')}",
                metadata={"subreddit": subreddit, "comments": p.get("num_comments", 0)},
            ))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Source 2: Nitter (X/Twitter via mirror, no API key)
# ---------------------------------------------------------------------------

def _nitter_search(query: str, limit: int = 20, timeout: float = 10.0) -> list[TrendingTopic]:
    """Scrape Nitter search RSS. Falls through Nitter instances if some are down."""
    out: list[TrendingTopic] = []
    encoded = urllib.parse.quote(query)
    for base in NITTER_INSTANCES:
        try:
            url = f"{base}/search/rss?f=tweets&q={encoded}&since=1d"
            headers = {"User-Agent": "Rufus-Trends/2.0"}
            r = httpx.get(url, headers=headers, timeout=timeout)
            if r.status_code != 200:
                continue
            # Crude RSS parse — avoids xml lib for size
            titles = re.findall(r"<title>(.*?)</title>", r.text, re.DOTALL)[1:]  # skip channel title
            for t in titles[:limit]:
                t = re.sub(r"\s+", " ", t).strip()
                if not t or t.lower().startswith("rt @"):
                    continue
                out.append(TrendingTopic(
                    keyword=t[:200],
                    source="twitter",
                    momentum=50,  # we don't have engagement counts via RSS — flat score
                    raw_score=1.0,
                    url=base,
                    metadata={"query": query},
                ))
            return out
        except Exception:
            continue
    return out


# ---------------------------------------------------------------------------
# Source 3: Google News RSS
# ---------------------------------------------------------------------------

def _gnews_rss(query: str, geo: str = "US", limit: int = 15, timeout: float = 10.0) -> list[TrendingTopic]:
    encoded = urllib.parse.quote(query)
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en-{geo}&gl={geo}&ceid={geo}:en"
    out: list[TrendingTopic] = []
    try:
        r = httpx.get(url, timeout=timeout)
        if r.status_code != 200:
            return out
        # Crude RSS parse: <item> blocks
        items = re.findall(r"<item>(.*?)</item>", r.text, re.DOTALL)
        for item in items[:limit]:
            title_m = re.search(r"<title>(.*?)</title>", item, re.DOTALL)
            link_m = re.search(r"<link>(.*?)</link>", item, re.DOTALL)
            pub_m = re.search(r"<pubDate>(.*?)</pubDate>", item, re.DOTALL)
            if not title_m:
                continue
            title = re.sub(r"\s+", " ", title_m.group(1)).strip()
            title = re.sub(r"\s*-\s*[^-]+$", "", title)  # strip "- Source Name"
            if len(title) < 15:
                continue
            # Recency boost
            momentum = 50.0
            if pub_m:
                try:
                    pub = datetime.strptime(pub_m.group(1).strip()[:25], "%a, %d %b %Y %H:%M:%S")
                    hours_old = (datetime.utcnow() - pub).total_seconds() / 3600
                    momentum = max(10, 70 - hours_old)
                except Exception:
                    pass
            out.append(TrendingTopic(
                keyword=title,
                source="gnews",
                momentum=momentum,
                raw_score=momentum,
                url=link_m.group(1).strip() if link_m else "",
                metadata={"query": query},
            ))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Source 4: YouTube mostPopular (uses existing google API client)
# ---------------------------------------------------------------------------

def _youtube_trending(category_id: str = "0", region: str = "US",
                       max_results: int = 25) -> list[TrendingTopic]:
    """Reuses the existing uploader's authenticated YouTube client."""
    try:
        from src.uploader import get_youtube_client
        yt = get_youtube_client()
    except Exception:
        return []
    out: list[TrendingTopic] = []
    try:
        req = yt.videos().list(
            part="snippet,statistics",
            chart="mostPopular",
            regionCode=region,
            videoCategoryId=category_id if category_id != "0" else None,
            maxResults=max_results,
        )
        resp = req.execute()
        for item in resp.get("items", []):
            sn = item.get("snippet", {})
            st = item.get("statistics", {})
            title = sn.get("title", "").strip()
            if not title:
                continue
            views = int(st.get("viewCount", 0))
            momentum = min(100, views / 10000)  # 1M views → 100
            out.append(TrendingTopic(
                keyword=title,
                source="youtube",
                momentum=momentum,
                raw_score=float(views),
                url=f"https://youtube.com/watch?v={item.get('id', '')}",
                metadata={"channel": sn.get("channelTitle", "")},
            ))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Source 5: HackerNews
# ---------------------------------------------------------------------------

def _hn_top(limit: int = 25, timeout: float = 10.0) -> list[TrendingTopic]:
    out: list[TrendingTopic] = []
    try:
        ids = httpx.get("https://hacker-news.firebaseio.com/v0/topstories.json", timeout=timeout).json()
        # Fetch first `limit` story details in parallel
        with ThreadPoolExecutor(max_workers=10) as pool:
            futures = [
                pool.submit(httpx.get, f"https://hacker-news.firebaseio.com/v0/item/{sid}.json", timeout=timeout)
                for sid in ids[:limit]
            ]
            for fut in as_completed(futures):
                try:
                    item = fut.result().json()
                    title = (item.get("title") or "").strip()
                    if not title:
                        continue
                    score = item.get("score", 0)
                    age_hours = max(1, (time.time() - item.get("time", time.time())) / 3600)
                    score_per_hour = score / age_hours
                    momentum = min(100, score_per_hour * 2)  # 50/hr → 100
                    out.append(TrendingTopic(
                        keyword=title,
                        source="hn",
                        momentum=momentum,
                        raw_score=float(score),
                        url=item.get("url", "") or f"https://news.ycombinator.com/item?id={item.get('id')}",
                    ))
                except Exception:
                    continue
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Source 6: Wikipedia top-viewed pages (daily search volume proxy)
# ---------------------------------------------------------------------------

def _wikipedia_top(timeout: float = 10.0) -> list[TrendingTopic]:
    # Yesterday's top — today's may not be aggregated yet
    from datetime import timedelta
    yesterday = (datetime.utcnow() - timedelta(days=1)).strftime("%Y/%m/%d")
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/top/en.wikipedia/all-access/{yesterday}"
    out: list[TrendingTopic] = []
    try:
        r = httpx.get(url, timeout=timeout, headers={"User-Agent": "Rufus-Trends/2.0"})
        if r.status_code != 200:
            return out
        items = r.json().get("items", [])
        if not items:
            return out
        articles = items[0].get("articles", [])[:30]
        for a in articles:
            title = a.get("article", "").replace("_", " ")
            views = a.get("views", 0)
            if title in {"Main_Page", "Special:Search", "-"} or title.startswith("Special:"):
                continue
            momentum = min(100, views / 50000)  # 5M views → 100
            out.append(TrendingTopic(
                keyword=title,
                source="wikipedia",
                momentum=momentum,
                raw_score=float(views),
                url=f"https://en.wikipedia.org/wiki/{a.get('article')}",
            ))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def discover_trending_topics(
    niche: str = "general",
    limit: int = 20,
    dedupe_against_memory: bool = True,
    parallel: bool = True,
) -> list[TrendingTopic]:
    """
    Aggregate trending topics from all sources, score by momentum, dedupe
    against memory.topics, return top `limit`.

    Args:
        niche: Niche key from NICHE_QUERIES. Falls back to "general" if unknown.
        limit: Max topics to return.
        dedupe_against_memory: Filter out topics already used in this niche
            in the last 60 days (via was_topic_recently_used).
        parallel: Fetch sources concurrently. Set False for debugging.

    Returns:
        Topics sorted by momentum DESC.
    """
    cfg = NICHE_QUERIES.get(niche, NICHE_QUERIES["general"])

    jobs: list = []
    # Reddit
    for sub in cfg.get("subreddits", []):
        jobs.append(("reddit", sub))
    # Twitter via Nitter
    for q in cfg.get("twitter", []):
        jobs.append(("nitter", q))
    # Google News
    for q in cfg.get("google_news", []):
        jobs.append(("gnews", q))
    # YouTube
    for cat in cfg.get("yt_categories", []):
        jobs.append(("youtube", cat))
    # HN and Wikipedia don't take args — always include
    jobs.append(("hn", None))
    jobs.append(("wikipedia", None))

    all_topics: list[TrendingTopic] = []

    def run_one(job):
        src, arg = job
        if src == "reddit":
            return _reddit_rising(arg, limit=20)
        if src == "nitter":
            return _nitter_search(arg, limit=15)
        if src == "gnews":
            return _gnews_rss(arg, limit=10)
        if src == "youtube":
            return _youtube_trending(category_id=arg)
        if src == "hn":
            return _hn_top()
        if src == "wikipedia":
            return _wikipedia_top()
        return []

    if parallel:
        with ThreadPoolExecutor(max_workers=8) as pool:
            for batch in pool.map(run_one, jobs):
                all_topics.extend(batch)
    else:
        for job in jobs:
            all_topics.extend(run_one(job))

    # Dedupe by fingerprint, keep highest-momentum copy
    by_fp: dict[str, TrendingTopic] = {}
    for t in all_topics:
        fp = t.fingerprint()
        if not fp or len(fp) < 10:
            continue
        if fp not in by_fp or t.momentum > by_fp[fp].momentum:
            by_fp[fp] = t

    candidates = list(by_fp.values())

    # Filter against memory (already-used topics)
    if dedupe_against_memory:
        try:
            from src.memory import was_topic_recently_used
            candidates = [t for t in candidates
                          if not was_topic_recently_used(niche, t.keyword, days=60)]
        except Exception:
            pass

    # Sort by momentum
    candidates.sort(key=lambda t: t.momentum, reverse=True)

    # Persist new candidates to memory.topics for cross-run tracking
    try:
        from src.memory import store_topic
        for t in candidates[:limit]:
            store_topic(niche, t.keyword, source=t.source, momentum=t.momentum)
    except Exception:
        pass

    return candidates[:limit]


def pick_top_topic(niche: str = "general") -> Optional[str]:
    """Convenience: return the single highest-momentum topic for a niche."""
    topics = discover_trending_topics(niche, limit=5)
    return topics[0].keyword if topics else None


def print_topics(topics: list[TrendingTopic]) -> None:
    """Debug helper — prints a table of discovered topics."""
    try:
        from rich.table import Table
        from rich.console import Console
        console = Console()
        table = Table(title="Trending Topics (multi-source)")
        table.add_column("Momentum", justify="right")
        table.add_column("Source")
        table.add_column("Topic", overflow="fold")
        for t in topics:
            color = "green" if t.momentum > 60 else "yellow" if t.momentum > 30 else "dim"
            table.add_row(f"[{color}]{t.momentum:5.1f}[/{color}]",
                          t.source,
                          t.keyword[:120])
        console.print(table)
    except ImportError:
        for t in topics:
            print(f"{t.momentum:5.1f}  [{t.source:10s}]  {t.keyword[:120]}")


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    niche = sys.argv[1] if len(sys.argv) > 1 else "general"
    print(f"Discovering trending topics for niche: {niche}\n")
    topics = discover_trending_topics(niche, limit=25, dedupe_against_memory=False)
    print_topics(topics)
