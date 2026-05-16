"""
Trending topic discovery — 4 free sources, no rate limits.

Sources (in fallback order):
  1. Google Trends RSS  — daily trending searches, no scraping
  2. YouTube Autocomplete — what people type into YouTube right now
  3. Reddit hot posts  — top posts from niche subreddits
  4. YouTube trending  — mostPopular videos via existing API auth

All sources are merged and deduplicated, then Ollama picks the
best topic for the given niche.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional

import httpx
from rich.console import Console
from rich.table import Table

console = Console()

# Niche → relevant subreddits
NICHE_SUBREDDITS: dict[str, list[str]] = {
    "finance":        ["personalfinance", "financialindependence", "investing", "stocks"],
    "motivation":     ["getmotivated", "selfimprovement", "productivity"],
    "tech":           ["technology", "artificial", "MachineLearning", "OpenAI"],
    "health":         ["nutrition", "loseit", "fitness", "HealthyFood"],
    "relationships":  ["relationship_advice", "dating_advice", "psychology"],
    "general":        ["todayilearned", "interestingasfuck", "YouShouldKnow"],
}

# Niche → seed words for YouTube autocomplete
NICHE_SEEDS: dict[str, list[str]] = {
    "finance":        ["how to make money", "passive income", "investing for beginners"],
    "motivation":     ["morning routine", "self discipline", "success habits"],
    "tech":           ["AI tools", "ChatGPT", "best apps"],
    "health":         ["lose weight fast", "healthy habits", "sleep better"],
    "relationships":  ["how to be confident", "body language", "psychology tricks"],
    "general":        ["did you know", "life hacks", "facts"],
}


@dataclass
class TrendingTopic:
    keyword: str
    search_volume_index: int = 0
    related_queries: list[str] = field(default_factory=list)
    source: str = "google_trends"


# ---------------------------------------------------------------------------
# Source 1 — Google Trends RSS (no rate limits)
# ---------------------------------------------------------------------------

def fetch_trends_rss(geo: str = "US", max_results: int = 20) -> list[str]:
    """Pull daily trending searches — tries the current Google Trends endpoint first."""
    # Google renamed the endpoint in 2024; try both
    urls = [
        f"https://trends.google.com/trending/rss?geo={geo}",
        f"https://trends.google.com/trends/trendingsearches/daily/rss?geo={geo}",
    ]
    for url in urls:
        try:
            r = httpx.get(url, timeout=10, follow_redirects=True)
            r.raise_for_status()
            try:
                root = ET.fromstring(r.text)
            except ET.ParseError:
                continue
            topics = []
            for item in root.findall(".//item"):
                title = item.findtext("title", "").strip()
                if title:
                    topics.append(title)
                if len(topics) >= max_results:
                    break
            if topics:
                console.print(f"[dim]Trends RSS: {len(topics)} topics[/dim]")
                return topics
        except Exception:
            pass
    console.print("[dim]Trends RSS: unavailable[/dim]")
    return []


def fetch_pytrends(niche: str, geo: str = "US", max_results: int = 10) -> list[str]:
    """Use pytrends to get real-time trending searches from Google."""
    try:
        from pytrends.request import TrendReq
        pt = TrendReq(hl="en-US", tz=360, timeout=(5, 15))
        df = pt.trending_searches(pn="united_states" if geo == "US" else geo.lower())
        topics = df[0].tolist()[:max_results]
        console.print(f"[dim]pytrends: {len(topics)} trending searches[/dim]")
        return topics
    except Exception as exc:
        console.print(f"[dim]pytrends unavailable ({exc.__class__.__name__})[/dim]")
        return []


def fetch_google_news_rss(niche: str, max_results: int = 15) -> list[str]:
    """Scrape Google News RSS for current viral headlines in the niche."""
    seeds = NICHE_SEEDS.get(niche, NICHE_SEEDS["general"])
    topics: list[str] = []
    for seed in seeds[:2]:
        try:
            url = f"https://news.google.com/rss/search?q={httpx.utils.primitive_value_to_str(seed)}&hl=en-US&gl=US&ceid=US:en"
            r = httpx.get(url, timeout=10, follow_redirects=True,
                          headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            root = ET.fromstring(r.text)
            for item in root.findall(".//item"):
                title = item.findtext("title", "").strip()
                # Strip " - Publisher" suffix that Google News appends
                title = re.sub(r"\s*-\s*[^-]{3,40}$", "", title).strip()
                if title and len(title) > 10:
                    topics.append(title)
                if len(topics) >= max_results:
                    break
        except Exception:
            pass
    topics = list(dict.fromkeys(topics))[:max_results]
    if topics:
        console.print(f"[dim]Google News: {len(topics)} headlines[/dim]")
    return topics


# ---------------------------------------------------------------------------
# Source 2 — YouTube Search Autocomplete
# ---------------------------------------------------------------------------

def fetch_youtube_autocomplete(niche: str, max_results: int = 15) -> list[str]:
    """Get YouTube search suggestions for niche seed queries."""
    seeds = NICHE_SEEDS.get(niche, NICHE_SEEDS["general"])
    suggestions: list[str] = []

    for seed in seeds[:2]:
        try:
            r = httpx.get(
                "https://suggestqueries.google.com/complete/search",
                params={"client": "youtube", "ds": "yt", "q": seed},
                timeout=8,
            )
            # Response is JSONP: window.google.ac.h([...])
            text = r.text
            match = re.search(r'\[.*\]', text, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                if isinstance(data, list) and len(data) > 1:
                    for item in data[1]:
                        if isinstance(item, list) and item:
                            suggestions.append(str(item[0]))
        except Exception as exc:
            console.print(f"[yellow]Autocomplete failed ({exc})[/yellow]")

    suggestions = list(dict.fromkeys(suggestions))[:max_results]
    console.print(f"[dim]YouTube autocomplete: {len(suggestions)} suggestions[/dim]")
    return suggestions


# ---------------------------------------------------------------------------
# Source 3 — Reddit hot posts
# ---------------------------------------------------------------------------

def fetch_reddit_hot(niche: str, max_results: int = 20) -> list[str]:
    """Pull hot + rising post titles from niche subreddits, sorted by upvote score."""
    subreddits = NICHE_SUBREDDITS.get(niche, NICHE_SUBREDDITS["general"])
    scored: list[tuple[int, str]] = []

    for sub in subreddits[:3]:
        for sort in ("hot", "rising"):
            try:
                r = httpx.get(
                    f"https://www.reddit.com/r/{sub}/{sort}.json",
                    params={"limit": 15},
                    headers={"User-Agent": "rufus-trends/1.0"},
                    timeout=10,
                    follow_redirects=True,
                )
                r.raise_for_status()
                posts = r.json().get("data", {}).get("children", [])
                for post in posts:
                    d = post.get("data", {})
                    title = d.get("title", "").strip()
                    score = int(d.get("score", 0))
                    if title and not title.lower().startswith("[") and score > 10:
                        scored.append((score, title))
            except Exception:
                pass

    # Sort by upvotes so the most-viral topics surface first
    scored.sort(key=lambda x: x[0], reverse=True)
    topics = list(dict.fromkeys(t for _, t in scored))[:max_results]
    console.print(f"[dim]Reddit hot+rising: {len(topics)} topics[/dim]")
    return topics


# ---------------------------------------------------------------------------
# Source 4 — YouTube trending videos (uses existing API auth)
# ---------------------------------------------------------------------------

def fetch_youtube_trending_titles(region_code: str = "US", max_results: int = 10) -> list[str]:
    """Return titles of currently trending YouTube videos."""
    try:
        from src.uploader import get_youtube_client
        youtube = get_youtube_client()
        response = youtube.videos().list(
            part="snippet",
            chart="mostPopular",
            regionCode=region_code,
            maxResults=max_results,
        ).execute()
        titles = [item["snippet"]["title"] for item in response.get("items", [])]
        console.print(f"[dim]YouTube trending: {len(titles)} videos[/dim]")
        return titles
    except Exception as exc:
        console.print(f"[yellow]YouTube trending failed ({exc})[/yellow]")
        return []


# ---------------------------------------------------------------------------
# Topic selector — Ollama picks best topic for the niche
# ---------------------------------------------------------------------------

def select_best_topic(
    all_topics: list[str],
    niche: str,
    model: str = "mistral",
) -> str:
    """Feed all trending topics to Ollama and get the single best one for the niche."""
    if not all_topics:
        return f"Top {niche} tips that will change your life"

    # Deduplicate and limit
    unique = list(dict.fromkeys(t.strip() for t in all_topics if t.strip()))[:40]
    topics_str = "\n".join(f"- {t}" for t in unique)

    prompt = f"""You are a viral YouTube content strategist. Today is right now — pick what will get clicks TODAY.

Here are today's trending topics and viral news:
{topics_str}

My YouTube channel niche is: {niche}

Pick the SINGLE best topic. It must:
1. Be trending TODAY (news, event, or pattern happening right now)
2. Fit the {niche} niche
3. Have shock/surprise/curiosity value (the "wait, really?" factor)
4. Work as a hook: "[Shocking fact] about [familiar thing]" or "Why [company/person] [failed/succeeded]"
5. Be coverable with stock footage only (no face or interview needed)
6. Work for both a 10-minute video AND a 60-second Short

Frame it as a specific, punchy video hook — not a generic topic.
Examples of GOOD: "Groupon rejected Google's $6 billion offer and lost 95% of its value"
Examples of BAD: "finance tips", "investing basics"

Reply with ONLY the topic/hook. One sentence. No explanation."""

    try:
        import httpx as _httpx
        r = _httpx.post(
            "http://localhost:11434/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=60,
        )
        r.raise_for_status()
        topic = r.json().get("response", "").strip().strip('"').strip("'")
        return topic or unique[0]
    except Exception as exc:
        console.print(f"[yellow]Topic selection failed ({exc}) — using top trending[/yellow]")
        return unique[0]


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def get_trending_topic(
    niche: str = "general",
    geo: str = "US",
    model: str = "mistral",
) -> str:
    """
    Fetch trends from all sources, merge, and let Ollama pick the best topic.
    Returns a single topic string ready to pass to the pipeline.
    """
    console.print("[cyan]Fetching trending topics...[/cyan]")

    all_topics: list[str] = []

    # Source 1: Google Trends RSS (fixed URL)
    all_topics.extend(fetch_trends_rss(geo=geo))

    # Source 2: pytrends — real Google trending searches
    all_topics.extend(fetch_pytrends(niche=niche, geo=geo))

    # Source 3: Google News headlines (today's viral news in the niche)
    all_topics.extend(fetch_google_news_rss(niche=niche))

    # Source 4: YouTube autocomplete
    all_topics.extend(fetch_youtube_autocomplete(niche=niche))

    # Source 5: Reddit hot + rising (upvote-sorted)
    all_topics.extend(fetch_reddit_hot(niche=niche))

    # Source 6: YouTube trending (only if already authenticated)
    all_topics.extend(fetch_youtube_trending_titles(region_code=geo))

    if not all_topics:
        console.print("[yellow]All trend sources failed — using fallback topic[/yellow]")
        fallbacks = {
            "finance": "5 money habits that separate the rich from the poor",
            "motivation": "morning habits that will change your life",
            "tech": "AI tools you need to know about",
            "health": "daily habits for a longer life",
            "general": "facts most people don't know",
        }
        return fallbacks.get(niche, "life lessons everyone should know")

    console.print(f"[green]{len(all_topics)} trending signals collected[/green]")
    topic = select_best_topic(all_topics, niche=niche, model=model)
    console.print(f"[bold green]Selected topic:[/bold green] {topic}")
    return topic


# ---------------------------------------------------------------------------
# Legacy compatibility — keep old functions working
# ---------------------------------------------------------------------------

def fetch_google_trends(keywords: list[str], geo: str = "US", timeframe: str = "now 7-d"):
    """Legacy wrapper — tries pytrends, falls back to RSS."""
    try:
        from pytrends.request import TrendReq
        from dataclasses import dataclass, field as dc_field
        pytrends = TrendReq(hl="en-US", tz=360)
        pytrends.build_payload(keywords[:5], cat=0, timeframe=timeframe, geo=geo)
        interest_df = pytrends.interest_over_time()
        topics = []
        for kw in keywords[:5]:
            avg_volume = int(interest_df[kw].mean()) if kw in interest_df.columns else 0
            try:
                related = pytrends.related_queries()
                rising = related.get(kw, {}).get("rising")
                related_list = rising["query"].tolist()[:5] if rising is not None and not rising.empty else []
            except Exception:
                related_list = []
            topics.append(TrendingTopic(keyword=kw, search_volume_index=avg_volume, related_queries=related_list))
        return sorted(topics, key=lambda t: t.search_volume_index, reverse=True)
    except Exception:
        return [TrendingTopic(keyword=kw, source="rss") for kw in keywords]


def fetch_youtube_trending(youtube_client, region_code: str = "US", max_results: int = 10) -> list[dict]:
    response = youtube_client.videos().list(
        part="snippet,statistics", chart="mostPopular",
        regionCode=region_code, maxResults=max_results, videoCategoryId="",
    ).execute()
    videos = []
    for item in response.get("items", []):
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        videos.append({
            "title": snippet.get("title"), "channel": snippet.get("channelTitle"),
            "views": int(stats.get("viewCount", 0)), "likes": int(stats.get("likeCount", 0)),
            "tags": snippet.get("tags", []), "category_id": snippet.get("categoryId"),
        })
    return sorted(videos, key=lambda v: v["views"], reverse=True)


def print_trends_table(topics) -> None:
    from rich.table import Table
    table = Table(title="Trending Topics", show_lines=True)
    table.add_column("Keyword", style="cyan")
    table.add_column("Interest", justify="right", style="green")
    table.add_column("Related", style="yellow")
    for t in topics:
        table.add_row(t.keyword, str(t.search_volume_index), ", ".join(t.related_queries) or "—")
    console.print(table)


def print_yt_trending_table(videos: list[dict]) -> None:
    from rich.table import Table
    table = Table(title="YouTube Trending", show_lines=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Title", style="cyan")
    table.add_column("Channel", style="blue")
    table.add_column("Views", justify="right", style="green")
    for i, v in enumerate(videos, 1):
        table.add_row(str(i), v["title"][:60], v["channel"][:30], f"{v['views']:,}")
    console.print(table)
