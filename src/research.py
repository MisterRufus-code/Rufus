"""
research.py — RAG layer that grounds scripts in real, current facts.

The single biggest quality lever in the pipeline. Without this, every
script is the LLM's pre-trained opinion stitched together. With this,
every script is grounded in 5-10 current sources.

Pipeline:
  topic → fetch sources → extract key facts → return compact context
  → orchestrator passes context to generate_script_from_media(research_context=...)

Sources (in order):
  1. Brave Search API  (2000 free queries/month, best quality)
  2. DuckDuckGo HTML   (free, no key, lower quality)
  3. Wikipedia REST    (free, structured, authoritative for evergreen)

The output is text ≤2000 chars — enough context for grounding, small
enough to not blow Ollama's num_ctx=8192.

Drop in as `src/research.py`. Wire into orchestrator.py at Step 6 by
adding ONE line before script generation:

    from src.research import research_topic
    research_ctx = research_topic(best_idea.title, niche=niche)
    # then pass research_context=research_ctx into generate_script_from_media
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")
BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

MAX_RESEARCH_CHARS = 2000           # cap on returned context
SOURCE_TIMEOUT = 12                 # per-source timeout (seconds)
SOURCES_PER_TOPIC = 8               # target snippets to collect


@dataclass
class ResearchSnippet:
    """A single factual snippet pulled from a source."""
    text: str
    source: str        # "brave", "ddg", "wikipedia"
    url: str = ""
    title: str = ""

    def has_numbers(self) -> bool:
        """Snippets with numbers are higher quality for factual scripts."""
        return bool(re.search(r"\b\d{2,}\b|\d+(?:\.\d+)?%", self.text))


# ---------------------------------------------------------------------------
# Source 1: Brave Search API (best quality, free tier)
# ---------------------------------------------------------------------------

def _brave_search(query: str, count: int = 10) -> list[ResearchSnippet]:
    if not BRAVE_API_KEY:
        return []
    out: list[ResearchSnippet] = []
    try:
        r = httpx.get(
            BRAVE_ENDPOINT,
            params={"q": query, "count": count, "freshness": "pm"},  # past month
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": BRAVE_API_KEY,
            },
            timeout=SOURCE_TIMEOUT,
        )
        if r.status_code != 200:
            return out
        data = r.json()
        for item in data.get("web", {}).get("results", []):
            desc = item.get("description") or item.get("snippet") or ""
            desc = _clean_snippet(desc)
            if len(desc) < 40:
                continue
            out.append(ResearchSnippet(
                text=desc,
                source="brave",
                url=item.get("url", ""),
                title=item.get("title", ""),
            ))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Source 2: DuckDuckGo HTML (no key, free, lower quality)
# ---------------------------------------------------------------------------

def _ddg_search(query: str, count: int = 10) -> list[ResearchSnippet]:
    """Scrape DDG HTML results page. Their API has been deprecated; HTML works."""
    out: list[ResearchSnippet] = []
    try:
        r = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query, "kl": "us-en"},
            headers={"User-Agent": "Mozilla/5.0 Rufus-Research/1.0"},
            timeout=SOURCE_TIMEOUT,
        )
        if r.status_code != 200:
            return out
        # Parse result snippets. DDG's HTML has <a class="result__snippet"> blocks.
        snippets = re.findall(
            r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
            r.text, flags=re.DOTALL,
        )
        titles = re.findall(
            r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            r.text, flags=re.DOTALL,
        )
        for i, snip in enumerate(snippets[:count]):
            text = _clean_snippet(_strip_html(snip))
            if len(text) < 40:
                continue
            url = titles[i][0] if i < len(titles) else ""
            title = _strip_html(titles[i][1]) if i < len(titles) else ""
            out.append(ResearchSnippet(text=text, source="ddg", url=url, title=title))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Source 3: Wikipedia REST API (free, structured, ideal for evergreen facts)
# ---------------------------------------------------------------------------

def _wiki_summary(query: str) -> list[ResearchSnippet]:
    """
    Try to find a Wikipedia article matching the query and return its summary.
    Best for evergreen/factual topics (history, science, economics).
    """
    out: list[ResearchSnippet] = []
    try:
        # Step 1: search for the article
        search_r = httpx.get(
            "https://en.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search", "format": "json",
                "srsearch": query, "srlimit": 3,
            },
            timeout=SOURCE_TIMEOUT,
            headers={"User-Agent": "Rufus-Research/1.0"},
        )
        if search_r.status_code != 200:
            return out
        results = search_r.json().get("query", {}).get("search", [])
        # Step 2: fetch summary for each
        for hit in results[:2]:
            title = hit.get("title", "")
            if not title:
                continue
            summary_r = httpx.get(
                f"https://en.wikipedia.org/api/rest_v1/page/summary/{title.replace(' ', '_')}",
                timeout=SOURCE_TIMEOUT,
                headers={"User-Agent": "Rufus-Research/1.0"},
            )
            if summary_r.status_code != 200:
                continue
            data = summary_r.json()
            extract = data.get("extract", "").strip()
            if len(extract) < 60:
                continue
            # Take first 2 sentences only — Wikipedia summaries can be long
            sentences = re.split(r"(?<=[.!?])\s+", extract)
            short = " ".join(sentences[:3])[:500]
            out.append(ResearchSnippet(
                text=short,
                source="wikipedia",
                url=data.get("content_urls", {}).get("desktop", {}).get("page", ""),
                title=title,
            ))
    except Exception:
        pass
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_ENTITY_MAP = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&#39;": "'", "&apos;": "'", "&nbsp;": " ",
}


def _strip_html(s: str) -> str:
    s = _HTML_TAG_RE.sub("", s)
    for ent, ch in _ENTITY_MAP.items():
        s = s.replace(ent, ch)
    s = re.sub(r"&#x?([0-9a-fA-F]+);", "", s)  # other entities → drop
    return s


def _clean_snippet(s: str) -> str:
    s = _strip_html(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _dedup_snippets(snippets: list[ResearchSnippet]) -> list[ResearchSnippet]:
    """Drop near-duplicates based on first 80 chars of normalized text."""
    seen: set[str] = set()
    out: list[ResearchSnippet] = []
    for s in snippets:
        fp = re.sub(r"\W", "", s.text.lower())[:80]
        if fp in seen:
            continue
        seen.add(fp)
        out.append(s)
    return out


def _score_snippet(s: ResearchSnippet) -> float:
    """Higher is better. Prefer numbers, named institutions, recent dates."""
    score = 1.0
    if s.has_numbers():
        score += 2.0
    if re.search(r"\b(?:McKinsey|Goldman|WEF|IMF|BIS|OECD|World Bank|MIT|Stanford|Harvard|Oxford|Pew|Forbes|Reuters|Bloomberg)\b", s.text):
        score += 1.5
    if re.search(r"\b(?:2024|2025|2026)\b", s.text):
        score += 1.0
    if s.source == "brave":
        score += 0.5  # higher-quality source
    if s.source == "wikipedia":
        score += 0.3
    return score


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def research_topic(
    topic: str,
    niche: str = "general",
    max_chars: int = MAX_RESEARCH_CHARS,
    include_wikipedia: bool = True,
) -> str:
    """
    Gather factual research context for a topic.

    Returns a compact text block ready to inject into the script prompt.
    Empty string if no usable research found (script falls back to LLM knowledge).

    Args:
        topic: The video topic / title.
        niche: Niche key — may be used to bias queries in future versions.
        max_chars: Hard cap on returned context length.
        include_wikipedia: Include Wikipedia summary lookups.

    Returns:
        Formatted research block: "FACT 1: ...\\nSOURCE: ...\\n\\nFACT 2: ..."
    """
    # Build query variants to widen the net
    queries = [
        topic,
        f"{topic} statistics",
        f"{topic} study report",
    ]

    snippets: list[ResearchSnippet] = []

    # Brave (if key set)
    for q in queries[:2]:
        snippets.extend(_brave_search(q, count=5))

    # DDG as backup (always run — costs nothing)
    if len(snippets) < SOURCES_PER_TOPIC:
        for q in queries[:2]:
            snippets.extend(_ddg_search(q, count=5))

    # Wikipedia for evergreen anchoring
    if include_wikipedia:
        snippets.extend(_wiki_summary(topic))

    # Dedup, score, sort
    snippets = _dedup_snippets(snippets)
    snippets.sort(key=_score_snippet, reverse=True)

    # Build the context block
    parts: list[str] = []
    char_count = 0
    used = 0
    for s in snippets:
        # Format: "FACT N (source): text"
        line = f"FACT {used + 1} ({s.source}): {s.text}"
        if s.url:
            line += f"\n  ↳ SOURCE: {s.url[:120]}"
        if char_count + len(line) > max_chars:
            break
        parts.append(line)
        char_count += len(line) + 2  # +2 for the join newlines
        used += 1
        if used >= SOURCES_PER_TOPIC:
            break

    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# CLI for quick testing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    topic = " ".join(sys.argv[1:]) or "AI job displacement 2025"
    print(f"Researching: {topic}\n" + "=" * 60)
    ctx = research_topic(topic)
    print(ctx if ctx else "(no research returned)")
    print("=" * 60)
    print(f"\nLength: {len(ctx)} chars")
