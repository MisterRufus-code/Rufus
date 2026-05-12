"""Fetch trending topics from Google Trends and YouTube."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

from pytrends.request import TrendReq
from rich.console import Console
from rich.table import Table

console = Console()


@dataclass
class TrendingTopic:
    keyword: str
    search_volume_index: int = 0
    related_queries: list[str] = field(default_factory=list)
    source: str = "google_trends"


def fetch_google_trends(
    keywords: list[str],
    geo: str = "US",
    timeframe: str = "now 7-d",
) -> list[TrendingTopic]:
    """Return interest-over-time data for the given keywords."""
    pytrends = TrendReq(hl="en-US", tz=360)
    pytrends.build_payload(keywords[:5], cat=0, timeframe=timeframe, geo=geo)

    interest_df = pytrends.interest_over_time()
    topics: list[TrendingTopic] = []

    for kw in keywords[:5]:
        if kw in interest_df.columns:
            avg_volume = int(interest_df[kw].mean())
        else:
            avg_volume = 0

        try:
            related = pytrends.related_queries()
            rising = related.get(kw, {}).get("rising")
            if rising is not None and not rising.empty:
                related_list = rising["query"].tolist()[:5]
            else:
                related_list = []
        except Exception:
            related_list = []

        topics.append(
            TrendingTopic(
                keyword=kw,
                search_volume_index=avg_volume,
                related_queries=related_list,
            )
        )

    topics.sort(key=lambda t: t.search_volume_index, reverse=True)
    return topics


def fetch_youtube_trending(youtube_client, region_code: str = "US", max_results: int = 10) -> list[dict]:
    """Return trending YouTube videos using the YouTube Data API."""
    response = (
        youtube_client.videos()
        .list(
            part="snippet,statistics",
            chart="mostPopular",
            regionCode=region_code,
            maxResults=max_results,
            videoCategoryId="",
        )
        .execute()
    )

    videos = []
    for item in response.get("items", []):
        snippet = item.get("snippet", {})
        stats = item.get("statistics", {})
        videos.append(
            {
                "title": snippet.get("title"),
                "channel": snippet.get("channelTitle"),
                "views": int(stats.get("viewCount", 0)),
                "likes": int(stats.get("likeCount", 0)),
                "tags": snippet.get("tags", []),
                "category_id": snippet.get("categoryId"),
            }
        )

    videos.sort(key=lambda v: v["views"], reverse=True)
    return videos


def print_trends_table(topics: list[TrendingTopic]) -> None:
    table = Table(title="Google Trends — Keyword Interest", show_lines=True)
    table.add_column("Keyword", style="cyan", no_wrap=True)
    table.add_column("Avg Interest (0-100)", justify="right", style="green")
    table.add_column("Rising Related Queries", style="yellow")

    for t in topics:
        table.add_row(
            t.keyword,
            str(t.search_volume_index),
            ", ".join(t.related_queries) or "—",
        )

    console.print(table)


def print_yt_trending_table(videos: list[dict]) -> None:
    table = Table(title="YouTube Trending Videos", show_lines=True)
    table.add_column("#", justify="right", style="dim")
    table.add_column("Title", style="cyan")
    table.add_column("Channel", style="blue")
    table.add_column("Views", justify="right", style="green")
    table.add_column("Likes", justify="right", style="magenta")

    for i, v in enumerate(videos, 1):
        table.add_row(
            str(i),
            v["title"][:60],
            v["channel"][:30],
            f"{v['views']:,}",
            f"{v['likes']:,}",
        )

    console.print(table)
