"""Fetch and display channel + video analytics via YouTube Data API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.uploader import get_youtube_client

console = Console()


@dataclass
class VideoStats:
    video_id: str
    title: str
    views: int
    likes: int
    comments: int
    published_at: str
    duration: str


@dataclass
class ChannelStats:
    channel_id: str
    name: str
    subscribers: int
    total_views: int
    video_count: int


def get_channel_stats() -> ChannelStats:
    """Return stats for the authenticated user's channel."""
    youtube = get_youtube_client()
    response = (
        youtube.channels()
        .list(part="snippet,statistics", mine=True)
        .execute()
    )

    item = response["items"][0]
    stats = item["statistics"]
    snippet = item["snippet"]

    return ChannelStats(
        channel_id=item["id"],
        name=snippet["title"],
        subscribers=int(stats.get("subscriberCount", 0)),
        total_views=int(stats.get("viewCount", 0)),
        video_count=int(stats.get("videoCount", 0)),
    )


def get_recent_videos_stats(max_results: int = 10) -> list[VideoStats]:
    """Return stats for the most recently uploaded videos."""
    youtube = get_youtube_client()

    channel_resp = (
        youtube.channels()
        .list(part="contentDetails", mine=True)
        .execute()
    )
    uploads_playlist_id = (
        channel_resp["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    )

    playlist_resp = (
        youtube.playlistItems()
        .list(
            part="contentDetails",
            playlistId=uploads_playlist_id,
            maxResults=max_results,
        )
        .execute()
    )

    video_ids = [
        item["contentDetails"]["videoId"]
        for item in playlist_resp.get("items", [])
    ]

    if not video_ids:
        return []

    videos_resp = (
        youtube.videos()
        .list(
            part="snippet,statistics,contentDetails",
            id=",".join(video_ids),
        )
        .execute()
    )

    result = []
    for item in videos_resp.get("items", []):
        stats = item.get("statistics", {})
        snippet = item.get("snippet", {})
        result.append(
            VideoStats(
                video_id=item["id"],
                title=snippet.get("title", ""),
                views=int(stats.get("viewCount", 0)),
                likes=int(stats.get("likeCount", 0)),
                comments=int(stats.get("commentCount", 0)),
                published_at=snippet.get("publishedAt", "")[:10],
                duration=item.get("contentDetails", {}).get("duration", ""),
            )
        )

    result.sort(key=lambda v: v.views, reverse=True)
    return result


def print_channel_stats(stats: ChannelStats) -> None:
    console.print(
        Panel(
            f"[cyan]Channel:[/cyan] {stats.name}\n"
            f"[green]Subscribers:[/green] {stats.subscribers:,}\n"
            f"[green]Total Views:[/green] {stats.total_views:,}\n"
            f"[blue]Videos:[/blue] {stats.video_count:,}",
            title="Channel Analytics",
            border_style="cyan",
        )
    )


def print_video_stats(videos: list[VideoStats]) -> None:
    table = Table(title="Recent Video Performance", show_lines=True)
    table.add_column("Title", style="cyan", max_width=45)
    table.add_column("Published", style="dim")
    table.add_column("Views", justify="right", style="green")
    table.add_column("Likes", justify="right", style="magenta")
    table.add_column("Comments", justify="right", style="yellow")

    for v in videos:
        table.add_row(
            v.title[:44],
            v.published_at,
            f"{v.views:,}",
            f"{v.likes:,}",
            f"{v.comments:,}",
        )

    console.print(table)
