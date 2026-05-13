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

    items = response.get("items", [])
    if not items:
        raise RuntimeError(
            "YouTube API returned no channel data. "
            "Check that OAuth scopes include youtube.readonly and re-authenticate."
        )
    item = items[0]
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


def sync_feedback_from_youtube(max_videos: int = 20, niche: str = "general") -> int:
    """
    Pull real YouTube analytics and auto-record ML feedback for each video.
    Returns the number of records saved.
    """
    from src.ml.feedback import VideoFeedback, record_feedback, load_all_feedback

    existing_ids = {r.video_id for r in load_all_feedback()}

    try:
        videos = get_recent_videos_stats(max_results=max_videos)
    except Exception as exc:
        console.print(f"[red]Could not fetch YouTube stats: {exc}[/red]")
        return 0

    if not videos:
        console.print("[yellow]No videos found on channel.[/yellow]")
        return 0

    # Fetch YouTube Analytics for CTR and retention if possible
    analytics_data: dict[str, dict] = {}
    try:
        youtube = get_youtube_client()
        from googleapiclient.discovery import build as yt_build
        try:
            yta = yt_build("youtubeAnalytics", "v2", credentials=youtube._http.credentials)
            ids_str = ",".join(f"video=={v.video_id}" for v in videos)
            resp = yta.reports().query(
                ids="channel==MINE",
                startDate="2020-01-01",
                endDate="2030-01-01",
                metrics="views,estimatedMinutesWatched,averageViewDuration,averageViewPercentage,annotationClickThroughRate",
                dimensions="video",
                filters=ids_str,
            ).execute()
            for row in resp.get("rows", []):
                if not row or len(row) < 1:
                    continue
                vid_id = row[0]
                analytics_data[vid_id] = {
                    "views": int(row[1]) if len(row) > 1 else 0,
                    "watch_minutes": float(row[2]) if len(row) > 2 else 0.0,
                    "avg_duration": float(row[3]) if len(row) > 3 else 0.0,
                    "retention_pct": float(row[4]) if len(row) > 4 else 0.0,
                    "ctr": float(row[5]) if len(row) > 5 else 0.0,
                }
        except Exception:
            pass
    except Exception:
        pass

    saved = 0
    for v in videos:
        if v.video_id in existing_ids:
            continue

        adata = analytics_data.get(v.video_id, {})
        avg_duration = adata.get("avg_duration", 0.0)
        retention = adata.get("retention_pct", 0.0) / 100.0
        ctr = adata.get("ctr", 0.0)

        # Estimate retention from likes/views ratio if Analytics unavailable
        # Only trust this heuristic when we have enough views to matter
        if retention == 0.0 and v.views >= 100:
            like_ratio = v.likes / v.views
            retention = min(like_ratio * 8, 0.8)

        fb = VideoFeedback(
            video_id=v.video_id,
            title=v.title,
            niche=niche,
            topic=v.title,
            prompt_model="unknown",
            ctr=ctr,
            avg_watch_time_seconds=avg_duration,
            retention_rate=retention,
            views=v.views,
            likes=v.likes,
        )
        record_feedback(fb)
        saved += 1
        console.print(
            f"[green]Synced:[/green] {v.title[:50]} "
            f"— {v.views:,} views, score: {fb.performance_score:.3f}"
        )

    if saved == 0:
        console.print("[dim]All videos already in feedback DB.[/dim]")
    else:
        console.print(f"[bold green]{saved} feedback records saved.[/bold green]")

    return saved
