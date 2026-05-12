"""
Auto-download free stock footage and images from Pexels.

Free API key: https://www.pexels.com/api/  (instant, no credit card)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, DownloadColumn

import config

console = Console()
PEXELS_API = "https://api.pexels.com/v1"
PEXELS_VIDEO_API = "https://api.pexels.com/videos"


def _headers() -> dict:
    key = os.getenv("PEXELS_API_KEY", "")
    if not key:
        raise RuntimeError(
            "PEXELS_API_KEY not set in .env\n"
            "Get a free key at: https://www.pexels.com/api/"
        )
    return {"Authorization": key}


def search_videos(
    query: str,
    per_page: int = 10,
    min_duration: int = 3,
    max_duration: int = 30,
    orientation: str = "landscape",
) -> list[dict]:
    """Search Pexels for videos matching *query*. Returns metadata list."""
    params = {
        "query": query,
        "per_page": per_page,
        "orientation": orientation,
        "size": "medium",
    }
    r = httpx.get(f"{PEXELS_VIDEO_API}/search", headers=_headers(), params=params, timeout=15)
    r.raise_for_status()
    videos = r.json().get("videos", [])
    return [
        v for v in videos
        if min_duration <= v.get("duration", 0) <= max_duration
    ]


def search_images(query: str, per_page: int = 10, orientation: str = "landscape") -> list[dict]:
    """Search Pexels for images matching *query*."""
    params = {"query": query, "per_page": per_page, "orientation": orientation}
    r = httpx.get(f"{PEXELS_API}/search", headers=_headers(), params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("photos", [])


def _best_video_url(video: dict) -> Optional[str]:
    """Pick the best quality video file URL (HD preferred)."""
    files = video.get("video_files", [])
    # prefer hd, then sd
    for quality in ("hd", "sd"):
        for f in files:
            if f.get("quality") == quality and f.get("file_type") == "video/mp4":
                return f["link"]
    return files[0]["link"] if files else None


def download_videos(
    queries: list[str],
    dest_dir: Optional[Path] = None,
    videos_per_query: int = 3,
    min_duration: int = 3,
    max_duration: int = 30,
) -> list[Path]:
    """Download free stock videos for a list of search queries."""
    dest = Path(dest_dir or config.MEDIA_LIBRARY_PATH / "pexels")
    dest.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []

    for query in queries:
        console.print(f"[cyan]Pexels search:[/cyan] {query}")
        try:
            results = search_videos(query, per_page=videos_per_query,
                                    min_duration=min_duration, max_duration=max_duration)
        except Exception as exc:
            console.print(f"[red]Search failed: {exc}[/red]")
            continue

        for video in results:
            url = _best_video_url(video)
            if not url:
                continue
            vid_id = video.get("id", "unknown")
            filename = dest / f"pexels_{vid_id}.mp4"
            if filename.exists():
                downloaded.append(filename)
                continue
            try:
                _download_file(url, filename)
                downloaded.append(filename)
            except Exception as exc:
                console.print(f"[red]Download error: {exc}[/red]")
            time.sleep(0.3)   # be polite to the API

    return downloaded


def download_images(
    queries: list[str],
    dest_dir: Optional[Path] = None,
    images_per_query: int = 5,
) -> list[Path]:
    """Download free stock images for a list of search queries."""
    dest = Path(dest_dir or config.MEDIA_LIBRARY_PATH / "pexels_images")
    dest.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []

    for query in queries:
        console.print(f"[cyan]Pexels image search:[/cyan] {query}")
        try:
            results = search_images(query, per_page=images_per_query)
        except Exception as exc:
            console.print(f"[red]Image search failed: {exc}[/red]")
            continue

        for photo in results:
            src = photo.get("src", {})
            url = src.get("large") or src.get("medium")
            if not url:
                continue
            photo_id = photo.get("id", "unknown")
            filename = dest / f"pexels_{photo_id}.jpg"
            if filename.exists():
                downloaded.append(filename)
                continue
            try:
                _download_file(url, filename)
                downloaded.append(filename)
            except Exception as exc:
                console.print(f"[red]Download error: {exc}[/red]")
            time.sleep(0.3)

    return downloaded


def _download_file(url: str, dest: Path) -> None:
    with httpx.stream("GET", url, timeout=60, follow_redirects=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 64):
                f.write(chunk)
    console.print(f"  [green]✓[/green] {dest.name}")
