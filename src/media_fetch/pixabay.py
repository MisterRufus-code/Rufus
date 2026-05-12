"""
Auto-download free stock footage and images from Pixabay.

Free API key: https://pixabay.com/api/docs/  (instant, no credit card)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import httpx
from rich.console import Console

import config

console = Console()
PIXABAY_API = "https://pixabay.com/api"


def _key() -> str:
    key = os.getenv("PIXABAY_API_KEY", "")
    if not key:
        raise RuntimeError(
            "PIXABAY_API_KEY not set in .env\n"
            "Get a free key at: https://pixabay.com/api/docs/"
        )
    return key


def search_videos(
    query: str,
    per_page: int = 10,
    min_duration: int = 3,
    max_duration: int = 30,
) -> list[dict]:
    params = {
        "key": _key(),
        "q": query,
        "video_type": "all",
        "per_page": min(per_page, 20),
        "safesearch": "true",
    }
    r = httpx.get(f"{PIXABAY_API}/videos/", params=params, timeout=15)
    r.raise_for_status()
    hits = r.json().get("hits", [])
    return [
        h for h in hits
        if min_duration <= h.get("duration", 0) <= max_duration
    ]


def search_images(query: str, per_page: int = 10) -> list[dict]:
    params = {
        "key": _key(),
        "q": query,
        "image_type": "photo",
        "per_page": min(per_page, 20),
        "safesearch": "true",
        "orientation": "horizontal",
    }
    r = httpx.get(f"{PIXABAY_API}/", params=params, timeout=15)
    r.raise_for_status()
    return r.json().get("hits", [])


def _best_video_url(hit: dict) -> Optional[str]:
    videos = hit.get("videos", {})
    for quality in ("large", "medium", "small"):
        v = videos.get(quality, {})
        url = v.get("url")
        if url:
            return url
    return None


def download_videos(
    queries: list[str],
    dest_dir: Optional[Path] = None,
    videos_per_query: int = 3,
    min_duration: int = 3,
    max_duration: int = 30,
) -> list[Path]:
    dest = Path(dest_dir or config.MEDIA_LIBRARY_PATH / "pixabay")
    dest.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for query in queries:
        import re
        query = re.sub(r"^\d+\.\s*", "", query).strip()  # remove leading "1. "
        if not query:
            continue
        console.print(f"[cyan]Pixabay search:[/cyan] {query}")
        try:
            results = search_videos(query, per_page=videos_per_query,
                                    min_duration=min_duration, max_duration=max_duration)
        except Exception as exc:
            console.print(f"[red]Search failed: {exc}[/red]")
            continue

        for hit in results[:videos_per_query]:
            url = _best_video_url(hit)
            if not url:
                continue
            hit_id = hit.get("id", "unknown")
            filename = dest / f"pixabay_{hit_id}.mp4"
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


def download_images(
    queries: list[str],
    dest_dir: Optional[Path] = None,
    images_per_query: int = 5,
) -> list[Path]:
    dest = Path(dest_dir or config.MEDIA_LIBRARY_PATH / "pixabay_images")
    dest.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for query in queries:
        console.print(f"[cyan]Pixabay image search:[/cyan] {query}")
        try:
            results = search_images(query, per_page=images_per_query)
        except Exception as exc:
            console.print(f"[red]Image search failed: {exc}[/red]")
            continue

        for hit in results[:images_per_query]:
            url = hit.get("largeImageURL") or hit.get("webformatURL")
            if not url:
                continue
            hit_id = hit.get("id", "unknown")
            filename = dest / f"pixabay_{hit_id}.jpg"
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
        with open(dest, "wb") as f:
            for chunk in r.iter_bytes(chunk_size=1024 * 64):
                f.write(chunk)
    console.print(f"  [green]✓[/green] {dest.name}")
