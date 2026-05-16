"""
Download Creative Commons licensed videos from YouTube using yt-dlp.

No API key required. Only downloads videos explicitly marked CC-BY.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Optional

from rich.console import Console

import config

console = Console()


def _ytdlp():
    try:
        import yt_dlp
        return yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp not installed — run: pip install yt-dlp")


def search_and_download(
    query: str,
    dest_dir: Path,
    max_videos: int = 3,
    max_duration: int = 30,
    min_duration: int = 3,
) -> list[Path]:
    yt_dlp = _ytdlp()
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    search_url = f"ytsearch{max_videos * 3}:{query} creative commons"

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(search_url, download=False)
        except Exception as exc:
            console.print(f"[red]yt-dlp search failed: {exc}[/red]")
            return []

    entries = (info or {}).get("entries", [])
    candidates = []
    for e in entries:
        dur = e.get("duration") or 0
        if min_duration <= dur <= max_duration:
            candidates.append(e)
        if len(candidates) >= max_videos:
            break

    for entry in candidates:
        video_id = entry.get("id", "")
        url = entry.get("url") or f"https://www.youtube.com/watch?v={video_id}"
        safe_id = hashlib.md5(video_id.encode()).hexdigest()[:12]
        out_path = dest_dir / f"ytcc_{safe_id}.mp4"

        if out_path.exists():
            downloaded.append(out_path)
            continue

        dl_opts = {
            "quiet": True,
            "no_warnings": True,
            "outtmpl": str(out_path),
            # Prefer H264 mp4; exclude AV1 (av01) which OpenCV can't decode in software
            "format": (
                "bestvideo[ext=mp4][vcodec!*=av01][height<=720]"
                "+bestaudio[ext=m4a]"
                "/bestvideo[ext=mp4][vcodec!*=av01][height<=720]+bestaudio[ext=m4a]"
                "/best[ext=mp4][vcodec!*=av01][height<=720]"
                "/best[height<=720]"
            ),
            "merge_output_format": "mp4",
            "match_filter": yt_dlp.utils.match_filter_func("license ~= 'Creative Commons'"),
        }
        with yt_dlp.YoutubeDL(dl_opts) as ydl:
            try:
                ydl.download([url])
                if out_path.exists():
                    console.print(f"  [green]✓[/green] {out_path.name}")
                    downloaded.append(out_path)
            except yt_dlp.utils.RejectedVideoReached:
                pass
            except Exception as exc:
                console.print(f"  [yellow]skipped ({exc.__class__.__name__})[/yellow]")
        time.sleep(0.5)

    return downloaded


def download_videos(
    queries: list[str],
    dest_dir: Optional[Path] = None,
    videos_per_query: int = 3,
    min_duration: int = 3,
    max_duration: int = 30,
) -> list[Path]:
    dest = Path(dest_dir or config.MEDIA_LIBRARY_PATH / "ytcc")
    downloaded: list[Path] = []
    for query in queries:
        console.print(f"[cyan]YouTube CC search:[/cyan] {query}")
        paths = search_and_download(
            query, dest,
            max_videos=videos_per_query,
            min_duration=min_duration,
            max_duration=max_duration,
        )
        downloaded.extend(paths)
    return downloaded
