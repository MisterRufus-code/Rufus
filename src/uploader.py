"""YouTube video uploader using the YouTube Data API v3."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from googleapiclient.http import MediaFileUpload
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

import config

console = Console()

RETRIABLE_STATUS_CODES = {500, 502, 503, 504}
MAX_RETRIES = 5


def get_youtube_client():
    """Build and return an authenticated YouTube API client."""
    creds: Optional[Credentials] = None

    if os.path.exists(config.YOUTUBE_TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(
            config.YOUTUBE_TOKEN_FILE, config.YOUTUBE_SCOPES
        )

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not os.path.exists(config.YOUTUBE_CLIENT_SECRETS_FILE):
                raise FileNotFoundError(
                    f"OAuth client secrets not found at '{config.YOUTUBE_CLIENT_SECRETS_FILE}'.\n"
                    "Download it from Google Cloud Console and place it in the project root."
                )
            flow = InstalledAppFlow.from_client_secrets_file(
                config.YOUTUBE_CLIENT_SECRETS_FILE, config.YOUTUBE_SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open(config.YOUTUBE_TOKEN_FILE, "w") as token_file:
            token_file.write(creds.to_json())

    return build("youtube", "v3", credentials=creds)


def upload_video(
    video_path: str,
    title: str,
    description: str,
    tags: list[str],
    category_id: str = config.DEFAULT_VIDEO_CATEGORY,
    privacy_status: str = config.DEFAULT_VIDEO_PRIVACY,
    thumbnail_path: Optional[str] = None,
) -> Optional[str]:
    """Upload a video file to YouTube. Returns the video ID on success."""
    path = Path(video_path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")

    youtube = get_youtube_client()

    body = {
        "snippet": {
            "title": title[:100],
            "description": description[:5000],
            "tags": tags[:500],
            "categoryId": category_id,
            "defaultLanguage": "en",
        },
        "status": {
            "privacyStatus": privacy_status,
            "selfDeclaredMadeForKids": False,
        },
    }

    media = MediaFileUpload(
        video_path,
        mimetype="video/*",
        resumable=True,
        chunksize=1024 * 1024 * 5,  # 5 MB chunks
    )

    request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        media_body=media,
    )

    video_id = _resumable_upload(request)

    if video_id and thumbnail_path and Path(thumbnail_path).exists():
        _set_thumbnail(youtube, video_id, thumbnail_path)

    return video_id


def _resumable_upload(request) -> Optional[str]:
    response = None
    retry = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Uploading to YouTube...", total=100)

        while response is None:
            try:
                status, response = request.next_chunk()
                if status:
                    pct = int(status.progress() * 100)
                    progress.update(task, completed=pct)
            except HttpError as e:
                if e.resp.status in RETRIABLE_STATUS_CODES:
                    retry += 1
                    if retry > MAX_RETRIES:
                        console.print(f"[red]Upload failed after {MAX_RETRIES} retries.[/red]")
                        return None
                    wait = 2**retry
                    console.print(f"[yellow]Retrying in {wait}s... (attempt {retry})[/yellow]")
                    time.sleep(wait)
                else:
                    raise

        progress.update(task, completed=100)

    video_id = response.get("id")
    url = f"https://www.youtube.com/watch?v={video_id}"
    console.print(f"\n[bold green]Upload complete![/bold green] {url}")
    return video_id


def _set_thumbnail(youtube, video_id: str, thumbnail_path: str) -> None:
    media = MediaFileUpload(thumbnail_path, mimetype="image/jpeg")
    youtube.thumbnails().set(videoId=video_id, media_body=media).execute()
    console.print("[green]Thumbnail set.[/green]")
