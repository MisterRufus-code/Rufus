#!/usr/bin/env python3
"""
youtube_uploader.py
Uploads a rendered Short to YouTube using the Data API v3.

First-time setup:
    1. Google Cloud Console → create project
    2. Enable YouTube Data API v3
    3. Create OAuth 2.0 credentials (Desktop app)
    4. Download as config/client_secrets.json
    5. Run this script once manually → browser opens → approve
    6. token.json is saved → future runs are fully automatic

Usage:
    python youtube_uploader.py /path/to/short.mp4 "Script text here"
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Google libs are imported lazily inside the functions that need them so unit
# tests can import this module without requiring the full Google auth stack.

CONFIG_DIR      = Path(__file__).parent.parent / "config"
NICHES_FILE     = CONFIG_DIR / "niches.json"
CLIENT_SECRETS  = CONFIG_DIR / "client_secrets.json"
TOKEN_FILE      = CONFIG_DIR / "youtube_token.json"

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# YouTube category IDs by niche (overridable via niches.json "youtube_category_id")
DEFAULT_CATEGORIES = {
    "finance":              "25",   # News & Politics
    "motivation":           "22",   # People & Blogs
    "mindset":              "27",   # Education
    "business":             "27",   # Education
    "personal_development": "27",   # Education
}

PEAK_HOURS_ET = [8, 12, 17, 20]  # US Eastern hours; EDT = UTC-4 (most of the year)
ET_UTC_DELTA  = timedelta(hours=4)   # UTC = ET + 4 (EDT). Switch to 5 during EST winter.

NICHE_HASHTAGS = {
    "finance":              ["#finance", "#investing", "#wealth", "#money", "#stockmarket", "#Shorts"],
    "motivation":           ["#motivation", "#mindset", "#grind", "#discipline", "#success", "#Shorts"],
    "mindset":              ["#mindset", "#psychology", "#selfimprovement", "#mentalhealth", "#Shorts"],
    "business":             ["#business", "#entrepreneur", "#startup", "#hustle", "#success", "#Shorts"],
    "personal_development": ["#personaldevelopment", "#habits", "#growth", "#selfimprovement", "#Shorts"],
}


def _next_peak_utc() -> str:
    """Return ISO 8601 UTC timestamp for the next US-ET peak hour, ≥5 min from now."""
    now_utc = datetime.now(tz=timezone.utc)
    now_et  = now_utc - ET_UTC_DELTA   # ET = UTC - 4

    for day_delta in range(3):
        day = now_et.date() + timedelta(days=day_delta)
        for hour in PEAK_HOURS_ET:
            et_naive  = datetime(day.year, day.month, day.day, hour, 0, 0)
            utc_aware = (et_naive + ET_UTC_DELTA).replace(tzinfo=timezone.utc)
            if utc_aware > now_utc + timedelta(minutes=5):
                return utc_aware.strftime("%Y-%m-%dT%H:%M:%SZ")

    return (now_utc + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_authenticated_service():
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    creds = None

    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CLIENT_SECRETS.exists():
                raise FileNotFoundError(
                    f"Missing {CLIENT_SECRETS}\n"
                    "Download OAuth credentials from Google Cloud Console → "
                    "APIs & Services → Credentials → Download JSON"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
            creds = flow.run_local_server(port=0)

        TOKEN_FILE.write_text(creds.to_json())

    return build("youtube", "v3", credentials=creds)


def load_niche():
    niches = json.loads(NICHES_FILE.read_text())
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or niches["active"]
    return niches["niches"][active], active


def build_metadata(script: str, niche_name: str, niche_cfg: dict) -> dict:
    hashtags   = " ".join(NICHE_HASHTAGS.get(niche_name, ["#Shorts"]))
    first_line = script.strip().split("\n")[0][:80]
    title      = first_line if first_line else "Daily Short"

    description = (
        f"{script}\n\n"
        f"{hashtags}\n\n"
        f"{niche_cfg.get('cta', '')}"
    )

    tags     = [t.lstrip("#") for t in NICHE_HASHTAGS.get(niche_name, [])]
    category = niche_cfg.get("youtube_category_id") or DEFAULT_CATEGORIES.get(niche_name, "22")

    return {
        "title":       title,
        "description": description,
        "tags":        tags,
        "categoryId":  category,
    }


def upload(video_path: Path, script: str, thumbnail_path: Path = None) -> tuple[str, str]:
    """Upload video (+ optional thumbnail); return (video_url, video_id)."""
    from googleapiclient.http import MediaFileUpload

    niche_cfg, niche_name = load_niche()
    youtube               = get_authenticated_service()
    metadata              = build_metadata(script, niche_name, niche_cfg)

    publish_at = _next_peak_utc()
    print(f"[youtube] uploading: {video_path.name}")
    print(f"[youtube] title: {metadata['title']}  category: {metadata['categoryId']}")
    print(f"[youtube] scheduled: publish at {publish_at} UTC")

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title":       metadata["title"],
                "description": metadata["description"],
                "tags":        metadata["tags"],
                "categoryId":  metadata["categoryId"],
            },
            "status": {
                "privacyStatus":           "private",
                "publishAt":               publish_at,
                "selfDeclaredMadeForKids": False,
            },
        },
        media_body=MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=1024 * 1024,
        ),
    )

    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"\r[youtube] {int(status.progress() * 100)}%", end="", flush=True)

    print()
    video_id  = response["id"]
    video_url = f"https://youtube.com/shorts/{video_id}"
    print(f"[youtube] uploaded → {video_url}")

    if thumbnail_path and Path(thumbnail_path).exists():
        try:
            from googleapiclient.http import MediaFileUpload as _MFU
            youtube.thumbnails().set(
                videoId=video_id,
                media_body=_MFU(str(thumbnail_path), mimetype="image/jpeg"),
            ).execute()
            print(f"[youtube] custom thumbnail uploaded")
        except Exception as e:
            print(f"[youtube] thumbnail upload skipped: {e}")

    return video_url, video_id


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python youtube_uploader.py <video.mp4> '<script text>'")
        sys.exit(1)

    path   = Path(sys.argv[1])
    script = sys.argv[2]
    url, _ = upload(path, script)
    print(f"\nYOUTUBE_URL={url}")
