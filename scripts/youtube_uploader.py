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
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

CONFIG_DIR      = Path(__file__).parent.parent / "config"
NICHES_FILE     = CONFIG_DIR / "niches.json"
CLIENT_SECRETS  = CONFIG_DIR / "client_secrets.json"
TOKEN_FILE      = CONFIG_DIR / "youtube_token.json"

SCOPES = ["https://www.googleapis.com/auth/youtube.upload"]

# Niche-specific hashtags
NICHE_HASHTAGS = {
    "finance":            ["#finance", "#investing", "#wealth", "#money", "#stockmarket", "#Shorts"],
    "motivation":         ["#motivation", "#mindset", "#grind", "#discipline", "#success", "#Shorts"],
    "mindset":            ["#mindset", "#psychology", "#selfimprovement", "#mentalhealth", "#Shorts"],
    "business":           ["#business", "#entrepreneur", "#startup", "#hustle", "#success", "#Shorts"],
    "personal_development": ["#personaldevelopment", "#habits", "#growth", "#selfimprovement", "#Shorts"],
}


def get_authenticated_service():
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
    active = niches["active"]
    return niches["niches"][active], active


def build_metadata(script: str, niche_name: str, niche_cfg: dict) -> dict:
    hashtags   = " ".join(NICHE_HASHTAGS.get(niche_name, ["#Shorts"]))
    # Title: first sentence of script, max 80 chars
    first_line = script.strip().split("\n")[0][:80]
    title      = first_line if first_line else "Daily Short"

    description = (
        f"{script}\n\n"
        f"{hashtags}\n\n"
        f"{niche_cfg.get('cta', '')}"
    )

    tags = [t.lstrip("#") for t in NICHE_HASHTAGS.get(niche_name, [])]

    return {
        "title":       title,
        "description": description,
        "tags":        tags,
        "categoryId":  "22",        # People & Blogs (works for all our niches)
    }


def upload(video_path: Path, script: str) -> str:
    niche_cfg, niche_name = load_niche()
    youtube               = get_authenticated_service()
    metadata              = build_metadata(script, niche_name, niche_cfg)

    print(f"[youtube] uploading: {video_path.name}")
    print(f"[youtube] title: {metadata['title']}")

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
                "privacyStatus":           "public",
                "selfDeclaredMadeForKids": False,
            },
        },
        media_body=MediaFileUpload(
            str(video_path),
            mimetype="video/mp4",
            resumable=True,
            chunksize=1024 * 1024,    # 1MB chunks
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
    return video_url


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python youtube_uploader.py <video.mp4> '<script text>'")
        sys.exit(1)

    path   = Path(sys.argv[1])
    script = sys.argv[2]
    url    = upload(path, script)
    print(f"\nYOUTUBE_URL={url}")
