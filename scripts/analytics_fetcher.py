#!/usr/bin/env python3
"""
analytics_fetcher.py
Pulls YouTube Analytics for each tracked video and saves metrics to SQLite.

Run daily via cron:
    0 10 * * * cd ~/Rufus && venv/bin/python scripts/analytics_fetcher.py
"""

import sys
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

sys.path.insert(0, str(Path(__file__).parent))
from db_manager import get_untracked_videos, save_metrics

CONFIG_DIR     = Path(__file__).parent.parent / "config"
CLIENT_SECRETS = CONFIG_DIR / "client_secrets.json"
TOKEN_FILE     = CONFIG_DIR / "youtube_token.json"

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]


def _auth() -> Credentials:
    creds = None
    if TOKEN_FILE.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow  = InstalledAppFlow.from_client_secrets_file(str(CLIENT_SECRETS), SCOPES)
            creds = flow.run_local_server(port=0)
        TOKEN_FILE.write_text(creds.to_json())
    return creds


def fetch_analytics():
    creds    = _auth()
    yt       = build("youtube",          "v3", credentials=creds)
    yt_analy = build("youtubeAnalytics", "v2", credentials=creds)

    videos = get_untracked_videos()
    if not videos:
        print("No tracked videos found.")
        return

    print(f"Fetching analytics for {len(videos)} videos...")

    for row in videos:
        vid_id = row["youtube_id"]
        db_id  = row["id"]
        try:
            # Basic stats from Data API
            resp  = yt.videos().list(part="statistics", id=vid_id).execute()
            items = resp.get("items", [])
            if not items:
                print(f"[analytics] {vid_id}: not found on YouTube")
                continue
            stats = items[0]["statistics"]
            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))

            # Detailed analytics (watch %, CTR) from Analytics API
            analy = yt_analy.reports().query(
                ids="channel==MINE",
                startDate="2020-01-01",
                endDate="2099-12-31",
                metrics="views,averageViewPercentage,annotationClickThroughRate",
                filters=f"video=={vid_id}",
            ).execute()

            rows      = analy.get("rows", [])
            watch_pct = float(rows[0][1]) if rows else 0.0
            ctr       = float(rows[0][2]) if rows else 0.0

            save_metrics(db_id, views=views, watch_pct=watch_pct, ctr=ctr, likes=likes)
            print(f"[analytics] {vid_id}: {views} views, {watch_pct:.1f}% watch, {ctr:.2f}% CTR")

        except Exception as e:
            print(f"[analytics] {vid_id} failed: {e}")


if __name__ == "__main__":
    fetch_analytics()
