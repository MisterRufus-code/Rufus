#!/usr/bin/env python3
"""
analytics_fetcher.py
Pulls YouTube Analytics for recently uploaded videos and saves metrics to SQLite.
Multi-channel: loops every channel with YouTube enabled, authenticating with
that channel's own token (channel_config resolves legacy single-channel paths).

Run daily via cron:
    0 10 * * * cd ~/Rufus && venv/bin/python scripts/analytics_fetcher.py
"""

import sys
from datetime import date, timedelta
from pathlib import Path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

sys.path.insert(0, str(Path(__file__).parent))
from db_manager import get_recent_tracked_videos, save_metrics
from channel_config import load_channel, list_channels

SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]

# Only fetch metrics for videos uploaded in the last N days.
RECENT_WINDOW_DAYS = 60


def _auth(channel) -> Credentials:
    token_file     = channel.token_path("youtube")
    client_secrets = channel.client_secrets_path()
    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow  = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
            creds = flow.run_local_server(port=0)
        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json())
    return creds


def fetch_analytics(channel_id: str = None):
    channels = [channel_id] if channel_id else list_channels()
    for cid in channels:
        channel = load_channel(cid)
        if not channel.platform_enabled("youtube"):
            continue
        try:
            _fetch_channel(channel)
        except Exception as e:
            print(f"[analytics] channel {channel.id} failed: {e}")


def _fetch_channel(channel):
    videos = get_recent_tracked_videos(days=RECENT_WINDOW_DAYS, channel=channel.id)
    if not videos:
        print(f"[analytics] {channel.id}: no videos uploaded in last {RECENT_WINDOW_DAYS} days.")
        return

    creds    = _auth(channel)
    yt       = build("youtube",          "v3", credentials=creds)
    yt_analy = build("youtubeAnalytics", "v2", credentials=creds)

    print(f"[analytics] {channel.id}: fetching metrics for {len(videos)} recent videos...")

    today_str = date.today().strftime("%Y-%m-%d")
    earliest  = (date.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    for row in videos:
        vid_id = row["youtube_id"]
        db_id  = row["id"]
        try:
            resp  = yt.videos().list(part="statistics", id=vid_id).execute()
            items = resp.get("items", [])
            if not items:
                print(f"[analytics] {vid_id}: not found on YouTube")
                continue
            stats = items[0]["statistics"]
            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))

            analy = yt_analy.reports().query(
                ids="channel==MINE",
                startDate=earliest,
                endDate=today_str,
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
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", help="Fetch one channel (default: all enabled)")
    args = ap.parse_args()
    fetch_analytics(args.channel)
