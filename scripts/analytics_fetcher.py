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

# Google libs are imported lazily inside the functions that need them (same
# pattern as youtube_uploader) so this module can be imported without the
# full Google auth stack — e.g. by tests asserting on SCOPES.

sys.path.insert(0, str(Path(__file__).parent))
from db_manager import get_recent_tracked_videos, save_metrics
from channel_config import load_channel, list_channels
# ONE scope list for the shared token file — declaring a different list here
# than the uploader's made the first scheduled run after an uploader auth fall
# into an interactive OAuth flow (see the SCOPES note in youtube_uploader.py).
from youtube_uploader import SCOPES

# Only fetch metrics for videos uploaded in the last N days.
RECENT_WINDOW_DAYS = 60


def _auth(channel):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow

    token_file     = channel.token_path("youtube")
    client_secrets = channel.client_secrets_path()
    creds = None
    if token_file.exists():
        creds = Credentials.from_authorized_user_file(str(token_file), SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                print(f"[analytics] token refresh failed ({e}) — re-auth required")
                creds = None
        if not creds or not creds.valid:
            # Analytics is a nice-to-have that runs BEFORE main.py in the
            # scheduled .bat — an interactive OAuth prompt here used to hang
            # the whole Task Scheduler job forever (no browser, no timeout),
            # killing that day's video. Fail fast instead; fetch_analytics'
            # per-channel try/except turns this into a skipped channel.
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
            try:
                creds = flow.run_local_server(port=0)
            except Exception as e:
                raise RuntimeError(
                    f"YouTube analytics auth needs a browser and none is "
                    f"available ({e}). Run `python scripts/analytics_fetcher.py` "
                    f"once interactively to regenerate {token_file}."
                ) from e
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
    from googleapiclient.discovery import build

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

            # annotationClickThroughRate was retired by YouTube in 2019 — it
            # returned 0 for every video, and feedback_analyzer multiplied its
            # engagement score by that 0, zeroing ALL scores (the "winners"
            # the hook prompt learned from were just the newest rows). The
            # Analytics API exposes no public impressions-CTR either, so CTR
            # is stored as 0.0 (schema kept) and the engagement formula no
            # longer uses it.
            analy = yt_analy.reports().query(
                ids="channel==MINE",
                startDate=earliest,
                endDate=today_str,
                metrics="views,averageViewPercentage",
                filters=f"video=={vid_id}",
            ).execute()

            rows      = analy.get("rows", [])
            watch_pct = float(rows[0][1]) if rows else 0.0

            save_metrics(db_id, views=views, watch_pct=watch_pct, ctr=0.0, likes=likes)
            print(f"[analytics] {vid_id}: {views} views, {watch_pct:.1f}% watch")

        except Exception as e:
            print(f"[analytics] {vid_id} failed: {e}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", help="Fetch one channel (default: all enabled)")
    args = ap.parse_args()
    fetch_analytics(args.channel)
