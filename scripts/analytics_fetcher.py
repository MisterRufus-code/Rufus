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
        token_file.write_text(creds.to_json(), encoding="utf-8")
    return creds


def fetch_analytics(channel_id: str = None, *, digest: bool = True):
    """Refresh metrics for every enabled channel, then (optionally) post one
    digest of what actually moved.

    The digest is the point of running this daily for a human: the numbers
    were already landing in SQLite, but nothing ever said them out loud, so
    "how did last week do" meant opening the dashboard on purpose."""
    channels = [channel_id] if channel_id else list_channels()
    collected: list[dict] = []
    for cid in channels:
        channel = load_channel(cid)
        if not channel.platform_enabled("youtube"):
            continue
        try:
            collected += _fetch_channel(channel)
        except Exception as e:
            print(f"[analytics] channel {channel.id} failed: {e}")
    if digest and collected:
        _post_digest(collected)
    return collected


def _post_digest(rows: list[dict]) -> None:
    """Summarize the fetch into one notification. Best effort by contract —
    analytics is a reporting step and must never fail the run that calls it."""
    try:
        import notify
    except Exception:
        return
    top = sorted(rows, key=lambda r: r["views"], reverse=True)[:5]
    total_views = sum(r["views"] for r in rows)
    watched = [r["watch_pct"] for r in rows if r["watch_pct"]]
    avg_watch = sum(watched) / len(watched) if watched else 0.0

    lines = [f"{total_views:,} views across {len(rows)} tracked video"
             f"{'s' if len(rows) != 1 else ''}",
             f"average watch {avg_watch:.1f}%", ""]
    for r in top:
        title = (r.get("title") or r["youtube_id"])[:60]
        lines.append(f"• {r['views']:,} views · {r['watch_pct']:.0f}% · {title}")
    try:
        notify.notify_analytics("\n".join(lines), rows=len(rows))
    except Exception as e:
        print(f"[analytics] digest notification failed (non-fatal): {e}")


def _fetch_channel(channel) -> list[dict]:
    """Fetch + persist metrics for one channel. Returns what it saved, so the
    caller can build a digest without re-reading the DB."""
    from googleapiclient.discovery import build

    collected: list[dict] = []
    videos = get_recent_tracked_videos(days=RECENT_WINDOW_DAYS, channel=channel.id)
    if not videos:
        print(f"[analytics] {channel.id}: no videos uploaded in last {RECENT_WINDOW_DAYS} days.")
        return collected

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
            # snippet as well as statistics: the digest needs a human-readable
            # title, and get_recent_tracked_videos only carries ids. Parts are
            # free here — videos.list costs 1 quota unit either way.
            resp  = yt.videos().list(part="snippet,statistics", id=vid_id).execute()
            items = resp.get("items", [])
            if not items:
                print(f"[analytics] {vid_id}: not found on YouTube")
                continue
            stats = items[0]["statistics"]
            title = items[0].get("snippet", {}).get("title", "")
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
            collected.append({
                "youtube_id": vid_id, "views": views, "likes": likes,
                "watch_pct": watch_pct, "channel": channel.id, "title": title,
            })

        except Exception as e:
            print(f"[analytics] {vid_id} failed: {e}")

    return collected


if __name__ == "__main__":
    # Saved settings first — this runs from a scheduled task, so without them
    # the daily digest is assembled and then posted nowhere. See settings_store.
    try:
        import settings_store
        settings_store.apply()
    except Exception as e:
        print(f"[analytics] saved settings not read ({e})")
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--channel", help="Fetch one channel (default: all enabled)")
    ap.add_argument("--no-digest", action="store_true",
                    help="Save metrics without posting the summary notification")
    args = ap.parse_args()
    fetch_analytics(args.channel, digest=not args.no_digest)
