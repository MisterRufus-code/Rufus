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

# force-ssl is needed for commentThreads().insert (the post-upload CTA comment);
# yt-analytics.readonly lets analytics_fetcher reuse the SAME token — it used to
# declare its own 3-scope list against the same token file, so the first
# scheduled run after an uploader auth hit an interactive OAuth prompt and hung
# the Task Scheduler job forever (run_scheduled.bat runs analytics BEFORE main).
# One superset, declared once, imported by analytics_fetcher.
# NOTE: adding a scope invalidates the old token — delete config/youtube_token.json
# and run one manual upload to re-OAuth (one time only).
SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.force-ssl",
    "https://www.googleapis.com/auth/yt-analytics.readonly",
]

# YouTube category IDs by niche (overridable via niches.json "youtube_category_id")
DEFAULT_CATEGORIES = {
    "finance":              "25",   # News & Politics
    "motivation":           "22",   # People & Blogs
    "mindset":              "27",   # Education
    "business":             "27",   # Education
    "personal_development": "27",   # Education
    "money_history":        "27",   # Education
}

PEAK_HOURS_ET = [8, 12, 17, 20]  # US Eastern hours

NICHE_HASHTAGS = {
    "finance":              ["#finance", "#investing", "#wealth", "#money", "#stockmarket", "#Shorts"],
    "motivation":           ["#motivation", "#mindset", "#grind", "#discipline", "#success", "#Shorts"],
    "mindset":              ["#mindset", "#psychology", "#selfimprovement", "#mentalhealth", "#Shorts"],
    "business":             ["#business", "#entrepreneur", "#startup", "#hustle", "#success", "#Shorts"],
    "personal_development": ["#personaldevelopment", "#habits", "#growth", "#selfimprovement", "#Shorts"],
    "money_history":        ["#history", "#money", "#economics", "#historyfacts", "#didyouknow", "#Shorts"],
}


def _channel():
    """Active channel (legacy shim returns the original single-channel paths)."""
    from channel_config import load_channel
    return load_channel()


def _next_peak_utc(peak_hours: list[int] = None, tz_name: str = None) -> str:
    """Return ISO 8601 UTC timestamp for the next peak hour, ≥5 min from now.
    Peak hours/timezone come from the channel config (US-ET defaults).
    Uses zoneinfo so DST is automatic — no hardcoded UTC offset."""
    from zoneinfo import ZoneInfo
    peaks   = peak_hours or PEAK_HOURS_ET
    tz      = ZoneInfo(tz_name or "America/New_York")
    now_utc = datetime.now(tz=timezone.utc)
    now_loc = now_utc.astimezone(tz)

    for day_delta in range(3):
        day = now_loc.date() + timedelta(days=day_delta)
        for hour in sorted(peaks):
            loc_aware = datetime(day.year, day.month, day.day, hour, 0, 0, tzinfo=tz)
            utc_aware = loc_aware.astimezone(timezone.utc)
            if utc_aware > now_utc + timedelta(minutes=5):
                return utc_aware.strftime("%Y-%m-%dT%H:%M:%SZ")

    return (now_utc + timedelta(hours=24)).strftime("%Y-%m-%dT%H:%M:%SZ")


def get_authenticated_service(channel=None):
    from google.auth.transport.requests import Request
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import InstalledAppFlow
    from googleapiclient.discovery import build

    channel        = channel or _channel()
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
                # Revoked/expired refresh token — don't crash a cron run on it.
                # Park the stale token and fall through to interactive re-auth.
                print(f"[youtube] token refresh failed ({e}) — re-auth required")
                try:
                    token_file.replace(token_file.with_suffix(".json.stale"))
                except OSError:
                    pass
                creds = None
        if not creds or not creds.valid:
            if not client_secrets.exists():
                raise FileNotFoundError(
                    f"Missing {client_secrets}\n"
                    "Download OAuth credentials from Google Cloud Console → "
                    "APIs & Services → Credentials → Download JSON"
                )
            flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets), SCOPES)
            try:
                creds = flow.run_local_server(port=0)
            except Exception as e:
                raise RuntimeError(
                    f"YouTube re-authentication needs a browser and none is available "
                    f"({e}). Run `python scripts/youtube_uploader.py <video> '<script>'` "
                    f"once interactively to regenerate {token_file}."
                ) from e

        token_file.parent.mkdir(parents=True, exist_ok=True)
        token_file.write_text(creds.to_json())

    return build("youtube", "v3", credentials=creds)


def load_niche():
    niches = json.loads(NICHES_FILE.read_text())
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or niches["active"]
    return niches["niches"][active], active


def build_metadata(script: str, niche_name: str, niche_cfg: dict) -> dict:
    """GPT-optimized title/description/tags (metadata_writer), legacy on failure."""
    hashtags = NICHE_HASHTAGS.get(niche_name, ["#Shorts"])
    category = niche_cfg.get("youtube_category_id") or DEFAULT_CATEGORIES.get(niche_name, "22")

    try:
        from metadata_writer import generate_metadata
        meta = generate_metadata(script, niche_name, niche_cfg, hashtags=hashtags)
    except Exception as e:
        print(f"[youtube] metadata_writer unavailable ({e}) — legacy metadata")
        first_line = script.strip().split("\n")[0][:80]
        meta = {
            "title":       first_line if first_line else "Daily Short",
            "description": f"{script}\n\n{' '.join(hashtags)}\n\n{niche_cfg.get('cta', '')}",
            "tags":        [t.lstrip("#") for t in hashtags],
        }

    meta["categoryId"] = category
    return meta


def post_cta_comment(youtube, video_id: str, niche_cfg: dict) -> None:
    """Post an owner CTA comment right after upload (Shorts can't pin via API —
    pinning stays a manual 10-second step in the daily checklist). Never raises."""
    import random
    pool = niche_cfg.get("cta_pool") or [niche_cfg.get("cta", "")]
    text = random.choice([c for c in pool if c] or ["What would you add?"])
    try:
        youtube.commentThreads().insert(
            part="snippet",
            body={"snippet": {
                "videoId": video_id,
                "topLevelComment": {"snippet": {"textOriginal": text}},
            }},
        ).execute()
        print(f"[youtube] CTA comment posted: {text[:60]}")
    except Exception as e:
        print(f"[youtube] CTA comment skipped: {e}")


def upload(video_path: Path, script: str, thumbnail_path: Path = None,
           metadata: dict = None) -> tuple[str, str]:
    """Upload video (+ optional thumbnail); return (video_url, video_id).
    Pass `metadata` to reuse a pre-built dict (avoids a second GPT call)."""
    from googleapiclient.http import MediaFileUpload

    channel               = _channel()
    niche_cfg, niche_name = load_niche()
    niche_cfg             = {**niche_cfg, **channel.niche_overrides.get(niche_name, {})}
    youtube               = get_authenticated_service(channel)
    metadata              = metadata or build_metadata(script, niche_name, niche_cfg)
    if "categoryId" not in metadata:
        metadata["categoryId"] = (niche_cfg.get("youtube_category_id")
                                  or DEFAULT_CATEGORIES.get(niche_name, "22"))

    privacy = channel.upload.get("privacy")
    if privacy not in ("private", "unlisted", "public"):
        privacy = "private"
    # Every Rufus video is 100% synthetic — GPT script, FLUX/SVD-generated
    # imagery and motion, synthesized voice — so this is unconditionally True,
    # never a per-niche knob. YouTube's altered/synthetic-content disclosure
    # policy (API support added 2024-10-30) requires self-declaring this for
    # realistic AI-generated content; the API returns it back once set but
    # never infers it — an undisclosed synthetic-media channel risks a strike
    # or removal once it's actually public, not just a style choice.
    status = {"privacyStatus": privacy, "selfDeclaredMadeForKids": False,
             "containsSyntheticMedia": True}
    if privacy == "private":
        # publishAt is only valid on private uploads (YouTube then auto-publishes)
        status["publishAt"] = _next_peak_utc(channel.upload.get("peak_hours"),
                                             channel.upload.get("timezone"))

    print(f"[youtube] channel: {channel.id}")
    print(f"[youtube] uploading: {video_path.name}")
    print(f"[youtube] title: {metadata['title']}  category: {metadata['categoryId']}")
    if "publishAt" in status:
        print(f"[youtube] scheduled: publish at {status['publishAt']} UTC")
    else:
        print(f"[youtube] privacy: {privacy} (immediate)")

    request = youtube.videos().insert(
        part="snippet,status",
        body={
            "snippet": {
                "title":       metadata["title"],
                "description": metadata["description"],
                "tags":        metadata["tags"],
                "categoryId":  metadata["categoryId"],
            },
            "status": status,
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
        progress, response = request.next_chunk()   # don't shadow the status dict above
        if progress:
            print(f"\r[youtube] {int(progress.progress() * 100)}%", end="", flush=True)

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

    post_cta_comment(youtube, video_id, niche_cfg)

    return video_url, video_id


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python youtube_uploader.py <video.mp4> '<script text>'")
        sys.exit(1)

    path   = Path(sys.argv[1])
    script = sys.argv[2]
    url, _ = upload(path, script)
    print(f"\nYOUTUBE_URL={url}")
