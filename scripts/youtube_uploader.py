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
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    peaks   = peak_hours or PEAK_HOURS_ET
    now_utc = datetime.now(tz=timezone.utc)
    try:
        tz = ZoneInfo(tz_name or "America/New_York")
    except ZoneInfoNotFoundError:
        # WINDOWS HAS NO IANA DATABASE. zoneinfo reads the system tz files,
        # which exist on Linux and macOS and do not exist on Windows at all —
        # there, the `tzdata` package IS the database, and it was in no
        # requirements file. Live, this surfaced as
        #
        #     Upload failed (not uploaded, safe to retry):
        #     'No time zone found with key America/New_York'
        #
        # on a finished, rendered video: a scheduling nicety refusing to
        # publish work that was ready. The schedule is worth having and is not
        # worth the upload, so this says exactly what to install and puts the
        # video up now instead of holding it hostage.
        import paths
        print(f"[youtube] ⚠ no timezone database for "
              f"{tz_name or 'America/New_York'} — Windows has none of its own. "
              f"Install it with `{paths.pip_hint('tzdata')}`.")
        print(f"[youtube]   uploading WITHOUT a scheduled publish time; the "
              f"video goes up private and you publish it yourself.")
        return ""
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
        token_file.write_text(creds.to_json(), encoding="utf-8")

    return build("youtube", "v3", credentials=creds)


def load_niche():
    niches = json.loads(NICHES_FILE.read_text(encoding="utf-8"))
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or niches["active"]
    return niches["niches"][active], active


def _hashtags_for(niche_name: str) -> list[str]:
    """The niche's hashtags, minus the one that would miscategorise the video.

    EVERY LIST ENDS IN #Shorts, and that was right while every video was one.
    YouTube reads that tag as a declaration of format: a nine-minute landscape
    upload carrying it is asking to be filed as something it is not, shown in
    a feed it cannot compete in, and measured against retention curves that do
    not apply to it. The tag is not decoration, it is a category, and it is the
    kind of mistake that is invisible in the pipeline and obvious in the
    analytics three weeks later.
    """
    tags = list(NICHE_HASHTAGS.get(niche_name, ["#Shorts"]))
    try:
        import video_format
        if video_format.is_long():
            tags = [t for t in tags if t.lower() != "#shorts"]
            if not tags:
                tags = ["#documentary"]
    except Exception:
        pass
    return tags


def build_metadata(script: str, niche_name: str, niche_cfg: dict,
                   chapters: str = "") -> dict:
    """GPT-optimized title/description/tags (metadata_writer), legacy on failure.

    `chapters` is the already-formatted timestamp block, or "" — see
    chapters.py. It goes in ABOVE the hashtags and the CTA because YouTube
    reads the list from the description text and a viewer skimming for what is
    inside should not have to scroll past a wall of tags to find it.
    """
    hashtags = _hashtags_for(niche_name)
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

    if chapters:
        # After the opening paragraph, before everything else. Not at the very
        # top — the first two lines of a description are what search and the
        # watch page show, and spending them on "0:00 Intro" throws away the
        # copy metadata_writer wrote to earn the click. Not at the bottom
        # either, under the hashtags, where nobody scrolls. The rule YouTube
        # actually enforces is about the first TIMESTAMP being 0:00, not the
        # first line. Guarded so a re-run cannot stack two copies.
        desc = str(meta.get("description") or "")
        if "0:00 " not in desc:
            opening, sep, rest = desc.partition("\n\n")
            meta["description"] = (f"{opening}\n\n{chapters}\n\n{rest}".strip()
                                   if sep else f"{desc}\n\n{chapters}".strip())

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


def post_source_comment(youtube, video_id: str, source_url: str,
                        seed_source: str = None) -> None:
    """Post a comment citing the real source the script was grounded in
    (a Wikipedia article, a Stack Exchange question, ...) — trust/
    differentiation lever against generic "AI slop" history channels, not
    just an engagement CTA. Same API limitation as post_cta_comment: the
    public YouTube Data API has no endpoint to PIN a comment, only to post
    one — pinning stays a manual step (see the daily checklist). A no-op
    when there's no URL (older rows from before seed_url existed, or a
    seed type — the wisdom pool — that never carried one). Never raises."""
    if not source_url:
        return
    label = f" ({seed_source})" if seed_source else ""
    text = f"Source for this one{label}: {source_url}"
    try:
        youtube.commentThreads().insert(
            part="snippet",
            body={"snippet": {
                "videoId": video_id,
                "topLevelComment": {"snippet": {"textOriginal": text}},
            }},
        ).execute()
        print(f"[youtube] source comment posted: {source_url}")
    except Exception as e:
        print(f"[youtube] source comment skipped: {e}")


def upload(video_path: Path, script: str, thumbnail_path: Path = None,
           metadata: dict = None, source_url: str = None,
           seed_source: str = None) -> tuple[str, str]:
    """Upload video (+ optional thumbnail); return (video_url, video_id).
    Pass `metadata` to reuse a pre-built dict (avoids a second GPT call).
    Pass `source_url` (+ optionally `seed_source` for a nicer label) to also
    post a source-citation comment — see post_source_comment()."""
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
        # publishAt is only valid on private uploads (YouTube then auto-publishes).
        # "" means the timezone database is missing and _next_peak_utc already
        # said so — upload now rather than lose a finished render to a
        # scheduling nicety.
        when = _next_peak_utc(channel.upload.get("peak_hours"),
                              channel.upload.get("timezone"))
        if when:
            status["publishAt"] = when

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
    post_source_comment(youtube, video_id, source_url, seed_source)

    return video_url, video_id


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python youtube_uploader.py <video.mp4> '<script text>'")
        sys.exit(1)

    path   = Path(sys.argv[1])
    script = sys.argv[2]
    url, _ = upload(path, script)
    print(f"\nYOUTUBE_URL={url}")
