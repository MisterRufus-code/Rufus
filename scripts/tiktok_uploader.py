#!/usr/bin/env python3
"""
tiktok_uploader.py
Posts a rendered Short to TikTok using the Content Posting API v2.

First-time setup:
    1. Apply for the Content Posting API at developers.tiktok.com
    2. Get approval (can take days). Required scope: video.publish
    3. Create config/tiktok_keys.json:
        {"client_key": "...", "client_secret": "..."}
    4. Run this script once with --auth to do the OAuth dance:
        python scripts/tiktok_uploader.py --auth
       → opens browser, you log in, paste the redirect URL back
       → config/tiktok_token.json is saved
    5. Future runs are fully automatic

Usage:
    python tiktok_uploader.py /path/to/short.mp4 "Script text here"
    python tiktok_uploader.py --auth
"""

import json
import os
import sys
import time
import urllib.parse
import webbrowser
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

CONFIG_DIR  = Path(__file__).parent.parent / "config"
NICHES_FILE = CONFIG_DIR / "niches.json"
KEYS_FILE   = CONFIG_DIR / "tiktok_keys.json"
TOKEN_FILE  = CONFIG_DIR / "tiktok_token.json"

REDIRECT_URI = "http://localhost:8080/callback"
SCOPES       = "user.info.basic,video.publish,video.upload"

OAUTH_BASE   = "https://www.tiktok.com/v2/auth/authorize/"
TOKEN_URL    = "https://open.tiktokapis.com/v2/oauth/token/"
INIT_URL     = "https://open.tiktokapis.com/v2/post/publish/video/init/"
STATUS_URL   = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"

NICHE_HASHTAGS = {
    "finance":              "#finance #investing #money #wealth #fyp",
    "motivation":           "#motivation #mindset #grind #discipline #fyp",
    "mindset":              "#mindset #psychology #selfimprovement #fyp",
    "business":             "#business #entrepreneur #startup #hustle #fyp",
    "personal_development": "#personaldevelopment #growth #habits #fyp",
}


# ── Auth ────────────────────────────────────────────────────────────────────────

def _load_keys() -> dict:
    if not KEYS_FILE.exists():
        raise FileNotFoundError(
            f"Missing {KEYS_FILE}. Create it with "
            '{"client_key": "...", "client_secret": "..."}'
        )
    return json.loads(KEYS_FILE.read_text())


def _save_token(data: dict) -> None:
    data["_saved_at"] = int(time.time())
    TOKEN_FILE.write_text(json.dumps(data, indent=2))


def _load_token() -> dict | None:
    if not TOKEN_FILE.exists():
        return None
    return json.loads(TOKEN_FILE.read_text())


def _refresh_if_needed(token: dict, keys: dict) -> dict:
    age      = int(time.time()) - token.get("_saved_at", 0)
    lifetime = token.get("expires_in", 86400)
    if age < lifetime - 300:
        return token  # still fresh

    resp = requests.post(
        TOKEN_URL,
        data={
            "client_key":    keys["client_key"],
            "client_secret": keys["client_secret"],
            "grant_type":    "refresh_token",
            "refresh_token": token["refresh_token"],
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )
    resp.raise_for_status()
    new = resp.json()
    _save_token(new)
    return new


def interactive_auth() -> None:
    """One-time OAuth flow. Opens browser, asks user to paste redirect URL."""
    keys  = _load_keys()
    state = os.urandom(8).hex()

    params  = {
        "client_key":    keys["client_key"],
        "scope":         SCOPES,
        "response_type": "code",
        "redirect_uri":  REDIRECT_URI,
        "state":         state,
    }
    auth_url = OAUTH_BASE + "?" + urllib.parse.urlencode(params)

    print(f"\nOpening browser to authorize TikTok…\n  {auth_url}\n")
    try:
        webbrowser.open(auth_url)
    except Exception:
        pass

    print("After approving, you'll be redirected to a localhost URL that won't load.")
    print("Copy the FULL URL from your browser's address bar and paste it here:")
    pasted = input("redirect URL: ").strip()

    qs   = urllib.parse.urlparse(pasted).query
    code = urllib.parse.parse_qs(qs).get("code", [None])[0]
    if not code:
        raise ValueError("No ?code=... found in the URL you pasted")

    resp = requests.post(
        TOKEN_URL,
        data={
            "client_key":    keys["client_key"],
            "client_secret": keys["client_secret"],
            "code":          code,
            "grant_type":    "authorization_code",
            "redirect_uri":  REDIRECT_URI,
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=20,
    )
    resp.raise_for_status()
    _save_token(resp.json())
    print(f"\nToken saved → {TOKEN_FILE}\n")


def _access_token() -> str:
    token = _load_token()
    if not token:
        raise RuntimeError(
            "No tiktok_token.json. Run:  python scripts/tiktok_uploader.py --auth"
        )
    keys  = _load_keys()
    token = _refresh_if_needed(token, keys)
    return token["access_token"]


# ── Niche / metadata ────────────────────────────────────────────────────────────

def _active_niche() -> tuple[dict, str]:
    data   = json.loads(NICHES_FILE.read_text())
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]
    return data["niches"][active], active


def _build_caption(script: str, niche_name: str, niche_cfg: dict) -> str:
    hook    = script.strip().split("\n")[0]
    tags    = NICHE_HASHTAGS.get(niche_name, "#fyp")
    cta     = niche_cfg.get("cta", "")
    # TikTok caption hard limit: 2200 chars
    caption = f"{hook}\n\n{cta}\n\n{tags}"
    return caption[:2200]


# ── Upload ──────────────────────────────────────────────────────────────────────

def upload(video_path: Path, script: str) -> str:
    """Post video to TikTok. Returns publish_id (poll status with that)."""
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"TikTok upload: video file not found: {video_path}")

    niche_cfg, niche_name = _active_niche()
    caption = _build_caption(script, niche_name, niche_cfg)

    token = _access_token()
    size  = video_path.stat().st_size
    chunk = min(size, 64 * 1024 * 1024)  # ≤ 64 MB per chunk per TikTok docs

    init_payload = {
        "post_info": {
            "title":                  caption,
            "privacy_level":          "SELF_ONLY",   # private draft – review before public
            "disable_duet":           False,
            "disable_comment":        False,
            "disable_stitch":         False,
            "video_cover_timestamp_ms": 3000,
        },
        "source_info": {
            "source":               "FILE_UPLOAD",
            "video_size":           size,
            "chunk_size":           chunk,
            "total_chunk_count":    max(1, (size + chunk - 1) // chunk),
        },
    }

    print(f"[tiktok] init upload: {video_path.name}  ({size / 1e6:.1f} MB)")
    init = requests.post(
        INIT_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type":  "application/json",
        },
        json=init_payload,
        timeout=30,
    )
    init.raise_for_status()
    info        = init.json()["data"]
    publish_id  = info["publish_id"]
    upload_url  = info["upload_url"]

    # PUT video to upload_url. For multi-chunk, repeat with Content-Range.
    print(f"[tiktok] uploading bytes…")
    with open(video_path, "rb") as fh:
        offset = 0
        chunk_idx = 0
        total_chunks = init_payload["source_info"]["total_chunk_count"]
        while True:
            buf = fh.read(chunk)
            if not buf:
                break
            end = offset + len(buf) - 1
            put = requests.put(
                upload_url,
                data=buf,
                headers={
                    "Content-Range": f"bytes {offset}-{end}/{size}",
                    "Content-Type":  "video/mp4",
                },
                timeout=300,
            )
            put.raise_for_status()
            chunk_idx += 1
            print(f"[tiktok] chunk {chunk_idx}/{total_chunks} uploaded")
            offset += len(buf)

    # Poll status until processing completes
    print(f"[tiktok] processing on TikTok…")
    for _ in range(30):
        time.sleep(4)
        s = requests.post(
            STATUS_URL,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type":  "application/json",
            },
            json={"publish_id": publish_id},
            timeout=20,
        )
        s.raise_for_status()
        status = s.json()["data"]["status"]
        if status in ("PUBLISH_COMPLETE", "SEND_TO_USER_INBOX"):
            print(f"[tiktok] ✓ status: {status}")
            return publish_id
        if status == "FAILED":
            raise RuntimeError(f"TikTok publish failed: {s.json()}")

    print(f"[tiktok] still processing – publish_id: {publish_id}")
    return publish_id


# ── CLI ─────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--auth":
        interactive_auth()
        sys.exit(0)

    if len(sys.argv) < 3:
        print("Usage:")
        print("  python tiktok_uploader.py --auth")
        print("  python tiktok_uploader.py <video.mp4> '<script text>'")
        sys.exit(1)

    path        = Path(sys.argv[1])
    script_text = sys.argv[2]
    pid         = upload(path, script_text)
    print(f"\nTIKTOK_PUBLISH_ID={pid}")
