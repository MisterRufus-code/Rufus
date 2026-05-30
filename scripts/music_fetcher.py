#!/usr/bin/env python3
"""
music_fetcher.py — Provider-agnostic background music for Rufus.

Fetches royalty-free music by niche mood. Provider chain:
  1. Jamendo API  (free client_id from developers.jamendo.com — set in config/keys.json)
  2. archive.org  (no key needed — CC0 public domain audio)
  3. None         (graceful skip — render proceeds voice-only, never crashes)

Tracks are cached in media_library/music/ so repeat runs are instant.

Add to config/keys.json:
    "jamendo_client_id": "your_client_id_here"
"""

import hashlib
import json
import os
import random
from pathlib import Path

import requests

ROOT       = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"
MUSIC_DIR  = ROOT / "media_library" / "music"
KEYS_FILE  = CONFIG_DIR / "keys.json"

MOOD_MAP = {
    "finance":              ["corporate", "ambient business", "cinematic tension"],
    "motivation":           ["epic cinematic", "motivational orchestral", "powerful"],
    "mindset":              ["calm lofi", "ambient focus", "peaceful instrumental"],
    "business":             ["corporate", "upbeat business", "ambient professional"],
    "personal_development": ["calm lofi", "ambient focus", "inspirational soft"],
}
DEFAULT_MOODS = ["ambient", "instrumental", "calm"]

MIN_TRACK_BYTES = 200_000


def _load_jamendo_key() -> str:
    try:
        return json.loads(KEYS_FILE.read_text()).get("jamendo_client_id", "")
    except (json.JSONDecodeError, OSError):
        return ""


def _cache_path(url: str) -> Path:
    h = hashlib.md5(url.encode()).hexdigest()[:14]
    return MUSIC_DIR / f"{h}.mp3"


def _download(url: str, dest: Path) -> bool:
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=16_384):
                f.write(chunk)
        if dest.stat().st_size < MIN_TRACK_BYTES:
            dest.unlink(missing_ok=True)
            return False
        return True
    except Exception as e:
        print(f"[music] download error: {e}")
        dest.unlink(missing_ok=True)
        return False


def _jamendo(mood: str) -> Path | None:
    client_id = _load_jamendo_key()
    if not client_id or client_id.startswith("YOUR_"):
        return None
    try:
        r = requests.get(
            "https://api.jamendo.com/v3.0/tracks/",
            params={
                "client_id":     client_id,
                "format":        "json",
                "limit":         20,
                "search":        mood,
                "audioformat":   "mp31",
                "license_cc":    "1",
                "minDuration":   25,
                "boost":         "popularity_month",
            },
            timeout=10,
        )
        r.raise_for_status()
        results = r.json().get("results", [])
        if not results:
            return None

        track = random.choice(results[:10])
        audio_url = track.get("audio") or track.get("audiodownload")
        if not audio_url:
            return None

        dest = _cache_path(audio_url)
        if dest.exists() and dest.stat().st_size >= MIN_TRACK_BYTES:
            print(f"[music] Jamendo cached: {dest.name}")
            return dest

        print(f"[music] Jamendo: \"{track.get('name','?')}\" by {track.get('artist_name','?')}")
        return dest if _download(audio_url, dest) else None
    except Exception as e:
        print(f"[music] Jamendo failed: {e}")
        return None


def _archive_music(mood: str) -> Path | None:
    try:
        r = requests.get(
            "https://archive.org/advancedsearch.php",
            params={
                "q":      f'{mood} AND mediatype:audio AND format:mp3 AND licenseurl:creativecommons',
                "fl":     "identifier",
                "rows":   20,
                "output": "json",
            },
            timeout=15,
        )
        r.raise_for_status()
        docs = r.json().get("response", {}).get("docs", [])
        if not docs:
            return None

        random.shuffle(docs)
        for doc in docs[:8]:
            identifier = doc.get("identifier", "")
            if not identifier:
                continue
            try:
                files_r = requests.get(
                    f"https://archive.org/metadata/{identifier}/files",
                    timeout=10,
                )
                files_r.raise_for_status()
                mp3s = [
                    f for f in files_r.json().get("result", [])
                    if f.get("name", "").lower().endswith(".mp3")
                    and int(f.get("size", 0)) >= MIN_TRACK_BYTES
                ]
                if not mp3s:
                    continue
                chosen    = random.choice(mp3s[:5])
                audio_url = f"https://archive.org/download/{identifier}/{chosen['name']}"
                dest      = _cache_path(audio_url)
                if dest.exists() and dest.stat().st_size >= MIN_TRACK_BYTES:
                    print(f"[music] archive.org cached: {dest.name}")
                    return dest
                print(f"[music] archive.org: {identifier}/{chosen['name']}")
                if _download(audio_url, dest):
                    return dest
            except Exception:
                continue
    except Exception as e:
        print(f"[music] archive.org failed: {e}")
    return None


def fetch_music(niche: str) -> Path | None:
    """Return path to a cached music track for the niche, or None for voice-only."""
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    moods = MOOD_MAP.get(niche, DEFAULT_MOODS)

    for mood in moods:
        track = _jamendo(mood)
        if track:
            return track
        track = _archive_music(mood)
        if track:
            return track

    print("[music] all sources failed — proceeding voice-only")
    return None


if __name__ == "__main__":
    import sys
    niche = sys.argv[1] if len(sys.argv) > 1 else "motivation"
    os.makedirs(MUSIC_DIR, exist_ok=True)
    path = fetch_music(niche)
    print(f"MUSIC_PATH={path}")
