#!/usr/bin/env python3
"""
music_fetcher.py — Provider-agnostic background music for Rufus.

Fetches royalty-free music by niche mood. Provider chain:
  1. Jamendo API  (free client_id from developers.jamendo.com — set in config/keys.json)
  2. archive.org  (no key needed — CC0 public domain audio)
  3. music_gen    (synthesized local ambient bed — zero APIs, always available)
  4. None         (only if even local synthesis fails — render proceeds voice-only)

Tracks are cached in media_library/music/ so repeat runs are instant.

Add to config/keys.json:
    "jamendo_client_id": "your_client_id_here"
"""

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path

import paths

import requests

ROOT       = Path(__file__).parent.parent
CONFIG_DIR = ROOT / "config"
MUSIC_DIR  = paths.media_root() / "music"
KEYS_FILE  = CONFIG_DIR / "keys.json"

MOOD_MAP = {
    "finance":              ["corporate", "ambient business", "cinematic tension"],
    "motivation":           ["epic cinematic", "motivational orchestral", "powerful"],
    "mindset":              ["calm lofi", "ambient focus", "peaceful instrumental"],
    "business":             ["corporate", "upbeat business", "ambient professional"],
    "personal_development": ["calm lofi", "ambient focus", "inspirational soft"],
    "money_history":        ["cinematic documentary", "epic historical orchestral", "ambient tension"],
}
DEFAULT_MOODS = ["ambient", "instrumental", "calm"]

MIN_TRACK_BYTES  = 200_000
MIN_TRACK_SECS   = 55.0   # reject tracks shorter than this — Shorts are ~60s

# archive.org throttles/rejects the default python-requests UA — identify properly.
_UA = {"User-Agent": "Rufus-Shorts/1.0 (autonomous video pipeline; contact via repo)"}


def _check_audio_duration(path: Path, min_secs: float = MIN_TRACK_SECS) -> bool:
    """Return True if the audio file is at least min_secs long."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        info = json.loads(r.stdout)
        for stream in info.get("streams", []):
            dur = float(stream.get("duration", 0))
            if dur >= min_secs:
                return True
        return False
    except Exception:
        return True   # if ffprobe unavailable, don't reject the track


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
        r = requests.get(url, stream=True, timeout=60, headers=_UA)
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
                "minDuration":   60,  # Shorts are ~60s; 25s tracks loop badly
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
        if not _download(audio_url, dest):
            return None
        if not _check_audio_duration(dest):
            print(f"[music] Jamendo: track too short (<{MIN_TRACK_SECS:.0f}s) — skipping")
            dest.unlink(missing_ok=True)
            return None
        return dest
    except Exception as e:
        print(f"[music] Jamendo failed: {e}")
        return None


def _archive_music(mood: str) -> Path | None:
    try:
        r = requests.get(
            "https://archive.org/advancedsearch.php",
            params={
                # Require a music subject and exclude spoken-word formats so we
                # don't pull podcasts/interviews/lectures/audiobooks (which the
                # broad opensource_audio collection is full of).
                "q":      (f'({mood}) AND mediatype:(audio) AND format:(MP3) '
                           f'AND subject:(music) '
                           f'AND -subject:(podcast) AND -subject:(speech) '
                           f'AND -subject:(interview) AND -subject:(sermon) '
                           f'AND -subject:(lecture) AND -subject:(audiobook) '
                           f'AND -subject:(poetry) AND -subject:(talk)'),
                "fl":     "identifier",
                "rows":   20,
                "output": "json",
            },
            timeout=15,
            headers=_UA,
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
                    headers=_UA,
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
                    if _check_audio_duration(dest):
                        return dest
                    print(f"[music] archive.org: track too short — skipping")
                    dest.unlink(missing_ok=True)
            except Exception as e:
                print(f"[music] archive.org {identifier} failed: {e}")
                continue
    except Exception as e:
        print(f"[music] archive.org failed: {e}")
    return None


def fetch_music(niche: str) -> Path | None:
    """Return path to a music track for the niche, or None for voice-only.

    Provider chain (best-to-worst quality):
      1. Jamendo — real produced tracks with proper mood tagging (needs free key)
      2. MusicGen — AI-generated from a niche-specific text prompt (always on-tone)
      3. Synthesized local bed — deterministic mood bed (fallback, always works)
      4. archive.org — last resort (unpredictable; speech-filtered query)
    """
    MUSIC_DIR.mkdir(parents=True, exist_ok=True)
    moods = MOOD_MAP.get(niche, DEFAULT_MOODS)

    # 1. Jamendo — real, produced, mood-tagged INSTRUMENTAL music (free key).
    for mood in moods:
        track = _jamendo(mood)
        if track:
            return track

    # 2. MusicGen — AI-generated from a niche text description.
    #    Output is literally written for the niche's emotional tone: finance gets
    #    "cinematic dark ambient, documentary strings", not a random podcast track.
    try:
        from musicgen_gen import generate_music
        track = generate_music(niche)
        if track:
            return track
    except Exception as e:
        print(f"[music] MusicGen failed: {e}")

    # 3. Synthesized local bed — mood-matched and instrumental, always available.
    try:
        from music_gen import ensure_music
        track = ensure_music(niche)
        if track:
            return track
    except Exception as e:
        print(f"[music] local synthesis failed: {e}")

    # 4. archive.org — last resort only (unpredictable; query is speech-filtered).
    for mood in moods:
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
