#!/usr/bin/env python3
"""
media_fetcher.py
Downloads one background video based on the active niche.
Sources: Pixabay, Pexels, Vimeo, Coverr, NASA, Internet Archive
Falls back automatically if a source fails or hits rate limit.
"""

import json
import random
import time
from pathlib import Path

import requests

CONFIG_DIR  = Path(__file__).parent.parent / "config"
NICHES_FILE = CONFIG_DIR / "niches.json"
KEYS_FILE   = CONFIG_DIR / "keys.json"
CACHE_DIR   = Path(__file__).parent.parent / "media_library" / "cache"


# ── Config loaders ─────────────────────────────────────────────────────────────

def load_niche():
    niches = json.loads(NICHES_FILE.read_text())
    active = niches["active"]
    return niches["niches"][active], active


def load_keys():
    return json.loads(KEYS_FILE.read_text())


# ── Source: Pixabay ─────────────────────────────────────────────────────────────

def _pixabay(query: str, keys: dict) -> str:
    key = keys.get("pixabay", "")
    if not key or key.startswith("YOUR_"):
        raise ValueError("Pixabay key not set")

    resp = requests.get(
        "https://pixabay.com/api/videos/",
        params={
            "key":       key,
            "q":         query,
            "video_type": "film",
            "per_page":  20,
            "order":     "popular",
            "min_width": 1080,
        },
        timeout=10,
    )
    resp.raise_for_status()
    hits = resp.json().get("hits", [])
    if not hits:
        raise ValueError(f"Pixabay: no results for '{query}'")

    video = random.choice(hits[:10])
    for q in ("large", "medium", "small", "tiny"):
        url = video["videos"].get(q, {}).get("url")
        if url:
            return url
    raise ValueError("Pixabay: no usable URL")


# ── Source: Pexels ──────────────────────────────────────────────────────────────

def _pexels(query: str, keys: dict) -> str:
    key = keys.get("pexels", "")
    if not key or key.startswith("YOUR_"):
        raise ValueError("Pexels key not set")

    resp = requests.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": key},
        params={
            "query":        query,
            "per_page":     20,
            "min_width":    1080,
            "min_duration": 5,
            "max_duration": 60,
        },
        timeout=10,
    )
    resp.raise_for_status()
    videos = resp.json().get("videos", [])
    if not videos:
        raise ValueError(f"Pexels: no results for '{query}'")

    video = random.choice(videos[:10])
    files = sorted(video.get("video_files", []), key=lambda x: x.get("width", 0), reverse=True)
    for f in files:
        if f.get("link") and f.get("width", 0) >= 720:
            return f["link"]
    raise ValueError("Pexels: no HD file found")


# ── Source: Vimeo ───────────────────────────────────────────────────────────────

def _vimeo(query: str, keys: dict) -> str:
    token = keys.get("vimeo", "")
    if not token or token.startswith("YOUR_"):
        raise ValueError("Vimeo token not set")

    resp = requests.get(
        "https://api.vimeo.com/videos",
        headers={"Authorization": f"Bearer {token}"},
        params={
            "query":    query,
            "filter":   "CC",
            "per_page": 20,
            "sort":     "relevant",
            "fields":   "uri,download,files,name",
        },
        timeout=10,
    )
    resp.raise_for_status()
    data = resp.json().get("data", [])
    if not data:
        raise ValueError(f"Vimeo: no results for '{query}'")

    random.shuffle(data)
    for video in data[:10]:
        for dl in video.get("download", []) + video.get("files", []):
            if dl.get("link") and dl.get("width", 0) >= 720:
                return dl["link"]
    raise ValueError("Vimeo: no downloadable HD video")


# ── Source: Coverr (no key) ─────────────────────────────────────────────────────

def _coverr(_query: str, _keys: dict) -> str:
    page = random.randint(1, 40)
    resp = requests.get(
        "https://coverr.co/api/videos/featured",
        params={"page": page, "per_page": 10},
        timeout=10,
    )
    resp.raise_for_status()
    raw = resp.json()

    items = raw if isinstance(raw, list) else raw.get("hits", raw.get("videos", []))
    if not items:
        raise ValueError("Coverr: empty response")

    item = random.choice(items)
    for field in ("mp4_url", "url", "video_url", "file_url", "src"):
        if item.get(field):
            return item[field]
    raise ValueError("Coverr: no URL field found")


# ── Source: NASA (no key needed for search) ─────────────────────────────────────

def _nasa(query: str, keys: dict) -> str:
    resp = requests.get(
        "https://images.nasa.gov/api/v1/search",
        params={"q": query, "media_type": "video", "page_size": 20},
        timeout=10,
    )
    resp.raise_for_status()
    items = resp.json().get("collection", {}).get("items", [])
    if not items:
        raise ValueError(f"NASA: no results for '{query}'")

    item = random.choice(items[:10])
    # Each item has a href pointing to a JSON manifest
    manifest_url = item.get("href")
    if not manifest_url:
        raise ValueError("NASA: no manifest href")

    manifest = requests.get(manifest_url, timeout=10).json()
    # Find the mp4 (original or medium)
    mp4s = [u for u in manifest if u.endswith(".mp4")]
    if not mp4s:
        raise ValueError("NASA: no mp4 in manifest")
    return mp4s[0]


# ── Source: Internet Archive (no key) ──────────────────────────────────────────

def _archive(query: str, _keys: dict) -> str:
    resp = requests.get(
        "https://archive.org/advancedsearch.php",
        params={
            "q":      f'"{query}" AND mediatype:movies AND format:mp4',
            "fl":     "identifier",
            "rows":   20,
            "output": "json",
        },
        timeout=15,
    )
    resp.raise_for_status()
    docs = resp.json().get("response", {}).get("docs", [])
    if not docs:
        raise ValueError(f"Archive: no results for '{query}'")

    doc = random.choice(docs[:10])
    identifier = doc["identifier"]

    # Get file listing
    files_resp = requests.get(
        f"https://archive.org/metadata/{identifier}/files",
        timeout=10,
    )
    files_resp.raise_for_status()
    files = files_resp.json().get("result", [])
    mp4s  = [f for f in files if f.get("name", "").endswith(".mp4")]
    if not mp4s:
        raise ValueError(f"Archive: no mp4 in {identifier}")

    filename = mp4s[0]["name"]
    return f"https://archive.org/download/{identifier}/{filename}"


# ── Downloader ──────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path) -> Path:
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    total     = int(resp.headers.get("content-length", 0))
    received  = 0
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=16_384):
            f.write(chunk)
            received += len(chunk)
            if total:
                print(f"\r  {received/total*100:.0f}% ({received//1024}KB)", end="", flush=True)
    print()
    return dest


# ── Main entry ──────────────────────────────────────────────────────────────────

# Priority order – randomise top-2 for variety, keep fallbacks stable
SOURCES = [
    ("pixabay",  _pixabay),
    ("pexels",   _pexels),
    ("vimeo",    _vimeo),
    ("coverr",   _coverr),
    ("nasa",     _nasa),
    ("archive",  _archive),
]


def fetch_video() -> Path:
    niche_cfg, active = load_niche()
    keys              = load_keys()
    query             = random.choice(niche_cfg["video_keywords"])

    cache_dir = CACHE_DIR / active
    cache_dir.mkdir(parents=True, exist_ok=True)

    ordered = SOURCES[:]
    random.shuffle(ordered[:2])          # vary between pixabay / pexels

    for name, fetcher in ordered:
        try:
            print(f"[{name}] query='{query}'")
            video_url = fetcher(query, keys)
            dest      = cache_dir / f"{name}_{int(time.time())}.mp4"
            print(f"[{name}] downloading...")
            _download(video_url, dest)
            print(f"[{name}] saved → {dest}")
            return dest
        except Exception as exc:
            print(f"[{name}] failed: {exc}")

    raise RuntimeError("All video sources failed – check keys and network.")


if __name__ == "__main__":
    path = fetch_video()
    print(f"\nVIDEO_PATH={path}")
