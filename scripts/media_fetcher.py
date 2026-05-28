#!/usr/bin/env python3
"""
media_fetcher.py
Downloads background videos for the active niche.
- Single fetch (fetch_video) or multi-candidate parallel fetch (fetch_candidates).
- NASA removed from default sources – returns space content for every query.
"""

import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

ROOT        = Path(__file__).parent.parent
CONFIG_DIR  = ROOT / "config"
NICHES_FILE = CONFIG_DIR / "niches.json"
KEYS_FILE   = CONFIG_DIR / "keys.json"
CACHE_DIR   = ROOT / "media_library" / "cache"

MIN_FILE_SIZE = 100_000   # 100 KB minimum – reject corrupt/tiny downloads


# ── Config ──────────────────────────────────────────────────────────────────────

def _load_niche():
    data   = json.loads(NICHES_FILE.read_text())
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]
    return data["niches"][active], active


def _load_keys():
    return json.loads(KEYS_FILE.read_text())


# ── Source fetchers (return direct MP4 URL) ─────────────────────────────────────

def _pixabay(query: str, keys: dict) -> str:
    key = keys.get("pixabay", "")
    if not key or key.startswith("YOUR_"):
        raise ValueError("Pixabay key not set")

    r = requests.get(
        "https://pixabay.com/api/videos/",
        params={"key": key, "q": query, "video_type": "film",
                "per_page": 20, "order": "popular", "min_width": 1080},
        timeout=10,
    )
    r.raise_for_status()
    hits = r.json().get("hits", [])
    if not hits:
        raise ValueError(f"Pixabay: no results for '{query}'")

    video = random.choice(hits[:10])
    for q in ("large", "medium", "small", "tiny"):
        url = video["videos"].get(q, {}).get("url")
        if url:
            return url
    raise ValueError("Pixabay: no usable URL")


def _pexels(query: str, keys: dict) -> str:
    key = keys.get("pexels", "")
    if not key or key.startswith("YOUR_"):
        raise ValueError("Pexels key not set")

    r = requests.get(
        "https://api.pexels.com/videos/search",
        headers={"Authorization": key},
        params={"query": query, "per_page": 20, "min_width": 1080,
                "min_duration": 5, "max_duration": 60},
        timeout=10,
    )
    r.raise_for_status()
    videos = r.json().get("videos", [])
    if not videos:
        raise ValueError(f"Pexels: no results for '{query}'")

    video = random.choice(videos[:10])
    files = sorted(video.get("video_files", []),
                   key=lambda x: x.get("width", 0), reverse=True)
    for f in files:
        if f.get("link") and f.get("width", 0) >= 720:
            return f["link"]
    raise ValueError("Pexels: no HD file found")


def _vimeo(query: str, keys: dict) -> str:
    token = keys.get("vimeo", "")
    if not token or token.startswith("YOUR_"):
        raise ValueError("Vimeo token not set")

    r = requests.get(
        "https://api.vimeo.com/videos",
        headers={"Authorization": f"Bearer {token}"},
        params={"query": query, "filter": "CC", "per_page": 20,
                "sort": "relevant", "fields": "uri,download,files,name"},
        timeout=10,
    )
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise ValueError(f"Vimeo: no results for '{query}'")

    random.shuffle(data)
    for video in data[:10]:
        for dl in video.get("download", []) + video.get("files", []):
            if not dl.get("link"):
                continue
            quality = dl.get("quality", "")
            width   = dl.get("width", 0)
            if quality in ("hd", "1080", "720") or width >= 720:
                return dl["link"]
    raise ValueError("Vimeo: no downloadable HD video")


def _coverr(_query: str, _keys: dict) -> str:
    page = random.randint(1, 40)
    r = requests.get(
        "https://coverr.co/api/videos/featured",
        params={"page": page, "per_page": 10},
        timeout=10,
    )
    r.raise_for_status()
    raw   = r.json()
    items = raw if isinstance(raw, list) else raw.get("hits", raw.get("videos", []))
    if not items:
        raise ValueError("Coverr: empty response")

    item = random.choice(items)
    for field in ("mp4_url", "url", "video_url", "file_url", "src"):
        if item.get(field):
            return item[field]
    raise ValueError("Coverr: no URL field found")


def _archive(query: str, _keys: dict) -> str:
    r = requests.get(
        "https://archive.org/advancedsearch.php",
        params={"q": f'"{query}" AND mediatype:movies AND format:mp4',
                "fl": "identifier", "rows": 20, "output": "json"},
        timeout=15,
    )
    r.raise_for_status()
    docs = r.json().get("response", {}).get("docs", [])
    if not docs:
        raise ValueError(f"Archive: no results for '{query}'")

    doc        = random.choice(docs[:10])
    identifier = doc["identifier"]
    files_r    = requests.get(
        f"https://archive.org/metadata/{identifier}/files", timeout=10
    )
    files_r.raise_for_status()
    mp4s = [f for f in files_r.json().get("result", [])
            if f.get("name", "").endswith(".mp4")]
    if not mp4s:
        raise ValueError(f"Archive: no mp4 in {identifier}")
    return f"https://archive.org/download/{identifier}/{mp4s[0]['name']}"


# NASA removed from default chain – it returns space content for every query,
# which is off-topic for finance/motivation/mindset/business niches.
SOURCES = [
    # ("pixabay", _pixabay),  # re-enable once Pixabay API key is set
    ("pexels",  _pexels),
    ("vimeo",   _vimeo),
    ("coverr",  _coverr),
    ("archive", _archive),
]


# ── Downloader ──────────────────────────────────────────────────────────────────

def _download(url: str, dest: Path, retries: int = 3, quiet: bool = False) -> Path:
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, stream=True, timeout=120)
            r.raise_for_status()
            total    = int(r.headers.get("content-length", 0))
            received = 0
            with open(dest, "wb") as f:
                for chunk in r.iter_content(chunk_size=16_384):
                    f.write(chunk)
                    received += len(chunk)
                    if total and not quiet:
                        print(f"\r  {received/total*100:.0f}% ({received//1024} KB)",
                              end="", flush=True)
            if not quiet:
                print()
            if dest.stat().st_size < MIN_FILE_SIZE:
                dest.unlink(missing_ok=True)
                raise ValueError(f"Downloaded file too small: {dest.stat().st_size} bytes")
            return dest
        except Exception as e:
            if attempt < retries:
                wait = 2 ** attempt
                print(f"  Download attempt {attempt} failed ({e}), retrying in {wait}s…")
                time.sleep(wait)
            else:
                raise


# ── Public API ──────────────────────────────────────────────────────────────────

def fetch_video() -> Path:
    """Fetch one video using the first working source."""
    niche_cfg, active = _load_niche()
    keys              = _load_keys()
    query             = random.choice(niche_cfg["video_keywords"])
    cache_dir         = CACHE_DIR / active
    cache_dir.mkdir(parents=True, exist_ok=True)

    ordered = SOURCES[:]
    random.shuffle(ordered[:2])

    for name, fetcher in ordered:
        try:
            print(f"[{name}] query='{query}'")
            url  = fetcher(query, keys)
            dest = cache_dir / f"{name}_{int(time.time())}.mp4"
            print(f"[{name}] downloading…")
            _download(url, dest)
            print(f"[{name}] saved → {dest}")
            return dest
        except Exception as e:
            print(f"[{name}] failed: {e}")

    raise RuntimeError("All video sources failed – check keys and network.")


def _fetch_one_candidate(idx: int, n: int, query: str, sources: list,
                         keys: dict, cand_dir: Path) -> Path | None:
    """Try each source in order for one candidate slot. Returns Path or None."""
    for name, fetcher in sources:
        try:
            url  = fetcher(query, keys)
            dest = cand_dir / f"cand_{idx+1}_{name}_{int(time.time()*1000) % 100000}.mp4"
            print(f"[candidate {idx+1}/{n}][{name}] query='{query}'")
            _download(url, dest, quiet=True)
            print(f"[candidate {idx+1}/{n}][{name}] ✓")
            return dest
        except Exception as e:
            print(f"[candidate {idx+1}/{n}][{name}] failed: {e}")
    print(f"[candidate {idx+1}/{n}] all sources failed for '{query}'")
    return None


def fetch_candidates(n: int = 5) -> list[Path]:
    """
    Fetch n different candidate videos in PARALLEL using different queries.
    Returns list of local paths for AI selection.
    """
    niche_cfg, active = _load_niche()
    keys              = _load_keys()
    keywords          = niche_cfg["video_keywords"].copy()
    random.shuffle(keywords)

    cand_dir = CACHE_DIR / active / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)

    # Clear old candidates
    for old in cand_dir.glob("*.mp4"):
        try:
            old.unlink()
        except Exception:
            pass

    ordered = SOURCES[:]
    random.shuffle(ordered[:2])

    candidates: list[Path] = []
    with ThreadPoolExecutor(max_workers=n) as ex:
        futures = [
            ex.submit(_fetch_one_candidate, i, n, keywords[i % len(keywords)],
                      ordered, keys, cand_dir)
            for i in range(n)
        ]
        for fut in as_completed(futures):
            result = fut.result()
            if result:
                candidates.append(result)

    if not candidates:
        # Fallback: reuse any mp4 already in the niche cache rather than hard-crashing.
        cached = sorted((CACHE_DIR / active).glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True)
        cached = [p for p in cached if p.stat().st_size >= MIN_FILE_SIZE]
        if cached:
            print(f"[media] all sources failed — reusing {min(n, len(cached))} cached video(s)")
            return cached[:n]
        raise RuntimeError("Could not fetch any candidate videos and no cache available.")

    return candidates


if __name__ == "__main__":
    path = fetch_video()
    print(f"\nVIDEO_PATH={path}")
