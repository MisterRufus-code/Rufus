"""Scan a media library directory and yield asset paths by type."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import config


@dataclass
class RawAsset:
    path: Path
    asset_type: str          # "video" | "image"
    size_bytes: int = 0
    stem: str = ""

    def __post_init__(self):
        self.size_bytes = self.path.stat().st_size if self.path.exists() else 0
        self.stem = self.path.stem


def _video_codec(path: Path) -> str:
    """Return the video codec name via ffprobe, or '' on failure."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=codec_name",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip().lower()
    except Exception:
        return ""


def scan_library(library_path: Path | None = None) -> Iterator[RawAsset]:
    """Yield every supported media file found under *library_path*, skipping AV1."""
    root = Path(library_path or config.MEDIA_LIBRARY_PATH)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return

    _SKIP_DIRS = {"__pycache__", ".git", ".cache", ".DS_Store", "node_modules"}

    for p in root.rglob("*"):
        if any(part.startswith(".") or part in _SKIP_DIRS for part in p.parts[len(root.parts):]):
            continue
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in config.SUPPORTED_VIDEO_EXTS:
            # Skip AV1-encoded videos — OpenCV can't hardware-decode them
            codec = _video_codec(p)
            if "av1" in codec or "av01" in codec:
                continue
            yield RawAsset(path=p, asset_type="video")
        elif ext in config.SUPPORTED_IMAGE_EXTS:
            yield RawAsset(path=p, asset_type="image")


def scan_summary(library_path: Path | None = None) -> dict:
    videos, images = 0, 0
    total_bytes = 0
    for asset in scan_library(library_path):
        total_bytes += asset.size_bytes
        if asset.asset_type == "video":
            videos += 1
        else:
            images += 1
    return {"videos": videos, "images": images, "total_gb": round(total_bytes / 1e9, 2)}
