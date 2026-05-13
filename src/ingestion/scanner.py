"""Scan a media library directory and yield asset paths by type."""

from __future__ import annotations

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


def scan_library(library_path: Path | None = None) -> Iterator[RawAsset]:
    """Yield every supported media file found under *library_path*."""
    root = Path(library_path or config.MEDIA_LIBRARY_PATH)
    if not root.exists():
        root.mkdir(parents=True, exist_ok=True)
        return

    _SKIP_DIRS = {"__pycache__", ".git", ".cache", ".DS_Store", "node_modules"}

    for p in root.rglob("*"):
        # Skip hidden directories and system folders
        if any(part.startswith(".") or part in _SKIP_DIRS for part in p.parts[len(root.parts):]):
            continue
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in config.SUPPORTED_VIDEO_EXTS:
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
