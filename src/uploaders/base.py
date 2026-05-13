"""Abstract base class for all platform uploaders.

Adding a new platform:
  1. Create src/uploaders/<platform>.py
  2. Subclass BaseUploader
  3. Implement upload() and get_analytics()
  4. Register in get_uploader()
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class UploadResult:
    success: bool
    platform: str
    video_id: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class UploadRequest:
    video_path: Path
    title: str
    description: str
    tags: list[str] = field(default_factory=list)
    thumbnail_path: Optional[Path] = None
    privacy: str = "private"          # private | unlisted | public
    niche: str = "general"
    shorts: bool = False


class BaseUploader(ABC):
    """All platform uploaders implement this interface."""

    platform: str = "unknown"

    @abstractmethod
    def upload(self, req: UploadRequest) -> UploadResult:
        """Upload *req.video_path* and return result."""

    @abstractmethod
    def get_analytics(self, video_id: str) -> dict:
        """Fetch performance data for *video_id*. Return empty dict if unavailable."""

    def is_available(self) -> bool:
        """Return True if credentials/config are present and valid."""
        return True


def get_uploader(platform: str = "youtube") -> BaseUploader:
    """Factory — returns the correct uploader for *platform*."""
    if platform == "youtube":
        from src.uploaders.youtube import YouTubeUploader
        return YouTubeUploader()
    if platform == "tiktok":
        from src.uploaders.tiktok import TikTokUploader
        return TikTokUploader()
    raise ValueError(f"Unknown platform: {platform!r}. Available: youtube, tiktok")
