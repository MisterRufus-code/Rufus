"""TikTok uploader stub — ready for implementation when TikTok API access is available.

TikTok Content Posting API requires:
  - Business/Creator account
  - App registration at developers.tiktok.com
  - OAuth 2.0 token with video.upload scope

Until then, this stub lets the rest of the system reference TikTok as a target
without crashing.
"""

from __future__ import annotations

from src.uploaders.base import BaseUploader, UploadRequest, UploadResult


class TikTokUploader(BaseUploader):
    platform = "tiktok"

    def upload(self, req: UploadRequest) -> UploadResult:
        # Stub: log intent, skip actual upload
        from rich.console import Console
        Console().print(
            "[yellow]TikTok upload stub — set up TikTok API credentials to enable.[/yellow]\n"
            f"  Video: {req.video_path}\n"
            f"  Title: {req.title}"
        )
        return UploadResult(
            success=False,
            platform=self.platform,
            error="TikTok uploader not yet configured — see src/uploaders/tiktok.py",
        )

    def get_analytics(self, video_id: str) -> dict:
        return {}

    def is_available(self) -> bool:
        import os
        return bool(os.getenv("TIKTOK_ACCESS_TOKEN"))
