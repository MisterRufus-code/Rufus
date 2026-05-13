"""YouTube uploader — thin wrapper around src.uploader using the new BaseUploader interface."""

from __future__ import annotations

from src.uploaders.base import BaseUploader, UploadRequest, UploadResult


class YouTubeUploader(BaseUploader):
    platform = "youtube"

    def upload(self, req: UploadRequest) -> UploadResult:
        try:
            from src.uploader import upload_video
            video_id = upload_video(
                video_path=str(req.video_path),
                title=req.title,
                description=req.description,
                tags=req.tags,
                privacy=req.privacy,
                thumbnail_path=str(req.thumbnail_path) if req.thumbnail_path else None,
            )
            if video_id:
                return UploadResult(
                    success=True,
                    platform=self.platform,
                    video_id=video_id,
                    url=f"https://youtube.com/watch?v={video_id}",
                )
            return UploadResult(success=False, platform=self.platform, error="No video ID returned")
        except Exception as exc:
            return UploadResult(success=False, platform=self.platform, error=str(exc))

    def get_analytics(self, video_id: str) -> dict:
        try:
            from src.analytics import sync_analytics
            data = sync_analytics([video_id])
            return data.get(video_id, {})
        except Exception:
            return {}

    def is_available(self) -> bool:
        import os
        token_file = os.getenv("YOUTUBE_TOKEN_FILE", "tokens/token.json")
        from pathlib import Path
        return Path(token_file).exists()
