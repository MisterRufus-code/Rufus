"""TikTok uploader — TikTok Content Posting API v2.

Requirements
------------
* A TikTok developer app with the ``video.publish`` (or ``video.upload``) scope.
* An OAuth 2.0 access token stored in the ``TIKTOK_ACCESS_TOKEN`` environment
  variable (or refreshed externally and written there before each run).
* ``httpx`` (already a project dependency).

Upload flow
-----------
1. ``_init_upload``  — POST to the init endpoint; receive upload_url + publish_id.
2. ``_upload_video_file`` — PUT the raw video bytes to the signed upload_url.
3. ``_check_status`` — Poll the status endpoint until the video is published or
   an error is returned.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

import httpx

from src.uploaders.base import BaseUploader, UploadRequest, UploadResult

# ---------------------------------------------------------------------------
# API constants
# ---------------------------------------------------------------------------
_INIT_URL   = "https://open.tiktokapis.com/v2/post/publish/video/init/"
_STATUS_URL = "https://open.tiktokapis.com/v2/post/publish/status/fetch/"
_QUERY_URL  = "https://open.tiktokapis.com/v2/video/query/"

_PRIVACY_MAP: dict[str, str] = {
    "private":  "SELF_ONLY",
    "unlisted": "FOLLOWER_OF_CREATOR",
    "public":   "PUBLIC_TO_EVERYONE",
}

_POLL_INTERVAL = 5   # seconds between status checks
_POLL_TIMEOUT  = 300 # seconds before giving up on a pending publish


class TikTokUploader(BaseUploader):
    """Upload videos to TikTok using the Content Posting API v2."""

    platform = "tiktok"

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True only when TIKTOK_ACCESS_TOKEN is set."""
        return bool(os.getenv("TIKTOK_ACCESS_TOKEN"))

    def upload(self, req: UploadRequest) -> UploadResult:
        """Orchestrate the three-step TikTok upload flow."""
        try:
            upload_url, publish_id = self._init_upload(
                title=req.title,
                privacy=req.privacy,
            )
            self._upload_video_file(upload_url, req.video_path)
            video_id = self._check_status(publish_id)
            return UploadResult(
                success=True,
                platform=self.platform,
                video_id=video_id,
                url=f"https://www.tiktok.com/@me/video/{video_id}" if video_id else None,
            )
        except Exception as exc:
            return UploadResult(
                success=False,
                platform=self.platform,
                error=str(exc),
            )

    def get_analytics(self, video_id: str) -> dict:
        """Fetch basic metrics for *video_id* from the TikTok video query endpoint."""
        token = self._require_token()
        try:
            resp = httpx.post(
                _QUERY_URL,
                headers=self._auth_headers(token),
                json={
                    "filters": {"video_ids": [video_id]},
                    "fields": ["id", "title", "view_count", "like_count",
                               "comment_count", "share_count"],
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            videos = data.get("data", {}).get("videos", [])
            return videos[0] if videos else {}
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Upload step helpers
    # ------------------------------------------------------------------

    def _init_upload(self, title: str, privacy: str) -> tuple[str, str]:
        """POST to the init endpoint.

        Returns
        -------
        (upload_url, publish_id)
        """
        token = self._require_token()
        privacy_level = _PRIVACY_MAP.get(privacy, "SELF_ONLY")

        resp = httpx.post(
            _INIT_URL,
            headers=self._auth_headers(token),
            json={
                "post_info": {
                    "title": title[:150],        # API max is 150 chars
                    "privacy_level": privacy_level,
                    "disable_duet": False,
                    "disable_comment": False,
                    "disable_stitch": False,
                },
                "source_info": {
                    "source": "FILE_UPLOAD",
                },
            },
            timeout=30,
        )
        self._raise_for_tiktok_error(resp, "init upload")

        body = resp.json()
        upload_url = body["data"]["upload_url"]
        publish_id = body["data"]["publish_id"]
        return upload_url, publish_id

    def _upload_video_file(self, upload_url: str, video_path: Path) -> None:
        """PUT raw video bytes to the signed *upload_url*."""
        video_bytes = Path(video_path).read_bytes()
        resp = httpx.put(
            upload_url,
            content=video_bytes,
            headers={"Content-Type": "video/mp4"},
            timeout=600,  # large files can be slow
        )
        if resp.status_code not in (200, 204):
            raise RuntimeError(
                f"Video file upload failed (HTTP {resp.status_code}): {resp.text[:200]}"
            )

    def _check_status(self, publish_id: str) -> Optional[str]:
        """Poll the status endpoint until the video is published.

        Returns the video_id on success, or raises on failure/timeout.
        """
        token = self._require_token()
        deadline = time.monotonic() + _POLL_TIMEOUT

        while time.monotonic() < deadline:
            resp = httpx.post(
                _STATUS_URL,
                headers=self._auth_headers(token),
                json={"publish_id": publish_id},
                timeout=30,
            )
            self._raise_for_tiktok_error(resp, "check publish status")

            body  = resp.json()
            data  = body.get("data", {})
            state = data.get("status", "")

            if state == "PUBLISH_COMPLETE":
                return data.get("publicaly_available_post_id", [None])[0]

            if state in ("FAILED", "PUBLISH_FAILED"):
                fail_reason = data.get("fail_reason", "unknown")
                raise RuntimeError(
                    f"TikTok publish failed (publish_id={publish_id}): {fail_reason}"
                )

            # States like PROCESSING_UPLOAD / PROCESSING_DOWNLOAD → keep polling
            time.sleep(_POLL_INTERVAL)

        raise TimeoutError(
            f"TikTok publish timed out after {_POLL_TIMEOUT}s "
            f"(publish_id={publish_id})"
        )

    # ------------------------------------------------------------------
    # Internal utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _require_token() -> str:
        token = os.getenv("TIKTOK_ACCESS_TOKEN", "")
        if not token:
            raise EnvironmentError(
                "TIKTOK_ACCESS_TOKEN is not set. "
                "Obtain an OAuth 2.0 access token from developers.tiktok.com "
                "and export it before running the uploader."
            )
        return token

    @staticmethod
    def _auth_headers(token: str) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=UTF-8",
        }

    @staticmethod
    def _raise_for_tiktok_error(resp: httpx.Response, step: str) -> None:
        """Raise a descriptive RuntimeError on non-2xx or TikTok error code."""
        try:
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"TikTok API error at {step!r} (HTTP {resp.status_code}): "
                f"{resp.text[:300]}"
            ) from exc

        body = resp.json()
        error_block = body.get("error", {})
        if error_block.get("code", "ok") not in ("ok", ""):
            raise RuntimeError(
                f"TikTok API returned error at {step!r}: "
                f"{error_block.get('code')} — {error_block.get('message', '')}"
            )
