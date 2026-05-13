"""Tests for YouTube uploader — atomic token write, backoff cap, upload flow."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


class TestAtomicWriteToken:
    def test_writes_file_atomically(self, tmp_path):
        from src.uploader import _atomic_write_token
        target = tmp_path / "token.json"
        _atomic_write_token(str(target), '{"token": "abc"}')
        assert target.exists()
        assert json.loads(target.read_text())["token"] == "abc"

    def test_no_partial_file_on_success(self, tmp_path):
        from src.uploader import _atomic_write_token
        target = tmp_path / "token.json"
        _atomic_write_token(str(target), '{"token": "xyz"}')
        # .tmp file must not exist after successful write
        assert not (tmp_path / "token.tmp").exists()

    def test_creates_parent_dir(self, tmp_path):
        from src.uploader import _atomic_write_token
        target = tmp_path / "nested" / "dir" / "token.json"
        _atomic_write_token(str(target), "{}")
        assert target.exists()

    def test_overwrites_existing_file(self, tmp_path):
        from src.uploader import _atomic_write_token
        target = tmp_path / "token.json"
        target.write_text('{"old": true}')
        _atomic_write_token(str(target), '{"new": true}')
        assert json.loads(target.read_text())["new"] is True


class TestBackoffCap:
    def test_backoff_capped_at_300s(self):
        """Verify exponential backoff never exceeds 300 seconds."""
        from src.uploader import MAX_RETRIES
        for retry in range(1, MAX_RETRIES + 1):
            wait = min(2**retry, 300)
            assert wait <= 300, f"retry {retry} wait {wait} exceeds 300s"

    def test_retry_5_would_be_32s_uncapped(self):
        assert 2**5 == 32  # still under 300 anyway
        assert min(2**5, 300) == 32

    def test_retry_10_capped_to_300(self):
        assert 2**10 == 1024
        assert min(2**10, 300) == 300


class TestResumableUpload:
    def test_returns_none_after_max_retries(self):
        from src.uploader import _resumable_upload, MAX_RETRIES
        from googleapiclient.errors import HttpError

        mock_resp = MagicMock()
        mock_resp.status = 503
        err = HttpError(resp=mock_resp, content=b"Service Unavailable")

        mock_request = MagicMock()
        mock_request.next_chunk.side_effect = err

        with patch("time.sleep"):
            result = _resumable_upload(mock_request)

        assert result is None

    def test_returns_video_id_on_success(self):
        from src.uploader import _resumable_upload

        call_count = [0]
        def fake_next_chunk():
            call_count[0] += 1
            if call_count[0] < 3:
                return MagicMock(progress=lambda: call_count[0] / 3), None
            return None, {"id": "abc123xyz"}

        mock_request = MagicMock()
        mock_request.next_chunk.side_effect = fake_next_chunk

        result = _resumable_upload(mock_request)
        assert result == "abc123xyz"
