"""Tests for pre-flight system checks."""

from __future__ import annotations

from unittest.mock import patch

import pytest


class TestCheckFfmpeg:
    def test_ok_when_both_present(self):
        from src.preflight import check_ffmpeg
        with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
            result = check_ffmpeg()
        assert result is None

    def test_fails_when_ffmpeg_missing(self):
        from src.preflight import check_ffmpeg
        with patch("shutil.which", return_value=None):
            result = check_ffmpeg()
        assert result is not None
        assert "ffmpeg" in result.lower()


class TestCheckOllama:
    def test_ok_when_ollama_responds_with_models(self):
        from src.preflight import check_ollama
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": [{"name": "mistral"}]}
        mock_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=mock_resp):
            result = check_ollama()
        assert result is None

    def test_fails_when_ollama_not_running(self):
        from src.preflight import check_ollama
        with patch("httpx.get", side_effect=Exception("connection refused")):
            result = check_ollama()
        assert result is not None
        assert "ollama" in result.lower()

    def test_warns_when_no_models_installed(self):
        from src.preflight import check_ollama
        from unittest.mock import MagicMock
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"models": []}
        mock_resp.raise_for_status.return_value = None
        with patch("httpx.get", return_value=mock_resp):
            result = check_ollama()
        assert result is not None
        assert "model" in result.lower()


class TestCheckDiskSpace:
    def test_ok_when_plenty_of_space(self):
        from src.preflight import check_disk_space
        with patch("shutil.disk_usage", return_value=(100e9, 50e9, 50e9)):
            result = check_disk_space()
        assert result is None

    def test_fails_when_low_disk(self):
        from src.preflight import check_disk_space
        # Only 1 GB free, default minimum is 5 GB
        with patch("shutil.disk_usage", return_value=(100e9, 99e9, 1e9)):
            result = check_disk_space()
        assert result is not None
        assert "disk" in result.lower() or "space" in result.lower()


class TestRunPreflight:
    def test_raises_system_exit_on_failure(self):
        from src.preflight import run_preflight
        with patch("src.preflight.check_ffmpeg", return_value="ffmpeg missing"), \
             patch("src.preflight.check_ollama", return_value=None), \
             patch("src.preflight.check_disk_space", return_value=None):
            with pytest.raises(SystemExit):
                run_preflight(abort_on_failure=True)

    def test_returns_error_list_without_exit(self):
        from src.preflight import run_preflight
        with patch("src.preflight.check_ffmpeg", return_value="ffmpeg missing"), \
             patch("src.preflight.check_ollama", return_value=None), \
             patch("src.preflight.check_disk_space", return_value=None):
            errors = run_preflight(abort_on_failure=False)
        assert len(errors) == 1

    def test_passes_all_checks_returns_empty(self):
        from src.preflight import run_preflight
        with patch("src.preflight.check_ffmpeg", return_value=None), \
             patch("src.preflight.check_ollama", return_value=None), \
             patch("src.preflight.check_disk_space", return_value=None):
            errors = run_preflight(abort_on_failure=False)
        assert errors == []
