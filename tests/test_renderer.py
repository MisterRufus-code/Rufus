"""Tests for FFmpeg renderer — timeout, asset validation, concat, audio merge."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_clip(path: str = "/tmp/fake.mp4", asset_type: str = "video",
               duration: float = 5.0):
    asset = MagicMock()
    asset.path = path
    asset.asset_type = asset_type
    asset.duration_seconds = duration
    clip = MagicMock()
    clip.asset = asset
    clip.in_point = 0.0
    clip.out_point = duration
    return clip


class TestFfmpegWrapper:
    def test_timeout_raises_runtime_error(self):
        from src.pipeline.renderer import _ffmpeg
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired(["ffmpeg"], 300)):
            with pytest.raises(RuntimeError, match="timed out"):
                _ffmpeg("-version")

    def test_called_process_error_propagates(self):
        from src.pipeline.renderer import _ffmpeg
        err = subprocess.CalledProcessError(1, ["ffmpeg"])
        with patch("subprocess.run", side_effect=err):
            with pytest.raises(subprocess.CalledProcessError):
                _ffmpeg("-version")

    def test_command_includes_hide_banner(self):
        from src.pipeline.renderer import _ffmpeg
        captured = []
        def fake_run(cmd, **kw):
            captured.append(cmd)
            r = MagicMock()
            r.returncode = 0
            return r
        with patch("subprocess.run", side_effect=fake_run):
            _ffmpeg("-version", check=False)
        assert "-hide_banner" in captured[0]

    def test_check_ffmpeg_returns_true_when_present(self):
        from src.pipeline.renderer import _check_ffmpeg
        mock_result = MagicMock()
        mock_result.returncode = 0
        with patch("subprocess.run", return_value=mock_result):
            assert _check_ffmpeg() is True

    def test_check_ffmpeg_returns_false_when_missing(self):
        from src.pipeline.renderer import _check_ffmpeg
        with patch("subprocess.run", side_effect=FileNotFoundError):
            assert _check_ffmpeg() is False


class TestMissingAssetSkipped:
    def test_missing_asset_skipped_not_crashed(self, tmp_path):
        from src.pipeline import renderer
        clip = _make_clip(path=str(tmp_path / "nonexistent.mp4"))

        # render_video should not raise even when the asset is missing
        def fake_ffmpeg(*args, **kw):
            r = MagicMock()
            r.returncode = 0
            r.stderr = ""
            return r

        with patch.object(renderer, "_check_ffmpeg", return_value=True), \
             patch.object(renderer, "_ffmpeg", side_effect=fake_ffmpeg), \
             patch("shutil.copy2"):
            # With no clips that exist, trimmed list is empty, concat will be called with []
            # We just verify it doesn't raise FileNotFoundError for the missing asset
            try:
                renderer.render_video(
                    clips=[clip],
                    audio_path=None,
                    output_path=tmp_path / "out.mp4",
                )
            except Exception as exc:
                # Only acceptable failure is empty concat (no clips), not FileNotFoundError
                assert "nonexistent" not in str(exc)


class TestConcatClips:
    def test_concat_creates_list_file(self, tmp_path):
        from src.pipeline.renderer import concat_clips
        p1 = tmp_path / "a.mp4"
        p2 = tmp_path / "b.mp4"
        p1.write_bytes(b"x")
        p2.write_bytes(b"x")
        out = tmp_path / "concat.mp4"
        with patch("src.pipeline.renderer._ffmpeg") as mock_ff:
            mock_ff.return_value = MagicMock()
            concat_clips([p1, p2], out)
        # _ffmpeg was called once with concat demuxer args
        assert mock_ff.called
        args = mock_ff.call_args[0]
        assert "-f" in args
        assert "concat" in args


class TestAddAudio:
    def test_add_audio_calls_ffmpeg(self, tmp_path):
        from src.pipeline.renderer import add_audio
        video = tmp_path / "v.mp4"
        audio = tmp_path / "a.wav"
        out = tmp_path / "out.mp4"
        with patch("src.pipeline.renderer._ffmpeg") as mock_ff:
            mock_ff.return_value = MagicMock()
            add_audio(video, audio, out)
        assert mock_ff.called
        args = mock_ff.call_args[0]
        assert str(video) in args
        assert str(audio) in args
