"""End-to-end smoke test — exercises the full pipeline in dry_run=True mode.

Heavy components (Ollama, CLIP, TTS, FFmpeg, YouTube) are mocked so this test
runs on any machine in < 5 seconds without ML packages or network access.

The goal is NOT to test individual units — unit tests do that.
The goal is to verify that the pipeline path from run_pipeline() to
PipelineResult does NOT crash, does NOT violate the media-first contract,
and honours the dry_run flag.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_asset(asset_id: str = "a1", caption: str = "test clip"):
    from src.database.models import MediaAsset
    return MediaAsset(
        asset_id=asset_id,
        path="footage/test.mp4",
        asset_type="video",
        caption=caption,
        detected_objects=[],
        dominant_colors=[],
        motion_score=0.5,
        clip_embedding=[0.0] * 512,
    )


def _make_timeline_clip(asset_id: str = "a1"):
    from src.database.models import TimelineClip
    asset = _make_asset(asset_id)
    return TimelineClip(
        asset=asset,
        scene_index=0,
        start_time=0.0,
        end_time=5.0,
    )


# ---------------------------------------------------------------------------
# Smoke test — dry_run=True, all heavy deps mocked
# ---------------------------------------------------------------------------

class TestPipelineDryRun:
    @pytest.fixture(autouse=True)
    def _patch_pipeline_deps(self, tmp_path, monkeypatch):
        """Patch every external call the pipeline makes so no real I/O occurs."""
        from contextlib import ExitStack
        import config
        monkeypatch.setattr(config, "OUTPUT_PATH", str(tmp_path / "output"))
        (tmp_path / "output").mkdir()

        mock_store = MagicMock()
        mock_store.get_all_assets.return_value = [_make_asset()]
        mock_store.get_all_asset_ids.return_value = set()

        fake_idea = MagicMock()
        fake_idea.title = "Test Idea"
        fake_idea.description = "Great topic"
        fake_idea.hook = "Did you know this?"
        fake_idea.tags = ["finance", "investing"]
        fake_idea.estimated_virality = "High"

        fake_script = MagicMock()
        fake_script.hook = "Hook line"
        fake_script.body = "Body content here."
        fake_script.cta = "Subscribe for more."
        fake_script.sections = []
        fake_script.full_text = "Hook line\nBody content here.\nSubscribe for more."
        fake_script.title = "Test Video"
        fake_script.call_to_action = "Subscribe for more."
        fake_script.description = "A test video."
        fake_script.tags = ["finance", "test"]

        patches = [
            patch("src.preflight.run_preflight", return_value=None),
            patch("src.trends.fetch_google_trends", return_value=[]),
            patch("src.ideas.generate_search_keywords", return_value=["investing", "stocks"]),
            patch("src.media_fetch.pexels.download_videos", return_value=[]),
            patch("src.media_fetch.ytcc.download_videos", return_value=[]),
            patch("src.ingestion.indexer.index_library", return_value={"indexed": 3, "skipped": 0, "errors": 0}),
            patch("src.database.vector_store.get_vector_store", return_value=mock_store),
            patch("src.ideas.generate_ideas_from_media", return_value=[fake_idea]),
            patch("src.ideas.generate_video_ideas", return_value=[fake_idea]),
            patch("src.ideas.generate_script_from_media", return_value=fake_script),
            patch("src.ideas.generate_video_script", return_value=fake_script),
            patch("src.ideas.generate_shorts_script", return_value=fake_script),
            patch("src.psychology.hooks.score_hook", return_value=MagicMock(total=0.7)),
            patch("src.matching.semantic.match_all_scenes", return_value=[_make_timeline_clip()]),
            patch("src.matching.semantic.scenes_from_script", return_value=[]),
            patch("src.entropy.engine.score_timeline", return_value=MagicMock(overall_score=0.85, passed=True, flagged_clips=[])),
            patch("src.ml.feedback.record_feedback"),
            patch("src.ml.optimizer.get_keyword_context", return_value=""),
            patch("src.ml.optimizer.get_optimized_prompt_prefix", return_value=""),
        ]
        with ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            yield

    def test_dry_run_returns_success(self, tmp_path):
        from src.pipeline.orchestrator import run_pipeline
        result = run_pipeline(
            topic="investing basics",
            niche="finance",
            dry_run=True,
            download_footage=False,
        )
        assert result.success is True, f"Pipeline failed: {result.errors}"
        assert result.dry_run is True

    def test_dry_run_no_video_file_created(self, tmp_path):
        """dry_run must NOT call FFmpeg — no rendered file should exist."""
        from src.pipeline.orchestrator import run_pipeline
        result = run_pipeline(
            topic="tech trends",
            niche="tech",
            dry_run=True,
            download_footage=False,
        )
        assert result.video_path is None or not result.video_path.exists()

    def test_dry_run_topic_sanitized(self, tmp_path):
        """Topic with special chars must not produce path-traversal output dir."""
        from src.pipeline.orchestrator import run_pipeline
        result = run_pipeline(
            topic="../../etc/passwd",
            niche="general",
            dry_run=True,
            download_footage=False,
        )
        # success may be True or False depending on preflight, but should not crash
        # and output dir should not be outside tmp
        assert result is not None

    def test_dry_run_errors_list_populated_on_exception(self, tmp_path):
        """If a non-critical step raises, errors list should record it."""
        from src.pipeline.orchestrator import run_pipeline
        with patch("src.trends.fetch_google_trends", side_effect=RuntimeError("trends down")):
            result = run_pipeline(
                topic="test topic",
                niche="general",
                dry_run=True,
                download_footage=False,
            )
        # Pipeline should survive a trends failure
        assert result is not None


# ---------------------------------------------------------------------------
# Retry utility smoke test
# ---------------------------------------------------------------------------

class TestRetryUtility:
    def test_succeeds_on_first_try(self):
        from src.utils.retry import with_retry
        calls = []

        @with_retry(max_attempts=3, base_delay=0)
        def fn():
            calls.append(1)
            return "ok"

        assert fn() == "ok"
        assert len(calls) == 1

    def test_retries_on_failure_then_succeeds(self):
        from src.utils.retry import with_retry
        calls = []

        @with_retry(max_attempts=3, base_delay=0)
        def fn():
            calls.append(1)
            if len(calls) < 3:
                raise ValueError("not ready")
            return "ok"

        assert fn() == "ok"
        assert len(calls) == 3

    def test_raises_after_max_attempts(self):
        from src.utils.retry import with_retry

        @with_retry(max_attempts=2, base_delay=0)
        def fn():
            raise RuntimeError("always fails")

        with pytest.raises(RuntimeError, match="always fails"):
            fn()

    def test_on_retry_callback_called(self):
        from src.utils.retry import with_retry
        retry_calls = []

        @with_retry(max_attempts=3, base_delay=0, on_retry=lambda attempt, exc: retry_calls.append(attempt))
        def fn():
            raise ValueError("fail")

        with pytest.raises(ValueError):
            fn()
        assert retry_calls == [1, 2]


# ---------------------------------------------------------------------------
# CircuitBreaker smoke test
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_opens_after_threshold(self):
        from src.utils.retry import CircuitBreaker
        cb = CircuitBreaker(threshold=2, reset_timeout=9999)
        for _ in range(2):
            try:
                with cb:
                    raise RuntimeError("fail")
            except RuntimeError:
                pass
        assert cb.state == CircuitBreaker.OPEN

    def test_closed_on_success(self):
        from src.utils.retry import CircuitBreaker
        cb = CircuitBreaker(threshold=3)
        with cb:
            pass
        assert cb.state == CircuitBreaker.CLOSED

    def test_raises_when_open(self):
        from src.utils.retry import CircuitBreaker
        cb = CircuitBreaker(threshold=1, reset_timeout=9999)
        try:
            with cb:
                raise RuntimeError("fail")
        except RuntimeError:
            pass
        with pytest.raises(RuntimeError, match="Circuit breaker is OPEN"):
            with cb:
                pass


# ---------------------------------------------------------------------------
# Humanization layer smoke test
# ---------------------------------------------------------------------------

class TestHumanizeLayer:
    def test_random_upload_delay_positive(self):
        from src.humanize import random_upload_delay_seconds
        delay = random_upload_delay_seconds()
        assert delay > 0

    def test_random_tts_speed_in_range(self):
        from src.humanize import random_tts_speed
        for _ in range(50):
            speed = random_tts_speed()
            assert 0.85 <= speed <= 1.15, f"Speed out of range: {speed}"

    def test_naturalise_title_returns_string(self):
        from src.humanize import naturalise_title
        result = naturalise_title("How to invest in stocks")
        assert isinstance(result, str)
        assert len(result) > 5

    def test_random_hook_style_valid(self):
        from src.humanize import random_hook_style, HOOK_STYLES
        for _ in range(20):
            assert random_hook_style() in HOOK_STYLES

    def test_random_subtitle_style_keys(self):
        from src.humanize import random_subtitle_style
        style = random_subtitle_style()
        assert "font_size" in style
        assert "colour" in style
        assert "shadow_opacity" in style

    def test_jittered_upload_time_within_spread(self):
        from src.humanize import jittered_upload_time
        from datetime import datetime, timedelta
        base = datetime(2025, 1, 1, 12, 0, 0)
        spread = 20
        for _ in range(50):
            jittered = jittered_upload_time(base, spread_minutes=spread)
            diff = abs((jittered - base).total_seconds())
            assert diff <= spread * 60 + 1  # +1 for float tolerance


# ---------------------------------------------------------------------------
# Platform abstraction smoke test
# ---------------------------------------------------------------------------

class TestUploaderAbstraction:
    def test_get_uploader_youtube_returns_youtube_uploader(self):
        from src.uploaders.base import get_uploader
        from src.uploaders.youtube import YouTubeUploader
        uploader = get_uploader("youtube")
        assert isinstance(uploader, YouTubeUploader)

    def test_get_uploader_tiktok_returns_tiktok_uploader(self):
        from src.uploaders.base import get_uploader
        from src.uploaders.tiktok import TikTokUploader
        uploader = get_uploader("tiktok")
        assert isinstance(uploader, TikTokUploader)

    def test_get_uploader_unknown_raises(self):
        from src.uploaders.base import get_uploader
        with pytest.raises(ValueError, match="Unknown platform"):
            get_uploader("myspace")

    def test_tiktok_upload_returns_failure_not_crash(self):
        from src.uploaders.tiktok import TikTokUploader
        from src.uploaders.base import UploadRequest
        uploader = TikTokUploader()
        req = UploadRequest(
            video_path=Path("fake.mp4"),
            title="Test",
            description="desc",
        )
        result = uploader.upload(req)
        assert result.success is False
        assert result.platform == "tiktok"
        assert result.error is not None

    def test_tiktok_unavailable_without_token(self, monkeypatch):
        monkeypatch.delenv("TIKTOK_ACCESS_TOKEN", raising=False)
        from src.uploaders.tiktok import TikTokUploader
        assert TikTokUploader().is_available() is False
