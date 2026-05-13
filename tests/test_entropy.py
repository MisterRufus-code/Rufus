"""Tests for visual entropy engine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


def _make_clip(asset_id: str, embedding=None, scene_index: int = 0,
               motion_score: float = 0.5, dominant_color=None, caption: str = ""):
    import numpy as np
    asset = MagicMock()
    asset.asset_id = asset_id
    asset.embedding = embedding if embedding is not None else np.random.rand(512).tolist()
    asset.motion_score = motion_score
    asset.dominant_color = dominant_color or [128, 128, 128]
    asset.caption = caption or asset_id
    clip = MagicMock()
    clip.asset = asset
    clip.scene_index = scene_index
    return clip


class TestScoreTimeline:
    def test_empty_timeline_returns_report(self):
        from src.entropy.engine import score_timeline
        report = score_timeline([])
        # Empty timeline → no clips to compare → score is valid
        assert 0.0 <= report.overall_score <= 1.0

    def test_single_clip_returns_score(self):
        from src.entropy.engine import score_timeline
        clip = _make_clip("c1")
        report = score_timeline([clip])
        assert 0.0 <= report.overall_score <= 1.0

    def test_identical_clips_low_entropy(self):
        import numpy as np
        from src.entropy.engine import score_timeline
        emb = np.ones(512).tolist()
        clips = [_make_clip(f"c{i}", embedding=emb, scene_index=i) for i in range(5)]
        report = score_timeline(clips)
        assert report.overall_score < 0.7

    def test_diverse_clips_higher_entropy(self):
        import numpy as np
        from src.entropy.engine import score_timeline
        rng = np.random.default_rng(42)
        clips = [_make_clip(f"c{i}", embedding=rng.random(512).tolist(), scene_index=i)
                 for i in range(6)]
        identical_emb = np.ones(512).tolist()
        identical_clips = [_make_clip(f"d{i}", embedding=identical_emb, scene_index=i)
                           for i in range(6)]
        diverse_score = score_timeline(clips).overall_score
        boring_score = score_timeline(identical_clips).overall_score
        assert diverse_score >= boring_score

    def test_report_has_passed_field(self):
        from src.entropy.engine import score_timeline
        clip = _make_clip("c1")
        report = score_timeline([clip])
        assert hasattr(report, "passed")
        assert isinstance(report.passed, bool)

    def test_report_has_flagged_clips(self):
        from src.entropy.engine import score_timeline
        clips = [_make_clip(f"c{i}", scene_index=i) for i in range(3)]
        report = score_timeline(clips)
        assert hasattr(report, "flagged_clips")
        assert isinstance(report.flagged_clips, list)


class TestEntropyThreshold:
    def test_env_var_overrides_threshold(self, monkeypatch):
        monkeypatch.setenv("ENTROPY_SCENE_REPEAT", "0.85")
        import sys
        if "src.entropy.engine" in sys.modules:
            del sys.modules["src.entropy.engine"]
        from src.entropy.engine import _scene_repeat_threshold
        assert _scene_repeat_threshold() == pytest.approx(0.85)

    def test_default_threshold_is_reasonable(self):
        from src.entropy.engine import _scene_repeat_threshold
        t = _scene_repeat_threshold()
        assert 0.8 <= t <= 1.0
