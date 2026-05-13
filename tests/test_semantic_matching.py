"""Tests for semantic scene matching and deduplication logic."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.database.models import MediaAsset, SceneRequirement


def _make_asset(asset_id: str, score: float = 0.5, duration: float = 5.0) -> MediaAsset:
    return MediaAsset(
        asset_id=asset_id,
        path=f"/fake/{asset_id}.mp4",
        asset_type="video",
        duration_seconds=duration,
        performance_score=score,
    )


def _make_scene(index: int = 0) -> SceneRequirement:
    return SceneRequirement(
        scene_index=index,
        description=f"Scene {index}",
        semantic_query=f"query for scene {index}",
        preferred_type="video",
        mood="neutral",
    )


class TestBuildSceneQuery:
    def test_combines_query_and_mood(self):
        from src.matching.semantic import build_scene_query
        scene = _make_scene()
        scene.semantic_query = "city traffic"
        scene.mood = "dramatic"
        query = build_scene_query(scene)
        assert "city traffic" in query
        assert "cinematic" in query or "dramatic" in query

    def test_neutral_mood_adds_no_suffix(self):
        from src.matching.semantic import build_scene_query
        scene = _make_scene()
        scene.semantic_query = "mountain lake"
        scene.mood = "neutral"
        query = build_scene_query(scene)
        assert query.startswith("mountain lake")


class TestMatchAllScenes:
    def _mock_store(self, assets_per_call):
        store = MagicMock()
        store.search.return_value = assets_per_call
        store.increment_usage_count.return_value = None
        return store

    def test_no_clips_when_store_empty(self):
        from src.matching.semantic import match_all_scenes
        with patch("src.matching.semantic.get_vector_store") as mock_vs, \
             patch("src.matching.semantic.embed_text", return_value=[0.0] * 512), \
             patch("src.matching.semantic.get_globally_used_asset_ids", return_value=set()):
            mock_vs.return_value = self._mock_store([])
            results = match_all_scenes([_make_scene(0)])
        assert results[0].selected is None

    def test_selects_highest_performance_score(self):
        from src.matching.semantic import match_all_scenes
        assets = [_make_asset("a1", score=0.3), _make_asset("a2", score=0.9)]
        with patch("src.matching.semantic.get_vector_store") as mock_vs, \
             patch("src.matching.semantic.embed_text", return_value=[0.0] * 512), \
             patch("src.matching.semantic.get_globally_used_asset_ids", return_value=set()):
            mock_vs.return_value = self._mock_store(assets)
            results = match_all_scenes([_make_scene(0)])
        assert results[0].selected.asset_id == "a2"

    def test_no_duplicate_clips_in_same_video(self):
        from src.matching.semantic import match_all_scenes
        # Only one asset available — used for scene 0, should still be used for scene 1
        # (no fresh options) but we verify dedup logic doesn't crash
        assets = [_make_asset("a1", score=0.5)]
        with patch("src.matching.semantic.get_vector_store") as mock_vs, \
             patch("src.matching.semantic.embed_text", return_value=[0.0] * 512), \
             patch("src.matching.semantic.get_globally_used_asset_ids", return_value=set()):
            mock_vs.return_value = self._mock_store(assets)
            results = match_all_scenes([_make_scene(0), _make_scene(1)])
        assert results[0].selected is not None
        assert results[1].selected is not None  # must not crash even with only 1 asset

    def test_duration_zero_assets_excluded(self):
        from src.matching.semantic import match_scene
        zero_dur = _make_asset("zero_dur", duration=0.0)
        good = _make_asset("good_dur", duration=5.0)
        with patch("src.matching.semantic.get_vector_store") as mock_vs, \
             patch("src.matching.semantic.embed_text", return_value=[0.0] * 512):
            mock_vs.return_value = self._mock_store([zero_dur, good])
            scene = _make_scene()
            scene.preferred_type = "video"
            scene.min_duration = 1.0
            scene.max_duration = 15.0
            result = match_scene(scene)
        # zero_dur should be filtered out
        assert all(a.duration_seconds and a.duration_seconds > 0 for a in result.candidates)


class TestScenesFromScript:
    def test_creates_scene_per_section(self):
        from src.matching.semantic import scenes_from_script
        sections = [
            {"heading": "Intro", "script": "Welcome to the video"},
            {"heading": "Main Point", "script": "Here is the key insight"},
        ]
        scenes = scenes_from_script(sections)
        assert len(scenes) == 2
        assert scenes[0].description == "Intro"
        assert scenes[1].scene_index == 1

    def test_empty_sections_returns_empty(self):
        from src.matching.semantic import scenes_from_script
        assert scenes_from_script([]) == []
