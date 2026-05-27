"""Tests for vector store — FAISS fallback (no Qdrant required)."""

from __future__ import annotations

import pytest

pytest.importorskip("torch", reason="torch not installed — vector store requires CLIP models")

from src.database.vector_store import _asset_point_id, FaissStore  # noqa: E402
from src.ingestion.extractor import AssetFeatures  # noqa: E402


def _make_features(asset_id: str, embedding: list[float] | None = None) -> AssetFeatures:
    import config
    dim = config.CLIP_EMBEDDING_DIM
    return AssetFeatures(
        asset_id=asset_id,
        path=f"/fake/{asset_id}.mp4",
        asset_type="video",
        caption="test caption",
        clip_embedding=embedding or [0.1] * dim,
        duration_seconds=5.0,
    )


class TestAssetPointId:
    def test_same_id_same_hash(self):
        assert _asset_point_id("abc") == _asset_point_id("abc")

    def test_different_ids_different_hashes(self):
        assert _asset_point_id("abc") != _asset_point_id("xyz")

    def test_result_is_positive_int(self):
        result = _asset_point_id("test_asset_001")
        assert isinstance(result, int)
        assert result > 0

    def test_no_collision_on_similar_strings(self):
        # Python hash() has known collisions for these — SHA256 should not
        ids = [f"asset_{i:04d}" for i in range(100)]
        hashes = [_asset_point_id(x) for x in ids]
        assert len(set(hashes)) == len(hashes), "Hash collision detected"


class TestFaissStore:
    @pytest.fixture
    def store(self, tmp_path, monkeypatch):
        # Redirect to an empty temp dir so tests don't touch the production index
        monkeypatch.setattr(FaissStore, "_STORE_DIR", str(tmp_path / "faiss_store"))
        return FaissStore()

    def test_empty_search_returns_empty(self, store):
        import config
        result = store.search([0.1] * config.CLIP_EMBEDDING_DIM)
        assert result == []

    def test_upsert_and_search(self, store):
        import numpy as np
        import config
        dim = config.CLIP_EMBEDDING_DIM

        # Insert an asset with a specific embedding
        vec = [1.0 if i == 0 else 0.0 for i in range(dim)]
        features = _make_features("asset_001", embedding=vec)
        store.upsert(features)

        results = store.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0].asset_id == "asset_001"

    def test_duplicate_upsert_not_duplicated(self, store):
        features = _make_features("dup_asset")
        store.upsert(features)
        store.upsert(features)
        assert store.get_all_assets().__len__() == 1

    def test_exclude_ids_filters_results(self, store):
        import config
        dim = config.CLIP_EMBEDDING_DIM
        for i in range(3):
            store.upsert(_make_features(f"asset_{i}", embedding=[float(i)] * dim))

        results = store.search([0.0] * dim, top_k=3, exclude_ids={"asset_0"})
        ids = [r.asset_id for r in results]
        assert "asset_0" not in ids

    def test_asset_type_filter(self, store):
        import config
        dim = config.CLIP_EMBEDDING_DIM
        vid = _make_features("video_1")
        vid.asset_type = "video"
        img = _make_features("image_1")
        img.asset_type = "image"
        store.upsert(vid)
        store.upsert(img)

        video_results = store.search([0.1] * dim, top_k=5, asset_type_filter="video")
        assert all(r.asset_type == "video" for r in video_results)

    def test_update_usage_changes_score(self, store):
        store.upsert(_make_features("score_asset"))
        store.update_usage("score_asset", 0.95)
        asset = next(a for a in store.get_all_assets() if a.asset_id == "score_asset")
        assert asset.performance_score == 0.95

    def test_increment_usage_count(self, store):
        store.upsert(_make_features("count_asset"))
        store.increment_usage_count("count_asset")
        store.increment_usage_count("count_asset")
        asset = next(a for a in store.get_all_assets() if a.asset_id == "count_asset")
        assert asset.usage_count == 2
