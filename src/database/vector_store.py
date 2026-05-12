"""
Qdrant vector store wrapper.

Stores CLIP embeddings + rich metadata for every media asset.
Falls back to in-memory FAISS if Qdrant is unavailable.
"""

from __future__ import annotations

import json
from typing import Optional

import numpy as np

import config
from src.database.models import MediaAsset
from src.ingestion.extractor import AssetFeatures

_store_instance = None


class QdrantStore:
    def __init__(self):
        from qdrant_client import QdrantClient
        from qdrant_client.models import (
            Distance,
            VectorParams,
            PointStruct,
        )
        self._qdrant = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        self._collection = config.QDRANT_COLLECTION
        self._dim = config.CLIP_EMBEDDING_DIM
        self._PointStruct = PointStruct

        existing = {c.name for c in self._qdrant.get_collections().collections}
        if self._collection not in existing:
            self._qdrant.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._dim, distance=Distance.COSINE),
            )

    def upsert(self, features: AssetFeatures) -> None:
        from qdrant_client.models import PointStruct
        asset = MediaAsset(
            asset_id=features.asset_id,
            path=features.path,
            asset_type=features.asset_type,
            caption=features.caption,
            detected_objects=features.detected_objects,
            dominant_colors=features.dominant_colors,
            motion_score=features.motion_score,
            duration_seconds=features.duration_seconds,
            frame_count=features.frame_count,
            clip_embedding=features.clip_embedding,
        )
        # Qdrant needs an integer or UUID id — use integer hash of asset_id
        point_id = abs(hash(features.asset_id)) % (2**63)
        self._qdrant.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=features.clip_embedding,
                    payload={**asset.to_payload(), "_asset_id_str": features.asset_id},
                )
            ],
        )

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        asset_type_filter: Optional[str] = None,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[MediaAsset]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        query_filter = None
        if asset_type_filter:
            query_filter = Filter(
                must=[FieldCondition(key="asset_type", match=MatchValue(value=asset_type_filter))]
            )

        fetch_k = top_k + len(exclude_ids) + 10 if exclude_ids else top_k
        try:
            result = self._qdrant.query_points(
                collection_name=self._collection,
                query=query_vector,
                limit=fetch_k,
                query_filter=query_filter,
                with_payload=True,
            )
            hits = result.points
        except AttributeError:
            hits = self._qdrant.search(
                collection_name=self._collection,
                query_vector=query_vector,
                limit=fetch_k,
                query_filter=query_filter,
                with_payload=True,
            )
        assets = [MediaAsset.from_payload(h.payload) for h in hits]
        if exclude_ids:
            assets = [a for a in assets if a.asset_id not in exclude_ids]
        return assets[:top_k]

    def get_all_assets(self) -> list[MediaAsset]:
        """Return all assets stored in Qdrant with their metadata."""
        assets = []
        offset = None
        while True:
            results, offset = self._qdrant.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
            )
            for r in results:
                try:
                    assets.append(MediaAsset.from_payload(r.payload))
                except Exception:
                    pass
            if offset is None:
                break
        return assets

    def get_all_asset_ids(self) -> set[str]:
        ids = set()
        offset = None
        while True:
            results, offset = self._qdrant.scroll(
                collection_name=self._collection,
                limit=256,
                offset=offset,
                with_payload=True,
            )
            for r in results:
                ids.add(r.payload.get("_asset_id_str", ""))
            if offset is None:
                break
        return ids

    def update_usage(self, asset_id: str, performance_score: float) -> None:
        """Increment usage count and update learned performance score."""
        from qdrant_client.models import SetPayload
        import datetime
        point_id = abs(hash(asset_id)) % (2**63)
        self._qdrant.set_payload(
            collection_name=self._collection,
            payload={
                "last_used_at": datetime.datetime.utcnow().isoformat(),
                "performance_score": performance_score,
            },
            points=[point_id],
        )

    def increment_usage_count(self, asset_id: str) -> None:
        point_id = abs(hash(asset_id)) % (2**63)
        results = self._qdrant.retrieve(
            collection_name=self._collection,
            ids=[point_id],
            with_payload=True,
        )
        if results:
            count = results[0].payload.get("usage_count", 0) + 1
            self._qdrant.set_payload(
                collection_name=self._collection,
                payload={"usage_count": count},
                points=[point_id],
            )


class FaissStore:
    """In-memory FAISS fallback when Qdrant is not running."""

    def __init__(self):
        import faiss
        self._dim = config.CLIP_EMBEDDING_DIM
        self._index = faiss.IndexFlatIP(self._dim)  # cosine via normalized vectors
        self._assets: list[MediaAsset] = []
        self._id_set: set[str] = set()

    def upsert(self, features: AssetFeatures) -> None:
        if features.asset_id in self._id_set:
            return
        vec = np.array([features.clip_embedding], dtype=np.float32)
        self._index.add(vec)
        asset = MediaAsset(
            asset_id=features.asset_id,
            path=features.path,
            asset_type=features.asset_type,
            caption=features.caption,
            detected_objects=features.detected_objects,
            dominant_colors=features.dominant_colors,
            motion_score=features.motion_score,
            duration_seconds=features.duration_seconds,
            frame_count=features.frame_count,
            clip_embedding=features.clip_embedding,
        )
        self._assets.append(asset)
        self._id_set.add(features.asset_id)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        asset_type_filter: Optional[str] = None,
        exclude_ids: Optional[set[str]] = None,
    ) -> list[MediaAsset]:
        if self._index.ntotal == 0:
            return []
        vec = np.array([query_vector], dtype=np.float32)
        k = min(top_k * 4, self._index.ntotal)
        _, indices = self._index.search(vec, k)
        results = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(self._assets):
                continue
            asset = self._assets[idx]
            if asset_type_filter and asset.asset_type != asset_type_filter:
                continue
            if exclude_ids and asset.asset_id in exclude_ids:
                continue
            results.append(asset)
            if len(results) >= top_k:
                break
        return results

    def get_all_assets(self) -> list[MediaAsset]:
        return list(self._assets)

    def get_all_asset_ids(self) -> set[str]:
        return set(self._id_set)

    def update_usage(self, asset_id: str, performance_score: float) -> None:
        for asset in self._assets:
            if asset.asset_id == asset_id:
                asset.performance_score = performance_score
                break

    def increment_usage_count(self, asset_id: str) -> None:
        for asset in self._assets:
            if asset.asset_id == asset_id:
                asset.usage_count += 1
                break


def get_vector_store() -> QdrantStore | FaissStore:
    global _store_instance
    if _store_instance is not None:
        return _store_instance
    try:
        _store_instance = QdrantStore()
    except Exception:
        from rich.console import Console
        Console().print(
            "[yellow]Qdrant unavailable — using in-memory FAISS fallback.[/yellow]"
        )
        _store_instance = FaissStore()
    return _store_instance
