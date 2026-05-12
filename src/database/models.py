"""Shared data models for the Rufus media system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MediaAsset:
    """A fully indexed media asset with semantic metadata."""
    asset_id: str
    path: str
    asset_type: str                        # "video" | "image"
    caption: str = ""
    detected_objects: list[str] = field(default_factory=list)
    dominant_colors: list[str] = field(default_factory=list)
    motion_score: float = 0.0
    duration_seconds: float = 0.0
    frame_count: int = 0
    usage_count: int = 0
    last_used_at: Optional[str] = None
    performance_score: float = 0.5        # learned from ML feedback
    clip_embedding: list[float] = field(default_factory=list)

    def to_payload(self) -> dict:
        """Qdrant payload (everything except the embedding vector)."""
        return {
            "asset_id": self.asset_id,
            "path": self.path,
            "asset_type": self.asset_type,
            "caption": self.caption,
            "detected_objects": self.detected_objects,
            "dominant_colors": self.dominant_colors,
            "motion_score": self.motion_score,
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
            "usage_count": self.usage_count,
            "last_used_at": self.last_used_at,
            "performance_score": self.performance_score,
        }

    @classmethod
    def from_payload(cls, payload: dict, embedding: list[float] | None = None) -> "MediaAsset":
        return cls(
            asset_id=payload["asset_id"],
            path=payload["path"],
            asset_type=payload.get("asset_type", "image"),
            caption=payload.get("caption", ""),
            detected_objects=payload.get("detected_objects", []),
            dominant_colors=payload.get("dominant_colors", []),
            motion_score=payload.get("motion_score", 0.0),
            duration_seconds=payload.get("duration_seconds", 0.0),
            frame_count=payload.get("frame_count", 0),
            usage_count=payload.get("usage_count", 0),
            last_used_at=payload.get("last_used_at"),
            performance_score=payload.get("performance_score", 0.5),
            clip_embedding=embedding or [],
        )


@dataclass
class SceneRequirement:
    """What the pipeline needs for a single script scene."""
    scene_index: int
    description: str             # from the script
    semantic_query: str          # enriched for retrieval
    preferred_type: str = "video"   # "video" | "image" | "any"
    min_duration: float = 3.0
    max_duration: float = 15.0
    mood: str = "neutral"


@dataclass
class TimelineClip:
    """One clip in the final video timeline."""
    asset: MediaAsset
    start_time: float            # seconds in timeline
    end_time: float
    scene_index: int
    in_point: float = 0.0       # trim start within asset
    out_point: float = 0.0      # trim end within asset
