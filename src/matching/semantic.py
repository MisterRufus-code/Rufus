"""
Reverse semantic matching engine.

Script scenes → CLIP text embeddings → search vector DB → ranked media assets.

Instead of keyword search, every scene description is embedded into the same
512-dim CLIP space as the indexed media, so the system retrieves footage by
visual meaning rather than tag matching.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from rich.console import Console

from src.database.models import MediaAsset, SceneRequirement
from src.database.vector_store import get_vector_store
from src.ingestion.extractor import embed_text

console = Console()


@dataclass
class MatchResult:
    scene: SceneRequirement
    candidates: list[MediaAsset] = field(default_factory=list)
    selected: MediaAsset | None = None


def build_scene_query(scene: SceneRequirement) -> str:
    """
    Enrich a plain scene description into a retrieval-optimised query.

    Combines the scene description, mood, and preferred type into a single
    natural-language string that maps well to CLIP's visual embedding space.
    """
    mood_map = {
        "neutral": "",
        "dramatic": "dramatic cinematic dark moody",
        "upbeat": "bright vibrant energetic colourful",
        "sad": "melancholy grey rain lonely empty",
        "tense": "tense close-up suspenseful shadows",
        "inspirational": "golden hour sunlight hopeful wide landscape",
        "funny": "playful comic exaggerated expression",
    }
    mood_suffix = mood_map.get(scene.mood, "")
    type_suffix = "video footage" if scene.preferred_type == "video" else "photograph"
    parts = [scene.semantic_query or scene.description, mood_suffix, type_suffix]
    return " ".join(p for p in parts if p).strip()


def match_scene(
    scene: SceneRequirement,
    top_k: int = 8,
    asset_type_filter: str | None = None,
) -> MatchResult:
    """Find the best media assets for a single scene requirement."""
    store = get_vector_store()
    query_text = build_scene_query(scene)

    try:
        query_vec = embed_text(query_text)
    except Exception as exc:
        console.print(f"[yellow]CLIP embed failed ({exc}), falling back to empty results.[/yellow]")
        return MatchResult(scene=scene)

    asset_filter = asset_type_filter or (
        scene.preferred_type if scene.preferred_type != "any" else None
    )

    candidates = store.search(query_vec, top_k=top_k, asset_type_filter=asset_filter)

    # Filter by duration requirement for videos
    if scene.preferred_type == "video":
        candidates = [
            a for a in candidates
            if scene.min_duration <= (a.duration_seconds or 999) <= scene.max_duration
        ]

    result = MatchResult(scene=scene, candidates=candidates)
    if candidates:
        # Pick asset with best combined semantic + performance score
        result.selected = max(
            candidates,
            key=lambda a: a.performance_score,
        )
    return result


def match_all_scenes(
    scenes: list[SceneRequirement],
    top_k: int = 8,
    avoid_repeat: bool = True,
) -> list[MatchResult]:
    """
    Match every scene in a script.

    With *avoid_repeat=True* the same asset won't be selected for two scenes
    (entropy-style deduplication at the selection level).
    """
    used_ids: set[str] = set()
    results: list[MatchResult] = []

    for scene in scenes:
        result = match_scene(scene, top_k=top_k)

        if avoid_repeat and result.candidates:
            # Try to pick an unused asset
            for candidate in result.candidates:
                if candidate.asset_id not in used_ids:
                    result.selected = candidate
                    break
            else:
                result.selected = result.candidates[0]

        if result.selected:
            used_ids.add(result.selected.asset_id)
            store = get_vector_store()
            store.increment_usage_count(result.selected.asset_id)

        results.append(result)

    return results


def scenes_from_script(script_sections: list[dict], default_mood: str = "neutral") -> list[SceneRequirement]:
    """Convert raw script sections into SceneRequirement objects."""
    scenes = []
    for i, section in enumerate(script_sections):
        heading = section.get("heading", f"Scene {i + 1}")
        script_text = section.get("script", "")
        # First 120 chars of script text = semantic query
        semantic_query = script_text[:120].replace("\n", " ").strip()
        scenes.append(
            SceneRequirement(
                scene_index=i,
                description=heading,
                semantic_query=semantic_query or heading,
                preferred_type="video",
                mood=default_mood,
            )
        )
    return scenes
