"""
Niche configuration loader.

Loads YAML config for a niche from config/niches/<niche>.yaml.
Falls back to config/niches/general.yaml if the niche is unknown.

Usage:
    from src.niche_config import get_niche_config
    cfg = get_niche_config("finance")
    print(cfg.model)              # "mistral"
    print(cfg.upload_schedule)   # [{"weekday": "tuesday", "hour": 14}, ...]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_NICHES_DIR = Path(__file__).parent.parent / "config" / "niches"


@dataclass
class NicheConfig:
    name: str
    display_name: str = ""
    model: str = "mistral"
    script_style: str = ""
    tone: str = "conversational"
    topic_boosters: list[str] = field(default_factory=list)
    seed_topics: list[str] = field(default_factory=list)
    hook_weights: dict[str, float] = field(default_factory=dict)
    retention_weights: dict[str, float] = field(default_factory=dict)
    upload_schedule: list[dict] = field(default_factory=list)
    default_category_id: str = "22"
    default_tags: list[str] = field(default_factory=list)
    entropy_min_score: float = 0.50
    entropy_scene_repeat_threshold: float = 0.97
    shorts_target_duration: int = 58
    shorts_hook_max_words: int = 12
    system_prompt_injection: str = ""
    visual_style: str = ""
    visual_keywords: list[str] = field(default_factory=list)


def get_niche_config(niche: str) -> NicheConfig:
    """Load niche config. Falls back to general if niche file not found."""
    path = _NICHES_DIR / f"{niche}.yaml"
    if not path.exists():
        path = _NICHES_DIR / "general.yaml"
    if not path.exists():
        return NicheConfig(name=niche)

    try:
        import yaml  # type: ignore[import]
    except ImportError:
        return NicheConfig(name=niche)

    try:
        data: dict = yaml.safe_load(path.read_text()) or {}
    except Exception as exc:
        import sys
        print(f"[niche_config] WARNING: could not parse {path}: {exc}", file=sys.stderr)
        return NicheConfig(name=niche)

    return NicheConfig(
        name=data.get("name", niche),
        display_name=data.get("display_name", niche),
        model=data.get("model", "mistral"),
        script_style=data.get("script_style", ""),
        tone=data.get("tone", "conversational"),
        topic_boosters=data.get("topic_boosters") or [],
        seed_topics=data.get("seed_topics") or [],
        hook_weights=data.get("hook_weights") or {},
        retention_weights=data.get("retention_weights") or {},
        upload_schedule=data.get("upload_schedule") or [],
        default_category_id=str(data.get("default_category_id", "22")),
        default_tags=data.get("default_tags") or [],
        entropy_min_score=float(data.get("entropy_min_score", 0.50)),
        entropy_scene_repeat_threshold=float(data.get("entropy_scene_repeat_threshold", 0.97)),
        shorts_target_duration=int(data.get("shorts_target_duration", 58)),
        shorts_hook_max_words=int(data.get("shorts_hook_max_words", 12)),
        system_prompt_injection=data.get("system_prompt_injection", ""),
        visual_style=data.get("visual_style", ""),
        visual_keywords=data.get("visual_keywords") or [],
    )


def list_niches() -> list[str]:
    """Return names of all configured niches."""
    return sorted(p.stem for p in _NICHES_DIR.glob("*.yaml"))
