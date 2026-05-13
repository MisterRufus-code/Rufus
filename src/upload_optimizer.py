"""
Upload time optimizer + A/B thumbnail tracker.

Picks the best upload time for a niche based on:
  1. Niche YAML schedule (preferred windows)
  2. Historical feedback: highest avg performance_score by hour-of-week

A/B thumbnail tracking records which thumbnail variant was live when a
video reached its first-48-hour performance snapshot so future renders
can favour the winning style.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import config

_AB_DB_PATH = Path(os.getenv("RUFUS_AB_DB_PATH", "data/ab_thumbnails.json"))
_ab_lock = threading.Lock()


# ── Upload time optimizer ─────────────────────────────────────────────────────

@dataclass
class UploadWindow:
    weekday: int    # 0=Monday … 6=Sunday
    hour: int       # UTC
    score: float    # higher = more desirable


def best_upload_time(niche: str) -> datetime:
    """
    Return the next UTC datetime that falls in the optimal upload window
    for *niche*, considering both the niche schedule and historical data.

    If no schedule is configured, defaults to next Tuesday 14:00 UTC.
    """
    from src.niche_config import get_niche_config
    cfg = get_niche_config(niche)

    # Build candidate windows from niche YAML
    windows: list[UploadWindow] = []
    _weekday_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    for entry in cfg.upload_schedule:
        wd = _weekday_map.get(str(entry.get("weekday", "")).lower())
        hr = int(entry.get("hour", 14))
        if wd is not None:
            windows.append(UploadWindow(weekday=wd, hour=hr, score=1.0))

    # Boost windows that historically performed well
    _apply_historical_scores(niche, windows)

    if not windows:
        windows = [UploadWindow(weekday=1, hour=14, score=1.0)]  # default: Tue 14:00

    best = max(windows, key=lambda w: w.score)
    return _next_occurrence(best.weekday, best.hour)


def _apply_historical_scores(niche: str, windows: list[UploadWindow]) -> None:
    """Boost windows whose hour-of-week correlates with high performance."""
    try:
        from src.ml.feedback import load_all_feedback
        records = [r for r in load_all_feedback() if r.niche == niche and r.views > 0]
        if not records:
            return
        # Build avg perf per (weekday, hour) from recorded_at timestamps
        bucket: dict[tuple[int, int], list[float]] = {}
        for r in records:
            try:
                dt = datetime.fromisoformat(r.recorded_at)
                key = (dt.weekday(), dt.hour)
                bucket.setdefault(key, []).append(r.performance_score)
            except Exception:
                continue
        if not bucket:
            return
        max_avg = max(sum(v) / len(v) for v in bucket.values())
        if max_avg <= 0:
            return
        for w in windows:
            key = (w.weekday, w.hour)
            if key in bucket:
                avg = sum(bucket[key]) / len(bucket[key])
                w.score += avg / max_avg  # normalize boost 0–1
    except Exception:
        pass


def _next_occurrence(weekday: int, hour: int) -> datetime:
    """Return the next UTC datetime matching *weekday* (0=Mon) and *hour*."""
    now = datetime.utcnow()
    days_ahead = (weekday - now.weekday()) % 7
    if days_ahead == 0 and now.hour >= hour:
        days_ahead = 7
    target = (now + timedelta(days=days_ahead)).replace(
        hour=hour, minute=0, second=0, microsecond=0
    )
    return target


# ── A/B thumbnail tracker ─────────────────────────────────────────────────────

@dataclass
class ThumbnailVariant:
    video_id: str
    variant: str                        # "A" or "B"
    style: str                          # e.g. "face_close_up", "text_heavy"
    thumbnail_path: str
    uploaded_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    performance_score: float = 0.0
    ctr: float = 0.0
    winner: Optional[bool] = None       # None = pending, True = winner, False = loser


def record_thumbnail_variant(
    video_id: str,
    variant: str,
    style: str,
    thumbnail_path: str,
) -> None:
    """Record a thumbnail variant for A/B tracking."""
    entry = ThumbnailVariant(
        video_id=video_id,
        variant=variant,
        style=style,
        thumbnail_path=thumbnail_path,
    )
    with _ab_lock:
        db = _load_ab_db()
        db.append(entry.__dict__)
        _save_ab_db(db)


def update_thumbnail_performance(video_id: str, variant: str, ctr: float, performance_score: float) -> None:
    """Update performance metrics after 48-hour snapshot."""
    with _ab_lock:
        db = _load_ab_db()
        for entry in db:
            if entry["video_id"] == video_id and entry["variant"] == variant:
                entry["ctr"] = ctr
                entry["performance_score"] = performance_score
        # Determine winner if both variants exist
        variants = [e for e in db if e["video_id"] == video_id]
        if len(variants) == 2 and all(e["ctr"] > 0 for e in variants):
            winner = max(variants, key=lambda e: e["ctr"])
            loser = min(variants, key=lambda e: e["ctr"])
            winner["winner"] = True
            loser["winner"] = False
        _save_ab_db(db)


def best_thumbnail_style(niche: str) -> Optional[str]:
    """Return the thumbnail style with the highest avg CTR for *niche*."""
    db = _load_ab_db()
    style_ctrs: dict[str, list[float]] = {}
    for entry in db:
        if entry.get("winner") is True and entry.get("ctr", 0) > 0:
            style = entry.get("style", "unknown")
            style_ctrs.setdefault(style, []).append(entry["ctr"])
    if not style_ctrs:
        return None
    return max(style_ctrs, key=lambda s: sum(style_ctrs[s]) / len(style_ctrs[s]))


def _load_ab_db() -> list[dict]:
    if not _AB_DB_PATH.exists():
        return []
    try:
        return json.loads(_AB_DB_PATH.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_ab_db(records: list[dict]) -> None:
    _AB_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _AB_DB_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(records, indent=2))
    os.replace(tmp, _AB_DB_PATH)
