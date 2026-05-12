"""
ML Feedback Store.

Records per-video performance metrics (CTR, watch time, retention)
and links them back to the prompts, niche, and asset IDs used.

Data is stored in a local JSON file — no external service required.
"""

from __future__ import annotations

import json
import datetime
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import config

console_import_ok = True
try:
    from rich.console import Console
    console = Console()
except ImportError:
    console_import_ok = False


@dataclass
class VideoFeedback:
    video_id: str                        # YouTube video ID
    title: str
    niche: str
    topic: str
    prompt_model: str                    # e.g. "mistral"
    asset_ids: list[str] = field(default_factory=list)
    ctr: float = 0.0                     # click-through rate 0–1
    avg_watch_time_seconds: float = 0.0
    retention_rate: float = 0.0          # 0–1
    views: int = 0
    likes: int = 0
    estimated_virality: str = "Medium"
    recorded_at: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )

    @property
    def performance_score(self) -> float:
        """Composite 0–1 score used to update asset weights."""
        ctr_w = min(self.ctr / 0.10, 1.0) * 0.35          # 10% CTR = max
        ret_w = min(self.retention_rate, 1.0) * 0.45
        view_w = min(self.views / 10_000, 1.0) * 0.20
        return round(ctr_w + ret_w + view_w, 4)


def _load_db() -> list[dict]:
    db_path = Path(config.FEEDBACK_DB_PATH)
    if not db_path.exists():
        return []
    try:
        return json.loads(db_path.read_text())
    except (json.JSONDecodeError, OSError):
        return []


def _save_db(records: list[dict]) -> None:
    db_path = Path(config.FEEDBACK_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    db_path.write_text(json.dumps(records, indent=2))


def record_feedback(feedback: VideoFeedback) -> None:
    """Append a feedback record and update asset performance scores."""
    records = _load_db()
    records.append(asdict(feedback))
    _save_db(records)

    # Push updated performance scores back into the vector store
    if feedback.asset_ids:
        try:
            from src.database.vector_store import get_vector_store
            store = get_vector_store()
            for asset_id in feedback.asset_ids:
                store.update_usage(asset_id, feedback.performance_score)
        except Exception:
            pass

    if console_import_ok:
        console.print(
            f"[green]Feedback recorded[/green] for video [cyan]{feedback.video_id}[/cyan] "
            f"— performance score: [bold]{feedback.performance_score:.3f}[/bold]"
        )


def load_all_feedback() -> list[VideoFeedback]:
    records = _load_db()
    result = []
    for r in records:
        try:
            result.append(VideoFeedback(**r))
        except TypeError:
            continue
    return result


def feedback_summary() -> dict:
    records = load_all_feedback()
    if not records:
        return {"total": 0}
    scores = [r.performance_score for r in records]
    niches: dict[str, list[float]] = {}
    for r in records:
        niches.setdefault(r.niche, []).append(r.performance_score)
    best_niche = max(niches, key=lambda k: sum(niches[k]) / len(niches[k])) if niches else ""
    return {
        "total": len(records),
        "avg_performance": round(sum(scores) / len(scores), 3),
        "best_niche": best_niche,
        "niches": {k: round(sum(v) / len(v), 3) for k, v in niches.items()},
    }
