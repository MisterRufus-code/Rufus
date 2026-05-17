"""
Rufus Event Bus — lightweight in-process pub/sub.

Every major pipeline action emits an event. Subscribers react without
the emitter knowing who's listening. This keeps modules decoupled and
makes the whole system observable.

Usage:
    from src.events import emit, subscribe, Event

    # Subscribe (at startup)
    subscribe("hook.scored",    my_handler)
    subscribe("pipeline.stage", dashboard_handler)

    # Emit (anywhere in pipeline)
    emit("hook.scored", niche="finance", score=8.4, hook="Stop losing...")

Standard events:
    pipeline.start          niche, topic, mode
    pipeline.stage          stage, step, total, niche
    pipeline.done           niche, topic, success, runtime_s
    pipeline.error          niche, stage, error
    hook.scored             niche, hook, viral_score, hook_score
    hook.selected           niche, hook, viral_score
    script.ready            niche, sections, retention_score
    retention.scored        niche, score, grade, suggestions
    tts.done                niche, duration_s, voice
    render.done             niche, output_path, mode
    upload.done             niche, video_id, title
    dna.updated             niche, sample_size
    trend.sniped            niche, topic, momentum
    genome.evolved          niche, generation, best_score
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Callable

from rich.console import Console

console = Console()

# ---------------------------------------------------------------------------
# Core bus
# ---------------------------------------------------------------------------

_lock        = threading.Lock()
_subscribers: dict[str, list[Callable]] = defaultdict(list)
_history:     list[dict]                = []   # last 500 events for replay/debug
_MAX_HISTORY = 500


def subscribe(event: str, handler: Callable) -> None:
    """Register *handler* to be called whenever *event* is emitted."""
    with _lock:
        _subscribers[event].append(handler)


def subscribe_all(handler: Callable) -> None:
    """Register *handler* for every event (wildcard subscriber)."""
    subscribe("*", handler)


def emit(event: str, **data) -> None:
    """
    Emit an event, calling all matching subscribers synchronously.
    Errors in subscribers are swallowed so the pipeline never crashes.
    """
    payload = {"event": event, "ts": time.time(), **data}

    with _lock:
        handlers = list(_subscribers.get(event, []))
        handlers += list(_subscribers.get("*", []))
        _history.append(payload)
        if len(_history) > _MAX_HISTORY:
            _history.pop(0)

    for handler in handlers:
        try:
            handler(**payload)
        except Exception:
            pass


def get_history(event: str | None = None, last_n: int = 100) -> list[dict]:
    """Return recent events, optionally filtered by event name."""
    with _lock:
        history = list(_history[-last_n:])
    if event:
        history = [e for e in history if e.get("event") == event]
    return history


# ---------------------------------------------------------------------------
# Built-in subscribers
# ---------------------------------------------------------------------------

def _structured_log_handler(**payload) -> None:
    """Write every event to the structured JSON log."""
    try:
        from src.logging_config import get_logger
        log = get_logger("rufus.events")
        event = payload.pop("event", "unknown")
        payload.pop("ts", None)
        log.info(event, **payload)
    except Exception:
        pass


def _metrics_collector(**payload) -> None:
    """Accumulate pipeline metrics in a simple in-memory store."""
    _MetricsStore.record(**payload)


subscribe_all(_structured_log_handler)
subscribe_all(_metrics_collector)


# ---------------------------------------------------------------------------
# Metrics store (in-memory, queried by dashboard/supervisor)
# ---------------------------------------------------------------------------

class _MetricsStore:
    _data: dict = {
        "pipelines_run":    0,
        "pipelines_ok":     0,
        "pipelines_failed": 0,
        "hooks_generated":  0,
        "avg_viral_score":  0.0,
        "avg_retention":    0.0,
        "total_renders":    0,
        "total_uploads":    0,
        "_viral_scores":    [],
        "_retention_scores": [],
    }
    _lock = threading.Lock()

    @classmethod
    def record(cls, event: str, **kw) -> None:
        with cls._lock:
            d = cls._data
            if event == "pipeline.start":
                d["pipelines_run"] += 1
            elif event == "pipeline.done":
                if kw.get("success"):
                    d["pipelines_ok"] += 1
                else:
                    d["pipelines_failed"] += 1
            elif event == "hook.scored":
                d["hooks_generated"] += 1
                score = kw.get("viral_score", 0)
                d["_viral_scores"].append(score)
                d["_viral_scores"] = d["_viral_scores"][-200:]
                d["avg_viral_score"] = round(
                    sum(d["_viral_scores"]) / len(d["_viral_scores"]), 2
                )
            elif event == "retention.scored":
                score = kw.get("score", 0)
                d["_retention_scores"].append(score)
                d["_retention_scores"] = d["_retention_scores"][-200:]
                d["avg_retention"] = round(
                    sum(d["_retention_scores"]) / len(d["_retention_scores"]), 2
                )
            elif event == "render.done":
                d["total_renders"] += 1
            elif event == "upload.done":
                d["total_uploads"] += 1

    @classmethod
    def snapshot(cls) -> dict:
        with cls._lock:
            return {k: v for k, v in cls._data.items() if not k.startswith("_")}


def metrics() -> dict:
    """Return current aggregated pipeline metrics."""
    return _MetricsStore.snapshot()
