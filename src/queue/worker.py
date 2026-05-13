"""Celery-based distributed task queue for Rufus pipeline jobs.

Requires Redis and Celery:
    pip install celery redis
    docker run -d -p 6379:6379 redis:alpine

Start a worker:
    celery -A src.queue.worker worker --loglevel=info --concurrency=2

Submit a job from Python:
    from src.queue.worker import submit_pipeline, get_task_result
    task_id = submit_pipeline("finance tips", niche="finance")
    result  = get_task_result(task_id)

Environment variables:
    REDIS_URL  – default redis://localhost:6379/0
"""

from __future__ import annotations

import os
from typing import Any

# ---------------------------------------------------------------------------
# Optional Celery import — provide clear error if not installed
# ---------------------------------------------------------------------------
try:
    from celery import Celery
    from celery.result import AsyncResult
    _CELERY_AVAILABLE = True
except ImportError:  # pragma: no cover
    _CELERY_AVAILABLE = False

    class Celery:  # type: ignore[no-redef]
        """Stub that raises on instantiation when Celery is not installed."""
        def __init__(self, *a, **kw):
            raise ImportError(
                "Celery is not installed. Run: pip install celery redis\n"
                "Then start Redis: docker run -d -p 6379:6379 redis:alpine"
            )

    class AsyncResult:  # type: ignore[no-redef]
        pass


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
_REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = Celery(
    "rufus",
    broker=_REDIS_URL,
    backend=_REDIS_URL,
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    result_expires=86400,                # keep results for 24 h
    worker_prefetch_multiplier=1,        # one task at a time per worker process
    task_acks_late=True,                 # ack only after task completes
    worker_max_tasks_per_child=10,       # restart worker every 10 tasks (memory hygiene)
)


# ---------------------------------------------------------------------------
# Task definition
# ---------------------------------------------------------------------------

@app.task(
    bind=True,
    max_retries=3,
    default_retry_delay=60,   # seconds before automatic retry
    name="rufus.run_pipeline",
)
def run_pipeline_task(
    self,
    topic: str,
    niche: str = "general",
    model: str = "mistral",
    voice: str = "af_heart",
    upload: bool = False,
    privacy: str = "private",
    shorts: bool = False,
    geo: str = "US",
    dry_run: bool = False,
) -> dict:
    """Run a full Rufus pipeline in an isolated subprocess.

    Retried up to 3 times (with 60-second delays) on any exception.
    Each attempt runs in a fresh subprocess via run_pipeline_isolated().
    """
    from src.worker_process import run_pipeline_isolated

    try:
        result = run_pipeline_isolated(
            topic=topic,
            niche=niche,
            model=model,
            voice=voice,
            upload=upload,
            privacy=privacy,
            shorts=shorts,
            geo=geo,
            dry_run=dry_run,
        )
        return result
    except Exception as exc:
        # Celery will retry up to max_retries times.
        raise self.retry(exc=exc)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def submit_pipeline(topic: str, niche: str = "general", **kwargs: Any) -> str:
    """Submit a pipeline job and return the Celery task ID.

    Parameters
    ----------
    topic:  Video topic / subject.
    niche:  Niche name (must have a matching config/niches/<niche>.yaml).
    **kwargs: Any keyword arg accepted by run_pipeline_task
              (model, voice, upload, privacy, shorts, geo, dry_run).

    Returns
    -------
    str — Celery task ID that can be passed to get_task_result().
    """
    async_result = run_pipeline_task.apply_async(
        kwargs={"topic": topic, "niche": niche, **kwargs},
        # Routing can be customised via CELERY_ROUTES if needed.
    )
    return async_result.id


def get_task_result(task_id: str) -> dict:
    """Return current status and result for *task_id*.

    Returns
    -------
    dict with keys:
        task_id   – echoed back
        status    – Celery state string (PENDING, STARTED, SUCCESS, FAILURE, RETRY)
        result    – result dict on SUCCESS, error string on FAILURE, None otherwise
        ready     – bool, True when the task has finished (success or failure)
    """
    ar: AsyncResult = app.AsyncResult(task_id)
    status = ar.state
    ready = ar.ready()

    if status == "SUCCESS":
        result_value = ar.result
    elif status == "FAILURE":
        result_value = str(ar.result)
    else:
        result_value = None

    return {
        "task_id": task_id,
        "status": status,
        "result": result_value,
        "ready": ready,
    }
