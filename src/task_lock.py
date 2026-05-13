"""
Process-level task lock — ensures only one heavy operation runs at a time.

Replaces the old low_power CPU throttling with a proper serialization lock.
Works across threads (FastAPI background tasks) and can be extended to
cross-process locking via a lock file if needed.

Usage:
    from src.task_lock import task_lock

    with task_lock("clip_inference"):
        embeddings = model(frames)

The label is just for logging — any holder blocks all others.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Generator

from rich.console import Console

console = Console()

# Single reentrant lock — one heavy task at a time, same thread can nest
_lock = threading.Lock()
_current_task: str = ""
_lock_meta = threading.Lock()  # protects _current_task string


@contextmanager
def task_lock(label: str = "task", timeout: float = 300.0) -> Generator[None, None, None]:
    """
    Block until the global task lock is free, then hold it for the duration.

    Args:
        label:   Human-readable name shown in logs while waiting/running.
        timeout: Max seconds to wait before giving up (raises RuntimeError).
    """
    global _current_task

    deadline = time.monotonic() + timeout
    logged_wait = False

    while True:
        acquired = _lock.acquire(blocking=False)
        if acquired:
            break
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"task_lock timeout after {timeout}s waiting for [{_current_task}] to finish"
            )
        if not logged_wait:
            with _lock_meta:
                holder = _current_task
            console.print(f"[dim]⏳ Waiting for [{holder}] to finish before starting [{label}]…[/dim]")
            logged_wait = True
        time.sleep(0.5)

    with _lock_meta:
        _current_task = label

    console.print(f"[dim]🔒 Task lock acquired: [{label}][/dim]")
    try:
        yield
    finally:
        with _lock_meta:
            _current_task = ""
        _lock.release()
        console.print(f"[dim]🔓 Task lock released: [{label}][/dim]")
