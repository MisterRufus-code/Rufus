"""Retry utility with exponential backoff + jitter and circuit breaker."""

from __future__ import annotations

import functools
import random
import time
from typing import Callable, Type


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception], None] | None = None,
):
    """Decorator: retry *fn* up to *max_attempts* times with exponential backoff + jitter.

    Full-jitter formula: sleep = random(0, min(cap, base * 2^attempt))
    This avoids thundering-herd when multiple workers retry simultaneously.
    """
    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:
                    if attempt == max_attempts:
                        raise
                    cap = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    delay = random.uniform(0, cap)
                    if on_retry:
                        on_retry(attempt, exc)
                    time.sleep(delay)
        return wrapper
    return decorator


class CircuitBreaker:
    """Simple circuit breaker — opens after *threshold* consecutive failures,
    resets after *reset_timeout* seconds.

    Usage:
        cb = CircuitBreaker(threshold=5, reset_timeout=60)
        with cb:
            call_external_service()
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

    def __init__(self, threshold: int = 5, reset_timeout: float = 60.0):
        self._threshold = threshold
        self._reset_timeout = reset_timeout
        self._failures = 0
        self._opened_at: float | None = None
        self._state = self.CLOSED

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            if time.monotonic() - (self._opened_at or 0) >= self._reset_timeout:
                self._state = self.HALF_OPEN
        return self._state

    def __enter__(self):
        if self.state == self.OPEN:
            raise RuntimeError("Circuit breaker is OPEN — service unavailable")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is None:
            self._failures = 0
            self._state = self.CLOSED
        else:
            self._failures += 1
            if self._failures >= self._threshold:
                self._state = self.OPEN
                self._opened_at = time.monotonic()
        return False
