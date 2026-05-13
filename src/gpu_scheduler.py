"""GPU scheduler — tracks VRAM budget across model singletons.

Prevents CLIP + BLIP + YOLO + Kokoro from all loading at once and OOM-crashing.
Uses LRU eviction when budget is exceeded.

VRAM estimates (MB):
    CLIP_ViT_B32  = 600
    BLIP_BASE     = 900
    YOLO_N        = 150
    KOKORO_ONNX   = 200
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Generator, Optional

from src.logging_config import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# VRAM estimates (MB) — document these here so callers don't hard-code numbers
# ---------------------------------------------------------------------------
CLIP_ViT_B32: int = 600
BLIP_BASE: int = 900
YOLO_N: int = 150
KOKORO_ONNX: int = 200


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class ModelSlot:
    name: str
    estimated_vram_mb: int
    loaded_at: float = field(default_factory=time.monotonic)
    last_used: float = field(default_factory=time.monotonic)


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class GPUScheduler:
    """
    Tracks which ML models are loaded and how much VRAM they consume.

    Automatically evicts the least-recently-used model when the budget is
    exceeded so that new models can load without OOM errors.

    In CPU mode (no CUDA or budget==0) all operations are no-ops.
    """

    def __init__(self, vram_budget_mb: Optional[int] = None) -> None:
        if vram_budget_mb is not None:
            self._budget_mb: int = vram_budget_mb
        else:
            try:
                import torch  # type: ignore[import]
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    self._budget_mb = int(props.total_memory / 1024 ** 2)
                    log.info(
                        "GPU scheduler: auto-detected VRAM budget",
                        budget_mb=self._budget_mb,
                    )
                else:
                    self._budget_mb = 0
                    log.info("GPU scheduler: CUDA unavailable — running in CPU mode (no eviction)")
            except Exception as exc:
                self._budget_mb = 0
                log.warning(
                    "GPU scheduler: could not query CUDA properties, defaulting to CPU mode",
                    error=str(exc),
                )

        self._slots: dict[str, ModelSlot] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self, model_name: str, estimated_vram_mb: int) -> bool:
        """
        Request VRAM for *model_name*.

        Registers the slot immediately if budget allows.  If over-budget,
        LRU eviction runs until there is enough room.  In CPU mode always
        returns True without tracking anything.

        Returns:
            True — always (either fits cleanly or space was made via eviction).
        """
        if self._budget_mb == 0:
            return True  # CPU mode — no tracking needed

        with self._lock:
            # Already loaded — just refresh the timestamp
            if model_name in self._slots:
                self._slots[model_name].last_used = time.monotonic()
                log.debug("GPU scheduler: model already acquired, refreshed timestamp", model=model_name)
                return True

            # Evict until there is enough free VRAM
            while (
                self._budget_mb > 0
                and self.current_usage_mb() + estimated_vram_mb > self._budget_mb
                and self._slots
            ):
                self._evict_lru()

            slot = ModelSlot(
                name=model_name,
                estimated_vram_mb=estimated_vram_mb,
            )
            self._slots[model_name] = slot
            log.info(
                "GPU scheduler: model acquired",
                model=model_name,
                vram_mb=estimated_vram_mb,
                total_used_mb=self.current_usage_mb(),
                budget_mb=self._budget_mb,
            )
        return True

    def release(self, model_name: str) -> None:
        """Remove a model's VRAM reservation."""
        if self._budget_mb == 0:
            return

        with self._lock:
            slot = self._slots.pop(model_name, None)
            if slot is not None:
                log.info(
                    "GPU scheduler: model released",
                    model=model_name,
                    vram_mb=slot.estimated_vram_mb,
                    total_used_mb=self.current_usage_mb(),
                )
            else:
                log.debug("GPU scheduler: release called for unknown model (no-op)", model=model_name)

    def touch(self, model_name: str) -> None:
        """Update the last-used timestamp for *model_name* (call when the model is used)."""
        if self._budget_mb == 0:
            return
        with self._lock:
            if model_name in self._slots:
                self._slots[model_name].last_used = time.monotonic()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _evict_lru(self) -> None:
        """Remove the slot with the oldest *last_used* timestamp. Must hold _lock."""
        if not self._slots:
            return
        lru_name = min(self._slots, key=lambda k: self._slots[k].last_used)
        evicted = self._slots.pop(lru_name)
        log.info(
            "GPU scheduler: evicted LRU model",
            model=lru_name,
            freed_mb=evicted.estimated_vram_mb,
            age_seconds=round(time.monotonic() - evicted.last_used, 1),
        )

    # ------------------------------------------------------------------
    # Status / introspection
    # ------------------------------------------------------------------

    def current_usage_mb(self) -> int:
        """Sum of estimated VRAM for all loaded slots (does NOT acquire _lock)."""
        return sum(s.estimated_vram_mb for s in self._slots.values())

    def vram_status(self) -> dict:
        """
        Snapshot of the scheduler's current state.

        Returns a dict with keys: budget_mb, used_mb, free_mb, slots.
        """
        with self._lock:
            used = self.current_usage_mb()
            return {
                "budget_mb": self._budget_mb,
                "used_mb": used,
                "free_mb": max(0, self._budget_mb - used),
                "slots": list(self._slots.keys()),
            }

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    @contextmanager
    def model_context(
        self, model_name: str, estimated_vram_mb: int
    ) -> Generator[None, None, None]:
        """
        Acquire VRAM on enter, release on exit.

        Usage::

            with scheduler.model_context("clip", CLIP_ViT_B32):
                embeddings = clip_model(frames)
        """
        self.acquire(model_name, estimated_vram_mb)
        try:
            yield
        finally:
            self.release(model_name)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_scheduler: Optional[GPUScheduler] = None


def get_gpu_scheduler() -> GPUScheduler:
    """Return the process-wide GPUScheduler singleton, creating it on first call."""
    global _scheduler
    if _scheduler is None:
        _scheduler = GPUScheduler()
    return _scheduler
