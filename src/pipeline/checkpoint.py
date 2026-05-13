"""Pipeline checkpoint — saves stage progress to {output_dir}/.checkpoint.json.

On crash at step 8 (TTS), re-running the same topic resumes from step 8.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Optional


class PipelineStage(str, Enum):
    STARTED = "STARTED"
    TRENDS = "TRENDS"
    KEYWORDS = "KEYWORDS"
    FOOTAGE_DOWNLOADED = "FOOTAGE_DOWNLOADED"
    MEDIA_INDEXED = "MEDIA_INDEXED"
    IDEAS_GENERATED = "IDEAS_GENERATED"
    SCRIPT_GENERATED = "SCRIPT_GENERATED"
    HOOK_OPTIMISED = "HOOK_OPTIMISED"
    SCENES_MATCHED = "SCENES_MATCHED"
    ENTROPY_CHECKED = "ENTROPY_CHECKED"
    TTS_COMPLETED = "TTS_COMPLETED"
    VIDEO_RENDERED = "VIDEO_RENDERED"
    THUMBNAIL_DONE = "THUMBNAIL_DONE"
    UPLOADED = "UPLOADED"
    COMPLETED = "COMPLETED"


_STAGE_ORDER: list[str] = [s.value for s in PipelineStage]


@dataclass
class CheckpointData:
    run_id: str
    stage: PipelineStage
    topic: str
    niche: str
    output_dir: str
    payload: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: _now_iso())
    updated_at: str = field(default_factory=lambda: _now_iso())


def _now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _checkpoint_to_dict(cp: CheckpointData) -> dict:
    d = asdict(cp)
    d["stage"] = cp.stage.value
    return d


def _checkpoint_from_dict(d: dict) -> CheckpointData:
    d = dict(d)
    d["stage"] = PipelineStage(d["stage"])
    return CheckpointData(**d)


class CheckpointManager:
    """Thread-safe checkpoint manager for a single pipeline run."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)
        self._lock = threading.Lock()

    @property
    def _path(self) -> Path:
        return self._output_dir / ".checkpoint.json"

    def exists(self) -> bool:
        return self._path.exists()

    def load(self) -> Optional[CheckpointData]:
        """Return CheckpointData or None if file missing or corrupt."""
        with self._lock:
            if not self._path.exists():
                return None
            try:
                raw = self._path.read_text(encoding="utf-8")
                data = json.loads(raw)
                return _checkpoint_from_dict(data)
            except Exception:
                return None

    def save(
        self,
        stage: PipelineStage,
        payload: dict | None = None,
    ) -> CheckpointData:
        """Create or update the checkpoint atomically.

        Preserves run_id and created_at from the first save.
        Updates updated_at on every call.
        """
        if payload is None:
            payload = {}

        with self._lock:
            existing = None
            if self._path.exists():
                try:
                    existing = _checkpoint_from_dict(
                        json.loads(self._path.read_text(encoding="utf-8"))
                    )
                except Exception:
                    existing = None

            now = _now_iso()
            if existing is not None:
                # Merge payload — new values overwrite old ones
                merged_payload = {**existing.payload, **payload}
                cp = CheckpointData(
                    run_id=existing.run_id,
                    stage=stage,
                    topic=existing.topic,
                    niche=existing.niche,
                    output_dir=existing.output_dir,
                    payload=merged_payload,
                    created_at=existing.created_at,
                    updated_at=now,
                )
            else:
                cp = CheckpointData(
                    run_id=str(uuid.uuid4()),
                    stage=stage,
                    topic="",
                    niche="",
                    output_dir=str(self._output_dir),
                    payload=payload,
                    created_at=now,
                    updated_at=now,
                )

            self._output_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(_checkpoint_to_dict(cp), indent=2), encoding="utf-8")
            os.replace(tmp, self._path)

        return cp

    def save_full(
        self,
        stage: PipelineStage,
        topic: str,
        niche: str,
        payload: dict | None = None,
    ) -> CheckpointData:
        """Save checkpoint with explicit topic and niche (used for first save)."""
        if payload is None:
            payload = {}

        with self._lock:
            existing = None
            if self._path.exists():
                try:
                    existing = _checkpoint_from_dict(
                        json.loads(self._path.read_text(encoding="utf-8"))
                    )
                except Exception:
                    existing = None

            now = _now_iso()
            if existing is not None:
                merged_payload = {**existing.payload, **payload}
                cp = CheckpointData(
                    run_id=existing.run_id,
                    stage=stage,
                    topic=topic or existing.topic,
                    niche=niche or existing.niche,
                    output_dir=str(self._output_dir),
                    payload=merged_payload,
                    created_at=existing.created_at,
                    updated_at=now,
                )
            else:
                cp = CheckpointData(
                    run_id=str(uuid.uuid4()),
                    stage=stage,
                    topic=topic,
                    niche=niche,
                    output_dir=str(self._output_dir),
                    payload=payload,
                    created_at=now,
                    updated_at=now,
                )

            self._output_dir.mkdir(parents=True, exist_ok=True)
            tmp = self._path.with_suffix(".tmp")
            tmp.write_text(json.dumps(_checkpoint_to_dict(cp), indent=2), encoding="utf-8")
            os.replace(tmp, self._path)

        return cp

    def advance(self, stage: PipelineStage, **payload_kwargs) -> None:
        """Shorthand for saving a stage with keyword payload args.

        Example::

            manager.advance(PipelineStage.KEYWORDS, keywords=["stocks"])
        """
        self.save(stage, payload=payload_kwargs)

    def clear(self) -> None:
        """Delete the checkpoint file (call on successful pipeline completion)."""
        with self._lock:
            try:
                self._path.unlink(missing_ok=True)
            except Exception:
                pass

    def is_done(self) -> bool:
        """True if the checkpoint stage is COMPLETED."""
        cp = self.load()
        return cp is not None and cp.stage == PipelineStage.COMPLETED

    def stage_index(self, stage: PipelineStage) -> int:
        """Return the 0-based position of *stage* in the ordered enum."""
        return _STAGE_ORDER.index(stage.value)

    def completed_stage(self, stage: PipelineStage) -> bool:
        """True if the saved checkpoint stage is >= *stage* in pipeline order."""
        cp = self.load()
        if cp is None:
            return False
        return self.stage_index(cp.stage) >= self.stage_index(stage)


# ---------------------------------------------------------------------------
# Module-level factory
# ---------------------------------------------------------------------------

_managers: dict[Path, CheckpointManager] = {}
_managers_lock = threading.Lock()


def get_or_create_manager(output_dir: Path) -> CheckpointManager:
    """Return (and cache) a CheckpointManager for *output_dir*."""
    output_dir = Path(output_dir).resolve()
    with _managers_lock:
        if output_dir not in _managers:
            _managers[output_dir] = CheckpointManager(output_dir)
        return _managers[output_dir]
