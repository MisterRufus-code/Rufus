"""
Structured logging for Rufus.

Writes JSON lines to logs/rufus.jsonl and human-readable output to the console.
Import get_logger() anywhere — no global state mutation.

Usage:
    from src.logging_config import get_logger
    log = get_logger(__name__)
    log.info("pipeline started", topic="investing", niche="finance")
"""

from __future__ import annotations

import json
import logging
import os
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

# ── log file location ─────────────────────────────────────────────────────────
_LOG_DIR = Path(os.getenv("RUFUS_LOG_DIR", "logs"))
_LOG_FILE = _LOG_DIR / "rufus.jsonl"
_LOG_LEVEL = os.getenv("RUFUS_LOG_LEVEL", "INFO").upper()

# ── internal JSON file handler ────────────────────────────────────────────────

class _JsonlHandler(logging.Handler):
    """Appends one JSON object per line to the log file."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path

    def emit(self, record: logging.LogRecord) -> None:
        entry: dict[str, Any] = {
            "ts": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Structured fields attached via log.info("msg", **kwargs)
        for k, v in record.__dict__.items():
            if k not in logging.LogRecord.__dict__ and not k.startswith("_"):
                entry[k] = v
        if record.exc_info:
            entry["exc"] = traceback.format_exception(*record.exc_info)
        try:
            with self._path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, default=str) + "\n")
        except OSError:
            pass  # never crash the application due to logging


# ── root Rufus logger ─────────────────────────────────────────────────────────

_root = logging.getLogger("rufus")
_root.setLevel(getattr(logging, _LOG_LEVEL, logging.INFO))

if not _root.handlers:
    # JSON file handler
    _root.addHandler(_JsonlHandler(_LOG_FILE))
    # Console handler — plain text, not JSON
    _console_handler = logging.StreamHandler(sys.stdout)
    _console_handler.setFormatter(
        logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
                          datefmt="%H:%M:%S")
    )
    _root.addHandler(_console_handler)
    _root.propagate = False


class _StructuredLogger:
    """Thin wrapper that lets callers pass keyword args as structured fields."""

    def __init__(self, name: str) -> None:
        self._log = logging.getLogger(f"rufus.{name}")

    def _emit(self, level: int, msg: str, **kwargs: Any) -> None:
        if self._log.isEnabledFor(level):
            record = self._log.makeRecord(
                self._log.name, level, "(unknown)", 0, msg, (), None
            )
            record.__dict__.update(kwargs)
            self._log.handle(record)

    def debug(self, msg: str, **kw: Any) -> None:
        self._emit(logging.DEBUG, msg, **kw)

    def info(self, msg: str, **kw: Any) -> None:
        self._emit(logging.INFO, msg, **kw)

    def warning(self, msg: str, **kw: Any) -> None:
        self._emit(logging.WARNING, msg, **kw)

    def error(self, msg: str, **kw: Any) -> None:
        self._emit(logging.ERROR, msg, **kw)

    def critical(self, msg: str, **kw: Any) -> None:
        self._emit(logging.CRITICAL, msg, **kw)

    def exception(self, msg: str, **kw: Any) -> None:
        kw["exc_info"] = sys.exc_info()
        self._emit(logging.ERROR, msg, **kw)


def get_logger(name: str) -> _StructuredLogger:
    """Return a structured logger scoped to *name*."""
    return _StructuredLogger(name)
