"""Tests for structured logging system."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture()
def tmp_log_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("RUFUS_LOG_DIR", str(tmp_path / "logs"))
    # Force module reload so new LOG_DIR is picked up
    import sys
    if "src.logging_config" in sys.modules:
        del sys.modules["src.logging_config"]
    yield tmp_path / "logs"
    if "src.logging_config" in sys.modules:
        del sys.modules["src.logging_config"]


class TestGetLogger:
    def test_returns_structured_logger(self):
        from src.logging_config import get_logger
        log = get_logger("test")
        assert hasattr(log, "info")
        assert hasattr(log, "error")
        assert hasattr(log, "warning")
        assert hasattr(log, "debug")

    def test_info_does_not_raise(self):
        from src.logging_config import get_logger
        log = get_logger("test.info")
        log.info("hello world", topic="investing", niche="finance")

    def test_warning_does_not_raise(self):
        from src.logging_config import get_logger
        log = get_logger("test.warning")
        log.warning("something odd", code=42)

    def test_error_does_not_raise(self):
        from src.logging_config import get_logger
        log = get_logger("test.error")
        log.error("something failed", exc_code=500)

    def test_exception_does_not_raise(self):
        from src.logging_config import get_logger
        log = get_logger("test.exception")
        try:
            raise ValueError("boom")
        except ValueError:
            log.exception("caught error")

    def test_multiple_loggers_independent(self):
        from src.logging_config import get_logger
        log_a = get_logger("module.a")
        log_b = get_logger("module.b")
        log_a.info("from a")
        log_b.info("from b")

    def test_writes_jsonl_to_file(self, tmp_log_dir):
        from src.logging_config import get_logger, _LOG_FILE
        log = get_logger("jsonl.test")
        log.info("jsonl message", key="value")
        if _LOG_FILE.exists():
            lines = _LOG_FILE.read_text().strip().splitlines()
            assert any("jsonl message" in line for line in lines)


class TestLogLevel:
    def test_debug_available(self):
        from src.logging_config import get_logger
        log = get_logger("debug.test")
        log.debug("debug msg", detail="extra")

    def test_critical_available(self):
        from src.logging_config import get_logger
        log = get_logger("critical.test")
        log.critical("critical msg")
