"""Tests for the process-level task lock."""

from __future__ import annotations

import threading
import time

import pytest


class TestTaskLock:
    def test_lock_acquired_and_released(self):
        from src.task_lock import task_lock, _lock
        with task_lock("test_task"):
            assert _lock.locked()
        assert not _lock.locked()

    def test_two_threads_block_each_other(self):
        from src.task_lock import task_lock
        entered = []
        exited = []
        barrier = threading.Event()

        def first():
            with task_lock("first"):
                entered.append("first")
                barrier.set()
                time.sleep(0.15)
                exited.append("first")

        def second():
            barrier.wait()
            with task_lock("second"):
                entered.append("second")
                exited.append("second")

        t1 = threading.Thread(target=first)
        t2 = threading.Thread(target=second)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        # second must have entered AFTER first exited
        assert exited[0] == "first"
        assert entered[1] == "second"

    def test_timeout_raises_runtime_error(self):
        from src.task_lock import task_lock
        event = threading.Event()

        def holder():
            with task_lock("holder"):
                event.set()
                time.sleep(2)

        t = threading.Thread(target=holder, daemon=True)
        t.start()
        event.wait()

        with pytest.raises(RuntimeError, match="timeout"):
            with task_lock("waiter", timeout=0.2):
                pass

    def test_exception_inside_releases_lock(self):
        from src.task_lock import task_lock, _lock
        try:
            with task_lock("exploding"):
                raise ValueError("boom")
        except ValueError:
            pass
        assert not _lock.locked()
