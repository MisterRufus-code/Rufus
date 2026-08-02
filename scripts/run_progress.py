#!/usr/bin/env python3
"""
run_progress.py — what the pipeline is doing RIGHT NOW, readable from outside.

The dashboard could already answer "is a run in progress?" (it checks main.py's
per-channel FileLock non-blockingly), but not "what is it DOING?" — and those
are very different answers when a run takes 5-45 minutes. Watching a lock file
tells you something is happening; it never tells you whether you're 30 seconds
or 40 minutes from a finished video, or that the run has been stuck on the same
step for half an hour.

main.py writes one small JSON file per channel as it moves through its seven
steps; the dashboard reads it. Deliberately a FILE and not shared memory or a
socket: a dashboard-launched run, a Task Scheduler run and a hand-run
run.bat are three unrelated OS processes, and a file is the only channel all
three already share.

CONTRACT: nothing in here may ever raise into the pipeline. Progress reporting
is observability — a video must never fail because a status file couldn't be
written. Every public function swallows its own errors.

  begin(channel, run_id, niche)   — a run started
  update(step, label)             — it reached step N of 7
  finish(status, detail)          — it ended (done / failed / cancelled)
  read(channel) / read_all()      — what the dashboard calls
"""

import json
import os
import time
from pathlib import Path

import paths

TOTAL_STEPS = 7

# A run whose progress file hasn't been touched in this long is reported as
# stale rather than live. Step 5 (render) legitimately runs for many minutes
# without an update, so this is generous — it exists to catch a process that
# died without calling finish(), not to police slow steps.
STALE_AFTER_SECONDS = 45 * 60

_current: dict = {}     # this process's own state, so update() needn't re-read


def _progress_dir() -> Path:
    return paths.log_dir() / "progress"


def _path(channel: str) -> Path:
    # Channel ids are config-controlled, but this builds a filesystem path, so
    # anything path-like is neutralised rather than trusted.
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (channel or "default"))
    return _progress_dir() / f"{safe}.json"


def _write(data: dict) -> None:
    try:
        d = _progress_dir()
        d.mkdir(parents=True, exist_ok=True)
        path = _path(data.get("channel", "default"))
        # Write-then-rename so a reader never sees a half-written file. The
        # dashboard polls this every few seconds; a torn read would show a
        # broken status bar for no reason.
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(data), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        pass    # observability must never break a render


def begin(channel: str, run_id: str = "", niche: str = "", topic: str = "") -> None:
    global _current
    _current = {
        "channel": channel or "default",
        "run_id": run_id,
        "niche": niche,
        "topic": topic,
        "step": 0,
        "total": TOTAL_STEPS,
        "label": "starting",
        "started_at": time.time(),
        "updated_at": time.time(),
        "pid": os.getpid(),
        "status": "running",
        "detail": "",
    }
    _write(_current)


def update(step: int, label: str) -> None:
    """Record that the run reached step `step` of 7. No-op if begin() wasn't
    called — a caller that skipped begin() (an odd entry point, a test) must
    not crash here."""
    if not _current:
        return
    _current["step"] = step
    _current["label"] = label
    _current["updated_at"] = time.time()
    _write(_current)


def finish(status: str = "done", detail: str = "") -> None:
    """Mark the run ended. `status` is done / failed / cancelled.

    Called from a finally-block, so it must tolerate being called when begin()
    never ran (an exception before the pipeline really started)."""
    if not _current:
        return
    _current["status"] = status
    _current["detail"] = (detail or "")[:300]
    _current["label"] = {"done": "finished", "failed": "failed",
                         "cancelled": "cancelled"}.get(status, status)
    _current["updated_at"] = time.time()
    _current["ended_at"] = time.time()
    _write(_current)


def read(channel: str) -> dict | None:
    """One channel's last known progress, or None. Adds derived fields the
    dashboard would otherwise recompute: age_seconds, elapsed_seconds, stale."""
    try:
        data = json.loads(_path(channel).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    now = time.time()
    data["age_seconds"] = max(0.0, now - float(data.get("updated_at") or now))
    started = float(data.get("started_at") or now)
    end = float(data.get("ended_at") or now)
    data["elapsed_seconds"] = max(0.0, end - started)
    # Stale means "claims to be running but nothing has touched it in a long
    # time" — i.e. the process almost certainly died without calling finish().
    data["stale"] = (data.get("status") == "running"
                     and data["age_seconds"] > STALE_AFTER_SECONDS)
    return data


def read_all() -> list[dict]:
    """Every channel's progress, newest activity first."""
    d = _progress_dir()
    if not d.is_dir():
        return []
    out = []
    for f in d.glob("*.json"):
        data = read(f.stem)
        if data:
            out.append(data)
    return sorted(out, key=lambda r: r.get("updated_at", 0), reverse=True)


def clear(channel: str) -> None:
    """Forget a channel's progress entirely (used when clearing a stale run)."""
    try:
        _path(channel).unlink(missing_ok=True)
    except OSError:
        pass
