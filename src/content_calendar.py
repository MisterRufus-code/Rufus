# src/content_calendar.py
"""Weekly content calendar — plans uploads across channels and niches."""

from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

_CALENDAR_PATH = Path(os.getenv("RUFUS_CALENDAR_PATH", "data/content_calendar.json"))
_calendar_lock = threading.Lock()

_WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class CalendarSlot:
    channel_id: str
    niche: str
    topic: Optional[str]
    scheduled_at: datetime
    status: str  # "pending" | "done" | "skipped"
    video_id: Optional[str] = None


@dataclass
class WeeklyPlan:
    week_start: datetime
    slots: list[CalendarSlot]
    generated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


# ── Serialisation helpers ─────────────────────────────────────────────────────

def _slot_to_dict(slot: CalendarSlot) -> dict:
    d = asdict(slot)
    d["scheduled_at"] = slot.scheduled_at.isoformat()
    return d


def _slot_from_dict(d: dict) -> CalendarSlot:
    d = dict(d)
    d["scheduled_at"] = datetime.fromisoformat(d["scheduled_at"])
    return CalendarSlot(**d)


def _plan_to_dict(plan: WeeklyPlan) -> dict:
    return {
        "week_start": plan.week_start.isoformat(),
        "generated_at": plan.generated_at,
        "slots": [_slot_to_dict(s) for s in plan.slots],
    }


def _plan_from_dict(d: dict) -> WeeklyPlan:
    return WeeklyPlan(
        week_start=datetime.fromisoformat(d["week_start"]),
        generated_at=d.get("generated_at", datetime.utcnow().isoformat()),
        slots=[_slot_from_dict(s) for s in d.get("slots", [])],
    )


# ── Core logic ────────────────────────────────────────────────────────────────

def generate_weekly_plan(
    channels: list[str],
    niches: list[str],
    uploads_per_channel_per_week: int = 3,
) -> WeeklyPlan:
    """
    Generate a WeeklyPlan spreading uploads evenly across Mon–Sun.

    For each (channel, niche) pair, ``uploads_per_channel_per_week`` slots are
    created.  The preferred upload time comes from ``best_upload_time(niche)``
    in upload_optimizer; subsequent slots for the same niche are offset by
    whole-day increments so they never cluster on the same weekday.
    """
    from src.upload_optimizer import best_upload_time

    now = datetime.utcnow()
    # Align week_start to the most recent Monday (00:00 UTC)
    week_start = (now - timedelta(days=now.weekday())).replace(
        hour=0, minute=0, second=0, microsecond=0
    )

    slots: list[CalendarSlot] = []

    for channel_id in channels:
        for niche in niches:
            # Preferred anchor time for this niche
            anchor: datetime = best_upload_time(niche)
            anchor_hour = anchor.hour

            # Spread N uploads across 7 days; step = 7 // N (minimum 1)
            total = uploads_per_channel_per_week
            step = max(1, 7 // total)

            used_days: set[int] = set()
            anchor_weekday = anchor.weekday()

            for i in range(total):
                # Pick a weekday offset from anchor that hasn't been used yet
                candidate_day = (anchor_weekday + i * step) % 7
                # If that day is already taken, find the next free one
                while candidate_day in used_days:
                    candidate_day = (candidate_day + 1) % 7
                used_days.add(candidate_day)

                scheduled_at = (week_start + timedelta(days=candidate_day)).replace(
                    hour=anchor_hour, minute=0, second=0, microsecond=0
                )

                slots.append(CalendarSlot(
                    channel_id=channel_id,
                    niche=niche,
                    topic=None,
                    scheduled_at=scheduled_at,
                    status="pending",
                    video_id=None,
                ))

    # Sort chronologically
    slots.sort(key=lambda s: s.scheduled_at)

    return WeeklyPlan(week_start=week_start, slots=slots)


def get_next_slot(channel_id: str, niche: str) -> Optional[CalendarSlot]:
    """Return the earliest pending CalendarSlot for *channel_id* + *niche*, or None."""
    plan = load_plan()
    if plan is None:
        return None
    pending = [
        s for s in plan.slots
        if s.channel_id == channel_id and s.niche == niche and s.status == "pending"
    ]
    if not pending:
        return None
    return min(pending, key=lambda s: s.scheduled_at)


def mark_slot_done(channel_id: str, scheduled_at: datetime, video_id: str) -> None:
    """Mark a slot as done and record the *video_id*."""
    with _calendar_lock:
        plan = load_plan()
        if plan is None:
            return
        for slot in plan.slots:
            if (
                slot.channel_id == channel_id
                and slot.scheduled_at == scheduled_at
                and slot.status == "pending"
            ):
                slot.status = "done"
                slot.video_id = video_id
                break
        _atomic_write(plan)


# ── Persistence ───────────────────────────────────────────────────────────────

def save_plan(plan: WeeklyPlan) -> None:
    """Atomically save *plan* to the calendar JSON file."""
    with _calendar_lock:
        _atomic_write(plan)


def _atomic_write(plan: WeeklyPlan) -> None:
    """Write plan to disk using os.replace() for atomicity. Caller holds lock."""
    _CALENDAR_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _CALENDAR_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(_plan_to_dict(plan), indent=2))
    os.replace(tmp, _CALENDAR_PATH)


def load_plan() -> Optional[WeeklyPlan]:
    """Load and return the WeeklyPlan from disk, or None if not present."""
    if not _CALENDAR_PATH.exists():
        return None
    try:
        raw = json.loads(_CALENDAR_PATH.read_text())
        return _plan_from_dict(raw)
    except (json.JSONDecodeError, OSError, KeyError, TypeError):
        return None


# ── Display ───────────────────────────────────────────────────────────────────

def print_calendar(plan: WeeklyPlan) -> None:
    """Render the weekly schedule as a Rich table."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box
    except ImportError:
        # Graceful fallback when Rich is not installed
        print(f"Weekly plan — week of {plan.week_start.date()} (generated {plan.generated_at})")
        for s in plan.slots:
            day = _WEEKDAYS[s.scheduled_at.weekday()]
            print(
                f"  {day} {s.scheduled_at.strftime('%H:%M')} UTC | "
                f"{s.channel_id} | {s.niche} | {s.status}"
                + (f" | {s.video_id}" if s.video_id else "")
            )
        return

    console = Console()
    table = Table(
        title=f"Content Calendar — week of {plan.week_start.date()}",
        box=box.ROUNDED,
        show_lines=True,
    )
    table.add_column("Day", style="bold cyan", no_wrap=True)
    table.add_column("Time (UTC)", style="dim")
    table.add_column("Channel", style="magenta")
    table.add_column("Niche", style="yellow")
    table.add_column("Topic")
    table.add_column("Status")
    table.add_column("Video ID", style="dim")

    status_styles = {"pending": "white", "done": "green", "skipped": "red"}

    for slot in plan.slots:
        day_name = _WEEKDAYS[slot.scheduled_at.weekday()]
        time_str = slot.scheduled_at.strftime("%H:%M")
        style = status_styles.get(slot.status, "white")
        table.add_row(
            day_name,
            time_str,
            slot.channel_id,
            slot.niche,
            slot.topic or "—",
            f"[{style}]{slot.status}[/{style}]",
            slot.video_id or "—",
        )

    console.print(table)
    console.print(f"[dim]Generated: {plan.generated_at}[/dim]")
