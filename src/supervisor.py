"""
Rufus Supervisor — autonomous pipeline brain.

Runs as a daemon, decides when to fire each niche, picks trending topics,
submits isolated pipeline jobs, monitors health & retries failures, and
learns from the ML feedback loop to prioritise better-performing niches.

Usage:
    python main.py supervisor                    # start daemon
    python main.py supervisor --once finance     # single run then exit
    python main.py supervisor --status           # print state and exit
"""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

class JobStatus(str, Enum):
    RUNNING = "running"
    DONE    = "done"
    FAILED  = "failed"
    TIMEOUT = "timeout"


# Pipeline stage markers — matched against job log output
_STAGE_PATTERNS = [
    (r"Step 1",           "1/10 Trend research"),
    (r"Step 2",           "2/10 AI keywords"),
    (r"Step 3",           "3/10 Footage download"),
    (r"Step 4",           "4/10 Indexing media"),
    (r"Step 5",           "5/10 Idea generation"),
    (r"Step 6b",          "6b/10 Hook optimisation"),
    (r"Step 6",           "6/10 Script writing"),
    (r"Step 7",           "7/10 Semantic matching"),
    (r"Step 8",           "8/10 Entropy check"),
    (r"Step 9",           "9/10 TTS voiceover"),
    (r"Step 10",          "10/10 Rendering video"),
    (r"Step 11",          "10/10 Subtitles"),
    (r"Step 12",          "10/10 Uploading"),
    (r"pipeline finished","✓ Done"),
    (r"pipeline resuming","↩ Resuming checkpoint"),
]

import re as _re
_STAGE_RE = [((_re.compile(p, _re.IGNORECASE)), label) for p, label in _STAGE_PATTERNS]

SPINNER_FRAMES = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]


@dataclass
class Job:
    job_id:        str
    niche:         str
    topic:         str
    mode:          str          # "both" | "long" | "shorts"
    status:        JobStatus    = JobStatus.RUNNING
    started_at:    Optional[datetime] = None
    finished_at:   Optional[datetime] = None
    pid:           Optional[int]      = None
    attempt:       int                = 1
    error:         Optional[str]      = None
    log_path:      Optional[Path]     = None
    current_stage: str                = "starting…"
    proc:          Optional[subprocess.Popen] = field(default=None, repr=False)

    def runtime_str(self) -> str:
        if not self.started_at:
            return "—"
        delta = (self.finished_at or datetime.now()) - self.started_at
        m = int(delta.total_seconds() // 60)
        s = int(delta.total_seconds() % 60)
        return f"{m}m{s:02d}s"

    def refresh_stage(self) -> None:
        """Parse the tail of the job log to find the current pipeline stage."""
        if not self.log_path or not self.log_path.exists():
            return
        try:
            with open(self.log_path, "rb") as f:
                # Read last 4 KB — enough to catch the latest stage line
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 4096))
                tail = f.read().decode("utf-8", errors="replace")
            for pattern, label in _STAGE_RE:
                if pattern.search(tail):
                    self.current_stage = label
        except Exception:
            pass


@dataclass
class SupervisorConfig:
    niches:              list[str] = field(
        default_factory=lambda: ["finance", "tech", "health", "general"]
    )
    mode:                str   = "both"        # pipeline mode per run
    check_interval:      int   = 30            # seconds between loop ticks
    run_interval_hours:  float = 6.0           # hours between runs per niche
    max_runtime:         int   = 5400          # kill job after 90 min
    max_retries:         int   = 3             # max consecutive failures per niche
    retry_backoff_min:   int   = 10            # base backoff minutes on failure
    min_disk_gb:         float = 8.0           # pause if disk drops below this
    ollama_url:          str   = "http://localhost:11434"
    state_file:          Path  = Path("logs/supervisor_state.json")
    log_file:            Path  = Path("logs/supervisor.log")
    max_concurrent:      int   = 1             # parallel pipelines (1 = serial)


# ---------------------------------------------------------------------------
# Supervisor
# ---------------------------------------------------------------------------

class Supervisor:

    def __init__(self, config: SupervisorConfig):
        self.config = config
        self.jobs:           list[Job]            = []
        self.running:        bool                 = True
        self._last_run:      dict[str, datetime]  = {}
        self._fail_count:    dict[str, int]       = {}
        self._niche_scores:  dict[str, float]     = {}  # ML-learned performance

        self.config.state_file.parent.mkdir(parents=True, exist_ok=True)
        self.config.log_file.parent.mkdir(parents=True, exist_ok=True)

        self._load_state()
        self._load_ml_scores()

        signal.signal(signal.SIGINT,  self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load_state(self) -> None:
        if not self.config.state_file.exists():
            return
        try:
            data = json.loads(self.config.state_file.read_text())
            for niche, ts in data.get("last_run", {}).items():
                self._last_run[niche] = datetime.fromisoformat(ts)
            for niche, n in data.get("fail_count", {}).items():
                self._fail_count[niche] = n
        except Exception:
            pass

    def _save_state(self) -> None:
        data = {
            "last_run":   {n: dt.isoformat() for n, dt in self._last_run.items()},
            "fail_count": dict(self._fail_count),
            "updated":    datetime.now().isoformat(),
        }
        self.config.state_file.write_text(json.dumps(data, indent=2))

    def _load_ml_scores(self) -> None:
        """Pull average view-through-rate per niche from the ML feedback store."""
        try:
            from src.ml.feedback_store import FeedbackStore
            store = FeedbackStore()
            for niche in self.config.niches:
                rows = store.query(niche=niche, limit=20)
                if rows:
                    avg = sum(r.get("vtr", 0.5) for r in rows) / len(rows)
                    self._niche_scores[niche] = avg
        except Exception:
            pass  # ML store not available — use equal weights

    def _log(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}\n"
        with open(self.config.log_file, "a") as f:
            f.write(line)

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def _health_check(self) -> tuple[bool, str]:
        free_gb = shutil.disk_usage("/").free / 1024**3
        if free_gb < self.config.min_disk_gb:
            return False, f"low disk ({free_gb:.1f} GB free)"

        try:
            import httpx
            r = httpx.get(f"{self.config.ollama_url}/api/tags", timeout=5)
            if r.status_code != 200:
                return False, "Ollama not responding"
        except Exception:
            return False, "Ollama unreachable"

        return True, "ok"

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def _due_niches(self) -> list[str]:
        """Return niches whose next-run time has arrived, sorted by ML score."""
        now   = datetime.now()
        interval = timedelta(hours=self.config.run_interval_hours)
        due: list[str] = []

        for niche in self.config.niches:
            fails = self._fail_count.get(niche, 0)
            if fails >= self.config.max_retries:
                continue  # suspended until operator resets

            last = self._last_run.get(niche)
            # exponential backoff grows the effective interval on failure
            backoff = timedelta(minutes=self.config.retry_backoff_min * (2 ** min(fails, 4)))
            effective = max(interval, backoff)

            if last is None or (now - last) >= effective:
                due.append(niche)

        # Prioritise niches with higher ML performance scores
        due.sort(key=lambda n: self._niche_scores.get(n, 0.5), reverse=True)
        return due

    def _active_niches(self) -> set[str]:
        return {j.niche for j in self.jobs if j.status == JobStatus.RUNNING}

    # ------------------------------------------------------------------
    # Topic selection
    # ------------------------------------------------------------------

    def _pick_topic(self, niche: str) -> str:
        """
        Choose a trending topic for *niche*.
        Falls back gracefully if trends module is unavailable.
        """
        try:
            from src.trends import fetch_trending_topic
            topic = fetch_trending_topic(niche=niche)
            if topic:
                return topic
        except Exception:
            pass

        # Generic fallback per niche
        fallbacks = {
            "finance":  "top 5 money mistakes costing you thousands",
            "tech":     "AI tools that will change your life in 2025",
            "health":   "daily habits that transform your health in 30 days",
            "wellness": "morning routines of the most successful people",
            "general":  "shocking facts most people don't know",
        }
        return fallbacks.get(niche, f"top {niche} tips this week")

    # ------------------------------------------------------------------
    # Job lifecycle
    # ------------------------------------------------------------------

    def _start_job(self, niche: str, topic: str, attempt: int = 1) -> Job:
        job = Job(
            job_id     = str(uuid.uuid4())[:8],
            niche      = niche,
            topic      = topic,
            mode       = self.config.mode,
            started_at = datetime.now(),
            attempt    = attempt,
        )

        cmd = [sys.executable, "main.py", "pipeline",
               "--niche", niche, "--topic", topic]
        if self.config.mode == "both":
            cmd.append("--both")
        elif self.config.mode == "shorts":
            cmd.append("--shorts")

        log_path = self.config.log_file.parent / f"job_{job.job_id}.log"
        job.log_path = log_path
        try:
            log_fh = open(log_path, "w")
            proc = subprocess.Popen(
                cmd,
                stdout=log_fh, stderr=subprocess.STDOUT,
                text=True, cwd=Path(__file__).parent.parent,
            )
            job.pid  = proc.pid
            job.proc = proc
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error  = str(e)
            self._fail_count[niche] = self._fail_count.get(niche, 0) + 1

        self.jobs.append(job)
        self._last_run[niche] = datetime.now()
        self._save_state()
        self._log(f"START  niche={niche} topic='{topic}' pid={job.pid} attempt={attempt}")
        return job

    def _poll_jobs(self) -> None:
        """Advance state of every running job; detect timeouts."""
        now = datetime.now()
        for job in self.jobs:
            if job.status != JobStatus.RUNNING or job.proc is None:
                continue

            job.refresh_stage()
            ret = job.proc.poll()
            if ret is not None:
                job.finished_at = now
                if ret == 0:
                    job.status = JobStatus.DONE
                    job.current_stage = "✓ complete"
                    self._fail_count[job.niche] = 0      # reset on success
                    self._log(f"DONE   niche={job.niche} runtime={job.runtime_str()}")
                else:
                    job.status = JobStatus.FAILED
                    job.current_stage = "✗ failed"
                    self._fail_count[job.niche] = self._fail_count.get(job.niche, 0) + 1
                    self._log(f"FAIL   niche={job.niche} exit={ret} fails={self._fail_count[job.niche]}")
                self._save_state()
                continue

            elapsed = (now - job.started_at).total_seconds()
            if elapsed > self.config.max_runtime:
                job.proc.terminate()
                job.status      = JobStatus.TIMEOUT
                job.finished_at = now
                self._fail_count[job.niche] = self._fail_count.get(job.niche, 0) + 1
                self._save_state()
                self._log(f"TIMEOUT niche={job.niche} after {job.runtime_str()}")

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def _render_dashboard(self) -> Panel:
        now = datetime.now()

        # ── Active / recent jobs table ──────────────────────────────
        spin_frame = SPINNER_FRAMES[int(now.timestamp() * 4) % len(SPINNER_FRAMES)]

        t = Table(box=None, header_style="bold cyan", show_edge=False)
        t.add_column("ID",      style="dim",  width=9)
        t.add_column("Niche",                 width=10)
        t.add_column("Stage",                 width=22)
        t.add_column("Status",                width=12)
        t.add_column("Elapsed",               width=8)
        t.add_column("#", style="dim",        width=3)

        STATUS_STYLE = {
            JobStatus.RUNNING: f"[yellow]{spin_frame} running[/yellow]",
            JobStatus.DONE:    "[green]✓ done[/green]",
            JobStatus.FAILED:  "[red]✗ failed[/red]",
            JobStatus.TIMEOUT: "[red]⏱ timeout[/red]",
        }

        for job in reversed(self.jobs[-15:]):
            stage = job.current_stage if job.status == JobStatus.RUNNING else ""
            t.add_row(
                job.job_id,
                job.niche,
                stage,
                STATUS_STYLE[job.status],
                job.runtime_str(),
                str(job.attempt),
            )

        # ── Niche schedule panel ────────────────────────────────────
        lines: list[str] = []
        for niche in self.config.niches:
            last  = self._last_run.get(niche)
            fails = self._fail_count.get(niche, 0)
            score = self._niche_scores.get(niche)

            if fails >= self.config.max_retries:
                eta_str = "[red]SUSPENDED[/red]"
            elif last is None:
                eta_str = "[green]now[/green]"
            else:
                nxt = last + timedelta(hours=self.config.run_interval_hours)
                secs = max(0, (nxt - now).total_seconds())
                h, rem = divmod(int(secs), 3600)
                m = rem // 60
                eta_str = f"{h}h{m:02d}m" if h else f"{m}m"

            score_str = f"  score={score:.2f}" if score is not None else ""
            fail_str  = f"  [red]⚠ {fails} fail(s)[/red]" if fails else ""
            lines.append(
                f"  [cyan]{niche:<12}[/cyan]"
                f" next: [white]{eta_str:<12}[/white]"
                f"{score_str}{fail_str}"
            )

        # ── Health ─────────────────────────────────────────────────
        free_gb = shutil.disk_usage("/").free / 1024**3
        health  = f"disk [green]{free_gb:.0f} GB[/green]"

        footer = (
            f"[dim]Health: {health}  |  Updated: {now.strftime('%H:%M:%S')}[/dim]"
        )
        body = Group(
            Text("Jobs (last 15):", style="bold"),
            t,
            Text(""),
            Text("Niche schedule:", style="bold"),
            Text.from_markup("\n".join(lines)),
            Text(""),
            Text.from_markup(footer),
        )
        return Panel(body, title="[bold green]Rufus Supervisor[/bold green]",
                     border_style="green")

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _handle_shutdown(self, *_) -> None:
        console.print("\n[yellow]Supervisor shutting down — waiting for jobs...[/yellow]")
        self.running = False
        for job in self.jobs:
            if job.status == JobStatus.RUNNING and job.proc:
                job.proc.terminate()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_once(self, niche: str) -> bool:
        """Run a single pipeline for *niche* and return True on success."""
        ok, reason = self._health_check()
        if not ok:
            console.print(f"[red]Health check failed: {reason}[/red]")
            return False

        topic = self._pick_topic(niche)
        console.print(f"[cyan]Running {niche}: {topic}[/cyan]")
        job = self._start_job(niche, topic)
        if job.status == JobStatus.FAILED:
            return False

        while job.proc and job.proc.poll() is None:
            time.sleep(5)

        self._poll_jobs()
        return job.status == JobStatus.DONE

    def run(self) -> None:
        """Main daemon loop."""
        console.print(
            f"[bold green]Rufus Supervisor started[/bold green]  "
            f"niches={self.config.niches}  interval={self.config.run_interval_hours}h\n"
            f"Press [bold]Ctrl+C[/bold] to stop."
        )

        with Live(self._render_dashboard(), refresh_per_second=0.5,
                  console=console) as live:
            while self.running:
                # 1. Update running jobs
                self._poll_jobs()

                # 2. Check concurrency budget
                active_count = len(self._active_niches())
                slots = self.config.max_concurrent - active_count

                if slots > 0:
                    # 3. Find niches due to run
                    due = [n for n in self._due_niches()
                           if n not in self._active_niches()]

                    for niche in due[:slots]:
                        ok, reason = self._health_check()
                        if not ok:
                            self._log(f"PAUSE  health={reason}")
                            break

                        topic = self._pick_topic(niche)
                        attempt = self._fail_count.get(niche, 0) + 1
                        self._start_job(niche, topic, attempt=attempt)

                live.update(self._render_dashboard())
                time.sleep(self.config.check_interval)

        console.print("[yellow]Supervisor stopped.[/yellow]")

    def print_status(self) -> None:
        """Print current schedule state and exit."""
        console.print(self._render_dashboard())
