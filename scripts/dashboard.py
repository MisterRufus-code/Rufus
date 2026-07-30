#!/usr/bin/env python3
"""
dashboard.py — Rufus's approval queue and status dashboard: one page
answering "what's actually happening with this pipeline," without scrolling
PowerShell logs — and the ONLY place a video can be approved for upload.

main.py no longer auto-uploads anything (RUFUS_AUTO_UPLOAD=1 is the explicit
opt-out back to the old behavior). Every rendered video lands here as
'pending'; a reviewer (you, or whoever you've shared this with) clicks
Approve to actually publish it, or Reject to never publish it. Title/
description are editable before that click.

Reads/writes rufus.db (WAL mode — safe alongside a live main.py run) and
reads media_library/debug/<run_id>/ for the per-run script/voiceover/
keyframes. No external assets (self-contained HTML/CSS/inline SVG — nothing
to break when you're not on the local network). No login of its own.

DEFAULT IS LOOPBACK-ONLY (127.0.0.1) — not reachable from your home WiFi at
all, only from this PC. This page also has "/system" routes that can START
AND KILL PROCESSES on this machine (launch a run, stop ComfyUI), so it needs
more than "no login" once those exist — loopback-only means literally
nothing on the network can reach ANY route here, including those.

Run:
    python scripts\\dashboard.py
    → http://localhost:8765            (this machine only, by default)

For access from your phone or another device, do NOT change
RUFUS_DASHBOARD_HOST to 0.0.0.0 and do NOT port-forward — either one puts
the process-control routes on your open WiFi/the internet with no login.
Use Tailscale instead (free, ~2 min):
    1. Install Tailscale on this PC and on your phone, sign into the same account.
    2. On this PC:  tailscale serve --bg 8765
       (proxies http://127.0.0.1:8765 onto your private tailnet over https,
       auto-renewed cert, only devices signed into YOUR tailnet can reach it —
       the dashboard itself never has to leave loopback.)
    3. On your phone (with Tailscale connected): open the https URL
       `tailscale serve status` prints.
    `tailscale serve --bg off` to stop sharing it again.

Environment:
  RUFUS_DASHBOARD_HOST   127.0.0.1 (default — loopback-only; 0.0.0.0 opens
                         it to your whole LAN, not recommended once
                         process-control routes exist — use tailscale serve
                         instead)
  RUFUS_DASHBOARD_PORT   8765
"""

import html
import json
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import paths
from urllib.parse import quote as _urlquote

import requests
from filelock import FileLock, Timeout
from flask import Flask, abort, redirect, request, send_from_directory

sys.path.insert(0, str(Path(__file__).parent))
import db_manager

ROOT       = Path(__file__).parent.parent
DEBUG_ROOT = paths.debug_root()

app = Flask(__name__)

UPLOAD_THRESHOLD_DEFAULT = 8   # visual reference line on the score sparkline

# Hard floor enforced in approve_video() below — no video below this score
# can be approved for upload, even by a human clicking the button, whether
# it's a misclick or a reviewer other than you (Tailscale-shared access).
# Keep in sync with main.py's HARD_MIN_UPLOAD_SCORE (duplicated, not
# imported, so this file has zero import-time dependency on main.py).
HARD_MIN_UPLOAD_SCORE = 7


# ── Process control (/system) — status, launch, cancel ───────────────────────
# Loopback-only binding (see module docstring) already keeps these off the
# network by default; _require_localhost() is defense-in-depth against
# someone later widening RUFUS_DASHBOARD_HOST without re-reading why.

def _require_localhost() -> None:
    if request.remote_addr not in ("127.0.0.1", "::1"):
        abort(403)


def _comfy_host() -> str:
    # Duplicated from comfy_client._host() rather than imported — same
    # reasoning as HARD_MIN_UPLOAD_SCORE above: this file stays free of an
    # import-time dependency on the ComfyUI client module.
    return os.environ.get("COMFY_HOST", "http://localhost:8188").rstrip("/")


def _comfyui_reachable() -> bool:
    try:
        r = requests.get(f"{_comfy_host()}/system_stats", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def _lock_path(channel_id: str) -> Path:
    # Exact naming main.py's _acquire_lock() uses — same lock, checked
    # non-blockingly instead of held.
    return ROOT / f"rufus.{channel_id}.lock.lock"


def _run_in_progress(channel_id: str) -> bool:
    """True if main.py (any entry point, anywhere) currently holds this
    channel's run lock. Works regardless of who started the run — a manual
    run.bat, Task Scheduler, or this dashboard — because it checks the same
    FileLock main.py itself uses, non-blockingly."""
    lock = FileLock(str(_lock_path(channel_id)))
    try:
        lock.acquire(timeout=0)
    except Timeout:
        return True
    except Exception:
        return False   # fail open — never let a lock-check bug block the page
    lock.release()
    return False


# ── Settings editor (/settings) ──────────────────────────────────────────────
# Most Rufus tunables are read via os.environ.get(...) in the process that's
# actually running — this dashboard editing a value can't retroactively
# change an already-running process, and won't reach a Task-Scheduler-
# launched run_scheduled.bat unless that file is separately updated to
# source this same JSON (documented in the /settings page itself, not done
# automatically here). What it DOES do immediately: every run this
# dashboard launches (_launch_run below — /system/run, /request-topic,
# /trending's queue button) picks these up as env overrides.

SETTINGS_FILE = ROOT / "config" / "dashboard_settings.json"

# (env var, label, kind, help). kind: "bool" (tri-state — blank means "don't
# override, use whatever's already configured") or "select:opt1,opt2,...".
SETTINGS_SCHEMA = [
    ("RUFUS_STILLS_ONLY", "Stills only", "bool",
     "Force Ken Burns on stills only, overriding Hunyuan/Wan/LTX/SVD all at once."),
    ("RUFUS_RENDERER", "Renderer", "select:ffmpeg,remotion",
     "remotion needs `cd remotion && npm install` once; falls back to ffmpeg on any failure."),
    ("RUFUS_LTX", "LTX motion engine", "bool", "Enable/disable the LTX-2.3 engine."),
    ("RUFUS_HUNYUAN", "Hunyuan motion engine", "bool", "Enable/disable the Hunyuan engine."),
    ("RUFUS_WAN", "Wan motion engine", "bool", "Enable/disable the Wan engine."),
]


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text())
    except (OSError, json.JSONDecodeError, ValueError):
        return {}


def _save_settings(values: dict) -> None:
    SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    SETTINGS_FILE.write_text(json.dumps(values, indent=2))


# Processes THIS dashboard launched, keyed by channel id — the only ones it
# can cancel. A run started elsewhere (Task Scheduler, a manual run.bat) has
# no Popen handle here; _run_in_progress() above still reports it as
# running (shared lock file), but cancelling it needs the actual process
# handle, which only exists for runs this page started.
_LAUNCHED: dict[str, subprocess.Popen] = {}


def _launch_run(*, niche: str | None = None, topic: str | None = None,
                channel: str | None = None) -> tuple[subprocess.Popen, Path]:
    """Fire-and-forget: a genuinely separate OS process, not an in-process
    call — this Flask app runs threaded=False (see approve_video's
    _scoped_env note), so a call that blocks for the 5-45+ minutes a video
    can take would freeze every other request. No --skip-upload: safety
    comes from the review-queue gate itself, same as run_scheduled.bat.
    Output goes to a log file (not the dashboard's own stdout) so a
    dashboard-launched run doesn't interleave console output with whatever
    else is running; the single shared launch path behind both
    /request-topic and /system/run. Settings saved via /settings layer on
    top of the current env (they don't replace it) as overrides for THIS
    child process only."""
    cmd = [sys.executable, str(ROOT / "scripts" / "main.py")]
    if niche:
        cmd += ["--niche", niche]
    if topic:
        cmd += ["--topic", topic]
    if channel:
        cmd += ["--channel", channel]
    env = os.environ.copy()
    env.update(_load_settings())
    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"dashboard_run_{int(time.time())}.log"
    with open(log_path, "wb") as logf:
        proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf, env=env,
                               stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    _LAUNCHED[channel or "default"] = proc
    return proc, log_path


def _cancel_run(channel: str | None = None) -> bool:
    """Terminate a run this dashboard launched. Returns False (no-op) for a
    run it has no handle to, or one that already finished."""
    proc = _LAUNCHED.get(channel or "default")
    if proc is None or proc.poll() is not None:
        return False
    try:
        proc.terminate()
    except OSError:
        return False
    return True


# ── Data access (read-only) ───────────────────────────────────────────────────

def _channels() -> list[str]:
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import channel_config
        return channel_config.list_channels()
    except Exception:
        return []


def _recent_videos(limit: int = 60, channel: str | None = None,
                   status: str | None = None) -> list[dict]:
    q = ("SELECT id, upload_date, niche, script_hook, title, score, "
         "hold_reason, youtube_id, run_id, channel, upload_status FROM videos")
    where, args = [], []
    if channel:
        where.append("channel = ?"); args.append(channel)
    if status:
        where.append("upload_status = ?"); args.append(status)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    try:
        with db_manager._conn() as c:
            rows = c.execute(q, args).fetchall()
    except Exception:
        return []
    cols = ["id", "upload_date", "niche", "script_hook", "title", "score",
            "hold_reason", "youtube_id", "run_id", "channel", "upload_status"]
    return [dict(zip(cols, r)) for r in rows]


def _video_detail(video_id: int) -> dict | None:
    q = ("SELECT id, upload_date, niche, script_hook, script_full, scene_desc, "
         "seed_type, seed_source, seed_content, youtube_id, video_file, score, "
         "run_id, score_specificity, score_hook, score_compression, score_loop, "
         "score_human, attempts_used, final_temperature, score_reasoning, "
         "title, channel, hold_reason, description, upload_status "
         "FROM videos WHERE id = ?")
    try:
        with db_manager._conn() as c:
            row = c.execute(q, (video_id,)).fetchone()
    except Exception:
        return None
    if not row:
        return None
    cols = ["id", "upload_date", "niche", "script_hook", "script_full",
            "scene_desc", "seed_type", "seed_source", "seed_content",
            "youtube_id", "video_file", "score", "run_id",
            "score_specificity", "score_hook", "score_compression",
            "score_loop", "score_human", "attempts_used", "final_temperature",
            "score_reasoning", "title", "channel", "hold_reason",
            "description", "upload_status"]
    return dict(zip(cols, row))


def _stats(limit: int = 100, channel: str | None = None) -> dict:
    rows = _recent_videos(limit=limit, channel=channel)
    total = len(rows)
    if not total:
        return {"total": 0, "avg_score": 0.0, "hold_rate": 0.0,
                "uploaded": 0, "held": 0, "pending": 0, "rejected": 0}
    scores = [r["score"] for r in rows if r["score"] is not None]
    pending  = sum(1 for r in rows if r["upload_status"] == "pending")
    approved = sum(1 for r in rows if r["upload_status"] == "approved")
    rejected = sum(1 for r in rows if r["upload_status"] == "rejected")
    return {
        "total": total,
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0.0,
        "hold_rate": round(100 * pending / total, 1),   # legacy key, kept for template compat
        "uploaded": approved,
        "held": pending,
        "pending": pending,
        "rejected": rejected,
    }


def _top_rejections(limit: int = 8, channel: str | None = None) -> list[dict]:
    q = ("SELECT rejected_reason, COUNT(*) c FROM script_attempts "
         "WHERE accepted = 0 AND rejected_reason IS NOT NULL AND rejected_reason != ''")
    args: list = []
    if channel:
        q += " AND channel = ?"
        args.append(channel)
    q += " GROUP BY rejected_reason ORDER BY c DESC LIMIT ?"
    args.append(limit)
    try:
        with db_manager._conn() as c:
            rows = c.execute(q, args).fetchall()
    except Exception:
        return []
    return [{"reason": r[0], "count": r[1]} for r in rows]


def _orphaned_debug_runs(limit: int = 40) -> list[dict]:
    """Debug folders with NO matching videos.run_id — a run that started
    (RUFUS_DEBUG wrote script/keyframes) but crashed before reaching Step 6's
    DB save. These are invisible everywhere else in the app; that's exactly
    the "every failure, not just the successes" gap this page closes."""
    if not DEBUG_ROOT.is_dir():
        return []
    try:
        with db_manager._conn() as c:
            known = {r[0] for r in c.execute(
                "SELECT run_id FROM videos WHERE run_id IS NOT NULL").fetchall()}
    except Exception:
        known = set()

    orphans = []
    try:
        entries = sorted((d for d in DEBUG_ROOT.iterdir() if d.is_dir()),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    for d in entries:
        if d.name in known:
            continue
        try:
            files = sorted(f.name for f in d.iterdir() if f.is_file())
        except OSError:
            files = []
        preview = ""
        script_file = d / "script.txt"
        if script_file.exists():
            try:
                preview = script_file.read_text(encoding="utf-8", errors="replace")[:300]
            except OSError:
                pass
        orphans.append({"run_id": d.name, "mtime": d.stat().st_mtime,
                        "files": files, "preview": preview})
        if len(orphans) >= limit:
            break
    return orphans


def _rejected_attempts(limit: int = 200, channel: str | None = None,
                       niche: str | None = None, phase: str | None = None) -> list[dict]:
    """Every rejected hook/body attempt (script_attempts already logs these —
    the homepage only ever showed the top-8 aggregate; this is the full,
    filterable browser."""
    q = ("SELECT ts, niche, phase, attempt_n, hook, body, rejected_reason, channel "
         "FROM script_attempts WHERE accepted = 0 AND rejected_reason IS NOT NULL "
         "AND rejected_reason != ''")
    args: list = []
    if channel:
        q += " AND channel = ?"; args.append(channel)
    if niche:
        q += " AND niche = ?"; args.append(niche)
    if phase:
        q += " AND phase = ?"; args.append(phase)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    try:
        with db_manager._conn() as c:
            rows = c.execute(q, args).fetchall()
    except Exception:
        return []
    cols = ["ts", "niche", "phase", "attempt_n", "hook", "body",
            "rejected_reason", "channel"]
    return [dict(zip(cols, r)) for r in rows]


# Root-cause taxonomy for script_attempts.rejected_reason — a fixed, small
# set of buckets instead of counting distinct free-text strings (which
# mostly differ only by which specific word got banned). After enough
# volume this answers "which STAGE of the pipeline is actually the
# bottleneck" at a glance instead of requiring someone to read every row.
# Order matters: checked top-to-bottom, first match wins (e.g. a banned-
# phrase rejection on a hook attempt is "safety", not "weak_hook").
_REJECTION_CATEGORIES = [
    ("safety",          ("banned phrase", "hedging", "conspiracy")),
    ("accuracy",        ("specificity", "invented", "fabricat", "sensory")),
    ("weak_hook",       ("forbidden opener", "hook too short", "hook too long")),
    ("loose_structure", ("loop no echo", "opinion word", "sentences too long",
                        "sentences too short", "cadence", "too few sentences")),
    ("boring",          ("boring", "no tension", "flat")),
]

# supervisor.py's three gates (seed_gate, fact_check, footage_gate) return
# free-form LLM prose, not script_writer's controlled-vocabulary strings —
# keyword matching against that prose would be unreliable. Their PHASE alone
# already says exactly what kind of failure it is, so those are categorized
# directly instead of by keyword.
_PHASE_CATEGORY = {
    "seed_gate":    "weak_seed",
    "fact_check":   "accuracy",
    "footage_gate": "footage_drift",
}
_CATEGORY_ORDER = ([c for c, _ in _REJECTION_CATEGORIES]
                  + ["weak_seed", "footage_drift", "other"])


def _categorize_rejection(reason: str, phase: str | None = None) -> str:
    if phase in _PHASE_CATEGORY:
        return _PHASE_CATEGORY[phase]
    r = (reason or "").lower()
    for category, keywords in _REJECTION_CATEGORIES:
        if any(k in r for k in keywords):
            return category
    return "other"


def _rejection_category_counts(channel: str | None = None) -> list[dict]:
    """Aggregate ALL rejected attempts (not just the last N shown in the
    browser) into the fixed taxonomy above — covers every gate in the
    pipeline (hook/body phases AND the three supervisor gates), so this
    can answer e.g. "is Hook Scorer or Fact-check the real bottleneck"."""
    reasons = _rejected_attempts(limit=100_000, channel=channel)
    from collections import Counter
    counts = Counter(_categorize_rejection(r["rejected_reason"], r["phase"]) for r in reasons)
    total = sum(counts.values())
    if not total:
        return []
    order = _CATEGORY_ORDER
    return [{"category": c, "count": counts[c],
            "pct": round(100 * counts[c] / total, 1)}
           for c in order if counts.get(c)]


def _distinct(column: str) -> list[str]:
    try:
        with db_manager._conn() as c:
            rows = c.execute(
                f"SELECT DISTINCT {column} FROM script_attempts "
                f"WHERE {column} IS NOT NULL ORDER BY {column}").fetchall()
        return [r[0] for r in rows]
    except Exception:
        return []


def _fmt_ts(epoch: float) -> str:
    from datetime import datetime
    return datetime.fromtimestamp(epoch).strftime("%Y-%m-%d %H:%M")


# Real audience performance, not just the internal LLM score. The data pipe
# already exists (analytics_fetcher.py fetches views/watch% via the YouTube
# API, report.py's correlate() already buckets score-vs-views for the log
# digest) — this is that same signal in the browser, since nothing here ever
# queried the `metrics` table before.
MIN_VIDEOS_FOR_CORRELATION = 5   # matches report.py's --correlate guard


def _performance_rows(channel: str | None = None, days: int = 90) -> list[dict]:
    """videos LEFT JOIN latest metrics row per video — same shape as
    report.py's _latest_metrics_join() / feedback_analyzer.py's query.
    LEFT JOIN (not INNER) so a video with no metrics yet still shows up
    with blank views/watch%, rather than silently vanishing."""
    q = """
        SELECT v.id, v.upload_date, v.niche, v.channel,
               COALESCE(v.title, v.script_hook) AS title, v.score, v.youtube_id,
               m.views, m.watch_pct, m.likes
        FROM videos v
        LEFT JOIN (
            SELECT video_id, views, watch_pct, ctr, likes
            FROM metrics WHERE id IN (SELECT MAX(id) FROM metrics GROUP BY video_id)
        ) m ON m.video_id = v.id
        WHERE v.upload_date >= date('now', ?)
    """
    args: list = [f"-{days} days"]
    if channel:
        q += " AND v.channel = ?"
        args.append(channel)
    q += " ORDER BY v.id DESC"
    try:
        with db_manager._conn() as c:
            rows = c.execute(q, args).fetchall()
    except Exception:
        return []
    cols = ["id", "upload_date", "niche", "channel", "title", "score",
            "youtube_id", "views", "watch_pct", "likes"]
    return [dict(zip(cols, r)) for r in rows]


def _score_vs_views(rows: list[dict]) -> list[dict]:
    """Bucket by score, average views per bucket — does the 1-10 gate
    actually predict real performance? Same question report.py's
    correlate() answers for the log digest."""
    from collections import defaultdict
    buckets: dict[int, list[int]] = defaultdict(list)
    for r in rows:
        if r["score"] is not None and r["views"] is not None:
            buckets[r["score"]].append(r["views"])
    return [{"score": s, "avg_views": round(sum(vs) / len(vs)), "n": len(vs)}
           for s, vs in sorted(buckets.items())]


def _debug_assets(run_id: str | None) -> list[dict]:
    """Files in this run's debug folder (script/voiceover/keyframes). Every run
    keeps these now (not just RUFUS_DEBUG=1 runs), and they're retained
    permanently as the quality-review record."""
    if not run_id:
        return []
    d = DEBUG_ROOT / run_id
    if not d.is_dir():
        return []
    out = []
    for f in sorted(d.iterdir()):
        if f.is_file():
            out.append({"name": f.name, "size_kb": max(1, f.stat().st_size // 1024)})
    return out


def _image_prompts(run_id: str | None) -> list[dict]:
    """The per-beat image-generation prompts for this run, paired with their
    keyframe. comfy_client/sd_client write NN.png (the still) + NN.txt (the
    exact FLUX/SD prompt that produced it) per beat. This surfaces the full
    script→images chain on the review page instead of leaving each prompt as a
    file the reviewer has to download one by one.

    Returns [{n, prompt, image}] ordered by beat, where `image` is the NN.png
    name (served via /debug/<run_id>/<image>) or None if only the prompt exists."""
    if not run_id:
        return []
    d = DEBUG_ROOT / run_id
    if not d.is_dir():
        return []
    out = []
    for txt in sorted(d.glob("[0-9]*.txt")):
        try:
            raw = txt.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            continue
        # Files are written as "FLUX PROMPT:\n<prompt>" — strip the label.
        prompt = raw.split(":", 1)[1].strip() if raw.lower().startswith("flux prompt") else raw
        png = txt.with_suffix(".png")
        out.append({"n": txt.stem, "prompt": prompt,
                    "image": png.name if png.is_file() else None})
    return out


def _gallery_images(limit: int = 60) -> list[dict]:
    """Every keyframe still across every recent run, newest first — a
    browsable portfolio instead of hunting through one video's detail page
    at a time. Same source _orphaned_debug_runs()/_debug_assets() already
    read (paths.debug_root()), just flattened across runs."""
    if not DEBUG_ROOT.is_dir():
        return []
    out: list[dict] = []
    try:
        run_dirs = sorted((d for d in DEBUG_ROOT.iterdir() if d.is_dir()),
                         key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    for run_dir in run_dirs:
        try:
            pngs = sorted(run_dir.glob("[0-9]*.png"))
        except OSError:
            continue
        for png in pngs:
            out.append({"run_id": run_dir.name, "image": png.name,
                       "mtime": png.stat().st_mtime})
            if len(out) >= limit:
                return out
    return out


# ── Rendering helpers (no template engine — a handful of small f-strings) ────

def _esc(s) -> str:
    return html.escape(str(s)) if s is not None else ""


def _score_color(score) -> str:
    if score is None:
        return "#888"
    if score >= 8:
        return "#22c55e"
    if score >= 6:
        return "#eab308"
    return "#ef4444"


def _sparkline_svg(scores: list[int], width: int = 320, height: int = 64,
                   threshold: int = UPLOAD_THRESHOLD_DEFAULT) -> str:
    """Score trend, oldest → newest, left to right. Self-contained inline SVG
    — no chart library, nothing that can fail to load remotely."""
    if not scores:
        return "<p class='muted'>No scored videos yet.</p>"
    n = len(scores)
    pad = 6
    step_x = (width - 2 * pad) / max(1, n - 1)

    def y(s):
        s = max(0, min(10, s))
        return height - pad - (s / 10) * (height - 2 * pad)

    points = " ".join(f"{pad + i*step_x:.1f},{y(s):.1f}" for i, s in enumerate(scores))
    thresh_y = y(threshold)
    dots = "".join(
        f'<circle cx="{pad + i*step_x:.1f}" cy="{y(s):.1f}" r="3" '
        f'fill="{_score_color(s)}"><title>{s}/10</title></circle>'
        for i, s in enumerate(scores)
    )
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'role="img" aria-label="Score trend">'
        f'<line x1="{pad}" y1="{thresh_y:.1f}" x2="{width-pad}" y2="{thresh_y:.1f}" '
        f'stroke="#6b7280" stroke-dasharray="4,4" stroke-width="1"/>'
        f'<polyline points="{points}" fill="none" stroke="#3b82f6" stroke-width="2"/>'
        f'{dots}</svg>'
    )


PAGE_HEAD = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Rufus Dashboard</title>
<style>
  :root { color-scheme: dark light; }
  body { font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
         background: #0f1115; color: #e5e7eb; }
  @media (prefers-color-scheme: light) { body { background: #f7f7f8; color: #1a1a1a; } }
  header { padding: 16px 24px; border-bottom: 1px solid #2a2d34; }
  header a { color: inherit; text-decoration: none; }
  main { padding: 20px 24px; max-width: 1100px; margin: 0 auto; }
  h1 { font-size: 20px; margin: 0; }
  h2 { font-size: 15px; color: #9ca3af; text-transform: uppercase;
       letter-spacing: 0.05em; margin: 28px 0 10px; }
  .cards { display: flex; gap: 12px; flex-wrap: wrap; }
  .card { background: #171a21; border: 1px solid #2a2d34; border-radius: 10px;
          padding: 14px 18px; min-width: 120px; }
  @media (prefers-color-scheme: light) { .card { background: #fff; border-color: #e5e7eb; } }
  .card .num { font-size: 26px; font-weight: 700; }
  .card .label { font-size: 12px; color: #9ca3af; }
  table { width: 100%; border-collapse: collapse; margin-top: 6px; }
  th, td { text-align: left; padding: 8px 10px; border-bottom: 1px solid #2a2d34; font-size: 14px; }
  @media (prefers-color-scheme: light) { th, td { border-color: #e5e7eb; } }
  th { color: #9ca3af; font-weight: 600; font-size: 12px; text-transform: uppercase; }
  tr:hover td { background: rgba(59,130,246,0.06); }
  a.row-link { color: inherit; text-decoration: none; display: block; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px;
           font-size: 12px; font-weight: 600; }
  .badge.ok { background: rgba(34,197,94,0.15); color: #22c55e; }
  .badge.held { background: rgba(239,68,68,0.15); color: #ef4444; }
  .badge.pending { background: rgba(234,179,8,0.15); color: #eab308; }
  .muted { color: #9ca3af; font-size: 13px; }
  .msg { padding: 10px 14px; border-radius: 8px; margin-bottom: 14px; font-size: 14px; }
  .msg.ok { background: rgba(34,197,94,0.12); color: #22c55e; }
  .msg.error { background: rgba(239,68,68,0.12); color: #ef4444; }
  .actions { margin: 16px 0; display: flex; gap: 10px; flex-wrap: wrap; }
  .btn { border: none; border-radius: 8px; padding: 10px 18px; font-size: 14px;
         font-weight: 600; cursor: pointer; }
  .btn.approve { background: #22c55e; color: #06210f; }
  .btn.reject  { background: #ef4444; color: #2a0a0a; }
  .btn.save    { background: #3b82f6; color: #06122a; }
  .field { display: block; width: 100%; box-sizing: border-box; margin: 6px 0 14px;
           padding: 8px 10px; border-radius: 6px; border: 1px solid #2a2d34;
           background: #171a21; color: inherit; font-family: inherit; font-size: 14px; }
  @media (prefers-color-scheme: light) { .field { background: #fff; border-color: #d1d5db; } }
  label { font-size: 12px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.05em; }
  .filters { margin: 12px 0; }
  .filters a { margin-right: 10px; font-size: 13px; color: #3b82f6; text-decoration: none; }
  .back { color: #3b82f6; text-decoration: none; font-size: 14px; }
  .script { white-space: pre-wrap; font-size: 15px; line-height: 1.5;
            background: #171a21; border: 1px solid #2a2d34; border-radius: 8px;
            padding: 14px; }
  @media (prefers-color-scheme: light) { .script { background: #fff; border-color: #e5e7eb; } }
  .grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
  @media (max-width: 700px) { .grid2 { grid-template-columns: 1fr; } }
  .assets a { display: inline-block; margin: 4px 8px 4px 0; font-size: 13px;
              color: #3b82f6; text-decoration: none; }
  .navlink { color: #9ca3af; text-decoration: none; font-size: 14px; margin-left: 16px; }
  .navlink:hover { color: #3b82f6; }
  .orphan { background: #171a21; border: 1px solid #2a2d34; border-radius: 8px;
            padding: 12px 14px; margin-bottom: 10px; }
  @media (prefers-color-scheme: light) { .orphan { background: #fff; border-color: #e5e7eb; } }
</style></head><body>
<header><a href="/"><h1>🎬 Rufus Dashboard</h1></a>
<a class="navlink" href="/failures">⚠ Failures &amp; rejected attempts</a>
<a class="navlink" href="/performance">📈 Performance</a>
<a class="navlink" href="/system">🖥 System</a>
<a class="navlink" href="/trending">🔥 Trending</a>
<a class="navlink" href="/gallery">🖼 Gallery</a>
<a class="navlink" href="/settings">⚙ Settings</a></header>
<main>
"""
PAGE_TAIL = "</main></body></html>"


# ── Routes ─────────────────────────────────────────────────────────────────────

def _status_badge(status: str) -> str:
    if status == "approved":
        return '<span class="badge ok">approved</span>'
    if status == "rejected":
        return '<span class="badge held">rejected</span>'
    return '<span class="badge pending">pending</span>'


def _videos_table(videos: list[dict]) -> str:
    if not videos:
        return "<p class='muted'>Nothing here.</p>"
    rows = ""
    for v in videos:
        score = v["score"]
        score_html = (f'<span style="color:{_score_color(score)};font-weight:700">{score}/10</span>'
                      if score is not None else "—")
        title = _esc((v["title"] or v["script_hook"] or "")[:70])
        rows += (f'<tr><td><a class="row-link" href="/video/{v["id"]}">'
                 f'{_esc(v["upload_date"])}</a></td>'
                 f'<td><a class="row-link" href="/video/{v["id"]}">{_esc(v["niche"])}</a></td>'
                 f'<td><a class="row-link" href="/video/{v["id"]}">{title}</a></td>'
                 f'<td>{score_html}</td><td>{_status_badge(v["upload_status"])}</td></tr>\n')
    return (f"<table><tr><th>Date</th><th>Niche</th><th>Hook / Title</th>"
            f"<th>Score</th><th>Status</th></tr>{rows}</table>")


def _msg_banner() -> str:
    ok_msg  = request.args.get("ok")
    err_msg = request.args.get("error")
    if ok_msg:
        return f'<div class="msg ok">{_esc(ok_msg)}</div>'
    if err_msg:
        return f'<div class="msg error">{_esc(err_msg)}</div>'
    return ""


@app.route("/")
def index():
    channel = request.args.get("channel") or None
    stats   = _stats(channel=channel)
    videos  = _recent_videos(limit=60, channel=channel)
    pending = _recent_videos(limit=60, channel=channel, status="pending")
    rejects = _top_rejections(channel=channel)
    channels = _channels()

    # oldest -> newest for the trend line
    scored = [v["score"] for v in reversed(videos) if v["score"] is not None]

    filt_html = ""
    if channels:
        links = [f'<a href="/">all channels</a>']
        for ch in channels:
            links.append(f'<a href="/?channel={_esc(ch)}">{_esc(ch)}</a>')
        filt_html = f'<div class="filters">{"".join(links)}</div>'

    channel_options = "".join(f'<option value="{_esc(ch)}">{_esc(ch)}</option>' for ch in channels)
    topic_form = f"""
    <h2>🎯 Make a video about a specific topic</h2>
    <p class="muted">Runs in the background (can take a while) — resolved to a
       real Wikipedia article so it's still fact-grounded, then shows up in
       the pending list below like any other video. Never auto-uploads.</p>
    <form method="post" action="/request-topic" style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end;margin-bottom:24px">
      <div style="flex:1;min-width:220px">
        <label for="topic">Topic</label>
        <input class="field" style="margin:6px 0 0" type="text" id="topic" name="topic"
               placeholder="e.g. Bretton Woods, Tulip mania..." required>
      </div>
      {f'''<div><label for="channel">Channel</label>
        <select class="field" style="margin:6px 0 0" id="channel" name="channel">
          <option value="">(default)</option>{channel_options}
        </select></div>''' if channels else ""}
      <button class="btn save" type="submit" style="height:38px">Queue it</button>
    </form>
    """

    cards = f"""
    <div class="cards">
      <div class="card"><div class="num">{stats['pending']}</div><div class="label">awaiting review</div></div>
      <div class="card"><div class="num">{stats['uploaded']}</div><div class="label">approved / uploaded</div></div>
      <div class="card"><div class="num">{stats['rejected']}</div><div class="label">rejected</div></div>
      <div class="card"><div class="num">{stats['avg_score']}</div><div class="label">avg score</div></div>
    </div>
    """

    reject_html = ""
    if rejects:
        items = "".join(f"<li>{_esc(r['reason'])} — <b>{r['count']}×</b></li>" for r in rejects)
        reject_html = f"<ul>{items}</ul>"
    else:
        reject_html = "<p class='muted'>No rejected attempts recorded yet.</p>"

    body = f"""
    {_msg_banner()}
    {topic_form}
    {filt_html}
    {cards}
    <h2>⏳ Awaiting your review ({len(pending)})</h2>
    {_videos_table(pending)}
    <h2>Score trend (oldest → newest)</h2>
    {_sparkline_svg(scored)}
    <div class="grid2">
      <div>
        <h2>All recent videos</h2>
        {_videos_table(videos)}
      </div>
      <div>
        <h2>Most common script rejections</h2>
        {reject_html}
      </div>
    </div>
    """
    return PAGE_HEAD + body + PAGE_TAIL


@app.route("/failures")
def failures():
    """Every failure the automation produced, not just the successes — a
    crashed run has NO row in `videos` at all (it never reached Step 6), so
    without this page it's invisible everywhere else in the app."""
    channel = request.args.get("channel") or None
    niche   = request.args.get("niche") or None
    phase   = request.args.get("phase") or None

    orphans = _orphaned_debug_runs()
    rejects = _rejected_attempts(channel=channel, niche=niche, phase=phase)
    categories = _rejection_category_counts(channel=channel)

    niche_links = "".join(
        f'<a href="/failures?niche={_esc(n)}">{_esc(n)}</a> ' for n in _distinct("niche"))
    phase_links = "".join(
        f'<a href="/failures?phase={_esc(p)}">{_esc(p)}</a> ' for p in _distinct("phase"))
    filt_html = (f'<div class="filters"><a href="/failures">all niches</a> {niche_links}'
                f'<br><a href="/failures">all phases</a> {phase_links}</div>')

    orphan_html = "<p class='muted'>No crashed/incomplete runs found — every RUFUS_DEBUG run reached the database.</p>"
    if orphans:
        blocks = ""
        for o in orphans:
            file_links = "".join(
                f'<a href="/debug/{_esc(o["run_id"])}/{_esc(f)}" target="_blank">{_esc(f)}</a> '
                for f in o["files"]
            ) or "<span class='muted'>(no files saved)</span>"
            preview = f"<div class='muted' style='margin:6px 0'>{_esc(o['preview'])}</div>" if o["preview"] else ""
            blocks += (f'<div class="orphan"><b>{_esc(o["run_id"])}</b> '
                      f'<span class="muted">· {_fmt_ts(o["mtime"])}</span>'
                      f'{preview}<div class="assets">{file_links}</div></div>\n')
        orphan_html = blocks

    reject_html = "<p class='muted'>No rejected attempts recorded.</p>"
    if rejects:
        rows = ""
        for r in rejects:
            preview = _esc((r["body"] or r["hook"] or "")[:90])
            cat = _esc(_categorize_rejection(r["rejected_reason"], r["phase"]))
            rows += (f"<tr><td class='muted'>{_esc(r['ts'])}</td>"
                     f"<td>{_esc(r['niche'])}</td><td>{_esc(r['phase'])}</td>"
                     f"<td><span class='badge pending'>{cat}</span></td>"
                     f"<td>{_esc(r['rejected_reason'])}</td><td>{preview}</td></tr>\n")
        reject_html = (f"<table><tr><th>When</th><th>Niche</th><th>Phase</th>"
                       f"<th>Category</th><th>Reason</th><th>Preview</th></tr>{rows}</table>")

    category_html = "<p class='muted'>Not enough rejected attempts yet to show a breakdown.</p>"
    if categories:
        bars = ""
        for c in categories:
            bars += (f"<div style='margin:6px 0'>"
                     f"<span style='display:inline-block;width:130px'>{_esc(c['category'])}</span>"
                     f"<span style='display:inline-block;width:200px;background:#2a2d34;"
                     f"border-radius:4px;overflow:hidden;vertical-align:middle'>"
                     f"<span style='display:block;height:10px;width:{c['pct']}%;"
                     f"background:#3b82f6'></span></span> "
                     f"<b>{c['count']}</b> <span class='muted'>({c['pct']}%)</span></div>\n")
        category_html = bars

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Crashed / incomplete runs ({len(orphans)})</h2>
    <p class="muted">Debug folders with no matching database row — the run
       started (RUFUS_DEBUG was on) but never finished (Step 4/5 failure,
       crash, or a stopped process).</p>
    {orphan_html}
    <h2>Bottleneck breakdown (all-time)</h2>
    <p class="muted">Every rejected attempt, ever, grouped into a fixed
       taxonomy instead of counted by exact wording — this is what actually
       answers "which stage of the pipeline is the bottleneck" once there's
       enough volume.</p>
    {category_html}
    <h2>Rejected script attempts</h2>
    {filt_html}
    {reject_html}
    """
    return PAGE_HEAD + body + PAGE_TAIL


@app.route("/performance")
def performance():
    """Score vs real audience performance — does the internal 1-10 gate
    actually predict views/watch%? analytics_fetcher.py already fetches this
    data and report.py's correlate() already computes this same signal for
    the log digest; this is that signal in the browser, since dashboard.py
    never queried the `metrics` table before this route existed."""
    channel = request.args.get("channel") or None
    rows = _performance_rows(channel=channel)

    channel_links = "".join(
        f'<a href="/performance?channel={_esc(c)}">{_esc(c)}</a> '
        for c in sorted({r["channel"] for r in rows if r["channel"]}))
    filt_html = f'<div class="filters"><a href="/performance">all channels</a> {channel_links}</div>'

    with_metrics = [r for r in rows if r["views"] is not None]
    correlation_html = (f"<p class='muted'>Need ≥{MIN_VIDEOS_FOR_CORRELATION} "
                        f"videos with metrics to correlate (have {len(with_metrics)}).</p>")
    if len(with_metrics) >= MIN_VIDEOS_FOR_CORRELATION:
        bars = ""
        for b in _score_vs_views(with_metrics):
            bars += (f"<div style='margin:6px 0'>"
                     f"<span style='display:inline-block;width:60px'>{b['score']}/10</span>"
                     f"<b>{b['avg_views']}</b> <span class='muted'>avg views "
                     f"(n={b['n']})</span></div>\n")
        correlation_html = bars

    table_html = "<p class='muted'>No uploaded videos in the last 90 days.</p>"
    if rows:
        trs = ""
        for r in rows:
            views = r["views"] if r["views"] is not None else "—"
            watch = f"{r['watch_pct']:.0f}%" if r["watch_pct"] is not None else "—"
            likes = r["likes"] if r["likes"] is not None else "—"
            link = (f'<a href="/video/{r["id"]}">{_esc(r["title"] or "(untitled)")}</a>'
                    if r["id"] else _esc(r["title"] or ""))
            trs += (f"<tr><td class='muted'>{_esc(r['upload_date'] or '')}</td>"
                   f"<td>{_esc(r['niche'] or '')}</td><td>{link}</td>"
                   f"<td>{r['score'] if r['score'] is not None else '—'}/10</td>"
                   f"<td>{views}</td><td>{watch}</td><td>{likes}</td></tr>\n")
        table_html = (f"<table><tr><th>Date</th><th>Niche</th><th>Title</th>"
                     f"<th>Score</th><th>Views</th><th>Watch%</th><th>Likes</th></tr>{trs}</table>")

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Does the score predict real performance?</h2>
    <p class="muted">Average views per internal score bucket — pulled from
       the same YouTube Analytics data analytics_fetcher.py already collects.</p>
    {correlation_html}
    <h2>Videos (last 90 days)</h2>
    {filt_html}
    {table_html}
    """
    return PAGE_HEAD + body + PAGE_TAIL


@app.route("/system")
def system_status():
    """Status + process control for THIS PC's automation: is ComfyUI up, is
    a run in progress per channel, launch a run, cancel one this dashboard
    started. Loopback-only binding is the primary guard (see module
    docstring); /system/run and /system/cancel additionally self-check
    remote_addr as defense in depth."""
    channels = _channels()
    comfy_up = _comfyui_reachable()

    rows = ""
    for cid in channels:
        running = _run_in_progress(cid)
        can_cancel = running and _LAUNCHED.get(cid, _LAUNCHED.get("default"))
        can_cancel = bool(can_cancel and can_cancel.poll() is None)
        status_badge = ("<span class='badge pending'>running</span>" if running
                        else "<span class='badge approved'>idle</span>")
        cancel_btn = (f'<form method="post" action="/system/cancel" style="display:inline">'
                     f'<input type="hidden" name="channel" value="{_esc(cid)}">'
                     f'<button type="submit">Cancel</button></form>' if can_cancel else "")
        rows += (f"<tr><td>{_esc(cid)}</td><td>{status_badge}</td><td>{cancel_btn}</td></tr>\n")
    channels_html = (f"<table><tr><th>Channel</th><th>Status</th><th></th></tr>{rows}</table>"
                     if rows else "<p class='muted'>No channels configured.</p>")

    comfy_badge = ("<span class='badge approved'>reachable</span>" if comfy_up
                  else "<span class='badge pending'>not reachable</span>")

    channel_options = "".join(f'<option value="{_esc(c)}">{_esc(c)}</option>' for c in channels)

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">ComfyUI</h2>
    <p>{comfy_badge} <span class="muted">({_esc(_comfy_host())})</span></p>
    <h2>Channels</h2>
    {channels_html}
    <h2>Run a video now</h2>
    <form method="post" action="/system/run" style="display:flex;gap:10px;flex-wrap:wrap;align-items:flex-end">
      {f'''<div><label for="sys-channel">Channel</label>
        <select class="field" style="margin:6px 0 0" id="sys-channel" name="channel">
          <option value="">(default)</option>{channel_options}
        </select></div>''' if channels else ""}
      <div><label for="sys-niche">Niche override</label>
        <input class="field" style="margin:6px 0 0" type="text" id="sys-niche" name="niche"
               placeholder="(optional)"></div>
      <button class="btn save" type="submit" style="height:38px">Start</button>
    </form>
    <p class="muted">Cancelling only works for a run started from this page
       — a Task Scheduler run or a manual run.bat run has no process handle
       here to stop. If one of those actually crashed rather than just
       running long, its channel still shows "running" until its .lock file
       is removed — check /failures for a crashed/orphaned run before
       deleting a lock file by hand.</p>
    """
    return PAGE_HEAD + body + PAGE_TAIL


@app.route("/system/run", methods=["POST"])
def system_run():
    _require_localhost()
    channel = request.form.get("channel", "").strip() or None
    niche   = request.form.get("niche", "").strip() or None
    from channel_config import load_channel
    resolved_id = load_channel(channel).id
    if _run_in_progress(resolved_id):
        return redirect("/system")
    try:
        _launch_run(niche=niche, channel=channel)
    except Exception:
        pass   # /system re-renders either way; no separate error channel here
    return redirect("/system")


@app.route("/system/cancel", methods=["POST"])
def system_cancel():
    _require_localhost()
    channel = request.form.get("channel", "").strip() or None
    _cancel_run(channel)
    return redirect("/system")


@app.route("/settings")
def settings():
    """Edit common tunables from a form instead of PowerShell/editing JSON
    by hand. Applies immediately to runs launched FROM this dashboard; a
    Task Scheduler run needs run_scheduled.bat updated separately to read
    config/dashboard_settings.json too — not wired automatically, since env
    vars don't propagate between independent processes."""
    values = _load_settings()
    rows = ""
    for key, label, kind, help_text in SETTINGS_SCHEMA:
        val = values.get(key, "")
        if kind == "bool":
            opts = [("", "(default — don't override)"), ("1", "on"), ("0", "off")]
        else:
            opts = [("", "(default)")] + [(o, o) for o in kind.split(":", 1)[1].split(",")]
        options = "".join(
            f'<option value="{_esc(v)}" {"selected" if val == v else ""}>{_esc(t)}</option>'
            for v, t in opts)
        rows += (f"<tr><td>{_esc(label)}</td>"
                f"<td><select name=\"{key}\">{options}</select></td></tr>"
                f"<tr><td colspan='2' class='muted' style='padding-top:0'>{_esc(help_text)}</td></tr>\n")

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Settings</h2>
    <p class="muted">Applies to runs launched from THIS dashboard (Run a
       video now, Queue a topic, Trending) immediately. A Task Scheduler
       run needs run_scheduled.bat updated separately to read the same
       file — leaving a setting at "(default)" never overrides anything.</p>
    <form method="post" action="/settings/save">
      <table>{rows}</table>
      <button class="btn save" type="submit" style="margin-top:12px">Save</button>
    </form>
    """
    return PAGE_HEAD + body + PAGE_TAIL


@app.route("/settings/save", methods=["POST"])
def settings_save():
    _require_localhost()
    values = {}
    for key, label, kind, help_text in SETTINGS_SCHEMA:
        v = request.form.get(key, "").strip()
        if v:
            values[key] = v
    _save_settings(values)
    return redirect("/settings")


@app.route("/gallery")
def gallery():
    """Browsable grid of generated stills across every recent run — the
    per-video detail page already shows one run's keyframes; this is every
    run's, for browsing past visual output rather than reviewing one video
    at a time."""
    images = _gallery_images()
    if not images:
        body = "<a class='back' href='/'>← back</a><p class='muted'>No keyframes saved yet.</p>"
        return PAGE_HEAD + body + PAGE_TAIL
    tiles = ""
    for img in images:
        src = f"/debug/{_esc(img['run_id'])}/{_esc(img['image'])}"
        tiles += (f'<a href="{src}" target="_blank" style="display:inline-block;margin:4px">'
                 f'<img src="{src}" style="width:120px;height:213px;object-fit:cover;'
                 f'border-radius:6px" loading="lazy" title="{_esc(img["run_id"])}"></a>\n')
    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Gallery ({len(images)} recent stills)</h2>
    <p class="muted">Every saved keyframe across recent runs, newest first.
       Click one to open full-size.</p>
    <div>{tiles}</div>
    """
    return PAGE_HEAD + body + PAGE_TAIL


@app.route("/trending")
def trending():
    """Browse this week's rising search queries per niche (Google Trends via
    pytrends) before committing a run to one — research.py already uses
    these to auto-pick topics (fetch_trending_wikipedia); this is that same
    signal, made browsable, with a "queue it" button that hands off to the
    existing /request-topic path (same fact-grounding, same gates) rather
    than a separate launch mechanism."""
    import research
    niches = list(research.NICHE_TREND_SEEDS.keys())
    niche = request.args.get("niche") or (niches[0] if niches else None)

    queries: list[str] = []
    error = None
    if niche:
        try:
            queries = research._trending_queries(niche)
        except Exception as e:
            error = str(e)

    niche_links = "".join(
        f'<a href="/trending?niche={_esc(n)}">{_esc(n)}</a> ' for n in niches)

    if error:
        list_html = f"<p class='muted'>Trend lookup failed: {_esc(error)}</p>"
    elif not queries:
        list_html = ("<p class='muted'>No rising queries right now for this "
                     "niche (pytrends not installed, rate-limited, or "
                     "nothing rising this week) — the same fail-open signal "
                     "research.py itself falls back on.</p>")
    else:
        items = ""
        for q in queries:
            items += (f'<form method="post" action="/request-topic" style="margin:6px 0">'
                     f'<input type="hidden" name="topic" value="{_esc(q)}">'
                     f'<button type="submit">Queue "{_esc(q)}"</button></form>\n')
        list_html = items

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">Trending queries — {_esc(niche or '(no niche configured)')}</h2>
    <p class="muted">This week's rising Google Trends searches for the
       niche. Queuing one resolves it to a real Wikipedia article through
       the same /request-topic path as typing a topic by hand — same
       fact-grounding, same gates, nothing skipped.</p>
    <div class="filters">{niche_links}</div>
    {list_html}
    """
    return PAGE_HEAD + body + PAGE_TAIL


@app.route("/request-topic", methods=["POST"])
def request_topic():
    """Kick off a real Rufus run for a topic YOU chose, in the background.
    Never blocks the request (a run takes minutes to hours) and never
    auto-uploads — it lands in the normal pending-review queue like any
    other video, going through every existing gate (fact-check, QC, score)
    exactly the same way. main.py's own per-channel FileLock is what
    actually prevents two overlapping runs of the same channel; this route
    doesn't need to duplicate that check."""
    topic = request.form.get("topic", "").strip()
    channel = request.form.get("channel", "").strip() or None
    if not topic:
        return _redirect_index(error="topic is required")
    try:
        _, log_path = _launch_run(topic=topic, channel=channel)
        return _redirect_index(
            ok=f'Queued "{topic}" — this can take a while. It will appear in '
               f'the pending list below when done. Log: logs/{log_path.name}')
    except Exception as e:
        return _redirect_index(error=f"failed to start the run: {e}")


def _redirect_index(ok: str = None, error: str = None):
    url = "/"
    if ok:
        url += f"?ok={_urlquote(ok)}"
    elif error:
        url += f"?error={_urlquote(error)}"
    return redirect(url)


@contextmanager
def _scoped_env(**overrides):
    """Set env vars for the duration of the block, then restore exactly what
    was there before (including "unset" if the key didn't exist). SAFE ONLY
    because app.run() below passes threaded=False — Flask 3.x defaults the
    dev server to threaded=True, under which two overlapping requests would
    interleave these mutations (audit finding: wrong-channel upload). Also,
    mutating os.environ permanently — as a naive assignment would — leaks
    across every later request in this long-lived process (confirmed live:
    it leaked into an unrelated test suite run in the same process)."""
    prev = {k: os.environ.get(k) for k in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for k, v in prev.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _redirect_detail(video_id: int, ok: str = None, error: str = None):
    url = f"/video/{video_id}"
    if ok:
        url += f"?ok={_urlquote(ok)}"
    elif error:
        url += f"?error={_urlquote(error)}"
    return redirect(url)


@app.route("/video/<int:video_id>")
def video_detail(video_id):
    v = _video_detail(video_id)
    if not v:
        abort(404)

    crit_rows = "".join(
        f"<tr><td>{label}</td><td>{v[key] if v[key] is not None else '—'}</td></tr>"
        for label, key in (
            ("Specificity", "score_specificity"), ("Hook", "score_hook"),
            ("Compression", "score_compression"), ("Loop", "score_loop"),
            ("Human", "score_human"),
        )
    )

    if v["upload_status"] == "approved" and v["youtube_id"]:
        status_html = (f'<span class="badge ok">approved → '
                       f'<a href="https://youtube.com/watch?v={_esc(v["youtube_id"])}" '
                       f'style="color:inherit" target="_blank">watch</a></span>')
    elif v["upload_status"] == "rejected":
        status_html = '<span class="badge held">rejected</span>'
    else:
        status_html = '<span class="badge pending">pending review</span>'
    if v["hold_reason"]:
        status_html += (f'<div class="muted" style="margin-top:6px">auto-gate note: '
                        f'{_esc(v["hold_reason"])}</div>')

    msg_html = _msg_banner()

    actions_html = ""
    if v["upload_status"] != "approved":
        reject_label = "Un-reject" if v["upload_status"] == "rejected" else "Reject"
        actions_html = f"""
        <div class="actions">
          <form method="post" action="/video/{v['id']}/approve" onsubmit="return confirm('Upload this video to YouTube now?');">
            <button class="btn approve" type="submit">✓ Approve &amp; Upload</button>
          </form>
          <form method="post" action="/video/{v['id']}/reject">
            <button class="btn reject" type="submit">{reject_label}</button>
          </form>
        </div>
        """

    edit_html = ""
    if v["upload_status"] != "approved":
        edit_html = f"""
        <h2>Title &amp; description (edit before approving)</h2>
        <form method="post" action="/video/{v['id']}/edit">
          <label for="title">Title</label>
          <input class="field" type="text" id="title" name="title" maxlength="100"
                 value="{_esc(v['title'] or '')}">
          <label for="description">Description</label>
          <textarea class="field" id="description" name="description" rows="6"
                    >{_esc(v['description'] or '')}</textarea>
          <button class="btn save" type="submit">Save</button>
        </form>
        """
    else:
        edit_html = f"""
        <h2>Title &amp; description</h2>
        <p><b>{_esc(v['title'] or '')}</b></p>
        <div class="script">{_esc(v['description'] or '—')}</div>
        """

    # Image-generation prompts (the script→images chain), rendered inline so a
    # reviewer sees exactly what each beat's still was told to draw, next to the
    # still itself — not a pile of files to download one by one.
    img_prompts = _image_prompts(v["run_id"])
    prompts_html = ("<p class='muted'>No per-beat image prompts saved for this "
                    "run yet.</p>")
    if img_prompts:
        cards = ""
        for p in img_prompts:
            thumb = ""
            if p["image"]:
                thumb = (f'<a href="/debug/{_esc(v["run_id"])}/{_esc(p["image"])}" '
                         f'target="_blank"><img src="/debug/{_esc(v["run_id"])}/'
                         f'{_esc(p["image"])}" loading="lazy" '
                         f'style="width:120px;border-radius:6px;flex:0 0 auto"></a>')
            cards += (
                f'<div style="display:flex;gap:12px;margin:10px 0;align-items:flex-start">'
                f'{thumb}'
                f'<div><b>beat {_esc(p["n"])}</b>'
                f'<div class="script" style="margin-top:4px">{_esc(p["prompt"])}</div>'
                f'</div></div>')
        prompts_html = cards

    assets = _debug_assets(v["run_id"])
    assets_html = "<p class='muted'>No debug artifacts for this run.</p>"
    if assets:
        links = "".join(
            f'<a href="/debug/{_esc(v["run_id"])}/{_esc(a["name"])}" target="_blank">'
            f'{_esc(a["name"])} ({a["size_kb"]}KB)</a>'
            for a in assets
        )
        assets_html = f'<div class="assets">{links}</div>'

    body = f"""
    <a class="back" href="/">← back</a>
    <h2 style="margin-top:14px">#{v['id']} · {_esc(v['upload_date'])} · {_esc(v['niche'])} · {_esc(v['channel'])}</h2>
    {msg_html}
    <p>{status_html}</p>
    <p><b>Score: <span style="color:{_score_color(v['score'])}">{v['score']}/10</span></b>
       ({v['attempts_used'] or '?'} attempts, temp {v['final_temperature'] or '?'})</p>
    <table style="max-width:320px">{crit_rows}</table>
    {actions_html}
    {edit_html}
    <h2>Script</h2>
    <div class="script">{_esc(v['script_full'] or v['script_hook'])}</div>
    <h2>Why this score (critic reasoning)</h2>
    <div class="script">{_esc(v['score_reasoning'] or '—')}</div>
    <h2>Seed / source</h2>
    <p class="muted">{_esc(v['seed_type'])} · {_esc(v['seed_source'])}</p>
    <div class="script">{_esc(v['seed_content'] or '—')}</div>
    <h2>Image prompts (what each beat was told to draw)</h2>
    {prompts_html}
    <h2>Debug artifacts (run {_esc(v['run_id'] or '—')})</h2>
    {assets_html}
    """
    return PAGE_HEAD + body + PAGE_TAIL


def _extra_publishers() -> list[tuple[str, object]]:
    """Platforms to cross-post to after YouTube, as (name, upload_fn).

    Opt-in per platform so an unconfigured account never turns a good approval
    into a scary error: RUFUS_PUBLISH_TIKTOK=1 enables TikTok. The module is
    imported lazily because it reads credentials at import time.

    Instagram is deliberately absent: the Graph API needs a Business/Creator
    account AND a PUBLICLY reachable video URL to pull from (it will not accept
    a local file), so it needs hosting decisions this box can't make on its
    own. Add it here once that's settled."""
    out = []
    if os.environ.get("RUFUS_PUBLISH_TIKTOK", "0").strip().lower() in ("1", "true", "yes", "on"):
        try:
            import tiktok_uploader
            out.append(("TikTok", tiktok_uploader.upload))
        except Exception as e:
            print(f"[dashboard] TikTok publisher unavailable ({e})")
    return out


@app.route("/video/<int:video_id>/approve", methods=["POST"])
def approve_video(video_id):
    """The ONLY path that actually uploads a video — see module docstring.
    Reuses the video's own stored channel/niche context (not whatever's
    'active' today) so the right voice/CTA-pool/category apply even if the
    approval happens days after generation."""
    v = _video_detail(video_id)
    if not v:
        abort(404)
    if v["youtube_id"]:
        return _redirect_detail(video_id, error="already uploaded")
    if v["score"] is None or v["score"] < HARD_MIN_UPLOAD_SCORE:
        shown = "unscored" if v["score"] is None else f"{v['score']}/10"
        return _redirect_detail(
            video_id,
            error=(f"blocked: score {shown} is below the {HARD_MIN_UPLOAD_SCORE}/10 "
                   f"minimum — this video cannot be approved for upload. Reject it "
                   f"or fix the underlying script/QC issue instead."))
    video_file = Path(v["video_file"] or "")
    if not video_file.exists():
        return _redirect_detail(video_id, error=f"video file missing on disk: {video_file}")

    thumb = video_file.with_suffix(".thumb.jpg")
    try:
        import youtube_uploader as yt_mod
        hashtags = yt_mod.NICHE_HASHTAGS.get(v["niche"], ["#Shorts"])
        meta = {
            "title": (v["title"] or v["script_hook"] or "Short")[:100],
            "description": v["description"] or (v["script_full"] or ""),
            "tags": [t.lstrip("#") for t in hashtags],
        }
        env_overrides = {"RUFUS_CHANNEL": v["channel"] or "main_en"}
        if v["niche"]:
            env_overrides["RUFUS_NICHE_OVERRIDE"] = v["niche"]

        with _scoped_env(**env_overrides):
            yt_url, yt_id = yt_mod.upload(video_file, v["script_full"] or "",
                                          thumbnail_path=thumb if thumb.exists() else None,
                                          metadata=meta)
    except Exception as e:
        # Upload itself failed — the video did NOT go up, so re-approving is
        # safe. Record it like main.py does so report.py's FAILED count sees
        # dashboard failures too (it used to miss them entirely).
        try:
            db_manager.mark_upload_failed(video_id, str(e))
        except Exception:
            pass
        return _redirect_detail(video_id, error=f"Upload failed (not uploaded, safe to retry): {e}")

    # Cross-post. TikTok has its own uploader module but was never wired to
    # the approve action, so approving only ever reached YouTube. Each extra
    # platform is best-effort and reported in the result message: YouTube
    # already succeeded by this point, so one platform being unconfigured or
    # erroring must not read as "the approval failed".
    extra = []
    for name, fn in _extra_publishers():
        try:
            fn(video_file, v["script_full"] or "")
            extra.append(f"{name} ok")
        except Exception as e:
            extra.append(f"{name} FAILED ({str(e)[:80]})")

    # The upload SUCCEEDED. A DB failure past this point must NOT read as
    # "upload failed" — that would tempt a re-approve and publish a DUPLICATE
    # public video. Separate block, explicit do-not-retry message.
    try:
        db_manager.update_youtube_id(video_id, yt_id)
        db_manager.set_upload_status(video_id, "approved")
    except Exception as db_err:
        return _redirect_detail(
            video_id,
            error=(f"UPLOADED OK ({yt_url}) but the status update failed "
                   f"({db_err}). Do NOT re-approve — it's already live. Fix "
                   f"the DB row manually if needed."))
    msg = f"Uploaded: {yt_url}"
    if extra:
        msg += "  |  " + "; ".join(extra)
    try:
        import notify
        notify.send("Rufus: video published",
                    f"{v['niche']} · {v['title'] or v['script_hook'] or ''}\n{yt_url}",
                    url=yt_url)
    except Exception:
        pass
    return _redirect_detail(video_id, ok=msg)


@app.route("/video/<int:video_id>/reject", methods=["POST"])
def reject_video(video_id):
    v = _video_detail(video_id)
    if not v:
        abort(404)
    if v["youtube_id"]:
        return _redirect_detail(video_id, error="already uploaded — can't reject")
    new_status = "pending" if v["upload_status"] == "rejected" else "rejected"
    db_manager.set_upload_status(video_id, new_status)
    return _redirect_detail(video_id, ok=f"marked {new_status}")


@app.route("/video/<int:video_id>/edit", methods=["POST"])
def edit_video(video_id):
    v = _video_detail(video_id)
    if not v:
        abort(404)
    title = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    db_manager.update_metadata(video_id, title=title or None, description=description or None)
    return _redirect_detail(video_id, ok="saved")


@app.route("/debug/<run_id>/<path:filename>")
def debug_file(run_id, filename):
    """Read-only static file serving for ONE run's debug folder — the real
    value of remote access (see the FLUX images / hear the voiceover from
    your phone). send_from_directory guards traversal in `filename`, but
    `run_id` is OUR path segment — an audit showed run_id=".." resolved to
    media_library/ itself, serving any rendered (incl. rejected) video. The
    resolve() check pins the folder to a direct child of DEBUG_ROOT."""
    folder = (DEBUG_ROOT / run_id).resolve()
    if folder.parent != DEBUG_ROOT.resolve() or not folder.is_dir():
        abort(404)
    return send_from_directory(folder, filename)


@app.errorhandler(404)
def not_found(e):
    return PAGE_HEAD + "<p>Not found. <a class='back' href='/'>← back</a></p>" + PAGE_TAIL, 404


if __name__ == "__main__":
    host = os.environ.get("RUFUS_DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("RUFUS_DASHBOARD_PORT", "8765"))
    db_manager.init_db()
    print(f"[dashboard] http://localhost:{port}  (LAN: http://<this PC's IP>:{port})")
    # threaded=False is LOAD-BEARING: approve_video mutates process env via
    # _scoped_env, which is only safe when requests are serialized. Flask 3.x
    # app.run() defaults threaded=True — two overlapping approvals could
    # interleave RUFUS_CHANNEL mutations and upload a video to the WRONG
    # channel. Single-threaded is fine for a 1-2 person review tool.
    app.run(host=host, port=port, debug=False, threaded=False)
