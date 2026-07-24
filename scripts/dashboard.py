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
to break when you're not on the local network). No login of its own — see
the module-level access note below before exposing this beyond your LAN.

Run:
    python scripts\\dashboard.py
    → http://localhost:8765            (this machine)
    → http://<PC's LAN IP>:8765        (phone/other device on the same wifi)

For access away from home (e.g. a friend reviewing/uploading remotely), do
NOT port-forward this — it has no login, so that exposes it AND your PC's
YouTube upload capability to the open internet. Use Tailscale (share this
one machine with their account, private, free, ~2 min) or a Cloudflare
Tunnel + Cloudflare Access (real https:// link, email login, no VPN app
needed on their end) — never a raw port-forward.

Environment:
  RUFUS_DASHBOARD_HOST   0.0.0.0 (default — LAN-visible; 127.0.0.1 for local-only)
  RUFUS_DASHBOARD_PORT   8765
"""

import html
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import quote as _urlquote

from flask import Flask, abort, redirect, request, send_from_directory

sys.path.insert(0, str(Path(__file__).parent))
import db_manager

ROOT       = Path(__file__).parent.parent
DEBUG_ROOT = ROOT / "media_library" / "debug"

app = Flask(__name__)

UPLOAD_THRESHOLD_DEFAULT = 8   # visual reference line on the score sparkline

# Hard floor enforced in approve_video() below — no video below this score
# can be approved for upload, even by a human clicking the button, whether
# it's a misclick or a reviewer other than you (Tailscale-shared access).
# Keep in sync with main.py's HARD_MIN_UPLOAD_SCORE (duplicated, not
# imported, so this file has zero import-time dependency on main.py).
HARD_MIN_UPLOAD_SCORE = 7


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


def _debug_assets(run_id: str | None) -> list[dict]:
    """Files in this run's debug folder (script/voiceover/keyframes), if
    RUFUS_DEBUG was on for that run and the ~30-day retention hasn't swept it."""
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
<a class="navlink" href="/failures">⚠ Failures &amp; rejected attempts</a></header>
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

    import subprocess
    main_py = str(ROOT / "scripts" / "main.py")
    cmd = [sys.executable, main_py, "--topic", topic]
    if channel:
        cmd += ["--channel", channel]

    log_dir = ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"topic_request_{int(time.time())}.log"
    try:
        with open(log_path, "wb") as logf:
            subprocess.Popen(cmd, cwd=str(ROOT), stdout=logf,
                             stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
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

    assets = _debug_assets(v["run_id"])
    assets_html = "<p class='muted'>No debug artifacts for this run (RUFUS_DEBUG was off, or they've aged out).</p>"
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
    <h2>Debug artifacts (run {_esc(v['run_id'] or '—')})</h2>
    {assets_html}
    """
    return PAGE_HEAD + body + PAGE_TAIL


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
    return _redirect_detail(video_id, ok=f"Uploaded: {yt_url}")


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
    host = os.environ.get("RUFUS_DASHBOARD_HOST", "0.0.0.0")
    port = int(os.environ.get("RUFUS_DASHBOARD_PORT", "8765"))
    db_manager.init_db()
    print(f"[dashboard] http://localhost:{port}  (LAN: http://<this PC's IP>:{port})")
    # threaded=False is LOAD-BEARING: approve_video mutates process env via
    # _scoped_env, which is only safe when requests are serialized. Flask 3.x
    # app.run() defaults threaded=True — two overlapping approvals could
    # interleave RUFUS_CHANNEL mutations and upload a video to the WRONG
    # channel. Single-threaded is fine for a 1-2 person review tool.
    app.run(host=host, port=port, debug=False, threaded=False)
