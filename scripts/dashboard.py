#!/usr/bin/env python3
"""
dashboard.py — Rufus's status dashboard: one page answering "what's actually
happening with this pipeline," without scrolling PowerShell logs.

Reads read-only from rufus.db (WAL mode — safe to read while main.py writes)
and media_library/debug/<run_id>/ for the per-run script/voiceover/keyframes.
No writes, no auth, no external assets (self-contained HTML/CSS/inline SVG —
nothing to break when you're not on the local network).

Run:
    python scripts\\dashboard.py
    → http://localhost:8765            (this machine)
    → http://<PC's LAN IP>:8765        (phone/other device on the same wifi)

For access when you're away from home, do NOT port-forward this (it has no
login) — install Tailscale on this PC and your phone instead: it's a free,
private VPN mesh, so the dashboard is reachable at this PC's Tailscale
address from anywhere, with zero public exposure.

Environment:
  RUFUS_DASHBOARD_HOST   0.0.0.0 (default — LAN-visible; 127.0.0.1 for local-only)
  RUFUS_DASHBOARD_PORT   8765
"""

import html
import os
import sys
from pathlib import Path

from flask import Flask, abort, request, send_from_directory

sys.path.insert(0, str(Path(__file__).parent))
import db_manager

ROOT       = Path(__file__).parent.parent
DEBUG_ROOT = ROOT / "media_library" / "debug"

app = Flask(__name__)

UPLOAD_THRESHOLD_DEFAULT = 8   # visual reference line on the score sparkline


# ── Data access (read-only) ───────────────────────────────────────────────────

def _channels() -> list[str]:
    try:
        sys.path.insert(0, str(Path(__file__).parent))
        import channel_config
        return channel_config.list_channels()
    except Exception:
        return []


def _recent_videos(limit: int = 60, channel: str | None = None) -> list[dict]:
    q = ("SELECT id, upload_date, niche, script_hook, title, score, "
         "hold_reason, youtube_id, run_id, channel FROM videos")
    args: list = []
    if channel:
        q += " WHERE channel = ?"
        args.append(channel)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(limit)
    try:
        with db_manager._conn() as c:
            rows = c.execute(q, args).fetchall()
    except Exception:
        return []
    cols = ["id", "upload_date", "niche", "script_hook", "title", "score",
            "hold_reason", "youtube_id", "run_id", "channel"]
    return [dict(zip(cols, r)) for r in rows]


def _video_detail(video_id: int) -> dict | None:
    q = ("SELECT id, upload_date, niche, script_hook, script_full, scene_desc, "
         "seed_type, seed_source, seed_content, youtube_id, video_file, score, "
         "run_id, score_specificity, score_hook, score_compression, score_loop, "
         "score_human, attempts_used, final_temperature, score_reasoning, "
         "title, channel, hold_reason FROM videos WHERE id = ?")
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
            "score_reasoning", "title", "channel", "hold_reason"]
    return dict(zip(cols, row))


def _stats(limit: int = 100, channel: str | None = None) -> dict:
    rows = _recent_videos(limit=limit, channel=channel)
    total = len(rows)
    if not total:
        return {"total": 0, "avg_score": 0.0, "hold_rate": 0.0,
                "uploaded": 0, "held": 0}
    scores = [r["score"] for r in rows if r["score"] is not None]
    held = sum(1 for r in rows if r["hold_reason"])
    uploaded = sum(1 for r in rows if r["youtube_id"])
    return {
        "total": total,
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0.0,
        "hold_rate": round(100 * held / total, 1),
        "uploaded": uploaded,
        "held": held,
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
  .muted { color: #9ca3af; font-size: 13px; }
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
</style></head><body>
<header><a href="/"><h1>🎬 Rufus Dashboard</h1></a></header>
<main>
"""
PAGE_TAIL = "</main></body></html>"


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    channel = request.args.get("channel") or None
    stats   = _stats(channel=channel)
    videos  = _recent_videos(limit=60, channel=channel)
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

    cards = f"""
    <div class="cards">
      <div class="card"><div class="num">{stats['total']}</div><div class="label">videos (recent)</div></div>
      <div class="card"><div class="num">{stats['avg_score']}</div><div class="label">avg score</div></div>
      <div class="card"><div class="num">{stats['uploaded']}</div><div class="label">uploaded</div></div>
      <div class="card"><div class="num">{stats['held']}</div><div class="label">held ({stats['hold_rate']}%)</div></div>
    </div>
    """

    rows = ""
    for v in videos:
        held = bool(v["hold_reason"])
        status = (f'<span class="badge held" title="{_esc(v["hold_reason"])}">held</span>'
                  if held else '<span class="badge ok">uploaded</span>' if v["youtube_id"]
                  else '<span class="muted">rendered</span>')
        score = v["score"]
        score_html = (f'<span style="color:{_score_color(score)};font-weight:700">{score}/10</span>'
                      if score is not None else "—")
        rows += (f'<tr><td><a class="row-link" href="/video/{v["id"]}">'
                 f'{_esc(v["upload_date"])}</a></td>'
                 f'<td><a class="row-link" href="/video/{v["id"]}">{_esc(v["niche"])}</a></td>'
                 f'<td><a class="row-link" href="/video/{v["id"]}">'
                 f'{_esc((v["title"] or v["script_hook"] or "")[:70])}</a></td>'
                 f'<td>{score_html}</td><td>{status}</td></tr>\n')
    table = (f"<table><tr><th>Date</th><th>Niche</th><th>Hook / Title</th>"
             f"<th>Score</th><th>Status</th></tr>{rows}</table>"
             if rows else "<p class='muted'>No videos yet — run Rufus at least once.</p>")

    reject_html = ""
    if rejects:
        items = "".join(f"<li>{_esc(r['reason'])} — <b>{r['count']}×</b></li>" for r in rejects)
        reject_html = f"<ul>{items}</ul>"
    else:
        reject_html = "<p class='muted'>No rejected attempts recorded yet.</p>"

    body = f"""
    {filt_html}
    {cards}
    <h2>Score trend (oldest → newest)</h2>
    {_sparkline_svg(scored)}
    <div class="grid2">
      <div>
        <h2>Recent videos</h2>
        {table}
      </div>
      <div>
        <h2>Most common script rejections</h2>
        {reject_html}
      </div>
    </div>
    """
    return PAGE_HEAD + body + PAGE_TAIL


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

    status_html = (f'<span class="badge held">HELD — {_esc(v["hold_reason"])}</span>'
                   if v["hold_reason"] else
                   (f'<span class="badge ok">uploaded → '
                    f'<a href="https://youtube.com/watch?v={_esc(v["youtube_id"])}" '
                    f'style="color:inherit" target="_blank">watch</a></span>'
                    if v["youtube_id"] else '<span class="muted">rendered, not uploaded</span>'))

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
    <p>{status_html}</p>
    <p><b>Score: <span style="color:{_score_color(v['score'])}">{v['score']}/10</span></b>
       ({v['attempts_used'] or '?'} attempts, temp {v['final_temperature'] or '?'})</p>
    <table style="max-width:320px">{crit_rows}</table>
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


@app.route("/debug/<run_id>/<path:filename>")
def debug_file(run_id, filename):
    """Read-only static file serving for ONE run's debug folder — the real
    value of remote access (see the FLUX images / hear the voiceover from
    your phone). send_from_directory guards path traversal internally."""
    folder = DEBUG_ROOT / run_id
    if not folder.is_dir():
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
    app.run(host=host, port=port, debug=False)
