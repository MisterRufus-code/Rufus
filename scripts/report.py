#!/usr/bin/env python3
"""
report.py — the 5-minute daily KPI report, read-only over rufus.db.

    python scripts/report.py                 # last 7 days
    python scripts/report.py --weeks 4       # trend over 4 weeks
    python scripts/report.py --channel main_en
    python scripts/report.py --correlate     # script score vs views (gate sanity)

Pure stdlib (sqlite3 + text tables) — runs anywhere, costs nothing.
"""

import argparse
import sqlite3
from pathlib import Path

ROOT    = Path(__file__).parent.parent
DB_FILE = ROOT / "rufus.db"


def _conn():
    c = sqlite3.connect(str(DB_FILE))
    c.row_factory = sqlite3.Row
    return c


def _table(headers: list[str], rows: list[list], widths: list[int]) -> str:
    def fit(s, w):
        s = str(s if s is not None else "")
        return s[: w - 1] + "…" if len(s) > w else s.ljust(w)
    out = ["  ".join(fit(h, w) for h, w in zip(headers, widths))]
    out.append("  ".join("-" * w for w in widths))
    for r in rows:
        out.append("  ".join(fit(v, w) for v, w in zip(r, widths)))
    return "\n".join(out)


def _latest_metrics_join() -> str:
    """SQL fragment: latest metrics row per video (LEFT so unposted videos show)."""
    return """
        LEFT JOIN (
            SELECT video_id, views, watch_pct, ctr, likes
            FROM metrics
            WHERE id IN (SELECT MAX(id) FROM metrics GROUP BY video_id)
        ) m ON m.video_id = v.id
    """


def recent_videos(days: int, channel: str | None) -> list[sqlite3.Row]:
    q = f"""
        SELECT v.id, v.upload_date, v.niche, v.channel,
               COALESCE(v.title, v.script_hook) AS title,
               v.score, v.youtube_id, v.score_reasoning,
               m.views, m.watch_pct, m.ctr, m.likes
        FROM videos v
        {_latest_metrics_join()}
        WHERE v.upload_date >= date('now', ?)
    """
    args: list = [f"-{days} days"]
    if channel:
        q += " AND v.channel = ?"
        args.append(channel)
    q += " ORDER BY v.id DESC"
    with _conn() as c:
        return c.execute(q, args).fetchall()


def report(days: int = 7, channel: str | None = None) -> str:
    rows = recent_videos(days, channel)
    out  = [f"\n══ RUFUS REPORT — last {days} days"
            + (f" — channel {channel}" if channel else "") + " ══\n"]

    if not rows:
        out.append("No videos in this window.")
        return "\n".join(out)

    table_rows = []
    for r in rows:
        status = ("LIVE" if r["youtube_id"]
                  else "FAILED" if (r["score_reasoning"] or "").find("UPLOAD FAILED") >= 0
                  else "held")
        table_rows.append([
            r["upload_date"], r["niche"][:12], r["title"], f"{r['score']}/10", status,
            r["views"] if r["views"] is not None else "—",
            f"{r['watch_pct']:.0f}%" if r["watch_pct"] else "—",
            r["likes"] if r["likes"] is not None else "—",
        ])
    out.append(_table(
        ["date", "niche", "title", "score", "status", "views", "watch", "likes"],
        table_rows, [10, 12, 34, 5, 6, 7, 5, 5]))

    n_live   = sum(1 for r in rows if r["youtube_id"])
    n_held   = sum(1 for r in rows if not r["youtube_id"]
                   and "UPLOAD FAILED" not in (r["score_reasoning"] or ""))
    n_failed = len(rows) - n_live - n_held
    views    = sum(r["views"] or 0 for r in rows)
    watched  = [r["watch_pct"] for r in rows if r["watch_pct"]]
    out.append(f"\nproduced {len(rows)}  ·  live {n_live}  ·  held-for-review {n_held}"
               f"  ·  upload-failed {n_failed}")
    out.append(f"total views {views:,}"
               + (f"  ·  avg watch {sum(watched)/len(watched):.0f}%" if watched else ""))

    # Watch% by niche — the steering signal for the schedule
    by_niche: dict[str, list[float]] = {}
    for r in rows:
        if r["watch_pct"]:
            by_niche.setdefault(r["niche"], []).append(r["watch_pct"])
    if by_niche:
        ranked = sorted(((sum(v) / len(v), k) for k, v in by_niche.items()), reverse=True)
        out.append("watch% by niche:  "
                   + "  ".join(f"{k} {avg:.0f}%" for avg, k in ranked))

    if n_failed:
        out.append("\n⚠ upload failures — check YouTube Studio before re-uploading:")
        for r in rows:
            if "UPLOAD FAILED" in (r["score_reasoning"] or ""):
                out.append(f"   id={r['id']}  {r['title']}")

    return "\n".join(out)


def correlate(channel: str | None = None) -> str:
    """Script score vs views — does the quality gate predict performance?"""
    q = f"""
        SELECT v.score, m.views, m.watch_pct
        FROM videos v {_latest_metrics_join()}
        WHERE m.views IS NOT NULL
    """
    args: list = []
    if channel:
        q += " AND v.channel = ?"
        args.append(channel)
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    if len(rows) < 5:
        return f"\nNeed ≥5 videos with metrics to correlate (have {len(rows)})."

    by_score: dict[int, list[int]] = {}
    for r in rows:
        by_score.setdefault(int(r["score"] or 0), []).append(int(r["views"] or 0))
    out = ["\n══ score → avg views (quality-gate sanity) ══"]
    for s in sorted(by_score, reverse=True):
        vs = by_score[s]
        out.append(f"  score {s}/10:  {sum(vs)//len(vs):>8,} avg views   ({len(vs)} videos)")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Rufus KPI report")
    ap.add_argument("--weeks", type=int, default=1, help="Window in weeks (default 1)")
    ap.add_argument("--channel", help="Filter to one channel")
    ap.add_argument("--correlate", action="store_true", help="Score vs views table")
    args = ap.parse_args()

    if not DB_FILE.exists():
        print("rufus.db not found — run the pipeline at least once first.")
        raise SystemExit(1)

    print(report(days=args.weeks * 7, channel=args.channel))
    if args.correlate:
        print(correlate(args.channel))
    print()
