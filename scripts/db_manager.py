#!/usr/bin/env python3
"""
db_manager.py
SQLite helpers for tracking produced videos and their analytics.
- WAL mode for concurrent-safe reads while writing
- Incremental fetch helper (get_recent_tracked_videos) so analytics_fetcher
  doesn't re-pull metrics for the entire history every day
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path

ROOT    = Path(__file__).parent.parent
DB_FILE = ROOT / "rufus.db"


def _conn():
    c = sqlite3.connect(str(DB_FILE))
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    return c


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_date  TEXT    DEFAULT (date('now')),
                niche        TEXT,
                script_hook  TEXT,
                script_full  TEXT,
                scene_desc   TEXT,
                seed_type    TEXT,
                seed_source  TEXT,
                seed_content TEXT,
                youtube_id   TEXT,
                video_file   TEXT,
                score        INTEGER DEFAULT 0
            )
        """)
        # Idempotent migration for older DBs missing the new columns
        for ddl in (
            "ALTER TABLE videos ADD COLUMN script_full TEXT",
            "ALTER TABLE videos ADD COLUMN seed_type TEXT",
            "ALTER TABLE videos ADD COLUMN seed_source TEXT",
            "ALTER TABLE videos ADD COLUMN seed_content TEXT",
            # Script-quality columns added in the standards/logging rework
            "ALTER TABLE videos ADD COLUMN run_id TEXT",
            "ALTER TABLE videos ADD COLUMN score_specificity INTEGER",
            "ALTER TABLE videos ADD COLUMN score_hook INTEGER",
            "ALTER TABLE videos ADD COLUMN score_compression INTEGER",
            "ALTER TABLE videos ADD COLUMN score_loop INTEGER",
            "ALTER TABLE videos ADD COLUMN score_human INTEGER",
            "ALTER TABLE videos ADD COLUMN attempts_used INTEGER",
            "ALTER TABLE videos ADD COLUMN final_temperature REAL",
            "ALTER TABLE videos ADD COLUMN score_reasoning TEXT",
            # Phase 1 (scale plan): GPT-optimized upload title, for CTR learning
            "ALTER TABLE videos ADD COLUMN title TEXT",
            # Phase 2 (scale plan): multi-channel attribution
            "ALTER TABLE videos ADD COLUMN channel TEXT",
            # Dashboard: WHY a video wasn't auto-uploaded (QC fail / factual
            # hold / below score threshold). NULL means it uploaded cleanly
            # (or the run predates this column).
            "ALTER TABLE videos ADD COLUMN hold_reason TEXT",
            # Approval queue: nothing uploads without a human clicking Approve
            # in the dashboard. 'pending' (default) / 'approved' (uploaded) /
            # 'rejected' (will never upload).
            "ALTER TABLE videos ADD COLUMN upload_status TEXT DEFAULT 'pending'",
            # Description was generated fresh at upload time and never saved —
            # the approval queue needs it persisted so it can be reviewed/
            # edited BEFORE the upload decision, not only after.
            "ALTER TABLE videos ADD COLUMN description TEXT",
            # seed_source already held a short LABEL ("Wikipedia",
            # "history.stackexchange.com") but never the actual clickable
            # link — meaning the one piece of data needed to cite a source
            # (the source-citation comment posted after upload) was thrown
            # away at save time. seed_url is the real link.
            "ALTER TABLE videos ADD COLUMN seed_url TEXT",
        ):
            try:
                c.execute(ddl)
            except Exception:
                pass  # column already exists
        # Backfill: rows already live on YouTube (from before the approval
        # queue existed) must not appear as "pending" just because the new
        # column's DEFAULT applied to them too.
        c.execute("UPDATE videos SET upload_status='approved' "
                  "WHERE youtube_id IS NOT NULL "
                  "AND (upload_status IS NULL OR upload_status='pending')")
        c.execute("""
            CREATE TABLE IF NOT EXISTS metrics (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                video_id   INTEGER REFERENCES videos(id),
                views      INTEGER DEFAULT 0,
                watch_pct  REAL    DEFAULT 0,
                ctr        REAL    DEFAULT 0,
                likes      INTEGER DEFAULT 0,
                fetched_at TEXT    DEFAULT (datetime('now'))
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS script_attempts (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id            TEXT,
                ts                TEXT DEFAULT (datetime('now')),
                niche             TEXT,
                seed_type         TEXT,
                phase             TEXT,
                attempt_n         INTEGER,
                hook              TEXT,
                body              TEXT,
                temperature       REAL,
                total_score       INTEGER,
                criterion_scores  TEXT,
                rejected_reason   TEXT,
                accepted          INTEGER,
                cost_usd          REAL,
                ms                INTEGER
            )
        """)
        # script_attempts.channel migration must run AFTER its CREATE (the table
        # may not have existed when the videos ALTER loop ran on a fresh DB).
        try:
            c.execute("ALTER TABLE script_attempts ADD COLUMN channel TEXT")
        except Exception:
            pass  # column already exists
        # Backfill: rows from the single-channel era belong to the default channel
        c.execute("UPDATE videos SET channel='main_en' WHERE channel IS NULL")
        c.execute("UPDATE script_attempts SET channel='main_en' WHERE channel IS NULL")
        c.execute("CREATE INDEX IF NOT EXISTS idx_metrics_video    ON metrics(video_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_videos_yt        ON videos(youtube_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_videos_channel   ON videos(channel)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_attempts_run     ON script_attempts(run_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_attempts_niche   ON script_attempts(niche)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_attempts_phase   ON script_attempts(phase)")


def save_video(niche: str, script_hook: str, scene_desc: str,
               video_file: str, youtube_id: str = None, score: int = 0,
               script_full: str = None,
               seed_type: str = None, seed_source: str = None,
               seed_content: str = None,
               seed_url: str = None,
               run_id: str = None,
               criterion_scores: dict = None,
               attempts_used: int = None,
               final_temperature: float = None,
               score_reasoning: str = None,
               title: str = None,
               channel: str = "main_en",
               hold_reason: str = None,
               description: str = None,
               upload_status: str = "pending") -> int:
    crits = criterion_scores or {}
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO videos "
            "(niche, script_hook, script_full, scene_desc, "
            " seed_type, seed_source, seed_content, seed_url, "
            " youtube_id, video_file, score, "
            " run_id, score_specificity, score_hook, score_compression, "
            " score_loop, score_human, attempts_used, final_temperature, "
            " score_reasoning, title, channel, hold_reason, description, "
            " upload_status) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (niche, script_hook, script_full, scene_desc,
             seed_type, seed_source, seed_content, seed_url,
             youtube_id, video_file, score,
             run_id,
             crits.get("specificity"), crits.get("hook"),
             crits.get("compression"), crits.get("loop"), crits.get("human"),
             attempts_used, final_temperature, score_reasoning,
             title, channel, hold_reason, description, upload_status),
        )
        return cur.lastrowid


def save_attempt(*, run_id: str, niche: str, seed_type: str, phase: str,
                 attempt_n: int, hook: str = None, body: str = None,
                 temperature: float = None, total_score: int = None,
                 criterion_scores: dict = None, rejected_reason: str = None,
                 accepted: bool = False, cost_usd: float = 0.0,
                 ms: int = 0) -> int:
    """Persist one script-writer attempt for offline analysis."""
    import json as _json
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO script_attempts "
            "(run_id, niche, seed_type, phase, attempt_n, hook, body, "
            " temperature, total_score, criterion_scores, "
            " rejected_reason, accepted, cost_usd, ms) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, niche, seed_type, phase, attempt_n, hook, body,
             temperature, total_score,
             _json.dumps(criterion_scores) if criterion_scores else None,
             rejected_reason, 1 if accepted else 0, cost_usd, ms),
        )
        return cur.lastrowid


def update_youtube_id(video_id: int, youtube_id: str):
    with _conn() as c:
        c.execute("UPDATE videos SET youtube_id=? WHERE id=?", (youtube_id, video_id))


def update_title(video_id: int, title: str):
    with _conn() as c:
        c.execute("UPDATE videos SET title=? WHERE id=?", (title, video_id))


def update_metadata(video_id: int, title: str = None, description: str = None):
    """Dashboard edit form: update whichever of title/description was given."""
    sets, args = [], []
    if title is not None:
        sets.append("title=?");       args.append(title)
    if description is not None:
        sets.append("description=?"); args.append(description)
    if not sets:
        return
    args.append(video_id)
    with _conn() as c:
        c.execute(f"UPDATE videos SET {', '.join(sets)} WHERE id=?", args)


def set_upload_status(video_id: int, status: str):
    """status: 'pending' | 'approved' | 'rejected'."""
    with _conn() as c:
        c.execute("UPDATE videos SET upload_status=? WHERE id=?", (status, video_id))


def mark_upload_failed(video_id: int, error: str):
    """Record that an upload was ATTEMPTED and failed — the video may or may
    not exist on YouTube, so ops must check manually before re-uploading
    (blind retry risks a duplicate public video)."""
    with _conn() as c:
        c.execute(
            "UPDATE videos SET score_reasoning = "
            "COALESCE(score_reasoning,'') || ? WHERE id=?",
            (f"\n[UPLOAD FAILED — verify on YouTube before retry] {error[:400]}", video_id),
        )


# A YouTube id is 11 characters of [A-Za-z0-9_-]. Accepting a whole URL is the
# point: nobody wants to extract an id from
# https://www.youtube.com/shorts/dQw4w9WgXcQ?feature=share by hand, and the
# person doing it at 1am will get it wrong.
_YT_ID_RE = re.compile(r"(?:youtu\.be/|/shorts/|/embed/|[?&]v=|^)"
                       r"([A-Za-z0-9_-]{11})(?:[?&/#]|$)")


def extract_youtube_id(text: str) -> str | None:
    """The video id out of a URL, a share link, or a bare id. None if absent."""
    text = (text or "").strip()
    if not text:
        return None
    m = _YT_ID_RE.search(text)
    return m.group(1) if m else None


def mark_published(video_id: int, youtube_id: str,
                   published_at: str | None = None) -> bool:
    """Record that a video is live on YouTube, however it got there.

    WHY THIS EXISTS. Analytics only looks at rows carrying a youtube_id, and
    only the pipeline's own uploader ever set one. The owner published several
    videos by hand — which is the correct thing to do while nothing
    auto-uploads — and every one of them was invisible to the whole learning
    loop: no metrics fetched, no views recorded, so feedback_analyzer had no
    winners to learn hooks from and the pipeline's quality judgements stayed
    guesses about what works.

    A manual upload is not a lesser kind of publish. This is the row it was
    missing.
    """
    yt = extract_youtube_id(youtube_id)
    if not yt:
        return False
    when = published_at or datetime.now().strftime("%Y-%m-%d")
    with _conn() as c:
        c.execute("UPDATE videos SET youtube_id=?, upload_status='approved', "
                  "upload_date=? WHERE id=?", (yt, when, video_id))
    return True


def published_without_metrics(channel: str | None = None) -> list[dict]:
    """Live videos that have never had a metrics row.

    The honest answer to "is the loop actually closed": a youtube_id means the
    video is trackable, a metrics row means it has actually been tracked, and
    the gap between those two numbers is how much of the feedback loop is
    still theoretical.
    """
    q = ("SELECT v.id, v.youtube_id, v.title, v.upload_date, v.score "
         "FROM videos v LEFT JOIN metrics m ON m.video_id = v.id "
         "WHERE v.youtube_id IS NOT NULL AND m.id IS NULL")
    args: list = []
    if channel:
        q += " AND v.channel = ?"
        args.append(channel)
    q += " ORDER BY v.upload_date DESC"
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    return [{"id": r[0], "youtube_id": r[1], "title": r[2],
             "upload_date": r[3], "score": r[4]} for r in rows]


def get_recent_tracked_videos(days: int = 30, channel: str = None) -> list[dict]:
    """Return videos uploaded in last N days that have a youtube_id.
    channel=None keeps legacy behavior (all channels)."""
    q = ("SELECT id, youtube_id FROM videos "
         "WHERE youtube_id IS NOT NULL AND upload_date >= date('now', ?)")
    args: list = [f"-{days} days"]
    if channel:
        q += " AND channel = ?"
        args.append(channel)
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    return [{"id": r[0], "youtube_id": r[1]} for r in rows]


# Back-compat alias (old name was misleading – it never returned "untracked").
def get_untracked_videos() -> list[dict]:
    return get_recent_tracked_videos(days=365)


def save_metrics(video_id: int, views: int, watch_pct: float,
                 ctr: float, likes: int):
    with _conn() as c:
        c.execute(
            "INSERT INTO metrics (video_id, views, watch_pct, ctr, likes) VALUES (?,?,?,?,?)",
            (video_id, views, watch_pct, ctr, likes),
        )


if __name__ == "__main__":
    init_db()
    print(f"DB initialized at {DB_FILE}")
