#!/usr/bin/env python3
"""
db_manager.py
SQLite helpers for tracking produced videos and their analytics.
- WAL mode for concurrent-safe reads while writing
- Incremental fetch helper (get_recent_tracked_videos) so analytics_fetcher
  doesn't re-pull metrics for the entire history every day
"""

import sqlite3
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
        ):
            try:
                c.execute(ddl)
            except Exception:
                pass  # column already exists
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
               run_id: str = None,
               criterion_scores: dict = None,
               attempts_used: int = None,
               final_temperature: float = None,
               score_reasoning: str = None,
               title: str = None,
               channel: str = "main_en") -> int:
    crits = criterion_scores or {}
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO videos "
            "(niche, script_hook, script_full, scene_desc, "
            " seed_type, seed_source, seed_content, "
            " youtube_id, video_file, score, "
            " run_id, score_specificity, score_hook, score_compression, "
            " score_loop, score_human, attempts_used, final_temperature, "
            " score_reasoning, title, channel) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (niche, script_hook, script_full, scene_desc,
             seed_type, seed_source, seed_content,
             youtube_id, video_file, score,
             run_id,
             crits.get("specificity"), crits.get("hook"),
             crits.get("compression"), crits.get("loop"), crits.get("human"),
             attempts_used, final_temperature, score_reasoning,
             title, channel),
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
