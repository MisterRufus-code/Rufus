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
        c.execute("CREATE INDEX IF NOT EXISTS idx_metrics_video ON metrics(video_id)")
        c.execute("CREATE INDEX IF NOT EXISTS idx_videos_yt    ON videos(youtube_id)")


def save_video(niche: str, script_hook: str, scene_desc: str,
               video_file: str, youtube_id: str = None, score: int = 0,
               script_full: str = None,
               seed_type: str = None, seed_source: str = None,
               seed_content: str = None) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO videos "
            "(niche, script_hook, script_full, scene_desc, "
            " seed_type, seed_source, seed_content, "
            " youtube_id, video_file, score) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (niche, script_hook, script_full, scene_desc,
             seed_type, seed_source, seed_content,
             youtube_id, video_file, score),
        )
        return cur.lastrowid


def update_youtube_id(video_id: int, youtube_id: str):
    with _conn() as c:
        c.execute("UPDATE videos SET youtube_id=? WHERE id=?", (youtube_id, video_id))


def get_recent_tracked_videos(days: int = 30) -> list[dict]:
    """Return videos uploaded in last N days that have a youtube_id."""
    with _conn() as c:
        rows = c.execute(
            "SELECT id, youtube_id FROM videos "
            "WHERE youtube_id IS NOT NULL "
            "  AND upload_date >= date('now', ?)",
            (f"-{days} days",),
        ).fetchall()
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
