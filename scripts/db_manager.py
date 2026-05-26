#!/usr/bin/env python3
"""
db_manager.py
SQLite helpers for tracking produced videos and their analytics.
"""

import sqlite3
from pathlib import Path

ROOT    = Path(__file__).parent.parent
DB_FILE = ROOT / "rufus.db"


def _conn():
    return sqlite3.connect(str(DB_FILE))


def init_db():
    with _conn() as c:
        c.execute("""
            CREATE TABLE IF NOT EXISTS videos (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                upload_date TEXT    DEFAULT (date('now')),
                niche       TEXT,
                script_hook TEXT,
                scene_desc  TEXT,
                youtube_id  TEXT,
                video_file  TEXT,
                score       INTEGER DEFAULT 0
            )
        """)
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


def save_video(niche: str, script_hook: str, scene_desc: str,
               video_file: str, youtube_id: str = None, score: int = 0) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO videos (niche, script_hook, scene_desc, youtube_id, video_file, score) "
            "VALUES (?,?,?,?,?,?)",
            (niche, script_hook, scene_desc, youtube_id, video_file, score),
        )
        return cur.lastrowid


def update_youtube_id(video_id: int, youtube_id: str):
    with _conn() as c:
        c.execute("UPDATE videos SET youtube_id=? WHERE id=?", (youtube_id, video_id))


def get_untracked_videos() -> list[dict]:
    with _conn() as c:
        rows = c.execute(
            "SELECT id, youtube_id FROM videos WHERE youtube_id IS NOT NULL"
        ).fetchall()
    return [{"id": r[0], "youtube_id": r[1]} for r in rows]


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
