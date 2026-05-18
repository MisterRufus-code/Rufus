"""
Rufus Memory Layer — persistent intelligence across pipeline runs.

PATCHED — fix 1.5 from REVIEW.md:
  SQLite is now in WAL mode with a 30s busy timeout. This is required for
  safe multi-process access (supervisor running niche pipelines in parallel).

The system remembers every hook, script, and topic it ever generated,
plus performance outcomes. Future runs query this memory to:
  - Avoid repeating weak hooks
  - Reuse proven phrase structures
  - Surface what worked in each niche
  - Feed the Hook Genome with fitness data
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()

_DB_PATH = Path("data/rufus_memory.db")
_lock = threading.Lock()
_INITIALIZED = False  # set once per process after WAL pragmas applied


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """One-time WAL setup. Idempotent — safe to call repeatedly."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")  # 30s — matches connect timeout


def _init_db(conn: sqlite3.Connection) -> None:
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS hooks (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        niche           TEXT    NOT NULL,
        topic           TEXT,
        hook_text       TEXT    NOT NULL,
        viral_score     REAL    DEFAULT 0,
        hook_score      REAL    DEFAULT 0,
        retention_score REAL    DEFAULT 0,
        views           INTEGER DEFAULT 0,
        watch_time_pct  REAL    DEFAULT 0,
        generation      INTEGER DEFAULT 0,
        genome_parent   TEXT,
        created_at      REAL    DEFAULT (unixepoch()),
        used_at         REAL
    );

    CREATE TABLE IF NOT EXISTS scripts (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        niche           TEXT    NOT NULL,
        topic           TEXT,
        title           TEXT,
        hook_id         INTEGER REFERENCES hooks(id),
        sections_json   TEXT,
        retention_score REAL    DEFAULT 0,
        viral_score     REAL    DEFAULT 0,
        views           INTEGER DEFAULT 0,
        created_at      REAL    DEFAULT (unixepoch())
    );

    CREATE TABLE IF NOT EXISTS topics (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        niche       TEXT    NOT NULL,
        topic       TEXT    NOT NULL,
        source      TEXT,
        momentum    REAL    DEFAULT 0,
        used        INTEGER DEFAULT 0,
        outcome     REAL    DEFAULT 0,
        created_at  REAL    DEFAULT (unixepoch())
    );

    CREATE TABLE IF NOT EXISTS footage_hashes (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        niche       TEXT    NOT NULL DEFAULT '',
        channel_id  TEXT    NOT NULL DEFAULT '',
        url         TEXT,
        file_hash   TEXT,
        filename    TEXT,
        render_id   TEXT    NOT NULL DEFAULT '',
        created_at  REAL    DEFAULT (unixepoch())
    );

    CREATE INDEX IF NOT EXISTS idx_hooks_niche ON hooks(niche);
    CREATE INDEX IF NOT EXISTS idx_hooks_score ON hooks(viral_score DESC);
    CREATE INDEX IF NOT EXISTS idx_scripts_niche ON scripts(niche);
    CREATE INDEX IF NOT EXISTS idx_topics_niche ON topics(niche);
    CREATE INDEX IF NOT EXISTS idx_footage_niche ON footage_hashes(niche, channel_id, created_at DESC);
    CREATE INDEX IF NOT EXISTS idx_topics_dedup ON topics(niche, topic);
    """)
    conn.commit()


@contextmanager
def _db():
    global _INITIALIZED
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        # 30s timeout on connect; WAL allows readers while writer is active.
        conn = sqlite3.connect(str(_DB_PATH), timeout=30.0)
        conn.row_factory = sqlite3.Row
        try:
            if not _INITIALIZED:
                _apply_pragmas(conn)
                _init_db(conn)
                _INITIALIZED = True
            yield conn
            conn.commit()
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Hook memory
# ---------------------------------------------------------------------------

@dataclass
class HookMemory:
    id:             int
    niche:          str
    topic:          str
    hook_text:      str
    viral_score:    float
    retention_score: float
    views:          int
    generation:     int
    created_at:     float

    def age_days(self) -> float:
        return (time.time() - self.created_at) / 86400

    def fitness(self) -> float:
        base = self.viral_score * 0.6 + self.retention_score * 0.4
        view_bonus = min(2.0, self.views / 5000) if self.views > 0 else 0
        recency = max(0.5, 1.0 - self.age_days() / 30)
        return base * recency + view_bonus


def store_hook(
    niche: str,
    topic: str,
    hook_text: str,
    viral_score: float,
    hook_score: float = 0,
    retention_score: float = 0,
    generation: int = 0,
    genome_parent: Optional[str] = None,
) -> int:
    with _db() as conn:
        cur = conn.execute(
            """INSERT INTO hooks
               (niche, topic, hook_text, viral_score, hook_score,
                retention_score, generation, genome_parent)
               VALUES (?,?,?,?,?,?,?,?)""",
            (niche, topic, hook_text, viral_score, hook_score,
             retention_score, generation, genome_parent),
        )
        return cur.lastrowid or 0


def top_hooks(
    niche: str,
    min_score: float = 6.0,
    limit: int = 20,
    days: int = 90,
) -> list[HookMemory]:
    since = time.time() - days * 86400
    with _db() as conn:
        rows = conn.execute(
            """SELECT id, niche, topic, hook_text, viral_score, retention_score,
                      views, generation, created_at
               FROM hooks
               WHERE niche = ? AND viral_score >= ? AND created_at >= ?
               ORDER BY viral_score DESC LIMIT ?""",
            (niche, min_score, since, limit),
        ).fetchall()
    return [HookMemory(**dict(r)) for r in rows]


def update_hook_performance(hook_id: int, views: int, watch_time_pct: float) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE hooks SET views=?, watch_time_pct=? WHERE id=?",
            (views, watch_time_pct, hook_id),
        )


# ---------------------------------------------------------------------------
# Script memory
# ---------------------------------------------------------------------------

def store_script(
    niche: str,
    topic: str,
    title: str,
    sections: list[dict],
    retention_score: float,
    viral_score: float,
    hook_id: Optional[int] = None,
) -> int:
    with _db() as conn:
        cur = conn.execute(
            """INSERT INTO scripts
               (niche, topic, title, hook_id, sections_json, retention_score, viral_score)
               VALUES (?,?,?,?,?,?,?)""",
            (niche, topic, title, hook_id,
             json.dumps(sections), retention_score, viral_score),
        )
        return cur.lastrowid or 0


def best_scripts(niche: str, limit: int = 5) -> list[dict]:
    with _db() as conn:
        rows = conn.execute(
            """SELECT title, retention_score, viral_score, sections_json
               FROM scripts WHERE niche=?
               ORDER BY retention_score * 0.5 + viral_score * 0.5 DESC LIMIT ?""",
            (niche, limit),
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Topic memory — with dedup helper for the new trends_v2 module
# ---------------------------------------------------------------------------

def store_topic(niche: str, topic: str, source: str = "", momentum: float = 0) -> None:
    with _db() as conn:
        # Skip duplicates within 30 days for the same niche
        existing = conn.execute(
            """SELECT id FROM topics
               WHERE niche=? AND LOWER(topic)=LOWER(?)
               AND created_at >= ?""",
            (niche, topic, time.time() - 30 * 86400),
        ).fetchone()
        if existing:
            return
        conn.execute(
            "INSERT INTO topics (niche, topic, source, momentum) VALUES (?,?,?,?)",
            (niche, topic, source, momentum),
        )


def mark_topic_used(niche: str, topic: str, outcome: float = 0) -> None:
    with _db() as conn:
        conn.execute(
            "UPDATE topics SET used=1, outcome=? WHERE niche=? AND topic=? AND used=0",
            (outcome, niche, topic),
        )


def unused_topics(niche: str, limit: int = 10) -> list[str]:
    with _db() as conn:
        rows = conn.execute(
            """SELECT topic FROM topics WHERE niche=? AND used=0
               ORDER BY momentum DESC, created_at DESC LIMIT ?""",
            (niche, limit),
        ).fetchall()
    return [r["topic"] for r in rows]


def was_topic_recently_used(niche: str, topic: str, days: int = 60) -> bool:
    """Return True if this topic (case-insensitive) was used in the last N days."""
    with _db() as conn:
        row = conn.execute(
            """SELECT id FROM topics
               WHERE niche=? AND LOWER(topic)=LOWER(?) AND used=1
               AND created_at >= ?""",
            (niche, topic, time.time() - days * 86400),
        ).fetchone()
    return row is not None


# ---------------------------------------------------------------------------
# Intelligence queries
# ---------------------------------------------------------------------------

def memory_context(niche: str) -> str:
    hooks = top_hooks(niche, min_score=7.0, limit=5)
    scripts = best_scripts(niche, limit=3)

    lines = [f"=== RUFUS MEMORY: {niche.upper()} ==="]

    if hooks:
        lines.append("\nBEST-PERFORMING HOOK STRUCTURES (replicate these patterns):")
        for h in hooks:
            lines.append(f"  [{h.viral_score:.1f}/10] {h.hook_text[:90]}")

    if scripts:
        lines.append("\nHIGH-RETENTION SCRIPT TITLES (study these structures):")
        for s in scripts:
            lines.append(f"  [ret={s['retention_score']:.0f}] {s['title']}")

    if not hooks and not scripts:
        lines.append("(No memory yet — first run in this niche)")

    lines.append("=== END MEMORY ===")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Footage hash tracking
# ---------------------------------------------------------------------------

def _hash_file(path: str) -> str:
    import hashlib
    h = hashlib.md5()
    try:
        with open(path, "rb") as f:
            h.update(f.read(1_048_576))
    except OSError:
        pass
    return h.hexdigest()


def record_footage_used(
    urls_or_paths: list[str],
    render_id: str,
    niche: str = "",
    channel_id: str = "",
) -> None:
    import hashlib
    with _db() as conn:
        for item in urls_or_paths:
            p = Path(item)
            file_hash = _hash_file(item) if p.exists() else hashlib.md5(item.encode()).hexdigest()
            url = item if item.startswith("http") else ""
            filename = p.name if p.exists() else ""
            conn.execute(
                """INSERT INTO footage_hashes
                   (niche, channel_id, url, file_hash, filename, render_id)
                   VALUES (?,?,?,?,?,?)""",
                (niche, channel_id, url, file_hash, filename, render_id),
            )


def get_recently_used_hashes(
    niche: str = "",
    channel_id: str = "",
    last_n_renders: int = 3,
) -> set[str]:
    with _db() as conn:
        rows = conn.execute(
            """SELECT DISTINCT render_id FROM footage_hashes
               WHERE niche=? AND channel_id=?
               ORDER BY created_at DESC LIMIT ?""",
            (niche, channel_id, last_n_renders),
        ).fetchall()
        if not rows:
            return set()
        render_ids = tuple(r["render_id"] for r in rows)
        placeholders = ",".join("?" * len(render_ids))
        hash_rows = conn.execute(
            f"SELECT file_hash FROM footage_hashes WHERE render_id IN ({placeholders})",
            render_ids,
        ).fetchall()
    return {r["file_hash"] for r in hash_rows}


def is_footage_used_recently(
    url_or_path: str,
    niche: str = "",
    channel_id: str = "",
    last_n_renders: int = 3,
) -> bool:
    import hashlib
    p = Path(url_or_path)
    file_hash = _hash_file(url_or_path) if p.exists() else hashlib.md5(url_or_path.encode()).hexdigest()
    return file_hash in get_recently_used_hashes(niche, channel_id, last_n_renders)


def stats() -> dict:
    with _db() as conn:
        hooks_total   = conn.execute("SELECT COUNT(*) FROM hooks").fetchone()[0]
        scripts_total = conn.execute("SELECT COUNT(*) FROM scripts").fetchone()[0]
        topics_total  = conn.execute("SELECT COUNT(*) FROM topics").fetchone()[0]
        avg_viral     = conn.execute(
            "SELECT AVG(viral_score) FROM hooks WHERE viral_score > 0"
        ).fetchone()[0] or 0
    return {
        "hooks":         hooks_total,
        "scripts":       scripts_total,
        "topics":        topics_total,
        "avg_viral_score": round(avg_viral, 2),
    }
