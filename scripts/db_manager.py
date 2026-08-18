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
            # When YouTube will make it visible, for an upload that went up
            # private with a publishAt. Empty for everything else. Without it
            # a scheduled video is indistinguishable from one that is private
            # forever, which is the state the owner was actually in.
            "ALTER TABLE videos ADD COLUMN publish_at TEXT",
            # WHEN, TO THE MINUTE, AND WHICH OF THE TWO WHENS.
            #
            # `upload_date` is date('now') — no time at all — and it means two
            # different things depending on how a video reached YouTube. The
            # pipeline's own uploader never touches it, so for those rows it
            # is the day the video was GENERATED. mark_published overwrites it
            # with today, so for a hand-published row it is the day it went
            # LIVE. One column, two meanings, and no way to tell them apart.
            #
            # A video generated Monday and approved Thursday is one row that
            # answers "when?" with Monday, and the owner asking when a video
            # went out cannot get the hour at all.
            #
            # So: two columns that each mean one thing, and neither is ever
            # rewritten to mean the other. upload_date stays exactly as it is
            # — report.py, review_recent.py and four dashboard queries filter
            # on it with date('now', ?), and breaking those to fix a display
            # is the wrong trade.
            "ALTER TABLE videos ADD COLUMN created_at TEXT",
            "ALTER TABLE videos ADD COLUMN uploaded_at TEXT",
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
        # Backfill created_at from upload_date — WITHOUT inventing a time.
        # Those rows genuinely do not have one, and "2026-08-15 00:00:00"
        # would read as "uploaded at midnight" to anyone glancing at the
        # column. A date with no time is the true answer for them.
        c.execute("UPDATE videos SET created_at = upload_date "
                  "WHERE created_at IS NULL AND upload_date IS NOT NULL")
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
        # WHAT THE SCOUT SAW, and the reason it is a table rather than a
        # longer prompt. "Research for days" is not a model thinking for
        # hours — it is the same competitor video seen on three consecutive
        # passes with its views climbing, which is a different fact from the
        # same video seen once. Only a store that survives between passes can
        # tell those apart.
        c.execute("""
            CREATE TABLE IF NOT EXISTS scout_observations (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                seen_at        TEXT DEFAULT (datetime('now')),
                video_id       TEXT,
                channel_id     TEXT,
                channel_title  TEXT,
                title          TEXT,
                published_at   TEXT,
                views          INTEGER DEFAULT 0,
                channel_median INTEGER DEFAULT 0,
                outperformance REAL    DEFAULT 0
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_obs_video "
                  "ON scout_observations(video_id, seen_at)")
        # A PROPOSAL IS NOT A VIDEO. A script is cents and seconds; a render is
        # hours of the 3090. The scout writes here and stops, a human approves,
        # and only then does a normal run render it — which is what makes being
        # wrong cheap enough to allow.
        c.execute("""
            CREATE TABLE IF NOT EXISTS proposals (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT DEFAULT (datetime('now')),
                channel     TEXT,
                niche       TEXT,
                topic       TEXT,
                hook        TEXT,
                script      TEXT,
                score       INTEGER DEFAULT 0,
                evidence    TEXT,
                status      TEXT DEFAULT 'pending',
                decided_at  TEXT,
                cost_usd    REAL DEFAULT 0
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


def _now() -> str:
    """Local wall-clock, to the second.

    LOCAL AND NOT UTC, deliberately. This timestamp exists to answer "when
    did that video go out" for one person looking at one dashboard on the
    machine that made it, and an owner in UTC+3 reading 00:51 for something
    they watched render at 03:51 would file that as a bug. YouTube's own
    publishAt stays RFC3339/UTC where the API requires it — that is a
    different field answering a different question.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


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
            " upload_status, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (niche, script_hook, script_full, scene_desc,
             seed_type, seed_source, seed_content, seed_url,
             youtube_id, video_file, score,
             run_id,
             crits.get("specificity"), crits.get("hook"),
             crits.get("compression"), crits.get("loop"), crits.get("human"),
             attempts_used, final_temperature, score_reasoning,
             title, channel, hold_reason, description, upload_status,
             _now()),
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
        c.execute("UPDATE videos SET youtube_id=?, uploaded_at=? WHERE id=?",
                  (youtube_id, _now(), video_id))


def set_publish_at(video_id: int, when: str):
    """Record when YouTube will make this video visible. "" = already visible.

    Only an upload that went up PRIVATE carries a publishAt — that is
    YouTube's own rule, not ours — so this is empty for a public or unlisted
    one, and empty is the honest answer rather than a missing row.
    """
    with _conn() as c:
        c.execute("UPDATE videos SET publish_at=? WHERE id=?",
                  (when or "", video_id))


def scheduled(channel: str | None = None) -> list[dict]:
    """Videos with a publish time still in the future, soonest first.

    The list nobody could see. A scheduled upload looks identical to one that
    is private forever from every page in this dashboard, and the difference
    is the whole question of whether the channel is publishing.
    """
    q = ("SELECT id, title, script_hook, youtube_id, publish_at, niche, channel "
         "FROM videos WHERE publish_at IS NOT NULL AND publish_at != '' "
         "AND publish_at > strftime('%Y-%m-%dT%H:%M:%SZ','now')")
    args: list = []
    if channel:
        q += " AND channel = ?"
        args.append(channel)
    q += " ORDER BY publish_at ASC"
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    cols = ["id", "title", "script_hook", "youtube_id", "publish_at",
            "niche", "channel"]
    return [dict(zip(cols, r)) for r in rows]


# ── The scout's memory ───────────────────────────────────────────────────────

def record_observations(videos: list[dict]) -> int:
    """Append this pass's competitor observations. Returns how many landed.

    APPEND, never update. The same video seen on Monday at 4,000 views and on
    Thursday at 40,000 is the observation worth having, and an UPDATE would
    throw the first half of it away — which is the whole difference between a
    scout that accumulates and one that only ever knows about today.
    """
    if not videos:
        return 0
    rows = [(v.get("video_id", ""), v.get("channel_id", ""),
             v.get("channel_title", ""), v.get("title", ""),
             v.get("published_at", ""), int(v.get("views", 0) or 0),
             int(v.get("channel_median", 0) or 0),
             float(v.get("outperformance", 0) or 0)) for v in videos]
    with _conn() as c:
        c.executemany(
            "INSERT INTO scout_observations (video_id, channel_id, "
            "channel_title, title, published_at, views, channel_median, "
            "outperformance) VALUES (?,?,?,?,?,?,?,?)", rows)
    return len(rows)


def rising(min_outperformance: float = 2.0, days: int = 14,
           limit: int = 20) -> list[dict]:
    """The strongest recent observations, one row per video (its latest).

    Latest-per-video rather than every sighting, because a video seen eight
    times would otherwise crowd out seven others it is not better than.
    """
    # MAX(id) AND NOT MAX(seen_at). SQLite's bare-column rule picks the row
    # belonging to the max, which is what makes "the latest sighting" work at
    # all — but seen_at has one-second resolution, so two passes in the same
    # second TIE and the row chosen is then arbitrary. A test that recorded a
    # video at 4,000 views and again at 40,000 got the 4,000 back. id is a
    # rowid: monotonic, and it cannot tie.
    q = ("SELECT video_id, channel_title, title, views, outperformance, "
         "       seen_at, COUNT(*) AS sightings, MAX(id) "
         "FROM scout_observations "
         "WHERE seen_at >= datetime('now', ?) AND outperformance >= ? "
         "GROUP BY video_id ORDER BY outperformance DESC LIMIT ?")
    with _conn() as c:
        rows = c.execute(q, (f"-{int(days)} days", float(min_outperformance),
                             int(limit))).fetchall()
    cols = ["video_id", "channel_title", "title", "views", "outperformance",
            "seen_at", "sightings", "_max_id"]
    return [{k: v for k, v in zip(cols, r) if k != "_max_id"} for r in rows]


# ── Proposals ────────────────────────────────────────────────────────────────

def recent_titles(limit: int = 200, channel: str | None = None) -> list[str]:
    """What this channel has already made, as titles/hooks.

    For the scout's duplicate check, which needs a cheap list of subjects
    rather than the full rows every other reader here wants.
    """
    q = ("SELECT COALESCE(NULLIF(title,''), script_hook) FROM videos "
         "WHERE COALESCE(NULLIF(title,''), script_hook) IS NOT NULL")
    args: list = []
    if channel:
        q += " AND channel = ?"
        args.append(channel)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    with _conn() as c:
        return [r[0] for r in c.execute(q, args).fetchall() if r[0]]


def save_proposal(*, channel: str, niche: str, topic: str, hook: str,
                  script: str, score: int, evidence: str,
                  cost_usd: float = 0.0) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO proposals (channel, niche, topic, hook, script, "
            "score, evidence, cost_usd) VALUES (?,?,?,?,?,?,?,?)",
            (channel, niche, topic, hook, script, int(score), evidence,
             float(cost_usd)))
        return cur.lastrowid


def proposals(status: str | None = "pending", limit: int = 50) -> list[dict]:
    q = ("SELECT id, created_at, channel, niche, topic, hook, script, score, "
         "evidence, status, decided_at, cost_usd FROM proposals")
    args: list = []
    if status:
        q += " WHERE status = ?"
        args.append(status)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    cols = ["id", "created_at", "channel", "niche", "topic", "hook", "script",
            "score", "evidence", "status", "decided_at", "cost_usd"]
    return [dict(zip(cols, r)) for r in rows]


def pending_proposal_count(channel: str | None = None) -> int:
    q = "SELECT COUNT(*) FROM proposals WHERE status='pending'"
    args: list = []
    if channel:
        q += " AND channel = ?"
        args.append(channel)
    with _conn() as c:
        return int(c.execute(q, args).fetchone()[0])


def decide_proposal(proposal_id: int, status: str) -> bool:
    """'approved' or 'rejected'. False if the id is unknown."""
    if status not in ("approved", "rejected"):
        return False
    with _conn() as c:
        cur = c.execute(
            "UPDATE proposals SET status=?, decided_at=datetime('now') "
            "WHERE id=? AND status='pending'", (status, int(proposal_id)))
        return cur.rowcount > 0


def proposal_cost_today() -> float:
    """What the scout has spent on prose today, for the daily ceiling."""
    with _conn() as c:
        row = c.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM proposals "
            "WHERE date(created_at) = date('now')").fetchone()
    return float(row[0] or 0.0)


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
                  "upload_date=?, uploaded_at=COALESCE(uploaded_at, ?) "
                  "WHERE id=?", (yt, when, published_at or _now(), video_id))
    return True


def history(limit: int = 200, channel: str | None = None) -> list[dict]:
    """Every video, newest first, with both timestamps and how it ended up.

    Ordered by when it was MADE, not when it went live: a video sitting in the
    approval queue has no upload time at all, and ordering by a column that is
    NULL for exactly the rows you are waiting on puts them somewhere arbitrary.
    """
    q = ("SELECT id, COALESCE(created_at, upload_date) AS created_at, "
         "uploaded_at, publish_at, title, script_hook, niche, channel, "
         "score, upload_status, youtube_id, hold_reason "
         "FROM videos")
    args: list = []
    if channel:
        q += " WHERE channel=?"
        args.append(channel)
    q += " ORDER BY COALESCE(created_at, upload_date) DESC, id DESC LIMIT ?"
    args.append(limit)
    cols = ["id", "created_at", "uploaded_at", "publish_at", "title",
            "script_hook", "niche", "channel", "score", "upload_status",
            "youtube_id", "hold_reason"]
    with _conn() as c:
        return [dict(zip(cols, r)) for r in c.execute(q, args).fetchall()]


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
