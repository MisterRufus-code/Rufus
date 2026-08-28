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
            # WHO DECIDED. Two people work this channel now, and until this
            # column every approval, rejection and pick was anonymous — the
            # database recorded what was chosen and never once recorded by
            # whom. "Who made this video" had no answer to give.
            #
            # A NAME, NOT A USER ID. config/users.json is a small hand-edited
            # file with no stable ids in it, and a revoked user must not turn
            # a year of history into dangling references. The name is what a
            # person recognises and what the page prints; if it is later
            # renamed the old rows keep saying who it was at the time, which
            # is the honest answer for a record of decisions.
            "ALTER TABLE videos ADD COLUMN decided_by TEXT",
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
        # WHAT WAS CHOSEN, AND WHAT IT WAS CHOSEN OVER. script_attempts already
        # records every draft the writer made on its way to one answer; this
        # records the drafts a PERSON was shown and ruled between. The
        # difference matters: an attempt the writer discarded says the writer
        # scored it low, and a candidate the owner passed over says a human
        # looked at both and preferred the other one.
        #
        # That is the labelled preference pair this channel cannot get any
        # other way yet. feedback_analyzer needs view counts and there are
        # none; a rejected sibling needs nothing but the click that already
        # happened. Rejected rows are therefore kept, not deleted — they are
        # half of every pair.
        c.execute("""
            CREATE TABLE IF NOT EXISTS script_candidates (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT DEFAULT (datetime('now')),
                proposal_id INTEGER,
                channel     TEXT,
                niche       TEXT,
                topic       TEXT,
                hook_style  TEXT,
                hook        TEXT,
                script      TEXT,
                score       INTEGER DEFAULT 0,
                run_id      TEXT,
                cost_usd    REAL DEFAULT 0,
                status      TEXT DEFAULT 'pending',
                decided_at  TEXT,
                fact_ok     INTEGER DEFAULT 1,
                fact_reason TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_cand_proposal "
                  "ON script_candidates(proposal_id, status)")
        # THE ONE LINK THE CHAIN WAS MISSING. project → script → gallery →
        # voice already threads on the far side: gallery_sets.candidate_id is
        # the chosen script, voice_takes.set_id is the chosen gallery. Only the
        # first hop had nothing, because candidates were invented for the scout
        # and the scout has proposals rather than projects.
        try:
            c.execute("ALTER TABLE script_candidates ADD COLUMN project_id INTEGER")
        except Exception:
            pass
        c.execute("CREATE INDEX IF NOT EXISTS idx_cand_project "
                  "ON script_candidates(project_id, status)")
        # A GATE THAT LABELS INSTEAD OF REJECTING. With a person ruling between
        # three finished scripts, a threshold that silently discards one is
        # deciding something the person is here to decide — and it was
        # discarding good scripts, which is the complaint that started this.
        # The score and the fact gate still run; they land in the row and on
        # the card, where a reviewer can weigh them.
        for ddl in ("ALTER TABLE script_candidates ADD COLUMN fact_ok INTEGER DEFAULT 1",
                    "ALTER TABLE script_candidates ADD COLUMN fact_reason TEXT",
                    "ALTER TABLE script_candidates ADD COLUMN decided_by TEXT"):
            try:
                c.execute(ddl)
            except Exception:
                pass  # column already exists
        # TWO FULL GALLERIES, AND THE SWAP THAT MAKES TWO ENOUGH.
        #
        # A gallery of sixteen pictures is sixteen independent draws, not one
        # artefact. Variant A comes back best on shot 3 and worst on shot 9;
        # variant B the other way round. Choosing a whole bundle throws away
        # the good half of the other one — so the unit of choice is the SHOT,
        # and a set is picked as a base and then corrected per shot.
        #
        # Two rather than three because of what that buys: at a defect rate
        # around one in five, the chance both variants fail the same shot is
        # about four per cent, so a sixteen-shot set expects well under one
        # unfixable shot. A third variant takes that to under a sixth of a shot
        # for another thirteen minutes of the 3090, which is not a trade worth
        # making.
        c.execute("""
            CREATE TABLE IF NOT EXISTS gallery_sets (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at   TEXT DEFAULT (datetime('now')),
                candidate_id INTEGER,
                channel      TEXT,
                niche        TEXT,
                topic        TEXT,
                script_file  TEXT,
                n_variants   INTEGER DEFAULT 2,
                status       TEXT DEFAULT 'pending',
                decided_at   TEXT,
                n_beats      INTEGER DEFAULT 0
            )
        """)
        # HOW MANY PICTURES THIS SET INTENDS TO DRAW, known before the first
        # one exists. Progress was inferred from the images already drawn, so
        # a set with none yet reported a target of zero — which reads as
        # finished, and the wizard showed the completed gallery view over an
        # empty table while ComfyUI was still rendering. A target has to come
        # from the plan, not from the output.
        for ddl in ("ALTER TABLE gallery_sets ADD COLUMN n_beats INTEGER DEFAULT 0",
                    "ALTER TABLE gallery_sets ADD COLUMN decided_by TEXT"):
            try:
                c.execute(ddl)
            except Exception:
                pass
        # status per IMAGE, the same idiom script_candidates uses and for the
        # same reason: the row that lost is half of a labelled preference pair,
        # and here there is one per shot rather than one per video. Sixteen
        # judgements a person makes anyway, recorded instead of discarded.
        c.execute("""
            CREATE TABLE IF NOT EXISTS gallery_images (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                set_id     INTEGER,
                variant    INTEGER,
                beat_index INTEGER,
                path       TEXT,
                prompt     TEXT,
                seed       INTEGER,
                status     TEXT DEFAULT 'pending'
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_gimg_set "
                  "ON gallery_images(set_id, beat_index, variant)")
        # Per-shot, because the swaps are where the attention went: taking a
        # base is one click and correcting eight shots is eight judgements, and
        # a set credited only to whoever pressed "take all from A" would hide
        # the person who actually did that work.
        for ddl in ("ALTER TABLE gallery_images ADD COLUMN decided_by TEXT",
                    # WHEN EACH PICTURE LANDED, which is the only honest basis
                    # for "how long is left". The estimate used to divide the
                    # time since the SET ROW was written by the pictures drawn
                    # — and that row is written before three voice takes and a
                    # storyboard call, so hours of setup were being averaged
                    # into the per-picture rate. Nine of thirty-eight drawn
                    # reported eleven hours remaining while ComfyUI was doing
                    # one every nineteen seconds.
                    "ALTER TABLE gallery_images ADD COLUMN created_at TEXT"):
            try:
                c.execute(ddl)
            except Exception:
                pass
        # THREE READS OF THE HOOK, AND ONLY THE HOOK.
        #
        # Audio is the one thing on this list that cannot be skimmed — it plays
        # at one times speed and there is no glancing at it. Three full
        # forty-five-second takes is two and a half minutes of listening; three
        # eight-second hooks is twenty-four seconds, and if the hook read lands
        # the rest follows it. So the take is the opening line.
        #
        # The VOICE is not what varies. A channel whose narrator changes every
        # video has no narrator, and that is channel identity rather than a
        # per-video decision. What varies is the tone the director assigns
        # beat 0 — the same lever that already sizes its pauses and grades its
        # picture.
        c.execute("""
            CREATE TABLE IF NOT EXISTS voice_takes (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at  TEXT DEFAULT (datetime('now')),
                set_id      INTEGER,
                channel     TEXT,
                topic       TEXT,
                tone        TEXT,
                text        TEXT,
                path        TEXT,
                status      TEXT DEFAULT 'pending',
                decided_at  TEXT,
                seconds     REAL DEFAULT 0,
                spans       TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_vtake_set "
                  "ON voice_takes(set_id, status)")
        # Measured from the audio by Whisper, not guessed from the word count.
        # A take is recorded before the pictures are drawn precisely so the
        # gallery stage can say how long each shot will be on screen.
        for ddl in ("ALTER TABLE voice_takes ADD COLUMN seconds REAL DEFAULT 0",
                    "ALTER TABLE voice_takes ADD COLUMN spans TEXT",
                    "ALTER TABLE voice_takes ADD COLUMN decided_by TEXT"):
            try:
                c.execute(ddl)
            except Exception:
                pass
        # ONE VIDEO IN PROGRESS, ACROSS ALL OF ITS STAGES.
        #
        # The four choosing stages each grew their own table and their own
        # page, and separately they work — but a person making a video does not
        # have four tasks, they have one, and nothing tied a chosen script back
        # to the topic it came from or forward to the pictures drawn for it.
        # Four tabs is what that looks like from the outside, and the owner
        # said so: "you made a mess".
        #
        # A project is the thread. It holds what has been decided so far and
        # which stage is open, which is what lets the wizard go BACKWARDS —
        # re-open the script stage, choose differently, and have the stages
        # after it know they are stale.
        c.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at   TEXT DEFAULT (datetime('now')),
                updated_at   TEXT DEFAULT (datetime('now')),
                channel      TEXT,
                niche        TEXT,
                title        TEXT,
                topic_source TEXT,
                script_id    INTEGER,
                script_file  TEXT,
                gallery_id   INTEGER,
                voice_id     INTEGER,
                stage        TEXT DEFAULT 'topic',
                status       TEXT DEFAULT 'open',
                video_id     INTEGER
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT (datetime('now')),
                author     TEXT,
                text       TEXT NOT NULL,
                priority   TEXT DEFAULT 'normal',
                notified   INTEGER DEFAULT 0,
                done_at    TEXT,
                done_by    TEXT
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_notes_open "
                  "ON notes(done_at, id)")
        # Who started it. Every stage below records who DECIDED; this records
        # who opened the project at all, which is the row the home page reads
        # to say whose video is in flight.
        try:
            c.execute("ALTER TABLE projects ADD COLUMN created_by TEXT")
        except Exception:
            pass
        # The topic options a project was offered, so REGEN has something to
        # replace and so the ones passed over are kept — same reason every
        # other stage keeps its losers.
        c.execute("""
            CREATE TABLE IF NOT EXISTS project_topics (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id INTEGER,
                title      TEXT,
                why        TEXT,
                status     TEXT DEFAULT 'pending'
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_ptopic_project "
                  "ON project_topics(project_id, status)")
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


# A PROPOSAL IS A TOPIC, and for a long time it was a topic with a script
# stapled to it that nothing ever read. The scout wrote a full script for every
# proposal; the dashboard showed it inside a <details>; and scout_approve then
# launched an ordinary run with topic= alone — which writes its own script from
# scratch. Six proposals meant six scripts paid for and six discarded, and the
# column was decoration the whole time.
#
# So the script arguments are optional now. The scout proposes topics, a person
# chooses one, and only then is prose paid for — against the one topic that
# survived rather than the five that did not.
def save_proposal(*, channel: str, niche: str, topic: str, evidence: str,
                  hook: str = "", script: str = "", score: int = 0,
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


def save_candidate(*, proposal_id: int | None, channel: str, niche: str,
                   topic: str, hook_style: str, hook: str, script: str,
                   score: int, run_id: str = "", cost_usd: float = 0.0,
                   fact_ok: bool = True, fact_reason: str = "",
                   project_id: int | None = None) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO script_candidates (proposal_id, channel, niche, "
            "topic, hook_style, hook, script, score, run_id, cost_usd, "
            "fact_ok, fact_reason, project_id) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (proposal_id, channel, niche, topic, hook_style, hook, script,
             int(score), run_id, float(cost_usd), 1 if fact_ok else 0,
             fact_reason, project_id))
        return cur.lastrowid


_CANDIDATE_COLS = ["id", "created_at", "proposal_id", "channel", "niche",
                   "topic", "hook_style", "hook", "script", "score", "run_id",
                   "cost_usd", "status", "decided_at", "fact_ok",
                   "fact_reason", "project_id", "decided_by"]


def candidates(proposal_id: int | None = None,
               status: str | None = None, limit: int = 50,
               project_id: int | None = None) -> list[dict]:
    """The scripts a person was shown, newest first.

    Ordered by score within a proposal so the set reads best-first, because a
    list of three scripts in generation order buries the one most likely to be
    picked under two that were not.
    """
    q = f"SELECT {', '.join(_CANDIDATE_COLS)} FROM script_candidates"
    where, args = [], []
    if proposal_id is not None:
        where.append("proposal_id = ?")
        args.append(int(proposal_id))
    if project_id is not None:
        where.append("project_id = ?")
        args.append(int(project_id))
    if status:
        where.append("status = ?")
        args.append(status)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY proposal_id DESC, score DESC, id ASC LIMIT ?"
    args.append(int(limit))
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    return [dict(zip(_CANDIDATE_COLS, r)) for r in rows]


def choose_candidate(candidate_id: int, by: str = "") -> dict | None:
    """Mark one candidate chosen and every sibling rejected. Returns it.

    BOTH SIDES IN ONE CALL, and that is the point rather than a convenience:
    the value here is the PAIR. A chosen row on its own says a script was made;
    a chosen row beside the two it beat says a person compared three and
    preferred this one, which is the only labelled preference this channel can
    collect before it has view counts.

    Returns None for an unknown id or one already decided, so a double-click on
    a slow page cannot re-decide a set and overwrite which sibling lost.
    """
    with _conn() as c:
        row = c.execute(
            f"SELECT {', '.join(_CANDIDATE_COLS)} FROM script_candidates "
            f"WHERE id = ? AND status = 'pending'", (int(candidate_id),)
        ).fetchone()
        if not row:
            return None
        chosen = dict(zip(_CANDIDATE_COLS, row))
        c.execute("UPDATE script_candidates SET status='chosen', "
                  "decided_at=datetime('now'), decided_by=? WHERE id=?",
                  (by or None, int(candidate_id)))
        # SIBLINGS SHARE A PROPOSAL *OR* A PROJECT, and only the first was
        # checked. The wizard writes candidates with a project and no proposal,
        # so choosing one rejected nothing: all three stayed pending, the stage
        # still offered them, and the preference pair — the entire reason the
        # losers are kept — was never recorded. NULL = NULL is not true in SQL,
        # which is why this has to name the column that is actually set rather
        # than matching on both and hoping.
        for column, value in (("proposal_id", chosen["proposal_id"]),
                              ("project_id", chosen.get("project_id"))):
            if value is None:
                continue
            c.execute(f"UPDATE script_candidates SET status='rejected', "
                      f"decided_at=datetime('now'), decided_by=? "
                      f"WHERE {column}=? AND id<>? AND status='pending'",
                      (by or None, int(value), int(candidate_id)))
        chosen["status"] = "chosen"
        return chosen


# ── projects: one video in progress, across all of its stages ────────────────

STAGES = ("topic", "script", "gallery", "voice", "render")

_PROJECT_COLS = ["id", "created_at", "updated_at", "channel", "niche", "title",
                 "topic_source", "script_id", "script_file", "gallery_id",
                 "voice_id", "stage", "status", "video_id", "created_by"]
_PTOPIC_COLS = ["id", "project_id", "title", "why", "status"]


def new_project(*, channel: str, niche: str, by: str = "") -> int:
    with _conn() as c:
        return c.execute(
            "INSERT INTO projects (channel, niche, created_by) VALUES (?,?,?)",
            (channel, niche, by or None)).lastrowid


def project(project_id: int) -> dict | None:
    with _conn() as c:
        row = c.execute(f"SELECT {', '.join(_PROJECT_COLS)} FROM projects "
                        f"WHERE id = ?", (int(project_id),)).fetchone()
    return dict(zip(_PROJECT_COLS, row)) if row else None


def projects(status: str | None = "open", limit: int = 20) -> list[dict]:
    q = f"SELECT {', '.join(_PROJECT_COLS)} FROM projects"
    args: list = []
    if status:
        q += " WHERE status = ?"
        args.append(status)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    return [dict(zip(_PROJECT_COLS, r)) for r in rows]


def update_project(project_id: int, **fields) -> bool:
    """Set any subset of a project's columns. Unknown keys are refused.

    Refused rather than ignored: a typo in a column name that silently does
    nothing is a stage that looks saved and is not, and this is the one table
    that has to be trustworthy for the wizard to move backwards at all.
    """
    bad = [k for k in fields if k not in _PROJECT_COLS or k == "id"]
    if bad:
        raise ValueError(f"projects has no column(s) {bad}")
    if not fields:
        return False
    sets = ", ".join(f"{k} = ?" for k in fields)
    with _conn() as c:
        cur = c.execute(
            f"UPDATE projects SET {sets}, updated_at = datetime('now') "
            f"WHERE id = ?", (*fields.values(), int(project_id)))
        return cur.rowcount > 0


def clear_project_from(project_id: int, stage: str) -> None:
    """Go back to `stage`: undecide it and everything after it.

    TWO HALVES, AND THE FIRST VERSION ONLY HAD ONE. Forgetting forward is
    right — pick a different script and the gallery drawn for the old one is
    not stale, it is pictures of a different video. But it cleared the
    project's POINTERS and left the options themselves marked chosen and
    rejected, and every stage lists what is still `pending`. So going back
    landed on a stage that looked empty and offered to regenerate: three
    scripts you had already paid for, invisible, and the only button on screen
    spending money to write three more. The owner hit it immediately —
    "when you go back it doesn't save".

    So going back also RE-OPENS. The options come back exactly as they were,
    your previous pick among them, and you choose again — the same one or a
    different one — without buying anything.

    Per-shot gallery picks are deliberately left alone. Re-opening the SET is
    what makes it choosable again; wiping which picture won each shot would
    throw away the one part of that stage that took real attention.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    p = project(project_id) or {}
    after = STAGES[STAGES.index(stage):]
    fields: dict = {"stage": stage}
    with _conn() as c:
        if "script" in after:
            fields["script_id"] = None
            fields["script_file"] = None
            c.execute("UPDATE script_candidates SET status='pending', "
                      "decided_at=NULL WHERE project_id=?", (int(project_id),))
        if "gallery" in after and p.get("gallery_id"):
            fields["gallery_id"] = None
            c.execute("UPDATE gallery_sets SET status='pending', "
                      "decided_at=NULL WHERE id=?", (int(p["gallery_id"]),))
        elif "gallery" in after:
            fields["gallery_id"] = None
        if "voice" in after and p.get("gallery_id"):
            fields["voice_id"] = None
            c.execute("UPDATE voice_takes SET status='pending', "
                      "decided_at=NULL WHERE set_id=?", (int(p["gallery_id"]),))
        elif "voice" in after:
            fields["voice_id"] = None
        if "topic" in after:
            fields["title"] = None
            fields["topic_source"] = None
            c.execute("UPDATE project_topics SET status='pending' "
                      "WHERE project_id=?", (int(project_id),))
    update_project(project_id, **fields)


# ── notes: the things the two of you have to remember ────────────────────────
#
# A NOTIFICATION IS NOT A RECORD. Discord and ntfy are a tap on the shoulder:
# read once, scrolled past, and gone by the next morning. Half of what gets
# sent between two people running a channel is not "look at this now", it is
# "do not forget this" — the pictures on 105 are wrong, the cfg needs raising,
# ask about the Bretton Woods hook. Those need somewhere to live until somebody
# actually does them.
#
# So a message is stored FIRST and pushed second, and the push is optional. A
# note nobody was pinged about is still a note; a ping nobody kept is nothing
# an hour later.

_NOTE_COLS = ["id", "created_at", "author", "text", "priority", "notified",
              "done_at", "done_by"]


def add_note(text: str, *, author: str = "", priority: str = "normal",
             notified: bool = False) -> int:
    with _conn() as c:
        return c.execute(
            "INSERT INTO notes (author, text, priority, notified) "
            "VALUES (?,?,?,?)",
            (author or None, text.strip(), priority, 1 if notified else 0)
        ).lastrowid


def notes(*, done: bool = False, limit: int = 50) -> list[dict]:
    """Open notes oldest first; done notes newest first.

    The orders differ on purpose. An open list is a queue — the thing waiting
    longest is the thing most likely to be forgotten, so it goes on top. A done
    list is a record, and what you want from a record is what happened last.
    """
    order = "id DESC" if done else "id ASC"
    where = "done_at IS NOT NULL" if done else "done_at IS NULL"
    with _conn() as c:
        rows = c.execute(
            f"SELECT {', '.join(_NOTE_COLS)} FROM notes WHERE {where} "
            f"ORDER BY {order} LIMIT ?", (int(limit),)).fetchall()
    return [dict(zip(_NOTE_COLS, r)) for r in rows]


def finish_note(note_id: int, by: str = "") -> bool:
    """Tick one off. False if it is already done, so two people clicking at
    once cannot overwrite who actually did it."""
    with _conn() as c:
        cur = c.execute(
            "UPDATE notes SET done_at=datetime('now'), done_by=? "
            "WHERE id=? AND done_at IS NULL", (by or None, int(note_id)))
        return cur.rowcount > 0


def reopen_note(note_id: int) -> bool:
    with _conn() as c:
        cur = c.execute("UPDATE notes SET done_at=NULL, done_by=NULL "
                        "WHERE id=? AND done_at IS NOT NULL", (int(note_id),))
        return cur.rowcount > 0


def open_note_count() -> int:
    try:
        with _conn() as c:
            return c.execute(
                "SELECT COUNT(*) FROM notes WHERE done_at IS NULL").fetchone()[0]
    except Exception:
        return 0


def retire_project_options(project_id: int) -> int:
    """Take everything still pending in a project out of the queues.

    WHY A FINISHED PROJECT MUST STOP ASKING. /scripts, /galleries and /voice
    count every pending row in the database, with no idea which project it
    belongs to. So the leftovers of a project you abandoned last week — the two
    scripts you did not pick, the gallery you moved on from, the reads you
    never chose — sit in those queues forever, next to today's, and the badges
    say seven reads are waiting when one is.

    The owner's words for it: "I don't want all this mass with all the scripts
    mixed and mixed gallery — I want to make the generation live, not save it
    for after." A queue that outlives the decision it belonged to is the
    opposite of live.

    Retired, not deleted, for the reason everything here is: the row that lost
    is half of a labelled pair, and that is why the losers are kept at all.
    """
    total = 0
    for stage in ("script", "gallery", "voice"):
        try:
            total += superseded_by_regen(project_id, stage)
        except Exception:
            pass
    with _conn() as c:
        c.execute("UPDATE project_topics SET status='superseded' "
                  "WHERE project_id=? AND status='pending'", (int(project_id),))
    return total


def retire_all_stale_options() -> int:
    """Clear the queues of everything not part of a project you have open.

    THE BACKLOG THAT ALREADY EXISTS. Retiring on close fixes this going
    forward and does nothing about what has already piled up. In the owner's
    database that is 3 scripts, 2 galleries and 21 voice takes — and only 4 of
    them belong to a project at all. The rest came from the older per-stage
    path, which starts a stage without opening a project, so there is nothing
    to finish and nothing that ever takes them out of the queues.

    An earlier version of this spared those, reasoning that a queue somebody
    might still be working from should not be swept. That was the wrong call:
    they are the whole pile the owner is looking at, and "I want to make the
    generation live, not save it for after" is not a request to preserve a
    queue nobody opened in a week. The protection that matters is the project
    you have open right now, and that is what this keeps.

    Retired, not deleted. Every row stays as the losing half of its pair.
    """
    with _conn() as c:
        open_ids = {r[0] for r in c.execute(
            "SELECT id FROM projects WHERE status = 'open'")}
        live_cands = set()
        if open_ids:
            marks = ",".join("?" * len(open_ids))
            live_cands = {r[0] for r in c.execute(
                f"SELECT id FROM script_candidates WHERE project_id IN ({marks})",
                tuple(open_ids))}
        live_sets = set()
        if live_cands:
            marks = ",".join("?" * len(live_cands))
            live_sets = {r[0] for r in c.execute(
                f"SELECT id FROM gallery_sets WHERE candidate_id IN ({marks})",
                tuple(live_cands))}

        n = 0
        def keep(col, ids):
            if not ids:
                return "", ()
            marks = ",".join("?" * len(ids))
            return f" AND ({col} IS NULL OR {col} NOT IN ({marks}))", tuple(ids)

        clause, args = keep("project_id", open_ids)
        n += c.execute(
            "UPDATE script_candidates SET status='superseded' "
            "WHERE status='pending'" + clause, args).rowcount or 0

        clause, args = keep("id", live_sets)
        n += c.execute(
            "UPDATE gallery_sets SET status='superseded' "
            "WHERE status='pending'" + clause, args).rowcount or 0

        clause, args = keep("set_id", live_sets)
        n += c.execute(
            "UPDATE voice_takes SET status='superseded' "
            "WHERE status='pending'" + clause, args).rowcount or 0

        clause, args = keep("project_id", open_ids)
        c.execute("UPDATE project_topics SET status='superseded' "
                  "WHERE status='pending'" + clause, args)
    return n


def superseded_by_regen(project_id: int, stage: str) -> int:
    """Retire the pending options AT `stage`, ahead of drawing new ones.

    WHY REGEN HAS TO CLEAR FIRST. save_project_topics has always deleted the
    pending options before writing the replacements, so the topic stage does
    exactly what pressing "again" means: three suggestions become three other
    suggestions. The other three stages never did. Regenerating scripts wrote
    three more alongside the three already there; regenerating the pictures
    started a whole second gallery set, and /galleries lists every pending set
    it can find, so the page grew a second forty-minute set of decisions
    underneath the first. Press it twice and there are nine scripts and three
    galleries, and the stage built to narrow a choice widens it instead.

    ONLY THIS STAGE. Regenerating the pictures says nothing about the script
    that was already settled, and clearing forward from here is a different
    operation with a different name — clear_project_from, which is what going
    BACK calls, and which re-opens rather than retires.

    RETIRED, NOT DELETED. A passed-over option is half of a labelled pair —
    what won is only meaningful beside what it beat — and that signal is the
    whole reason these are recorded instead of thrown away. So they become
    `superseded`: out of the pending lists every stage reads, still on disk for
    the comparison. Deleting the rows would trade a tidy table for the
    measurements.

    Returns how many were retired, so the caller can say so rather than
    silently discarding something that took forty minutes to draw.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}")
    p = project(project_id) or {}
    with _conn() as c:
        if stage == "script":
            cur = c.execute("UPDATE script_candidates SET status='superseded' "
                            "WHERE project_id=? AND status='pending'",
                            (int(project_id),))
        elif stage == "gallery":
            # A set belongs to this project through the chosen script, which is
            # what _launch_galleries passes as candidate_id — the only thread
            # between a project and the sets drawn for it.
            if not p.get("script_id"):
                return 0
            cur = c.execute("UPDATE gallery_sets SET status='superseded' "
                            "WHERE candidate_id=? AND status='pending'",
                            (int(p["script_id"]),))
        elif stage == "voice":
            if not p.get("gallery_id"):
                return 0
            cur = c.execute("UPDATE voice_takes SET status='superseded' "
                            "WHERE set_id=? AND status='pending'",
                            (int(p["gallery_id"]),))
        else:
            # topic: save_project_topics already replaces, and render has no
            # options to retire.
            return 0
        return cur.rowcount or 0


def save_project_topics(project_id: int, options: list[dict]) -> int:
    """Replace this project's topic options. Returns how many were stored."""
    with _conn() as c:
        c.execute("DELETE FROM project_topics WHERE project_id = ? "
                  "AND status = 'pending'", (int(project_id),))
        n = 0
        for o in options:
            title = str(o.get("title", "")).strip()
            if not title:
                continue
            c.execute("INSERT INTO project_topics (project_id, title, why) "
                      "VALUES (?,?,?)",
                      (int(project_id), title, str(o.get("why", ""))))
            n += 1
        return n


def project_topics(project_id: int, status: str | None = "pending") -> list[dict]:
    q = f"SELECT {', '.join(_PTOPIC_COLS)} FROM project_topics WHERE project_id = ?"
    args: list = [int(project_id)]
    if status:
        q += " AND status = ?"
        args.append(status)
    q += " ORDER BY id ASC"
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    return [dict(zip(_PTOPIC_COLS, r)) for r in rows]


def choose_project_topic(topic_id: int) -> dict | None:
    """Pick one option; its siblings are recorded as passed over."""
    with _conn() as c:
        row = c.execute(
            f"SELECT {', '.join(_PTOPIC_COLS)} FROM project_topics "
            f"WHERE id = ? AND status = 'pending'", (int(topic_id),)).fetchone()
        if not row:
            return None
        got = dict(zip(_PTOPIC_COLS, row))
        c.execute("UPDATE project_topics SET status='chosen' WHERE id=?",
                  (int(topic_id),))
        c.execute("UPDATE project_topics SET status='rejected' "
                  "WHERE project_id=? AND id<>? AND status='pending'",
                  (int(got["project_id"]), int(topic_id)))
        got["status"] = "chosen"
        return got


# ── galleries ────────────────────────────────────────────────────────────────

_GSET_COLS = ["id", "created_at", "candidate_id", "channel", "niche", "topic",
              "script_file", "n_variants", "status", "decided_at", "n_beats",
              "decided_by"]
_GIMG_COLS = ["id", "set_id", "variant", "beat_index", "path", "prompt",
              "seed", "status", "decided_by", "created_at"]


def save_gallery_set(*, candidate_id: int | None, channel: str, niche: str,
                     topic: str, script_file: str, n_variants: int,
                     n_beats: int = 0) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO gallery_sets (candidate_id, channel, niche, topic, "
            "script_file, n_variants, n_beats) VALUES (?,?,?,?,?,?,?)",
            (candidate_id, channel, niche, topic, script_file,
             int(n_variants), int(n_beats)))
        return cur.lastrowid


def set_gallery_beats(set_id: int, n_beats: int) -> None:
    """The picture count, once the prompts are planned.

    The set row is written before the prompts exist — it has to be, because the
    images reference it — and the count only settles after the shots that talk
    about the same thing have been merged. So it is filled in a moment later,
    and the progress panel reads it rather than counting what has arrived.
    """
    with _conn() as c:
        c.execute("UPDATE gallery_sets SET n_beats=? WHERE id=?",
                  (int(n_beats), int(set_id)))


def save_gallery_image(*, set_id: int, variant: int, beat_index: int,
                       path: str, prompt: str, seed: int) -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO gallery_images (set_id, variant, beat_index, path, "
            "prompt, seed, created_at) VALUES (?,?,?,?,?,?,datetime('now'))",
            (int(set_id), int(variant), int(beat_index), str(path), prompt,
             int(seed)))
        return cur.lastrowid


def seconds_since_last_picture(set_id: int) -> float | None:
    """How long since anything landed in this set, or None if nothing has.

    A set that is not finished and has not gained a picture in minutes is not
    slow, it is stopped — and the difference matters, because the dashboard
    was quoting a twelve-hour estimate for a draw that had died. The owner
    turned ComfyUI off trying to fix it, which is what a person does when the
    screen insists something is still happening.
    """
    with _conn() as c:
        row = c.execute(
            "SELECT MAX(created_at) FROM gallery_images WHERE set_id=? "
            "AND created_at IS NOT NULL", (int(set_id),)).fetchone()
    if not row or not row[0]:
        return None
    import time as _t
    from datetime import datetime, timezone
    try:
        t = datetime.fromisoformat(str(row[0])).replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return max(0.0, _t.time() - t.timestamp())


def gallery_draw_rate(set_id: int, window: int = 6) -> float | None:
    """Seconds per picture, measured from the last few that actually landed.

    THE ONLY HONEST BASIS FOR AN ESTIMATE. The old one divided the time since
    the SET ROW was written by the number drawn — and that row exists before
    three voice takes and a storyboard call, so a set created three hours ago
    whose drawing began five minutes ago reported twenty-three minutes per
    picture and eleven hours remaining, while ComfyUI was visibly doing one
    every nineteen seconds.

    A trailing window rather than the whole run, because the rate is not
    constant: the first picture pays for the model load, and a machine that
    starts thermal-throttling halfway through should move the estimate rather
    than be averaged away by the fast start.

    None when there is nothing to measure — fewer than two timestamped
    pictures, or a set drawn before this column existed. The caller shows no
    estimate at all rather than a made-up one.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT created_at FROM gallery_images WHERE set_id=? "
            "AND created_at IS NOT NULL ORDER BY id DESC LIMIT ?",
            (int(set_id), int(window) + 1)).fetchall()
    if len(rows) < 2:
        return None
    from datetime import datetime, timezone
    try:
        stamps = sorted(
            datetime.fromisoformat(str(r[0])).replace(
                tzinfo=timezone.utc).timestamp() for r in rows)
    except ValueError:
        return None
    span = stamps[-1] - stamps[0]
    return span / (len(stamps) - 1) if span > 0 else None


def gallery_sets(status: str | None = "pending", limit: int = 20) -> list[dict]:
    q = f"SELECT {', '.join(_GSET_COLS)} FROM gallery_sets"
    args: list = []
    if status:
        q += " WHERE status = ?"
        args.append(status)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(int(limit))
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    return [dict(zip(_GSET_COLS, r)) for r in rows]


def gallery_images(set_id: int, status: str | None = None) -> list[dict]:
    """Every image in a set, in beat then variant order — which is the order
    the page lays them out: one row per shot, the variants side by side."""
    q = f"SELECT {', '.join(_GIMG_COLS)} FROM gallery_images WHERE set_id = ?"
    args: list = [int(set_id)]
    if status:
        q += " AND status = ?"
        args.append(status)
    q += " ORDER BY beat_index ASC, variant ASC"
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    return [dict(zip(_GIMG_COLS, r)) for r in rows]


def choose_gallery_base(set_id: int, variant: int, by: str = "") -> int:
    """Take every shot from `variant`. Returns how many beats were set.

    THE BASE IS ONE CLICK AND THE SWAPS ARE THE CORRECTIONS. Marking the whole
    variant chosen first, then letting individual beats be swapped, is what
    keeps this to one judgement plus a handful — rather than sixteen.
    """
    with _conn() as c:
        c.execute("UPDATE gallery_images SET status='rejected' "
                  "WHERE set_id=?", (int(set_id),))
        cur = c.execute("UPDATE gallery_images SET status='chosen', "
                        "decided_by=? WHERE set_id=? AND variant=?",
                        (by or None, int(set_id), int(variant)))
        return cur.rowcount


def swap_gallery_beat(set_id: int, beat_index: int, variant: int,
                      by: str = "") -> bool:
    """Take this one shot from `variant` instead. False if there is no such
    image — a beat one variant failed to render has nothing to swap to."""
    with _conn() as c:
        exists = c.execute(
            "SELECT 1 FROM gallery_images WHERE set_id=? AND beat_index=? "
            "AND variant=?", (int(set_id), int(beat_index), int(variant))
        ).fetchone()
        if not exists:
            return False
        c.execute("UPDATE gallery_images SET status='rejected' "
                  "WHERE set_id=? AND beat_index=?",
                  (int(set_id), int(beat_index)))
        c.execute("UPDATE gallery_images SET status='chosen', decided_by=? "
                  "WHERE set_id=? AND beat_index=? AND variant=?",
                  (by or None, int(set_id), int(beat_index), int(variant)))
        return True


def chosen_gallery(set_id: int) -> list[dict]:
    """The picked image per beat, in beat order, with nothing missing.

    A beat with no chosen row is a hole the renderer would fill with the wrong
    picture — clip[i] belongs to beat[i] and everything downstream assumes it —
    so this returns [] rather than a short list, and the caller says why.
    """
    rows = gallery_images(set_id, status="chosen")
    by_beat = {r["beat_index"]: r for r in rows}
    if not by_beat:
        return []
    wanted = range(max(by_beat) + 1)
    if any(i not in by_beat for i in wanted):
        return []
    return [by_beat[i] for i in wanted]


def decide_gallery_set(set_id: int, status: str = "chosen",
                       by: str = "") -> bool:
    with _conn() as c:
        cur = c.execute(
            "UPDATE gallery_sets SET status=?, decided_at=datetime('now'), "
            "decided_by=? WHERE id=? AND status='pending'",
            (status, by or None, int(set_id)))
        return cur.rowcount > 0


# ── voice takes ──────────────────────────────────────────────────────────────

_VTAKE_COLS = ["id", "created_at", "set_id", "channel", "topic", "tone",
               "text", "path", "status", "decided_at", "seconds", "spans",
               "decided_by"]


def save_voice_take(*, set_id: int, channel: str, topic: str, tone: str,
                    text: str, path: str, seconds: float = 0.0,
                    spans: str = "") -> int:
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO voice_takes (set_id, channel, topic, tone, text, "
            "path, seconds, spans) VALUES (?,?,?,?,?,?,?,?)",
            (int(set_id), channel, topic, tone, text, str(path),
             float(seconds or 0), spans))
        return cur.lastrowid


def voice_takes(set_id: int | None = None, status: str | None = None,
                limit: int = 60) -> list[dict]:
    q = f"SELECT {', '.join(_VTAKE_COLS)} FROM voice_takes"
    where, args = [], []
    if set_id is not None:
        where.append("set_id = ?")
        args.append(int(set_id))
    if status:
        where.append("status = ?")
        args.append(status)
    if where:
        q += " WHERE " + " AND ".join(where)
    q += " ORDER BY set_id DESC, id ASC LIMIT ?"
    args.append(int(limit))
    with _conn() as c:
        rows = c.execute(q, args).fetchall()
    return [dict(zip(_VTAKE_COLS, r)) for r in rows]


def choose_voice_take(take_id: int, by: str = "") -> dict | None:
    """Mark one take chosen and its siblings passed over. Returns it.

    Same shape as choose_candidate, and for the same reason: the pair is what
    makes a click worth recording. Returns None for an unknown or already
    decided id, so a double tap on a phone cannot re-decide which read lost.
    """
    with _conn() as c:
        row = c.execute(
            f"SELECT {', '.join(_VTAKE_COLS)} FROM voice_takes "
            f"WHERE id = ? AND status = 'pending'", (int(take_id),)).fetchone()
        if not row:
            return None
        take = dict(zip(_VTAKE_COLS, row))
        c.execute("UPDATE voice_takes SET status='chosen', "
                  "decided_at=datetime('now'), decided_by=? WHERE id=?",
                  (by or None, int(take_id)))
        c.execute("UPDATE voice_takes SET status='rejected', "
                  "decided_at=datetime('now'), decided_by=? "
                  "WHERE set_id=? AND id<>? AND status='pending'",
                  (by or None, int(take["set_id"]), int(take_id)))
        take["status"] = "chosen"
        return take


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


def video_by_id(video_id: int) -> dict | None:
    """One video row by id, or None.

    _video_detail in the dashboard already does this, but it lives in a Flask
    module that imports auth and flask — a CLI that has to pull those in to
    read one row is a CLI that fails on a machine where the dashboard's
    dependencies are not installed.
    """
    q = ("SELECT id, upload_date, created_at, uploaded_at, niche, channel, "
         "script_hook, script_full, scene_desc, title, description, score, "
         "run_id, video_file, youtube_id, upload_status, hold_reason, "
         "publish_at, seed_type, seed_source, seed_content, seed_url, "
         "score_reasoning, score_specificity, score_hook, score_compression, "
         "score_loop, score_human FROM videos WHERE id=?")
    cols = ["id", "upload_date", "created_at", "uploaded_at", "niche",
            "channel", "script_hook", "script_full", "scene_desc", "title",
            "description", "score", "run_id", "video_file", "youtube_id",
            "upload_status", "hold_reason", "publish_at", "seed_type",
            "seed_source", "seed_content", "seed_url", "score_reasoning",
            "score_specificity", "score_hook", "score_compression",
            "score_loop", "score_human"]
    with _conn() as c:
        row = c.execute(q, (video_id,)).fetchone()
    return dict(zip(cols, row)) if row else None


def update_video_file(video_id: int, path: str) -> bool:
    """Repoint a row at a newly rendered file. False if the id is unknown.

    Called only AFTER a render succeeds. A row updated first would name a file
    that does not exist yet, and the review page would 404 on a video that was
    playing fine ten seconds earlier.
    """
    with _conn() as c:
        cur = c.execute("UPDATE videos SET video_file=? WHERE id=?",
                        (str(path), video_id))
        return cur.rowcount > 0


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


def set_upload_status(video_id: int, status: str, by: str = ""):
    """status: 'pending' | 'approved' | 'rejected'.

    `by` is stamped only when it is given, so the auto-approve sweep and any
    caller that predates attribution leave the existing name alone instead of
    blanking a decision a person actually made.
    """
    with _conn() as c:
        if by:
            c.execute("UPDATE videos SET upload_status=?, decided_by=? "
                      "WHERE id=?", (status, by, video_id))
        else:
            c.execute("UPDATE videos SET upload_status=? WHERE id=?",
                      (status, video_id))


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
        # ONE YOUTUBE ID BELONGS TO ONE VIDEO, and nothing here used to say so.
        # Six rows in the owner's database carry the id kGVAHaObJ38 — six
        # different mp4s, six different scripts, one link pasted six times. It
        # is an easy mistake to make from a phone and there was no guard.
        #
        # The damage is not the wrong link. It is that analytics joins metrics
        # on this column, so all six rows were credited with a SEVENTH video's
        # views — identical view counts and a watch percentage of zero,
        # spread across scripts that had nothing to do with each other. Five
        # videos that were never published looked published and performed
        # like a video they are not, which is worse than having no data: it
        # is data that teaches the wrong lesson.
        taken = c.execute("SELECT id FROM videos WHERE youtube_id=? AND id<>?",
                          (yt, int(video_id))).fetchone()
        if taken:
            raise ValueError(
                f"{yt} is already recorded as video #{taken[0]}. One YouTube "
                f"id belongs to one video — if #{taken[0]} is wrong, clear it "
                f"first; if this one is a different upload, use its own link.")
        c.execute("UPDATE videos SET youtube_id=?, upload_status='approved', "
                  "upload_date=?, uploaded_at=COALESCE(uploaded_at, ?) "
                  "WHERE id=?", (yt, when, published_at or _now(), video_id))
    return True


def clear_youtube_id(video_id: int, by: str = "") -> bool:
    """Take a wrong link off a video. False if it had none.

    THE OPERATION THE ADVICE ASSUMED EXISTED. duplicate_youtube_ids reports
    that six rows claim one link, feedback_analyzer excludes them and prints
    "clear the wrong ones and they rejoin the learning" — and there was no way
    to clear one. Not in the dashboard, not in the CLI, nowhere. Advice that
    names an action the software cannot perform is worse than no advice: it
    reads as a fix and delivers nothing.

    The row goes back to pending, because that is what it actually is. A video
    whose link belonged to a different video was never published, and leaving
    it 'approved' would keep it out of the review queue — invisible in both
    directions. Its metrics rows are left alone: they are a record of what was
    fetched under that id, and deleting them would hide that this happened
    rather than undo it.
    """
    with _conn() as c:
        cur = c.execute(
            "UPDATE videos SET youtube_id=NULL, uploaded_at=NULL, "
            "upload_status='pending', decided_by=COALESCE(?, decided_by) "
            "WHERE id=? AND youtube_id IS NOT NULL AND youtube_id <> ''",
            (by or None, int(video_id)))
        return cur.rowcount > 0


def duplicate_youtube_ids() -> list[dict]:
    """Ids claimed by more than one video, worst first.

    Read by the audit on the measure page: a guard added today does not clean
    up the rows that got in before it, and those rows are still feeding another
    video's views into every average the learning loop computes.
    """
    with _conn() as c:
        rows = c.execute(
            "SELECT youtube_id, COUNT(*) n, GROUP_CONCAT(id) ids FROM videos "
            "WHERE youtube_id IS NOT NULL AND youtube_id <> '' "
            "GROUP BY youtube_id HAVING n > 1 ORDER BY n DESC").fetchall()
    return [{"youtube_id": r[0], "count": r[1],
             "video_ids": [int(x) for x in str(r[2]).split(",")]} for r in rows]


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
