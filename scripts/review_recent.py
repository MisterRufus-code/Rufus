#!/usr/bin/env python3
"""
review_recent.py — prep the last N videos for a /watch-driven quality review.

Not an analysis tool itself: it just gathers what a review needs (the file
path, the exact script, the niche/score context) into one clean block per
video, so a LOCAL Claude Code session (with the claude-video /watch skill
installed — https://github.com/bradautomates/claude-video) can be pointed
at each video without you hunting for paths/scripts by hand. Also passes
the script straight to /watch's --no-whisper mode, so reviewing your own
videos never needs Whisper — the transcript is already known exactly.

Usage:
  python scripts/review_recent.py                  # last 5 videos, any channel
  python scripts/review_recent.py --n 10
  python scripts/review_recent.py --channel main_en
  python scripts/review_recent.py --niche money_history

Then, in a LOCAL Claude Code session (this only prints text — it can't
call /watch itself):
  "Here are my last N videos [paste output]. For each one whose file still
  exists, run /watch <path> --no-whisper on it, using the script text as
  context, and critique: hook strength, pacing, whether the images match
  the narration, visible AI artifacts, caption legibility. Then look
  across all of them for patterns that show up more than once — not
  one-off notes — and propose concrete changes to fix those."
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import db_manager


def recent_for_review(n: int = 5, channel: str | None = None,
                      niche: str | None = None) -> list[sqlite3.Row]:
    q = ("SELECT id, upload_date, niche, channel, score, video_file, "
         "script_full, script_hook FROM videos")
    clauses, args = [], []
    if channel:
        clauses.append("channel = ?")
        args.append(channel)
    if niche:
        clauses.append("niche = ?")
        args.append(niche)
    if clauses:
        q += " WHERE " + " AND ".join(clauses)
    q += " ORDER BY id DESC LIMIT ?"
    args.append(n)
    with db_manager._conn() as c:
        c.row_factory = sqlite3.Row
        return c.execute(q, args).fetchall()


def format_block(row: sqlite3.Row) -> str:
    path = Path(row["video_file"] or "")
    exists = path.exists()
    lines = [
        f"── video #{row['id']}  ({row['upload_date']})  "
        f"niche={row['niche']}  channel={row['channel']}  "
        f"score={row['score'] if row['score'] is not None else 'unscored'} ──",
        f"file: {path}" + ("" if exists else "  [MISSING ON DISK — skip]"),
        "script:",
        (row["script_full"] or row["script_hook"] or "(no script saved)").strip(),
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5, help="how many recent videos (default 5)")
    ap.add_argument("--channel", default=None)
    ap.add_argument("--niche", default=None)
    args = ap.parse_args()

    rows = recent_for_review(args.n, args.channel, args.niche)
    if not rows:
        print("No videos found in rufus.db matching that filter.")
        return

    for row in rows:
        print(format_block(row))
        print()


if __name__ == "__main__":
    main()
