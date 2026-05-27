#!/usr/bin/env python3
"""
review_scripts.py – Browse generated scripts stored in the SQLite DB.

Usage:
    python scripts/review_scripts.py           # latest 10 scripts
    python scripts/review_scripts.py --all     # all scripts
    python scripts/review_scripts.py --id 5    # single script by ID
    python scripts/review_scripts.py -n 20     # latest 20
"""

import argparse
import sqlite3
from pathlib import Path

ROOT    = Path(__file__).parent.parent
DB_FILE = ROOT / "rufus.db"

SEP  = "─" * 60
SEP2 = "═" * 60


def _conn():
    c = sqlite3.connect(str(DB_FILE))
    c.row_factory = sqlite3.Row
    return c


def _file_status(path: str) -> str:
    if not path:
        return "no file"
    return "✓ exists" if Path(path).exists() else "✗ missing"


def _print_video(row):
    yt = row["youtube_id"] or "—"
    yt_url = f"https://youtu.be/{yt}" if row["youtube_id"] else "not uploaded"
    file_ok = _file_status(row["video_file"])

    print(SEP2)
    print(f"  ID {row['id']}  |  {row['upload_date']}  |  niche: {row['niche']}  |  score: {row['score']}")
    print(f"  YouTube: {yt_url}")
    print(f"  File:    {row['video_file'] or '—'}  [{file_ok}]")
    print(SEP)
    script = row["script_full"] or row["script_hook"] or "(no script saved)"
    print(script)
    print()


def main():
    parser = argparse.ArgumentParser(description="Review generated scripts")
    parser.add_argument("--all",  action="store_true", help="Show all scripts")
    parser.add_argument("--id",   type=int,            help="Show single script by DB id")
    parser.add_argument("-n",     type=int, default=10, help="Number of latest scripts (default: 10)")
    args = parser.parse_args()

    if not DB_FILE.exists():
        print(f"No database found at {DB_FILE}")
        print("Run the pipeline at least once first.")
        return

    with _conn() as c:
        if args.id:
            rows = c.execute(
                "SELECT * FROM videos WHERE id = ?", (args.id,)
            ).fetchall()
            if not rows:
                print(f"No video with id={args.id}")
                return
        elif args.all:
            rows = c.execute(
                "SELECT * FROM videos ORDER BY id DESC"
            ).fetchall()
        else:
            rows = c.execute(
                "SELECT * FROM videos ORDER BY id DESC LIMIT ?", (args.n,)
            ).fetchall()

    if not rows:
        print("No scripts in the database yet.")
        return

    print(f"\n  Rufus Script Review — {len(rows)} record(s)\n")
    for row in rows:
        _print_video(row)
    print(SEP2)
    print(f"  Total: {len(rows)} script(s)\n")


if __name__ == "__main__":
    main()
