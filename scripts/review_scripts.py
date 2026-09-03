#!/usr/bin/env python3
"""
review_scripts.py – Browse generated scripts stored in the SQLite DB.

Usage:
    python scripts/review_scripts.py           # latest 10 scripts
    python scripts/review_scripts.py --all     # all scripts
    python scripts/review_scripts.py --id 5    # single script by ID
    python scripts/review_scripts.py -n 20     # latest 20

Prints the rubric's own reasoning under each script, which is where "why did
this score 3" is actually recorded — per-criterion marks, how many attempts it
took, and any hold reason.
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


def _safe(row, key, default=""):
    """sqlite3.Row may not contain newer columns on legacy DBs."""
    try:
        v = row[key]
        return v if v is not None else default
    except (IndexError, KeyError):
        return default


def _print_video(row):
    yt = _safe(row, "youtube_id", "")
    yt_url  = f"https://youtu.be/{yt}" if yt else "not uploaded"
    file_ok = _file_status(_safe(row, "video_file", ""))

    print(SEP2)
    print(f"  ID {row['id']}  |  {row['upload_date']}  |  niche: {row['niche']}  |  score: {_safe(row, 'score', 0)}")
    print(f"  YouTube: {yt_url}")
    print(f"  File:    {_safe(row, 'video_file', '—')}  [{file_ok}]")

    seed_type   = _safe(row, "seed_type", "")
    seed_source = _safe(row, "seed_source", "")
    seed_content = _safe(row, "seed_content", "")
    if seed_type:
        snippet = (seed_content[:150] + "…") if len(seed_content) > 150 else seed_content
        print(f"  Seed:    [{seed_type}] {seed_source} — {snippet}")

    # WHY THE SCORE IS THE SCORE. The rubric writes down its reasoning and
    # its per-criterion marks at scoring time, and until now nothing printed
    # them — so "why did the last four scripts all come back 3 or 4" had a
    # recorded answer that no tool would show. A score with no reasoning next
    # to it is a number to argue with; with the reasoning it is a note about
    # what to change.
    crits = [(k, _safe(row, f"score_{k}", None)) for k in
             ("specificity", "hook", "compression", "loop", "human")]
    have = [(k, v) for k, v in crits if v is not None]
    if have:
        print(f"  Rubric:  " + "  ".join(f"{k} {v}/3" for k, v in have))
    attempts = _safe(row, "attempts_used", None)
    if attempts:
        temp = _safe(row, "final_temperature", None)
        tail = f" at temperature {temp}" if temp else ""
        print(f"  Took:    {attempts} attempt(s){tail}")
    hold = _safe(row, "hold_reason", "")
    if hold:
        print(f"  Held:    {hold}")
    why = _safe(row, "score_reasoning", "")
    if why:
        print(SEP)
        print("  WHY THIS SCORE")
        for line in str(why).strip().splitlines():
            print(f"  {line}")

    print(SEP)
    script = _safe(row, "script_full", "") or _safe(row, "script_hook", "") or "(no script saved)"
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
            rows = c.execute("SELECT * FROM videos ORDER BY id DESC").fetchall()
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
