#!/usr/bin/env python3
"""Take everything you have made out of Rufus, in formats other things can read.

WHY A PRODUCT NEEDS THIS AND A SCRIPT DOES NOT. Somebody who buys software is
entitled to leave with their work. Not a copy of rufus.db — that is a
proprietary schema readable by this program and nothing else, which is a
hostage dressed as a backup — but the same content as CSV and JSON, openable
in a spreadsheet, loadable by pandas, diffable in git.

It is also the honest half of the argument for keeping the source closed. A
product that will not let you take your data out is relying on the export being
hard; one that hands it over in a morning is relying on being good.

WHAT COMES OUT, AND WHY THESE THINGS.

  videos.csv        every video, its score, its published id, its metrics
  decisions.csv     THE ONE THAT MATTERS. Every choice a person made and what
                    they chose it OVER — three scripts and which won, two draws
                    per shot and which shipped, three reads and which was used.
                    This project's own account of itself is that the labelled
                    preference pairs are the product; a preference pair is not
                    a row anywhere in the database, it is the SHAPE of several,
                    and reassembling it is exactly the work an export exists to
                    do rather than leave to whoever opens the file.
  scripts/*.txt     the writing, one file per chosen script, as plain text
  projects.csv      the five-stage thread each video came through
  spend.csv         what the model calls cost, per month

WHAT DOES NOT COME OUT. Credentials, tokens, and the users file. Not because
they are hard to include, but because an export is a file that gets emailed,
uploaded to a spreadsheet service and left in a downloads folder — the same
reasoning that keeps them out of a support bundle. The dashboard's sign-in
tokens are not the owner's DATA, they are the keys to the building.
"""

from __future__ import annotations

import csv
import json
import sqlite3
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent


def _db() -> Path:
    """The live database, asked of db_manager rather than assumed — tests
    repoint it, and an export module that hard-coded the path would quietly
    dump the developer's real channel during a test run."""
    import db_manager
    return Path(db_manager.DB_FILE)


def _rows(conn: sqlite3.Connection, sql: str, args: tuple = ()) -> list[dict]:
    """Query to a list of dicts, or [] when the table is not there.

    Fail-soft per query rather than per export: a database from an older
    version is missing tables a newer one writes, and one absent table must
    cost that file, not the whole export. Somebody exporting their work is
    usually about to stop using the program, which is the worst possible
    moment to hand them an exception instead of their data.
    """
    try:
        cur = conn.execute(sql, args)
    except sqlite3.Error as e:
        print(f"[export] skipped: {e}")
        return []
    names = [d[0] for d in cur.description]
    return [dict(zip(names, row)) for row in cur.fetchall()]


def _write_csv(path: Path, rows: list[dict], columns: list[str]) -> int:
    """Write rows, always creating the file, always with its header.

    An empty CSV with its header row is a true answer — "you have no voice
    takes" — and a missing or headerless file is an ambiguous one that reads as
    a failed export. The header is what makes the empty case legible, which is
    why the columns are declared by the CALLER rather than read off the first
    row: derived from the data, an empty table produced an empty file, which is
    exactly the ambiguity this was supposed to remove. A declared list also
    fixes the column ORDER, so two exports taken a month apart diff against
    each other instead of reshuffling.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return len(rows)


def _decisions(conn: sqlite3.Connection) -> list[dict]:
    """Every human choice, with what it was chosen over.

    A PREFERENCE PAIR IS NOT A ROW. It is one chosen sibling and its rejected
    siblings, grouped by whatever made them siblings — a proposal id for
    scripts, a set id and a beat for pictures, a set id for reads. The database
    stores the members; the pairing is implied by the grouping, and an export
    that dumped three tables verbatim would leave the reader to rediscover
    that. This is the file somebody would actually train on.
    """
    out: list[dict] = []

    scripts = _rows(conn,
                    "SELECT id, created_at, proposal_id, project_id, topic, "
                    "hook_style, hook, score, status, decided_by "
                    "FROM script_candidates ORDER BY id")
    groups: dict = {}
    for row in scripts:
        key = row.get("project_id") or f"proposal:{row.get('proposal_id')}"
        groups.setdefault(key, []).append(row)
    for key, siblings in groups.items():
        chosen = next((s for s in siblings if s["status"] == "chosen"), None)
        if not chosen:
            continue      # nobody ruled on this set; not a preference yet
        for other in siblings:
            if other["id"] == chosen["id"]:
                continue
            out.append({
                "kind": "script",
                "group": str(key),
                "decided_at": chosen.get("created_at"),
                "decided_by": chosen.get("decided_by") or "",
                "topic": chosen.get("topic") or "",
                "chosen_id": chosen["id"],
                "chosen_label": chosen.get("hook_style") or "",
                "chosen_score": chosen.get("score"),
                "chosen_text": (chosen.get("hook") or "")[:500],
                "passed_over_id": other["id"],
                "passed_over_label": other.get("hook_style") or "",
                "passed_over_score": other.get("score"),
                "passed_over_text": (other.get("hook") or "")[:500],
            })

    pictures = _rows(conn,
                     "SELECT id, set_id, beat_index, variant, prompt, status, "
                     "decided_by, created_at FROM gallery_images "
                     "ORDER BY set_id, beat_index, variant")
    shots: dict = {}
    for row in pictures:
        shots.setdefault((row["set_id"], row["beat_index"]), []).append(row)
    for (set_id, beat), draws in shots.items():
        chosen = next((d for d in draws if d["status"] == "chosen"), None)
        if not chosen or len(draws) < 2:
            continue
        for other in draws:
            if other["id"] == chosen["id"]:
                continue
            out.append({
                "kind": "picture",
                "group": f"set:{set_id} shot:{beat + 1}",
                "decided_at": chosen.get("created_at"),
                "decided_by": chosen.get("decided_by") or "",
                "topic": (chosen.get("prompt") or "")[:120],
                "chosen_id": chosen["id"],
                "chosen_label": f"variant {chosen['variant']}",
                "chosen_score": None,
                "chosen_text": (chosen.get("prompt") or "")[:500],
                "passed_over_id": other["id"],
                "passed_over_label": f"variant {other['variant']}",
                "passed_over_score": None,
                "passed_over_text": (other.get("prompt") or "")[:500],
            })

    reads = _rows(conn,
                  "SELECT id, set_id, tone, status, decided_by, created_at, "
                  "topic FROM voice_takes ORDER BY set_id, id")
    sets: dict = {}
    for row in reads:
        sets.setdefault(row["set_id"], []).append(row)
    for set_id, takes in sets.items():
        chosen = next((t for t in takes if t["status"] == "chosen"), None)
        if not chosen:
            continue
        for other in takes:
            if other["id"] == chosen["id"]:
                continue
            out.append({
                "kind": "voice",
                "group": f"set:{set_id}",
                "decided_at": chosen.get("created_at"),
                "decided_by": chosen.get("decided_by") or "",
                "topic": chosen.get("topic") or "",
                "chosen_id": chosen["id"],
                "chosen_label": chosen.get("tone") or "",
                "chosen_score": None,
                "chosen_text": "",
                "passed_over_id": other["id"],
                "passed_over_label": other.get("tone") or "",
                "passed_over_score": None,
                "passed_over_text": "",
            })

    return out


VIDEO_COLUMNS = [
    "id", "created_at", "upload_date", "channel", "niche", "title",
    "script_hook", "score", "youtube_id", "upload_status", "uploaded_at",
    "hold_reason", "rufus_version",
]

PROJECT_COLUMNS = [
    "id", "created_at", "updated_at", "channel", "niche", "title",
    "topic_source", "stage", "status", "video_id", "created_by",
]

SPEND_COLUMNS = ["month", "proposals", "script_candidates", "script_attempts",
                 "total_usd"]

DECISION_COLUMNS = [
    "kind", "group", "decided_at", "decided_by", "topic",
    "chosen_id", "chosen_label", "chosen_score", "chosen_text",
    "passed_over_id", "passed_over_label", "passed_over_score",
    "passed_over_text",
]


def _readme_text(counts: dict) -> str:
    lines = [
        "Rufus data export",
        f"Taken {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "WHAT IS IN HERE",
        "",
        "  videos.csv      every video, its score, its published id",
        "  decisions.csv   every choice a person made and what they chose it",
        "                  over — one row per PAIR, not per row of the",
        "                  database. This is the file worth keeping.",
        "  projects.csv    the five-stage thread each video came through",
        "  spend.csv       what the model calls cost, by month",
        "  scripts/        the writing, one plain text file per chosen script",
        "",
    ]
    for name, n in counts.items():
        lines.append(f"  {name:16} {n} row(s)")
    lines += [
        "",
        "WHAT IS NOT IN HERE",
        "",
        "  No API keys, no dashboard sign-in tokens, no users file. An export",
        "  gets emailed, uploaded to a spreadsheet service and left in a",
        "  downloads folder. Those are not your data, they are the keys to the",
        "  building.",
        "",
        "  No video, audio or image files — they are already on your disk, in",
        "  the output and media_library folders, in ordinary formats.",
        "",
        "EVERYTHING HERE IS CSV AND PLAIN TEXT. Nothing needs Rufus to read it.",
        "",
    ]
    return "\n".join(lines)


def export(out_dir: Path | str | None = None) -> Path:
    """Write the export and return the directory it went to."""
    dest = Path(out_dir) if out_dir else (
        ROOT / f"rufus-export-{time.strftime('%Y%m%d-%H%M%S')}")
    dest.mkdir(parents=True, exist_ok=True)

    db = _db()
    if not db.exists():
        raise FileNotFoundError(f"{db} does not exist — nothing to export")

    counts: dict = {}
    with sqlite3.connect(str(db)) as conn:
        counts["videos.csv"] = _write_csv(dest / "videos.csv", _rows(
            conn,
            "SELECT id, created_at, upload_date, channel, niche, title, "
            "script_hook, score, youtube_id, upload_status, uploaded_at, "
            "hold_reason, rufus_version FROM videos ORDER BY id"),
            VIDEO_COLUMNS)

        counts["decisions.csv"] = _write_csv(
            dest / "decisions.csv", _decisions(conn), DECISION_COLUMNS)

        counts["projects.csv"] = _write_csv(dest / "projects.csv", _rows(
            conn,
            "SELECT id, created_at, updated_at, channel, niche, title, "
            "topic_source, stage, status, video_id, created_by "
            "FROM projects ORDER BY id"), PROJECT_COLUMNS)

        # Spend by month, from the three tables that record it. Monthly rather
        # than per video because proposals and candidates are not linked to the
        # video that eventually came out of them, and inventing that link would
        # produce a per-video figure nobody could reconcile with a bill.
        spend: dict = {}
        for table, when in (("proposals", "created_at"),
                            ("script_candidates", "created_at"),
                            ("script_attempts", "ts")):
            for row in _rows(conn,
                             f"SELECT substr({when}, 1, 7) AS month, "
                             f"COALESCE(SUM(cost_usd), 0) AS usd "
                             f"FROM {table} GROUP BY month"):
                if row["month"]:
                    spend.setdefault(row["month"], {"month": row["month"]})
                    spend[row["month"]][table] = round(row["usd"] or 0, 4)
        rows = []
        for month in sorted(spend):
            row = spend[month]
            row["total_usd"] = round(
                sum(v for k, v in row.items() if k != "month"), 4)
            rows.append(row)
        counts["spend.csv"] = _write_csv(dest / "spend.csv", rows,
                                         SPEND_COLUMNS)

        # The writing itself, as files rather than as a column. A script is
        # something a person reads; buried in a CSV cell with its newlines
        # escaped, it is not.
        scripts_dir = dest / "scripts"
        written = 0
        for row in _rows(conn,
                         "SELECT id, topic, script FROM script_candidates "
                         "WHERE status = 'chosen' ORDER BY id"):
            slug = "".join(ch if ch.isalnum() else "-"
                           for ch in (row.get("topic") or "untitled"))[:60]
            slug = "-".join(part for part in slug.split("-") if part).lower()
            (scripts_dir).mkdir(parents=True, exist_ok=True)
            (scripts_dir / f"{row['id']:04d}-{slug or 'untitled'}.txt").write_text(
                row.get("script") or "", encoding="utf-8")
            written += 1
        counts["scripts/"] = written

    (dest / "README.txt").write_text(_readme_text(counts), encoding="utf-8")
    (dest / "counts.json").write_text(json.dumps(counts, indent=2) + "\n",
                                      encoding="utf-8")
    return dest


def _cli() -> int:
    import sys

    out = sys.argv[1] if len(sys.argv) > 1 else None
    try:
        dest = export(out)
    except Exception as e:
        print(f"\n  Export failed: {e}\n")
        return 1
    counts = json.loads((dest / "counts.json").read_text(encoding="utf-8"))
    print(f"\n  {dest}")
    for name, n in counts.items():
        print(f"    {name:16} {n}")
    print("\n  CSV and plain text. Nothing here needs Rufus to read it.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
