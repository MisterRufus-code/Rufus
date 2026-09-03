#!/usr/bin/env python3
"""
rewrite.py — write this video's script again, without rebuilding the video.

WHAT THIS IS FOR. A run produces one script and thirteen minutes of GPU on top
of it. When the script is the thing that is wrong, the only lever was to throw
away the whole run and start again — which costs the pictures too, and takes
long enough that it does not get used. So a script nobody was happy with went
out anyway.

This runs the script step ALONE, on the same source the original run used: the
same scene description, the same seed, the same niche. Seconds and cents, no
GPU. The result is written beside the run as a candidate, so the review page
can show it next to what is already there — the two scripts, with both rubric
breakdowns, and a choice.

IT DOES NOT REPLACE ANYTHING. The candidate sits in the run folder until
someone picks it. Overwriting the live script on the way to showing it to a
human would mean losing the original the moment you asked whether the new one
was better.

PRESSING IT TWICE GIVES TWO DIFFERENT SCRIPTS, which is the difference between
a regenerate button and a slot machine that keeps landing on the same fruit.
write_script_until_good already feeds a rejected attempt forward into the next
cycle; this passes the previous candidate in the same way, so each press knows
what the last one produced.

    python scripts/rewrite.py 93              write a candidate for video 93
    python scripts/rewrite.py 93 --show       print the candidate on disk

CONTRACT: never touches the database and never touches the video. The only
thing it writes is one json file in the run's own folder.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import console  # noqa: E402
import db_manager  # noqa: E402
import paths  # noqa: E402

CANDIDATE_NAME = "rewrite.json"


def candidate_path(run_id: str) -> Path:
    return paths.debug_root() / run_id / CANDIDATE_NAME


def latest(run_id: str) -> dict | None:
    """The candidate on disk for this run, or None."""
    try:
        return json.loads(candidate_path(run_id).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def discard(run_id: str) -> bool:
    try:
        candidate_path(run_id).unlink()
        return True
    except OSError:
        return False


def _seed_from(row: dict) -> dict | None:
    """Rebuild the seed dict the original run wrote from.

    Without it the writer would invent a fresh subject, and the rewrite would
    be a different video rather than another take on this one. The fact gate
    also checks the script against its source — with no source, a rewrite
    would be judged against nothing and score zero for specificity, which is
    exactly the failure mode four videos hit last week for a different reason.
    """
    if not (row.get("seed_content") or "").strip():
        return None
    return {"type": row.get("seed_type") or "",
            "source": row.get("seed_source") or "",
            "content": row.get("seed_content") or "",
            "url": row.get("seed_url") or ""}


def propose(video_id: int) -> dict:
    """Write one new script for this video. The candidate dict, or {"ok": False}."""
    row = db_manager.video_by_id(video_id)
    if not row:
        return {"ok": False, "why": f"no video #{video_id} in the database"}
    run_id = (row.get("run_id") or "").strip()
    if not run_id:
        return {"ok": False, "why": (
            f"video #{video_id} has no run_id, so there is nowhere to keep a "
            f"candidate beside it")}

    scene = (row.get("scene_desc") or "").strip()
    if not scene:
        return {"ok": False, "why": (
            f"video #{video_id} has no saved scene description — that is what "
            f"the writer works from, so there is nothing to rewrite against")}

    seed = _seed_from(row)
    if seed is None:
        print("[rewrite] ⚠ this run saved no seed content — the writer has no "
              "source to ground itself in, and the fact gate will have nothing "
              "to check against")

    # What the last press produced, so this one does not produce it again.
    previous = latest(run_id)
    if previous and previous.get("script"):
        scene = (f"{scene}\n\nDo not repeat this angle, which was already "
                 f"written and set aside:\n{previous['script'][:600]}")

    import script_writer
    t0 = time.time()
    try:
        result = script_writer.write_script_until_good(scene, seed=seed)
    except Exception as e:
        return {"ok": False, "why": f"the writer failed: {e}"}

    cand = {
        "ok": True,
        "video_id": video_id,
        "run_id": run_id,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seconds": round(time.time() - t0, 1),
        "script": result.get("script", ""),
        "score": result.get("score", 0),
        "criterion_scores": result.get("criterion_scores") or {},
        "reasoning": result.get("reasoning", ""),
        "attempts_used": result.get("attempts_used"),
        "cost_usd": round(float(result.get("cost_usd", 0) or 0), 4),
    }
    if not cand["script"].strip():
        return {"ok": False, "why": "the writer returned an empty script"}

    out = candidate_path(run_id)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(cand, indent=2, ensure_ascii=False),
                       encoding="utf-8")
    except OSError as e:
        return {"ok": False, "why": f"could not save the candidate: {e}"}

    print(f"[rewrite] video #{video_id}: {cand['score']}/10 in "
          f"{cand['seconds']}s for ${cand['cost_usd']:.4f} → {out}")
    return cand


def main(argv: list[str] | None = None) -> int:
    console.force_utf8()
    ap = argparse.ArgumentParser(description="Write a video's script again")
    ap.add_argument("video_id", type=int)
    ap.add_argument("--show", action="store_true",
                    help="print the candidate already on disk and write nothing")
    args = ap.parse_args(argv)
    db_manager.init_db()

    if args.show:
        row = db_manager.video_by_id(args.video_id)
        cand = latest((row or {}).get("run_id") or "") if row else None
        if not cand:
            print(f"[rewrite] no candidate for video #{args.video_id}")
            return 1
        print(f"  written {cand.get('written_at')} · {cand.get('score')}/10 · "
              f"${cand.get('cost_usd', 0):.4f}")
        print()
        print(cand.get("script", ""))
        return 0

    result = propose(args.video_id)
    if not result.get("ok"):
        print(f"[rewrite] {result.get('why')}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
