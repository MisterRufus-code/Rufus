#!/usr/bin/env python3
"""
recut.py — rebuild one run's video from the stills that are on disk right now.

WHY THIS IS NOT "RUN IT AGAIN". A gallery is usually right about most of its
frames and wrong about one: a contact sheet where a scene should be, a figure
with no arms, the wrong object on the table. Fixing that by re-running the
pipeline costs the script, the voice, and thirteen minutes of GPU to change one
picture — so it does not get done, and the video ships with the bad frame in it.

A re-cut redraws nothing and rewrites nothing. It takes the script the run
already wrote, the voiceover it already recorded, and whatever stills are in
its debug folder at this moment, and produces a new mp4 from them. Redraw one
beat with the dashboard's Regen button, re-cut, and the only thing that changed
in the finished video is that shot.

THE VOICEOVER IS REUSED, AND THAT IS THE WHOLE SAFETY PROPERTY. Cuts are placed
from Whisper's word timings. Synthesizing the voice again would produce audio a
few milliseconds different, Whisper would time the new audio, and every cut in
the video would move — so "I redrew beat 7" would silently reshuffle the other
nine. Handed the same bytes, the transcription is the same and the cuts land
exactly where they did before.

    python scripts/recut.py 93            rebuild video #93
    python scripts/recut.py 93 --dry-run  say what it would use, render nothing

CONTRACT: never destructive. The new file is written beside the old one and the
database row is repointed only after a successful render, so a failed re-cut
leaves a working video where it was.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import console  # noqa: E402
import db_manager  # noqa: E402
import paths  # noqa: E402


def stills_for(run_dir: Path) -> list[Path]:
    """Every beat frame in the folder, in narration order.

    The render loop writes `01.png` for a beat and `01a.png`, `01b.png` for its
    sub-frames, so lexical order IS narration order — "01" < "01a" < "01b" <
    "02" — and sorting by name preserves it exactly. Sorting by mtime would
    not: a regenerated frame is the newest file in the folder and would jump to
    the end of the video.
    """
    if not run_dir.is_dir():
        return []
    return sorted((p for p in run_dir.glob("*.png") if p.stem[:2].isdigit()),
                  key=lambda p: p.name)


def voiceover_for(run_dir: Path) -> Path | None:
    v = run_dir / "voiceover.mp3"
    return v if v.exists() else None


def plan(video_id: int) -> dict:
    """What a re-cut of this video would use. Read-only."""
    row = db_manager.video_by_id(video_id)
    if not row:
        return {"ok": False, "why": f"no video #{video_id} in the database"}
    run_id = (row.get("run_id") or "").strip()
    if not run_id:
        return {"ok": False, "why": (
            f"video #{video_id} has no run_id, so there is no folder of "
            f"stills to rebuild it from")}
    run_dir = paths.debug_root() / run_id
    frames = stills_for(run_dir)
    script = (row.get("script_full") or "").strip()
    out = {"ok": False, "video_id": video_id, "run_id": run_id,
           "run_dir": str(run_dir), "frames": [str(p) for p in frames],
           "voiceover": str(voiceover_for(run_dir) or ""),
           "script_chars": len(script), "channel": row.get("channel"),
           "niche": row.get("niche")}
    if not run_dir.is_dir():
        out["why"] = f"{run_dir} does not exist — the run's frames were removed"
        return out
    if not frames:
        out["why"] = f"no numbered stills in {run_dir}"
        return out
    if not script:
        out["why"] = (f"video #{video_id} has no saved script, and the render "
                      f"builds its captions from it")
        return out
    # A missing voiceover is NOT fatal — it means the render synthesizes one,
    # which is slower and moves the cuts, so it is worth saying out loud rather
    # than discovering in the finished file.
    if not out["voiceover"]:
        out["warning"] = ("no voiceover.mp3 in this run — the voice will be "
                          "regenerated, and the cuts may land differently")
    out["ok"] = True
    return out


def recut(video_id: int) -> Path | None:
    """Rebuild and repoint the database row. None on any failure."""
    p = plan(video_id)
    if not p["ok"]:
        print(f"[recut] {p['why']}")
        return None
    if p.get("warning"):
        print(f"[recut] ⚠ {p['warning']}")

    row = db_manager.video_by_id(video_id)
    frames = [Path(f) for f in p["frames"]]
    voice = Path(p["voiceover"]) if p["voiceover"] else None
    print(f"[recut] video #{video_id} · run {p['run_id']} · "
          f"{len(frames)} still(s)")

    out_dir = paths.output_dir()
    try:
        from remotion_renderer import render as remotion_render
        new_path = remotion_render(row["script_full"], frames, out_dir,
                                   voice_path=voice)
    except Exception as e:
        print(f"[recut] Remotion failed ({e}) — falling back to FFmpeg")
        try:
            from audio_gen import render as ffmpeg_render
            new_path = ffmpeg_render(row["script_full"], frames, out_dir)
        except Exception as e2:
            print(f"[recut] FFmpeg render also failed ({e2}) — the existing "
                  f"video is untouched")
            return None

    # Repointed only now. A row updated before the render succeeded would name
    # a file that does not exist, and the review page would 404 on a video that
    # was fine ten seconds ago.
    if not db_manager.update_video_file(video_id, str(new_path)):
        print(f"[recut] rendered {new_path} but could not update the database "
              f"— the row still points at the old file")
        return new_path
    print(f"[recut] → {new_path}")
    return Path(new_path)


def main(argv: list[str] | None = None) -> int:
    console.force_utf8()
    ap = argparse.ArgumentParser(description="Rebuild a video from its stills")
    ap.add_argument("video_id", type=int)
    ap.add_argument("--dry-run", action="store_true",
                    help="say what would be used and render nothing")
    args = ap.parse_args(argv)
    # Idempotent, and the same thing dashboard.py and main.py do before their
    # first read. Without it this CLI raises on a database that predates a
    # column it selects — which is every database that has not been opened by
    # a newer build yet.
    db_manager.init_db()

    if args.dry_run:
        p = plan(args.video_id)
        for k in ("run_id", "run_dir", "voiceover", "script_chars", "niche"):
            print(f"  {k:14} {p.get(k, '')}")
        print(f"  {'frames':14} {len(p.get('frames', []))}")
        for f in p.get("frames", []):
            print(f"                 {Path(f).name}")
        if p.get("warning"):
            print(f"  ⚠ {p['warning']}")
        if not p["ok"]:
            print(f"  NOT USABLE: {p['why']}")
        return 0 if p["ok"] else 1

    return 0 if recut(args.video_id) else 1


if __name__ == "__main__":
    raise SystemExit(main())
