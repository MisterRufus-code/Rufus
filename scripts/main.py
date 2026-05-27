#!/usr/bin/env python3
"""
main.py – Runs the full Rufus pipeline end to end.

Steps:
    1. Research a real seed (Reddit story → wisdom quote fallback)
    2. Fetch 5 candidate videos (parallel)
    3. GPT-4o Vision describes all → GPT picks the best
    4. Write script from seed + scene (35-50s, value-focused)
    5. Render: TTS + Whisper + FFmpeg → 1080x1920 mp4 (all clips cut together)
    6. Save to local SQLite DB (incl. full script + seed)
    7. Upload to YouTube (private, with thumbnail)

stdout is also tee'd to logs/rufus_YYYYMMDD.log so cron runs leave an audit trail.

Usage:
    python main.py                  # full run
    python main.py --skip-upload    # render only, no upload (for testing)
    python main.py --niche finance  # override active niche for this run
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT        = Path(__file__).parent.parent
CONFIG_DIR  = ROOT / "config"
NICHES_FILE = CONFIG_DIR / "niches.json"
OUTPUT_DIR  = ROOT / "media_library" / "output"
LOG_DIR     = ROOT / "logs"


# ── Tee stdout/stderr to a daily log file ───────────────────────────────────────

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self.streams:
            s.flush()


def _enable_file_logging():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"rufus_{time.strftime('%Y%m%d')}.log"
    log_fp   = open(log_path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, log_fp)
    sys.stderr = _Tee(sys.__stderr__, log_fp)
    return log_path


sys.path.insert(0, str(Path(__file__).parent))
from research        import get_seed
from media_fetcher   import fetch_candidates
from llava_tagger    import pick_best_video
from script_writer   import write_script, check_blacklist, add_to_blacklist
from audio_gen       import render
from db_manager      import init_db, save_video, update_youtube_id


def load_niche_cfg(override: str = None):
    data = json.loads(NICHES_FILE.read_text())
    if override:
        if override not in data["niches"]:
            print(f"Unknown niche '{override}'. Available: {list(data['niches'])}")
            sys.exit(1)
        # Set env var so all sub-modules pick it up without touching the file on disk.
        os.environ["RUFUS_NICHE_OVERRIDE"] = override
    active = os.environ.get("RUFUS_NICHE_OVERRIDE") or data["active"]
    return data["niches"][active], active


def _todays_niche() -> str:
    """Pick today's niche from config schedule. Day-of-year mod schedule length."""
    from datetime import datetime
    data     = json.loads(NICHES_FILE.read_text())
    schedule = data.get("schedule") or [data.get("active", "finance")]
    doy      = datetime.now().timetuple().tm_yday   # 1-366
    return schedule[(doy - 1) % len(schedule)]


def _all_scheduled_niches() -> list[str]:
    """Return unique niches present in schedule, preserving order."""
    data     = json.loads(NICHES_FILE.read_text())
    schedule = data.get("schedule") or [data.get("active", "finance")]
    seen     = []
    for n in schedule:
        if n not in seen:
            seen.append(n)
    return seen


def run(skip_upload: bool = False, niche_override: str = None):
    log_path = _enable_file_logging()
    start    = time.time()
    niche_cfg, active = load_niche_cfg(niche_override)

    print(f"\n{'='*52}")
    print(f"  RUFUS  |  niche: {active}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  log:   {log_path}")
    print(f"{'='*52}\n")

    init_db()

    # ── Step 1: Research a real seed ───────────────────────────────────────────
    print("[ 1 / 7 ]  Researching real source material...")
    try:
        seed = get_seed(active)
        if seed.get("type") == "reddit":
            print(f"           → Reddit: {seed.get('title', '')[:80]}")
        else:
            print(f"           → Quote:  \"{seed.get('content', '')[:80]}\" — {seed.get('source')}\n")
    except Exception as e:
        print(f"           ✗ Step 1 failed: {e}")
        sys.exit(1)

    # ── Step 2: Fetch candidate videos (parallel) ──────────────────────────────
    print("[ 2 / 7 ]  Fetching candidate videos (parallel)...")
    try:
        candidates = fetch_candidates(n=5)
        print(f"           → {len(candidates)} candidates downloaded\n")
    except Exception as e:
        print(f"           ✗ Step 2 failed: {e}")
        sys.exit(1)

    # ── Step 3: AI picks best video ─────────────────────────────────────────────
    print("[ 3 / 7 ]  AI selecting best video...")
    try:
        video_path, scene = pick_best_video(candidates, niche_cfg["llava_context"])
        print(f"           → selected: {video_path.name}")
        short = scene[:120] + "..." if len(scene) > 120 else scene
        print(f"           → {short}\n")
    except Exception as e:
        print(f"           ✗ Step 3 failed: {e}")
        sys.exit(1)

    # ── Step 4: Write script from seed + scene ─────────────────────────────────
    print("[ 4 / 7 ]  Writing script with GPT...")
    try:
        script = write_script(scene, seed=seed)

        if check_blacklist(script):
            print("           ⚠ Similar script already used – regenerating...")
            script = write_script(scene + " (make it different from previous versions)", seed=seed)

        add_to_blacklist(script)
        preview = script[:100] + "..." if len(script) > 100 else script
        print(f"           → {preview}\n")
    except Exception as e:
        print(f"           ✗ Step 4 failed: {e}")
        sys.exit(1)

    # ── Step 5: Render (all clips cut together) ─────────────────────────────────
    print("[ 5 / 7 ]  Rendering Short...")
    try:
        output_path = render(script, candidates, OUTPUT_DIR)
        print(f"           → {output_path}\n")
    except Exception as e:
        print(f"           ✗ Step 5 failed: {e}")
        sys.exit(1)

    # ── Step 6: Save to DB ──────────────────────────────────────────────────────
    print("[ 6 / 7 ]  Saving to database...")
    db_id = None
    try:
        hook  = script.strip().split("\n")[0][:100]
        db_id = save_video(
            niche=active,
            script_hook=hook,
            scene_desc=scene[:500],
            video_file=str(output_path),
            script_full=script,
            seed_type=seed.get("type"),
            seed_source=seed.get("source"),
            seed_content=(seed.get("content", "") or "")[:1000],
        )
        print(f"           → saved (id={db_id})\n")
    except Exception as e:
        print(f"           ⚠ DB save failed (non-fatal): {e}\n")

    # ── Step 7: Upload (with custom thumbnail) ─────────────────────────────────
    yt_url = None
    if skip_upload:
        print("[ 7 / 7 ]  Upload skipped (--skip-upload)\n")
    else:
        print("[ 7 / 7 ]  Generating thumbnail + uploading to YouTube...")
        try:
            from thumbnail_gen    import make_thumbnail
            from youtube_uploader import upload

            thumb = None
            try:
                thumb = make_thumbnail(output_path, script)
                print(f"           thumbnail: {thumb.name}")
            except Exception as e:
                print(f"           ⚠ thumbnail generation skipped: {e}")

            yt_url, yt_id = upload(output_path, script, thumbnail_path=thumb)
            print(f"           → {yt_url}\n")

            if db_id and yt_id:
                update_youtube_id(db_id, yt_id)
        except Exception as e:
            print(f"           ✗ Upload failed: {e}\n")

    # ── Done ────────────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"{'='*52}")
    print(f"  Done in {elapsed:.0f}s  |  {active}")
    if yt_url:
        print(f"  YouTube: {yt_url}")
    print(f"  File:    {output_path}")
    print(f"{'='*52}\n")

    return {"video": str(output_path), "youtube_url": yt_url, "script": script, "seed": seed}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rufus pipeline runner")
    parser.add_argument("--skip-upload", action="store_true", help="Render only, skip YouTube upload")
    parser.add_argument("--niche",       type=str,            help="Override niche (e.g. finance, motivation, mindset)")
    parser.add_argument("--scheduled",   action="store_true", help="Use today's niche from config schedule (for cron)")
    parser.add_argument("--rotate",      action="store_true", help="Run one video per unique niche in the schedule")
    args = parser.parse_args()

    if sum(bool(x) for x in (args.niche, args.scheduled, args.rotate)) > 1:
        print("Use only one of --niche, --scheduled, --rotate")
        sys.exit(1)

    if args.rotate:
        niches = _all_scheduled_niches()
        print(f"\n[rotate] producing {len(niches)} video(s): {niches}\n")
        for n in niches:
            # Clear any prior env override so each iteration starts clean
            os.environ.pop("RUFUS_NICHE_OVERRIDE", None)
            run(skip_upload=args.skip_upload, niche_override=n)
    elif args.scheduled:
        n = _todays_niche()
        print(f"\n[scheduled] today's niche: {n}\n")
        run(skip_upload=args.skip_upload, niche_override=n)
    else:
        run(skip_upload=args.skip_upload, niche_override=args.niche)
