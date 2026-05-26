#!/usr/bin/env python3
"""
main.py – Runs the full Rufus pipeline end to end.

Steps:
    1. Fetch 5 candidate videos
    2. LLaVA describes all → GPT picks the best
    3. Write script with GPT + scorer loop (score ≥ 7, max 3 attempts)
    4. Render: TTS + subtitles + FFmpeg → 1080x1920 mp4
    5. Save to local SQLite DB
    6. Upload to YouTube

Usage:
    python main.py                  # full run
    python main.py --skip-upload    # render only, no upload (for testing)
    python main.py --niche finance  # override active niche for this run
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT        = Path(__file__).parent.parent
CONFIG_DIR  = ROOT / "config"
NICHES_FILE = CONFIG_DIR / "niches.json"
OUTPUT_DIR  = ROOT / "media_library" / "output"

sys.path.insert(0, str(Path(__file__).parent))
from media_fetcher import fetch_candidates
from llava_tagger   import pick_best_video
from script_writer  import write_script, check_blacklist, add_to_blacklist
from audio_gen      import render
from db_manager     import init_db, save_video, update_youtube_id


def load_niche_cfg(override: str = None):
    data = json.loads(NICHES_FILE.read_text())
    if override:
        if override not in data["niches"]:
            print(f"Unknown niche '{override}'. Available: {list(data['niches'])}")
            sys.exit(1)
        data["active"] = override
        NICHES_FILE.write_text(json.dumps(data, indent=2))
    active = data["active"]
    return data["niches"][active], active


def run(skip_upload: bool = False, niche_override: str = None):
    start = time.time()
    niche_cfg, active = load_niche_cfg(niche_override)

    print(f"\n{'='*52}")
    print(f"  RUFUS  |  niche: {active}")
    print(f"{'='*52}\n")

    init_db()

    # ── Step 1: Fetch candidate videos ─────────────────────────────────────────
    print("[ 1 / 6 ]  Fetching candidate videos...")
    try:
        candidates = fetch_candidates(n=5)
        print(f"           → {len(candidates)} candidates downloaded\n")
    except Exception as e:
        print(f"           ✗ Step 1 failed: {e}")
        sys.exit(1)

    # ── Step 2: AI picks best video ─────────────────────────────────────────────
    print("[ 2 / 6 ]  AI selecting best video...")
    try:
        video_path, scene = pick_best_video(candidates, niche_cfg["llava_context"])
        print(f"           → selected: {video_path.name}")
        short = scene[:120] + "..." if len(scene) > 120 else scene
        print(f"           → {short}\n")
    except Exception as e:
        print(f"           ✗ Step 2 failed: {e}")
        sys.exit(1)

    # ── Step 3: Write script ────────────────────────────────────────────────────
    print("[ 3 / 6 ]  Writing script with GPT...")
    try:
        script = write_script(scene)

        if check_blacklist(script):
            print("           ⚠ Similar script already used – regenerating...")
            script = write_script(scene + " (make it different from previous versions)")

        add_to_blacklist(script)
        preview = script[:100] + "..." if len(script) > 100 else script
        print(f"           → {preview}\n")
    except Exception as e:
        print(f"           ✗ Step 3 failed: {e}")
        sys.exit(1)

    # ── Step 4: Render ──────────────────────────────────────────────────────────
    print("[ 4 / 6 ]  Rendering Short...")
    try:
        output_path = render(script, video_path, OUTPUT_DIR)
        print(f"           → {output_path}\n")
    except Exception as e:
        print(f"           ✗ Step 4 failed: {e}")
        sys.exit(1)

    # ── Step 5: Save to DB ──────────────────────────────────────────────────────
    print("[ 5 / 6 ]  Saving to database...")
    db_id = None
    try:
        hook  = script.strip().split("\n")[0][:100]
        db_id = save_video(
            niche=active,
            script_hook=hook,
            scene_desc=scene[:500],
            video_file=str(output_path),
        )
        print(f"           → saved (id={db_id})\n")
    except Exception as e:
        print(f"           ⚠ DB save failed (non-fatal): {e}\n")

    # ── Step 6: Upload ──────────────────────────────────────────────────────────
    yt_url = None
    if skip_upload:
        print("[ 6 / 6 ]  Upload skipped (--skip-upload)\n")
    else:
        print("[ 6 / 6 ]  Uploading to YouTube...")
        try:
            from youtube_uploader import upload
            yt_url = upload(output_path, script)
            print(f"           → {yt_url}\n")

            if db_id and yt_url:
                yt_id = yt_url.rstrip("/").split("/")[-1]
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

    return {"video": str(output_path), "youtube_url": yt_url, "script": script}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-upload", action="store_true", help="Render only, skip YouTube upload")
    parser.add_argument("--niche",       type=str,            help="Override active niche for this run")
    args = parser.parse_args()
    run(skip_upload=args.skip_upload, niche_override=args.niche)
