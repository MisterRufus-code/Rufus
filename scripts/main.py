#!/usr/bin/env python3
"""
main.py – Runs the full Rufus pipeline end to end.

Steps:
    1. Download video from best available source
    2. Analyze with LLaVA → scene description
    3. Write script with GPT based on description
    4. Render: TTS + subtitles + FFmpeg → 1080x1920 mp4
    5. Upload to YouTube

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

# ── Paths ───────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
CONFIG_DIR  = ROOT / "config"
NICHES_FILE = CONFIG_DIR / "niches.json"
OUTPUT_DIR  = ROOT / "media_library" / "output"

# ── Import pipeline modules ─────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from media_fetcher    import fetch_video
from llava_tagger     import tag_video
from script_writer    import write_script, check_blacklist, add_to_blacklist
from audio_gen        import render


def load_niche_cfg(override: str = None):
    data = json.loads(NICHES_FILE.read_text())
    if override:
        if override not in data["niches"]:
            print(f"Unknown niche: {override}")
            sys.exit(1)
        data["active"] = override
        NICHES_FILE.write_text(json.dumps(data, indent=2))
    active = data["active"]
    return data["niches"][active], active


def run(skip_upload: bool = False, niche_override: str = None):
    start = time.time()
    niche_cfg, active = load_niche_cfg(niche_override)

    print(f"\n{'='*50}")
    print(f" RUFUS  |  niche: {active}")
    print(f"{'='*50}\n")

    # ── Step 1: Download video ──────────────────────────────────────────────────
    print("[ 1 / 5 ]  Fetching video...")
    video_path = fetch_video()
    print(f"           → {video_path}\n")

    # ── Step 2: LLaVA scene analysis ───────────────────────────────────────────
    print("[ 2 / 5 ]  Analyzing scene with LLaVA...")
    scene = tag_video(video_path, niche_cfg["llava_context"])
    print(f"           → {scene[:120]}...\n" if len(scene) > 120 else f"           → {scene}\n")

    # ── Step 3: Write script ────────────────────────────────────────────────────
    print("[ 3 / 5 ]  Writing script with GPT...")
    script = write_script(scene)

    if check_blacklist(script):
        print("           ⚠ Similar script already used – regenerating...")
        script = write_script(scene + " (make it different from previous versions)")

    add_to_blacklist(script)
    print(f"           → {script[:100]}...\n" if len(script) > 100 else f"           → {script}\n")

    # ── Step 4: Render ──────────────────────────────────────────────────────────
    print("[ 4 / 5 ]  Rendering Short...")
    output_path = render(script, video_path, OUTPUT_DIR)
    print(f"           → {output_path}\n")

    # ── Step 5: Upload to YouTube ───────────────────────────────────────────────
    if skip_upload:
        print("[ 5 / 5 ]  Upload skipped (--skip-upload)\n")
        yt_url = None
    else:
        print("[ 5 / 5 ]  Uploading to YouTube...")
        from youtube_uploader import upload
        yt_url = upload(output_path, script)
        print(f"           → {yt_url}\n")

    # ── Done ────────────────────────────────────────────────────────────────────
    elapsed = time.time() - start
    print(f"{'='*50}")
    print(f" Done in {elapsed:.0f}s")
    if yt_url:
        print(f" YouTube: {yt_url}")
    print(f" File:    {output_path}")
    print(f"{'='*50}\n")

    return {"video": str(output_path), "youtube_url": yt_url, "script": script}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-upload", action="store_true", help="Render only, skip YouTube upload")
    parser.add_argument("--niche",       type=str,            help="Override active niche for this run")
    args = parser.parse_args()

    run(skip_upload=args.skip_upload, niche_override=args.niche)
