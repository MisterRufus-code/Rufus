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
OUTPUT_DIR  = Path(os.environ.get("RUFUS_OUTPUT_DIR", ROOT / "media_library" / "output"))
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
from script_writer   import write_script, preanalyze, check_blacklist, add_to_blacklist
from audio_gen       import render
from db_manager      import init_db, save_video, update_youtube_id


def _parse_video_queries(analysis: str) -> list[str]:
    """Extract VIDEO QUERIES line from pre-analysis output (item 7)."""
    for line in (analysis or "").split("\n"):
        if "VIDEO QUERIES" in line.upper() or "VIDEO QUERY" in line.upper():
            parts = line.split(":", 1)
            if len(parts) > 1:
                queries = [q.strip().strip("'\"") for q in parts[1].split(",")]
                return [q for q in queries if len(q) > 2][:3]
    return []


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


def run(skip_upload: bool = False, niche_override: str = None, output_dir: Path = None):
    log_path = _enable_file_logging()
    start    = time.time()
    niche_cfg, active = load_niche_cfg(niche_override)
    out_dir  = output_dir or OUTPUT_DIR

    print(f"\n{'='*52}")
    print(f"  RUFUS  |  niche: {active}  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  log:   {log_path}")
    print(f"{'='*52}\n")

    init_db()

    # ── Step 1: Research seed + pre-analyse ────────────────────────────────────
    print("[ 1 / 7 ]  Researching real source material...")
    try:
        seed = get_seed(active)
        if seed.get("type") == "reddit":
            print(f"           → Reddit: {seed.get('title', '')[:80]}")
        elif seed.get("type") == "hackernews":
            print(f"           → HN:     {seed.get('title', '')[:80]}")
        else:
            print(f"           → Quote:  \"{seed.get('content', '')[:80]}\" — {seed.get('source')}\n")
    except Exception as e:
        print(f"           ✗ Step 1 failed: {e}")
        sys.exit(1)

    # Pre-analysis runs here so the hook angle is available for video selection
    seed_analysis = ""
    script_run_id = None
    try:
        seed_analysis, script_run_id, _ = preanalyze(seed)
    except Exception as e:
        print(f"           ⚠ Pre-analysis failed (non-fatal): {e}")

    video_source  = os.environ.get("RUFUS_VIDEO_SOURCE", "pexels").strip().lower()
    video_queries = _parse_video_queries(seed_analysis)

    # ── Step 2: Get candidate clips — AI-generated (ComfyUI) or stock (Pexels) ──
    candidates = []
    scene = ""
    if video_source == "comfy":
        print("[ 2 / 7 ]  Generating AI clips with ComfyUI...")
        try:
            from comfy_client import generate_clips
            prompts = video_queries or [niche_cfg.get("llava_context", f"Describe this scene for {active} content.")]
            candidates = generate_clips(prompts, n=int(os.environ.get("COMFY_CLIPS", "5")))
            if candidates:
                scene = "AI-generated footage: " + "; ".join(prompts[:3])
                print(f"           → {len(candidates)} clips generated\n")
        except Exception as e:
            print(f"           ⚠ ComfyUI generation failed ({e}) — falling back to Pexels")
        if not candidates:
            print("           → no AI clips — falling back to Pexels stock footage")
            video_source = "pexels"

    if video_source != "comfy":
        print("[ 2 / 7 ]  Fetching candidate videos (parallel)...")
        try:
            if video_queries:
                print(f"           → script queries: {video_queries}")
            candidates = fetch_candidates(n=7, extra_keywords=video_queries or None)
            print(f"           → {len(candidates)} candidates downloaded\n")
        except Exception as e:
            print(f"           ✗ Step 2 failed: {e}")
            sys.exit(1)

    # ── Step 3: AI picks best video (stock only — generated clips are on-topic) ─
    if scene:
        print("[ 3 / 7 ]  Generated clips are purpose-built — skipping vision pick\n")
    else:
        print("[ 3 / 7 ]  AI selecting best video...")
        try:
            video_path, scene = pick_best_video(
                candidates, niche_cfg["llava_context"],
                seed=seed, analysis=seed_analysis or None,
            )
            print(f"           → selected: {video_path.name}")
            short = scene[:120] + "..." if len(scene) > 120 else scene
            print(f"           → {short}\n")
        except Exception as e:
            print(f"           ✗ Step 3 failed: {e}")
            sys.exit(1)

    # ── Step 4: Write script (reuses pre-analysis, no duplicate API call) ──────
    print("[ 4 / 7 ]  Writing script with GPT...")
    try:
        result = write_script(scene, seed=seed,
                              precomputed_analysis=seed_analysis or None,
                              run_id=script_run_id)
        script = result["script"]

        if check_blacklist(script):
            print("           ⚠ Similar script already used – regenerating...")
            try:
                result = write_script(scene + " (make it different from previous versions)",
                                      seed=seed,
                                      precomputed_analysis=seed_analysis or None,
                                      run_id=script_run_id)
                script = result["script"]
            except Exception as _regen_err:
                print(f"           ⚠ Blacklist regen failed ({_regen_err}) — using original script")

        add_to_blacklist(script)
        preview = script[:100] + "..." if len(script) > 100 else script
        print(f"           → {preview}")
        print(f"           → score {result['score']}/10  attempts={result['attempts_used']}  "
              f"cost=${result['cost_usd']:.4f}\n")
    except Exception as e:
        print(f"           ✗ Step 4 failed: {e}")
        sys.exit(1)

    # ── Step 5: Render (all clips cut together) ─────────────────────────────────
    # RUFUS_RENDERER=remotion uses the React engine (spring-pop captions, smooth
    # crossfades, progress bar); anything else uses the FFmpeg engine. Remotion
    # failures fall back to FFmpeg so a render always completes.
    renderer = os.environ.get("RUFUS_RENDERER", "ffmpeg").strip().lower()
    print(f"[ 5 / 7 ]  Rendering Short ({renderer})...")
    try:
        if renderer == "remotion":
            try:
                from remotion_renderer import render as remotion_render
                output_path = remotion_render(script, candidates, out_dir)
            except Exception as e:
                print(f"           ⚠ Remotion failed ({e}) — falling back to FFmpeg")
                output_path = render(script, candidates, out_dir)
        else:
            output_path = render(script, candidates, out_dir)
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
            run_id=result.get("run_id"),
            score=result.get("score", 0),
            criterion_scores=result.get("criterion_scores"),
            attempts_used=result.get("attempts_used"),
            final_temperature=result.get("final_temperature"),
            score_reasoning=(result.get("reasoning") or "")[:2000],
        )
        print(f"           → saved (id={db_id})\n")
    except Exception as e:
        print(f"           ⚠ DB save failed (non-fatal): {e}\n")

    # ── Step 7: Upload (with custom thumbnail) ─────────────────────────────────
    # Quality gate: only auto-upload videos whose script cleared the bar. A weak
    # script never reaches YouTube — it's saved locally for review instead.
    yt_url = None
    yt_id  = None   # guard: upload() may not be reached if quality gate holds
    min_score = int(os.environ.get("RUFUS_MIN_UPLOAD_SCORE", "8"))
    final_score = result.get("score", 0)
    if skip_upload:
        print("[ 7 / 7 ]  Upload skipped (--skip-upload)\n")
    elif final_score < min_score:
        print(f"[ 7 / 7 ]  Upload held — score {final_score}/10 < {min_score}/10 threshold.")
        print(f"           Video saved for review: {output_path}\n")
    else:
        print(f"[ 7 / 7 ]  Score {final_score}/10 ≥ {min_score} — generating thumbnail + uploading...")
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
            yt_url = None
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
    parser.add_argument("--output-dir",  type=str,            help="Directory to write rendered mp4 files (overrides RUFUS_OUTPUT_DIR env var)")
    args = parser.parse_args()

    if sum(bool(x) for x in (args.niche, args.scheduled, args.rotate)) > 1:
        print("Use only one of --niche, --scheduled, --rotate")
        sys.exit(1)

    out_dir_arg = Path(args.output_dir) if args.output_dir else None

    if args.rotate:
        niches = _all_scheduled_niches()
        print(f"\n[rotate] producing {len(niches)} video(s): {niches}\n")
        for n in niches:
            # Clear any prior env override so each iteration starts clean
            os.environ.pop("RUFUS_NICHE_OVERRIDE", None)
            run(skip_upload=args.skip_upload, niche_override=n, output_dir=out_dir_arg)
    elif args.scheduled:
        n = _todays_niche()
        print(f"\n[scheduled] today's niche: {n}\n")
        run(skip_upload=args.skip_upload, niche_override=n, output_dir=out_dir_arg)
    else:
        run(skip_upload=args.skip_upload, niche_override=args.niche, output_dir=out_dir_arg)
