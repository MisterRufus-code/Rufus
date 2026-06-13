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
LOCK_FILE   = ROOT / "rufus.lock"


# ── Single-instance lock (cron overlap protection) ───────────────────────────────

def _acquire_lock() -> None:
    """Refuse to start if another Rufus run is alive (overlapping cron + manual
    runs corrupt temp files and double-write the DB). Stale locks self-clear."""
    if LOCK_FILE.exists():
        try:
            pid = int(LOCK_FILE.read_text().strip())
        except (ValueError, OSError):
            pid = -1
        if pid > 0 and pid != os.getpid():
            try:
                os.kill(pid, 0)   # signal 0 = existence check only
                print(f"ERROR: another Rufus run (pid {pid}) is in progress. "
                      f"Wait for it, or delete {LOCK_FILE} if it crashed.")
                sys.exit(1)
            except ProcessLookupError:
                pass              # stale lock from a dead process — take over
            except PermissionError:
                print(f"ERROR: pid {pid} exists (no permission to signal) — assuming live run.")
                sys.exit(1)
    LOCK_FILE.write_text(str(os.getpid()))


def _release_lock() -> None:
    try:
        if LOCK_FILE.exists() and LOCK_FILE.read_text().strip() == str(os.getpid()):
            LOCK_FILE.unlink()
    except OSError:
        pass


# ── Housekeeping (disk + logs never grow unbounded) ──────────────────────────────

def _housekeeping(max_log_days: int = 90, max_cache_days: int = 14) -> None:
    """Delete old logs and stale cache/temp media. Cheap, runs every start."""
    cutoff_logs  = time.time() - max_log_days * 86400
    cutoff_cache = time.time() - max_cache_days * 86400
    removed = 0
    for d, cutoff in (
        (LOG_DIR, cutoff_logs),
        (LOG_DIR / "scripts", cutoff_logs),
        (ROOT / "media_library" / "cache", cutoff_cache),
        (ROOT / "media_library" / "temp", cutoff_cache),
        (ROOT / "media_library" / "music", cutoff_cache),
    ):
        if not d.exists():
            continue
        for f in d.rglob("*"):
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                continue
    if removed:
        print(f"[maint] cleaned {removed} stale file(s)")


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


def _build_sd_prompts(script: str, niche: str, n: int = 4) -> list[str]:
    """Generate n visually-distinct SD prompts that progress through the script arc.

    Each of the n scenes is anchored to a different camera distance, location,
    and time of day so consecutive clips never look the same.
    """
    import re

    # Hard visual anchors per scene slot — GPT must honour these.
    # They rotate if n > 4 so every slot stays unique.
    ANCHORS = [
        ("EXTREME CLOSE-UP",  "tight detail shot, face/hands/object",    "high-contrast dramatic light"),
        ("WIDE ESTABLISHING", "outdoor or large interior, no face needed","golden hour or overcast sky"),
        ("MEDIUM SHOT",       "person mid-frame, waist-up, in action",    "side or rim lighting"),
        ("AERIAL / ABSTRACT", "overhead view or symbolic still-life",     "cool blue or dark moody tones"),
    ]

    anchor_lines = "\n".join(
        f"  Scene {i+1}: camera={ANCHORS[i % 4][0]}, subject={ANCHORS[i % 4][1]}, "
        f"lighting={ANCHORS[i % 4][2]}"
        for i in range(n)
    )

    try:
        from openai import OpenAI
        keys_file = CONFIG_DIR / "keys.json"
        if keys_file.exists():
            key = json.loads(keys_file.read_text()).get("openai", "")
            if key and not key.startswith("YOUR_") and not key.startswith("FILL_"):
                client = OpenAI(api_key=key)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Create {n} Stable Diffusion image prompts for a YouTube Short.\n"
                            f"Niche: {niche}\n\n"
                            f"Script (for story context):\n{script}\n\n"
                            f"MANDATORY scene constraints — you MUST follow these exactly:\n"
                            f"{anchor_lines}\n\n"
                            "Rules:\n"
                            "- Each scene MUST use a DIFFERENT physical location (no two scenes in the same room/place).\n"
                            "- No two scenes can share the same camera distance.\n"
                            "- Each description: 15-25 words, concrete subject, concrete setting, specific lighting.\n"
                            f"- Output ONLY {n} plain lines, no numbering, no labels, nothing else."
                        ),
                    }],
                    max_tokens=400,
                    temperature=0.8,
                )
                raw_lines = resp.choices[0].message.content.strip().split("\n")
                lines = [
                    re.sub(r"^[\d\.\-\)\s]+", "", l).strip()
                    for l in raw_lines if l.strip()
                ]
                lines = [l for l in lines if len(l) > 10]

                # If GPT returned fewer than n, fill with anchor-driven fallbacks
                # (never cycle the same prompt — each fallback uses a unique anchor).
                while len(lines) < n:
                    i = len(lines)
                    cam, subj, light = ANCHORS[i % 4]
                    lines.append(
                        f"{niche} scene, {cam.lower()}, {subj}, {light}, "
                        "photorealistic, cinematic"
                    )
                return lines[:n]
    except Exception as e:
        print(f"[sd] GPT prompt generation skipped ({e}) — using anchor fallback")

    # Key-free fallback: anchor-driven prompts (always visually distinct).
    sentences = [s.strip() for s in re.split(r"[.!?]", script) if len(s.strip()) > 15]
    prompts = []
    for i in range(n):
        cam, subj, light = ANCHORS[i % 4]
        if sentences:
            idx = int(i * len(sentences) / n)
            cue = sentences[idx][:60]
        else:
            cue = f"{niche} concept"
        prompts.append(
            f"{cue}, {cam.lower()}, {subj}, {light}, photorealistic, cinematic"
        )
    return prompts


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


def run(skip_upload: bool = False, niche_override: str = None, output_dir: Path = None,
        channel_id: str = None):
    _acquire_lock()
    import atexit
    atexit.register(_release_lock)   # release on any exit path (idempotent)

    # Channel resolution (channel-in-a-box). Legacy installs without
    # channels.json get a synthesized "main_en" channel — behavior unchanged.
    from channel_config import load_channel
    channel = load_channel(channel_id)
    os.environ["RUFUS_CHANNEL"] = channel.id          # sub-modules inherit it
    if channel.voice:
        os.environ.setdefault("RUFUS_EDGE_VOICE", channel.voice)

    log_path = _enable_file_logging()
    _housekeeping()
    start    = time.time()
    niche_cfg, active = load_niche_cfg(niche_override)
    niche_cfg = {**niche_cfg, **channel.niche_overrides.get(active, {})}
    out_dir  = output_dir or channel.output_dir

    print(f"\n{'='*52}")
    print(f"  RUFUS  |  channel: {channel.id}  |  niche: {active}  |  "
          f"{time.strftime('%Y-%m-%d %H:%M:%S')}")
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
        elif seed.get("type") == "stackexchange":
            print(f"           → SE:     {seed.get('title', '')[:80]}")
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

    # Source resolution: explicit env > per-niche config > default "sd".
    # RUFUS_VIDEO_SOURCE=sd          → Stable Diffusion stills + Ken Burns (GPU)
    # RUFUS_VIDEO_SOURCE=hyperframes → HTML motion-graphics via HyperFrames (CPU)
    # RUFUS_VIDEO_SOURCE=hybrid      → SD photo + HyperFrames CSS overlay (best quality)
    # RUFUS_VIDEO_SOURCE=pexels      → Pexels stock footage
    video_source  = (os.environ.get("RUFUS_VIDEO_SOURCE")
                     or niche_cfg.get("video_source") or "sd").strip().lower()
    video_queries = _parse_video_queries(seed_analysis)

    # sd, hyperframes, and hybrid all GENERATE clips from the script, so they defer
    # to step 2.5 (after the script exists). Only stock sources fetch up front.
    DEFERRED_SOURCES = ("sd", "hyperframes", "hybrid")

    # ── Step 2: Get candidate clips — generated (SD/HyperFrames) or stock ───────
    candidates = []
    scene = ""
    if video_source in DEFERRED_SOURCES:
        # Clips are generated AFTER the script is written (step 2.5) so they can
        # be tailored to the actual content. Placeholder scene gives write_script
        # context; candidates are filled in later.
        scene = niche_cfg.get("llava_context", f"{active} scene")
        print(f"[ 2 / 7 ]  {video_source} mode — clip generation deferred until after scripting\n")

    if video_source not in DEFERRED_SOURCES:
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
    if video_source in DEFERRED_SOURCES:
        print(f"[ 3 / 7 ]  {video_source} mode — skipping vision pick\n")
    elif scene:
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

    # ── Step 2.5: Generate script-matched clips (SD / HyperFrames / Hybrid) ─────
    # Fallback chain so a render never dies:
    #   hybrid     → SD images + HF CSS overlay → Ken Burns on SD images → SD → pexels
    #   hyperframes → sd → pexels
    #   sd         → pexels
    if video_source in DEFERRED_SOURCES:
        print(f"[ 2.5/7 ]  Generating {video_source} clips from script content...")
        try:
            n_clips = int(os.environ.get("SD_CLIPS", "4"))
            prompts = _build_sd_prompts(script, active, n=n_clips)
            print(f"           → prompts: {prompts}")
            # Pass niche name so HF prompts can use niche-specific visual guides
            niche_cfg_tagged = {**niche_cfg, "name": active}

            if video_source == "hybrid":
                import sd_client as _sd
                import hyperframes_client as _hf
                from sd_client import (generate_images as sd_generate_images,
                                       generate_clips_from_images as sd_animate)
                from hyperframes_client import generate_clips as hf_generate

                # Step A: SD base images (raw PNGs, no animation)
                sd_imgs = sd_generate_images(prompts, n=n_clips) if _sd.is_available() else []
                if sd_imgs:
                    print(f"           → {len(sd_imgs)} SD base images ready for hybrid")
                else:
                    print("           ⚠ SD not running — HyperFrames will use pure CSS")

                # Step B: HyperFrames renders (hybrid if SD images exist, else pure CSS)
                if _hf.is_available():
                    candidates = hf_generate(prompts, n=n_clips, clip_duration=8.0,
                                             niche_cfg=niche_cfg_tagged,
                                             image_paths=sd_imgs or None)
                    if candidates:
                        mode  = "hybrid SD+HF" if sd_imgs else "HF css-only"
                        scene = f"{mode}: " + "; ".join(prompts[:2])
                        print(f"           → {len(candidates)} {mode} clips ready\n")

                # Step C: HF unavailable but SD images exist → Ken Burns fallback
                if not candidates and sd_imgs:
                    print("           ⚠ HyperFrames unavailable — Ken Burns on SD images")
                    candidates = sd_animate(sd_imgs, clip_duration=8.0)
                    if candidates:
                        scene = "SD Ken Burns: " + "; ".join(prompts[:2])
                        print(f"           → {len(candidates)} clips ready\n")

                if not candidates:
                    print("           ⚠ hybrid failed — trying full SD pipeline")
                    video_source = "sd"  # fall through to the SD block below

            if video_source == "hyperframes":
                from hyperframes_client import generate_clips as hf_generate
                candidates = hf_generate(prompts, n=n_clips,
                                         clip_duration=8.0, niche_cfg=niche_cfg_tagged)
                if candidates:
                    scene = "HyperFrames motion-graphic: " + "; ".join(prompts[:2])
                    print(f"           → {len(candidates)} clips ready\n")
                else:
                    print("           ⚠ HyperFrames produced nothing — trying SD")
                    video_source = "sd"   # fall through to the SD block below

            if video_source == "sd":
                from sd_client import generate_clips as sd_generate
                candidates = sd_generate(prompts, n=n_clips)
                if candidates:
                    scene = "SD-generated: " + "; ".join(prompts[:2])
                    print(f"           → {len(candidates)} clips ready\n")
                else:
                    print("           ⚠ SD failed — falling back to Pexels")
                    video_source = "pexels"

            if not candidates:
                if video_queries:
                    print(f"           → using script queries: {video_queries}")
                candidates = fetch_candidates(n=5, extra_keywords=video_queries or None)
                video_path, scene = pick_best_video(
                    candidates, niche_cfg["llava_context"],
                    seed=seed, analysis=seed_analysis or None,
                )
                print(f"           → Pexels fallback: {video_path.name}\n")
        except Exception as e:
            print(f"           ✗ Clip generation failed: {e}")
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
            channel=channel.id,
        )
        print(f"           → saved (id={db_id})\n")
    except Exception as e:
        print(f"           ⚠ DB save failed (non-fatal): {e}\n")

    # ── Step 7: Upload (with custom thumbnail) ─────────────────────────────────
    # Quality gate: only auto-upload videos whose script cleared the bar. A weak
    # script never reaches YouTube — it's saved locally for review instead.
    yt_url = None
    yt_id  = None   # guard: upload() may not be reached if quality gate holds
    min_score = int(os.environ.get("RUFUS_MIN_UPLOAD_SCORE",
                                   str(channel.upload.get("min_score", 8))))
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
            from youtube_uploader import upload, build_metadata

            thumb = None
            try:
                thumb = make_thumbnail(output_path, script)
                print(f"           thumbnail: {thumb.name}")
            except Exception as e:
                print(f"           ⚠ thumbnail generation skipped: {e}")

            # GPT title/description once here, persisted for CTR learning
            meta = None
            try:
                meta = build_metadata(script, active, niche_cfg)
                if db_id and meta.get("title"):
                    from db_manager import update_title
                    update_title(db_id, meta["title"])
            except Exception as e:
                print(f"           ⚠ metadata pre-build failed (uploader will retry): {e}")

            yt_url, yt_id = upload(output_path, script, thumbnail_path=thumb, metadata=meta)
            print(f"           → {yt_url}\n")

            if db_id and yt_id:
                try:
                    update_youtube_id(db_id, yt_id)
                except Exception as e:
                    print(f"           ⚠ DB youtube_id update failed (video IS uploaded): {e}")
        except Exception as e:
            yt_url = None
            print(f"           ✗ Upload failed: {e}")
            print(f"           Video saved locally: {output_path} — check YouTube "
                  f"Studio before re-uploading (may have partially gone through)\n")
            if db_id:
                try:
                    from db_manager import mark_upload_failed
                    mark_upload_failed(db_id, str(e))
                except Exception:
                    pass

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
    parser.add_argument("--scheduled",   action="store_true", help="Use today's niche from the channel/config schedule (for cron)")
    parser.add_argument("--rotate",      action="store_true", help="Run one video per unique niche in the schedule")
    parser.add_argument("--output-dir",  type=str,            help="Directory to write rendered mp4 files (overrides RUFUS_OUTPUT_DIR env var)")
    parser.add_argument("--channel",     type=str,            help="Channel id from config/channels.json (default: default_channel / legacy)")
    args = parser.parse_args()

    if sum(bool(x) for x in (args.niche, args.scheduled, args.rotate)) > 1:
        print("Use only one of --niche, --scheduled, --rotate")
        sys.exit(1)

    out_dir_arg = Path(args.output_dir) if args.output_dir else None

    # Channel schedule (if defined) takes precedence over niches.json schedule.
    def _channel_schedule() -> list[str]:
        try:
            from channel_config import load_channel
            ch = load_channel(args.channel)
            if ch.schedule:
                return ch.schedule
        except Exception:
            pass
        data = json.loads(NICHES_FILE.read_text())
        return data.get("schedule") or [data.get("active", "finance")]

    if args.rotate:
        seen: list[str] = []
        for n in _channel_schedule():
            if n not in seen:
                seen.append(n)
        print(f"\n[rotate] producing {len(seen)} video(s): {seen}\n")
        for n in seen:
            # Clear any prior env override so each iteration starts clean
            os.environ.pop("RUFUS_NICHE_OVERRIDE", None)
            run(skip_upload=args.skip_upload, niche_override=n,
                output_dir=out_dir_arg, channel_id=args.channel)
    elif args.scheduled:
        from datetime import datetime
        schedule = _channel_schedule()
        doy      = datetime.now().timetuple().tm_yday
        n        = schedule[(doy - 1) % len(schedule)]
        print(f"\n[scheduled] today's niche: {n}\n")
        run(skip_upload=args.skip_upload, niche_override=n,
            output_dir=out_dir_arg, channel_id=args.channel)
    else:
        run(skip_upload=args.skip_upload, niche_override=args.niche,
            output_dir=out_dir_arg, channel_id=args.channel)
